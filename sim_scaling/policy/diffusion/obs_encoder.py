"""
ResNet18 backbone with SpatialSoftmax replacing the final pooling.

This module provides:
- SpatialSoftmax: converts feature maps (N, C, H, W) into keypoint coordinates
  (N, 2*C) by treating each channel as a spatial probability distribution.
- ResNet18SpatialSoftmax: a ResNet-18 backbone (from torchvision) where the
  final avgpool + fc are removed and replaced with SpatialSoftmax. Optionally
  a 1x1 conv can reduce the channels to `num_keypoints` before spatial softmax.

Run this file directly for a quick smoke test.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18
import torchvision.models


class SpatialSoftmax(nn.Module):
	"""Spatial softmax as a module.

	Given input of shape (N, C, H, W) it computes softmax over H*W for each
	channel, then returns the expected 2D coordinates (x, y) per channel, so
	output shape is (N, 2*C). Coordinates are in range [-1, 1], where -1 is
	left/top and +1 is right/bottom.
	"""

	def __init__(self, height: Optional[int] = None, width: Optional[int] = None, temperature: Optional[float] = 1.0, learnable_temperature: bool = False):
		super().__init__()
		self.height = height
		self.width = width
		self.register_buffer("pos_x", torch.tensor([]))
		self.register_buffer("pos_y", torch.tensor([]))
		if temperature is None:
			temperature = 1.0
		if learnable_temperature:
			self.temperature = nn.Parameter(torch.tensor(float(temperature)))
		else:
			self.register_buffer("temperature", torch.tensor(float(temperature)))
		self._create_mesh(5, 5, device=None, dtype=torch.float32)

	def _create_mesh(self, h: int, w: int, device: torch.device, dtype: torch.dtype):
		# x: -1 -> 1 horizontally, y: -1 -> 1 vertically
		pos_x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
		pos_y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
		# meshgrid with indexing='xy'
		px, py = torch.meshgrid(pos_x, pos_y, indexing="xy")
		# px, py shapes are (w, h) due to the ordering; transpose to (h, w)
		px = px.t().contiguous()  # (h, w)
		py = py.t().contiguous()  # (h, w)
		# flatten to (H*W)
		self.pos_x = px.reshape(-1)
		self.pos_y = py.reshape(-1)
		self.height = h
		self.width = w

	def forward(self, feature: torch.Tensor) -> torch.Tensor:
		# feature: (N, C, H, W)
		assert feature.dim() == 4, "SpatialSoftmax expects a 4D tensor"
		N, C, H, W = feature.shape
		if self.pos_x.numel() == 0 or self.height != H or self.width != W:
			self._create_mesh(H, W, device=feature.device, dtype=feature.dtype)

		# Flatten spatial dimensions
		features = feature.view(N, C, H * W)
		# temperature may be a parameter or buffer
		temp = self.temperature if hasattr(self, "temperature") else self.temperature
		# safe divide; ensure temp is scalar tensor
		t = temp if torch.is_tensor(temp) else torch.tensor(float(temp), device=feature.device, dtype=feature.dtype)
		# apply softmax over spatial dim
		softmax_attention = F.softmax(features / t, dim=2)  # (N, C, H*W)

		# expected x/y: sum over spatial locations of position * attention
		# pos_x/pos_y shape (H*W,)
		pos_x = self.pos_x.unsqueeze(0).to(feature.device).to(feature.dtype)  # (1, H*W)
		pos_y = self.pos_y.unsqueeze(0).to(feature.device).to(feature.dtype)

		exp_x = torch.sum(softmax_attention * pos_x.unsqueeze(1), dim=2)  # (N, C)
		exp_y = torch.sum(softmax_attention * pos_y.unsqueeze(1), dim=2)  # (N, C)

		# concatenate x and y for each channel -> (N, 2*C)
		keypoints = torch.cat([exp_x, exp_y], dim=2 - 1) if False else torch.cat([exp_x, exp_y], dim=1)
		# keypoints shape: (N, 2*C)
		return keypoints


class ResNet18SpatialSoftmax(nn.Module):
	"""ResNet18 backbone with SpatialSoftmax replacing avgpool+fc.

	Args:
		num_keypoints: if provided, a 1x1 conv reduces the final channels (512)
			to `num_keypoints` before SpatialSoftmax. Output shape will be
			(N, 2 * num_keypoints). If None, SpatialSoftmax acts on all 512
			channels and output shape is (N, 2 * 512).
		pretrained: whether to attempt to load torchvision pretrained weights.
		temperature, learnable_temperature: passed to SpatialSoftmax.
	"""

	def __init__(self, num_keypoints: Optional[int] = None, pretrained: bool = False, temperature: Optional[float] = 1.0, learnable_temperature: bool = False):
		super().__init__()
		# Import torchvision lazily so module still imports when torchvision
		# not available (but will raise if user tries to instantiate without it).
			
		base = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None)

		# remove avgpool and fc -> keep up to the last conv layer
		# children() ordering: conv1, bn1, relu, maxpool, layer1..layer4, avgpool, fc
		modules = list(base.children())[:-2]
		self.backbone = nn.Sequential(*modules)

		self.reduce_channels = None
		if num_keypoints is not None:
			# 1x1 conv to reduce 512 -> num_keypoints
			self.reduce_channels = nn.Conv2d(512, num_keypoints, kernel_size=1)
			spatial_channels = num_keypoints
		else:
			spatial_channels = 512

		self.spatial_softmax = SpatialSoftmax(temperature=temperature, learnable_temperature=learnable_temperature)
		self._spatial_channels = spatial_channels

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Expect input x: (N, 3, H, W)
		features = self.backbone(x)  # (N, 512, H', W')
		if self.reduce_channels is not None:
			features = self.reduce_channels(features)  # (N, K, H', W')
		kp = self.spatial_softmax(features)  # (N, 2 * channels)
		return kp


__all__ = ["SpatialSoftmax", "ResNet18SpatialSoftmax"]


if __name__ == "__main__":
	# Quick smoke test
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = ResNet18SpatialSoftmax(num_keypoints=64, pretrained=True).to(device)
	# dummy batch (N=2, 3 channels, 160x160)
	x = torch.randn(2, 3, 160, 160, device=device)
	out = model(x)
	print("Output shape:", out.shape)
	print("Output:", out)
	# Expect (2, 32) when num_keypoints=16

