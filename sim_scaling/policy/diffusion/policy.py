from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from sim_scaling.policy.diffusion.obs_encoder import ResNet18SpatialSoftmax
from sim_scaling.policy.diffusion.conditional_unet1d import ConditionalUnet1D
from sim_scaling.policy.diffusion.linear_normalizer import LinearNormalizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class DiffusionPolicy(nn.Module):
    def __init__(self, device, n_obs=2, n_actions=8, num_train_timesteps=100, num_inference_timesteps=None, pretrained=False):
        super(DiffusionPolicy, self).__init__()
        self.device = device
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps if num_inference_timesteps is not None else num_train_timesteps
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            variance_type="fixed_small",
            clip_sample=True
        )
        self.encoder = ResNet18SpatialSoftmax(num_keypoints=64, pretrained=pretrained).to(device)
        self.model = ConditionalUnet1D(
            input_dim=3,
            local_cond_dim=None,
            global_cond_dim=128 * self.n_obs,
            down_dims=[128, 256, 512],
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=True
        ).to(device)
        self.image_normalizer = LinearNormalizer(min_val=0.0, max_val=255.0)
        self.action_normalizer = LinearNormalizer(min_val=-0.1, max_val=0.1)

    def conditional_sample(self, global_cond: torch.Tensor):
        batch_size = global_cond.shape[0]
        trajectory = torch.randn(
            size=(batch_size, self.n_actions, 3), 
            dtype=torch.float32,
            device=global_cond.device)
        
        scheduler = self.scheduler
        scheduler.set_timesteps(self.num_inference_timesteps)

        for t in scheduler.timesteps:
            model_output = self.model(trajectory, t.expand(batch_size).to(global_cond.device), global_cond=global_cond)
            trajectory = scheduler.step(model_output, t, trajectory).prev_sample

        return trajectory

    def predict(self, obs, agent_pos):
        normalized_images = self.image_normalizer.normalize(obs)
        global_cond = self.encoder(normalized_images.reshape(-1, 3, 160, 160))
        global_cond = global_cond.reshape(obs.shape[0], -1)  # (B, global_cond_dim)
        actions = self.conditional_sample(global_cond)
        actions = self.action_normalizer.denormalize(actions)
        actions += agent_pos[:, -1:, :]
        return actions

    def compute_loss(self, obs, agent_pos, action) -> torch.Tensor:
        normalized_images = self.image_normalizer.normalize(obs)
        action -= agent_pos[:, -1:, :]
        normalized_action = self.action_normalizer.normalize(action)

        global_cond = self.encoder(normalized_images.reshape(-1, 3, 160, 160))
        global_cond = global_cond.reshape(obs.shape[0], -1)  # (B, global_cond_dim)
        
        batch_size = obs.shape[0]
        noise = torch.randn_like(normalized_action, device=self.device)
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=self.device).long()
        noisy_actions = self.scheduler.add_noise(normalized_action, noise, timesteps)

        pred = self.model(noisy_actions, timesteps, global_cond=global_cond)
        # print("noisy_actions shape:", noisy_actions.shape, " dtype:", noisy_actions.dtype)
        # print("pred shape:", pred.shape, " dtype:", pred.dtype)

        loss = F.mse_loss(pred, noise)

        # print("Computed loss:", loss.item())
        
        return loss