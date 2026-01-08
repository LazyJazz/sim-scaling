import sim_scaling.policy.base_policy
import torch
import numpy as np
from sim_scaling.policy.diffusion.policy import DiffusionPolicy as DiffusionPolicyImpl

class DiffusionPolicy(sim_scaling.policy.base_policy.BasePolicy):
    def __init__(self, ckpt_path: str, device, **kargs):
        super().__init__(**kargs)
        self.device = device
        self.ckpt_path = ckpt_path
        print(f"Loading DiffusionPolicy from checkpoint: {ckpt_path}, device: {device}")
        self.policy = DiffusionPolicyImpl(device=device, **kargs)
        self.policy.load_state_dict(torch.load(ckpt_path, map_location=device))
        torch.set_float32_matmul_precision('high')
        self.policy = torch.compile(self.policy)
        self.policy.eval()

        self.imgs = None
        self.head_pose = None

    def get_action(self, obs):
        img = obs['rgb']
        img: torch.Tensor
        img = img.permute(0, 3, 1, 2)  # (B, 160, 160, 3) -> (B, 3, 160, 160)
        head_pose = obs['head_pose'][:, :3]
        if self.imgs is None:
            self.imgs = [img] * self.policy.n_obs
            self.head_pose = [head_pose] * self.policy.n_obs
        else:
            self.imgs = self.imgs[1:] + [img]
            self.head_pose = self.head_pose[1:] + [head_pose]
        obs_imgs = torch.stack(self.imgs, dim=1)  # (B, n_obs, 3, 160, 160)
        obs_head_pose = torch.stack(self.head_pose, dim=1)  # (B, n_obs, 7)
        obs_imgs = obs_imgs.float()
        action = obs['head_pose'].clone()
        with torch.no_grad():
            action[:, :3] = self.policy.predict(obs_imgs, obs_head_pose)[:, 0, ...]
        action[:, 3:] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)

        return action
    