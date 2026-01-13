import torch
import numpy as np
import zarr
from torch.utils.data import Dataset
import hydra
from omegaconf import OmegaConf 

class ImageDataset(Dataset):
    def __init__(self, path: str, success_only: bool = True, n_obs=2, n_actions=8):
        self.success_only = success_only
        self.z_handle = zarr.open(path, mode='r')
        self.image_rgb = self.z_handle['rgb']
        self.done = self.z_handle['done']
        self.head_pose = self.z_handle['head_pose']
        self.action = self.z_handle['action']
        self.success = self.z_handle['success']
        self.num_steps = self.z_handle['num_steps']
        
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.horizon = self.n_obs + self.n_actions - 1

        done = self.done[:]
        success = self.success[:]
        done_idx = np.where(done)
        done_idx = list(done_idx)
        done_idx = np.stack(done_idx, axis=1)
        num_steps = self.num_steps[done_idx[:,0], done_idx[:,1]]
        self.indices = []
        for i in range(len(done_idx)):
            env_id = done_idx[i, 0]
            step_id = done_idx[i, 1]
            if success_only and success[env_id, step_id] == 0:
                continue
            num_steps_i = num_steps[i]
            for start_step in range(self.n_obs - 1, num_steps_i - self.n_actions + 2):
                    self.indices.append((env_id, start_step + step_id - num_steps_i))
        
        self.indices = np.array(self.indices)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        env_id, step_id = self.indices[idx]
        images = self.image_rgb[env_id, step_id - self.n_obs + 1: step_id + 1]
        head_poses = self.head_pose[env_id, step_id - self.n_obs + 1: step_id + 1]
        actions = self.action[env_id, step_id: step_id + self.n_actions]
        # print(f"images shape: {images.shape}, head_poses shape: {head_poses.shape}, actions shape: {actions.shape}")
        return images, head_poses[:, :3], actions[:, :3]

class MixDataset(Dataset):
    def __init__(self, dataset1: OmegaConf, dataset2: OmegaConf, ratio: float, length: int, **kargs):
        self.dataset1_cls = hydra.utils.get_class(dataset1.__target__)
        self.dataset1 = self.dataset1_cls(**dataset1.args)
        self.dataset2_cls = hydra.utils.get_class(dataset2.__target__)
        self.dataset2 = self.dataset2_cls(**dataset2.args)
        self.ratio = ratio
        self.length = length
        self.length1 = int(self.length * self.ratio)
        self.length2 = self.length - self.length1
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx < self.length1:
            idx1 = idx % len(self.dataset1)
            return self.dataset1[idx1]
        else:
            idx2 = (idx - self.length1) % len(self.dataset2)
            return self.dataset2[idx2]

class ConcatDataset(Dataset):
    def __init__(self, dataset1: OmegaConf, dataset2: OmegaConf, **kargs):
        self.dataset1_cls = hydra.utils.get_class(dataset1.__target__)
        self.dataset1 = self.dataset1_cls(**dataset1.args)
        self.dataset2_cls = hydra.utils.get_class(dataset2.__target__)
        self.dataset2 = self.dataset2_cls(**dataset2.args)
        
    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)
    
    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        else:
            idx2 = idx - len(self.dataset1)
            return self.dataset2[idx2]

if __name__ == "__main__":
    dataset = ImageDataset("./data/pusht_rt4000.zarr")