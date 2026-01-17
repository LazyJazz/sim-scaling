import time
import torch
import numpy as np
import sim_scaling.policy.base_policy
import sim_scaling.task.base_env
import sim_scaling.util.zarr_util

class DataGenManager(sim_scaling.manager.base_manager.BaseManager):
    def __init__(self, path: str, succ_traj: int, **kargs):
        super().__init__(**kargs)
        self.recorder = sim_scaling.util.zarr_util.ZarrRecorder(path)
        self.succ_traj = succ_traj

    def step(self, obs, action):
        super().step(obs, action)
        self.recorder.record_frame(obs, action)

    def should_terminate(self):
        return self.env.success_count >= self.succ_traj

    def close(self):
        self.recorder.close()
    
    def __repr__(self):
        return f"Iter.{self.iter}: success_count: {self.env.success_count}/{self.succ_traj}, done_count: {self.env.done_count}, success_rate: {self.env.get_success_rate():.3f}, step_duration: {self.duration:.3f}s"