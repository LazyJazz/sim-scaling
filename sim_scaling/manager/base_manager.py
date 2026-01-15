import torch
import numpy as np
import sim_scaling.policy.base_policy
import sim_scaling.task.base_env
import time

class BaseManager:
    def __init__(self, env: sim_scaling.task.base_env.BaseEnv, policy: sim_scaling.policy.base_policy.BasePolicy, num_iter=2000, **kargs):
        self.env = env
        self.env: sim_scaling.task.base_env.BaseEnv
        self.policy = policy
        self.policy: sim_scaling.policy.base_policy.BasePolicy
        self.num_iter = num_iter
        self.iter = 0
        self.last_time = time.time()
        self.duration = 0.0

    def step(self, obs, action):
        self.iter += 1
        self.duration = time.time() - self.last_time
        self.last_time = time.time()

    def should_terminate(self):
        return self.iter > self.num_iter
    
    def __repr__(self):
        return f"Iter.{self.iter}/{self.num_iter}: success_count: {self.env.success_count}, done_count: {self.env.done_count}, success_rate: {self.env.get_success_rate():.3f}"