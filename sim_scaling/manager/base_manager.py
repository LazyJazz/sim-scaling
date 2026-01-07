import torch
import numpy as np
import sim_scaling.policy.base_policy
import sim_scaling.task.base_env

class BaseManager:
    def __init__(self, env: sim_scaling.task.base_env.BaseEnv, policy: sim_scaling.policy.base_policy.BasePolicy, num_iter=2000, **kargs):
        self.env = env
        self.env: sim_scaling.task.base_env.BaseEnv
        self.policy = policy
        self.policy: sim_scaling.policy.base_policy.BasePolicy
        self.num_iter = num_iter
        self.iter = 0

    def step(self, obs, action):
        self.iter += 1

    def should_terminate(self):
        return self.iter > self.num_iter