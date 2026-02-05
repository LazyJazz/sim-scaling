import torch
import numpy as np
import time

class BaseManager:
    def __init__(self, env, policy, num_iter=2000, **kargs):
        self.env = env
        self.policy = policy
        self.num_iter = num_iter
        self.iter = 0
        self.last_time = time.time()
        self.duration = 0.0

    def step(self, obs, action, step_result=False):
        if step_result:
            self.iter += 1
        self.duration = time.time() - self.last_time
        self.last_time = time.time()

    def should_terminate(self):
        return self.iter > self.num_iter
    
    def __repr__(self):
        # f"Iter.{self.iter}/{self.num_iter}: success_count: {self.env.success_count}, done_count: {self.env.done_count}, success_rate: {self.env.get_success_rate():.3f}"
        result = f"Iter.{self.iter}/{self.num_iter}"
        if hasattr(self.env, 'success_count'):
            result += f": success_count: {self.env.success_count}, done_count: {self.env.done_count}, success_rate: {self.env.get_success_rate():.3f}"
        return result

    def close(self):
        pass