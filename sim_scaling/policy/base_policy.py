import torch
import numpy as np

class BasePolicy:
    def __init__(self, **kargs):
        pass

    def get_action(self, obs):
        head_pose = obs["head_pose"]
        print(f"Head pose: {head_pose}")
        return head_pose.clone()