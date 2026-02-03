from sim_scaling.policy.base_policy import BasePolicy
import pygame
import torch
import numpy as np

class JoystickPolicy(BasePolicy):
    def __init__(self, joystick_id=0, **kargs):
        super().__init__(**kargs)
        pygame.init()
        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(joystick_id)
        self.joystick.init()

    def get_action(self, obs):
        pygame.event.pump()  # Process event queue

        # Assuming the joystick has at least 2 axes for movement
        axis_0 = self.joystick.get_axis(0)  # Left/Right
        axis_1 = self.joystick.get_axis(1)  # Forward/Backward
        axis_2 = (self.joystick.get_axis(2) + 1.0) * 0.5
        axis_5 = (self.joystick.get_axis(5) + 1.0) * 0.5
        
        move_vel = np.array([0.0, 0.0, 0.0])
        move_vel[1] = axis_0  # Scale to max 0.1 m/s
        move_vel[0] = axis_1  # Invert Y axis
        move_vel[2] = (axis_5 - axis_2)  # Scale to max 0.1 m/s
        move_vel *= 0.05

        if np.linalg.norm(move_vel) < 0.01:
            move_vel = np.array([0.0, 0.0, 0.0])

        # Create an action tensor based on joystick input
        targ_pose = obs['head_pose'].clone()
        action = torch.tensor(move_vel, dtype=torch.float32).to(targ_pose.device)
        targ_pose[..., :3] += action
        targ_pose[..., 3:] = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)  # No rotation change

        return targ_pose