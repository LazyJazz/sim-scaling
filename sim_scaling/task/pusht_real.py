import cv2
import pyrealsense2 as rs
import torch
import numpy as np
import matplotlib.pyplot as plt
from sim_scaling.util.franka_client import FrankaClient
import time
import pygame

class PushTRealEnv:
    def __init__(self, controller_ip="localhost", controller_port=8765, device="cuda"):
        self.device = torch.device(device)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        self.profile = self.pipeline.start(self.config)
        self.rs_device = self.profile.get_device()
        color_sensor = None
        for s in self.rs_device.query_sensors():
            name = s.get_info(rs.camera_info.name).lower()
            if "rgb" in name or "color" in name:
                color_sensor = s
                break

        if color_sensor is not None:
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            if color_sensor.supports(rs.option.enable_auto_white_balance):
                color_sensor.set_option(rs.option.enable_auto_white_balance, 1)

        for _ in range(30):
            self.pipeline.wait_for_frames()

        self.franka_client = FrankaClient(controller_ip, controller_port)

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(np.zeros((160, 160, 3), dtype=np.uint8))
        self.ax.axis('off')
        self.ax.set_title('RealSense RGB Stream')
        self.fig.canvas.draw()
        plt.show(block=False)
        self.done = False
        self.success = False
        self.pause = True

        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        self.last_tp = time.time()
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.done_count = 0
        self.success_count = 0
        self.num_steps = 0
    
    def get_success_rate(self):
        return self.success_count / max(1, self.done_count)

    def get_observations(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        x = 80
        y = 0
        width_height = 480
        color_image = color_image[y:y+width_height, x:x+width_height, :]

        # stretch and resize to 160x160
        color_image = cv2.resize(color_image, (160, 160), interpolation=cv2.INTER_LINEAR)

        color_image = torch.from_numpy(color_image).unsqueeze(0)

        self.im.set_data(color_image.squeeze(0).cpu().numpy())
        self.fig.canvas.draw()
        plt.pause(0.001)

        self.ee_pose, self.targ_pose = self.franka_client.get_pos()
        print(f"End-effector position: {self.ee_pose}, Target position: {self.targ_pose}")

        obs = {
            "rgb": color_image.to(self.device),
            "head_pose": torch.tensor([[self.ee_pose[0], self.ee_pose[1], self.ee_pose[2], 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32).to(self.device),
            "done": torch.tensor([self.done], dtype=torch.bool).to(self.device),
            "success": torch.tensor([self.success], dtype=torch.bool).to(self.device),
            "num_steps": torch.tensor([self.num_steps], dtype=torch.int32).to(self.device),
            "velocity": torch.tensor([self.velocity], dtype=torch.float32).to(self.device)
        }
        return obs
    
    def set_action(self, action):
        if self.pause:
            action[..., :3] = torch.tensor([[self.ee_pose[0], self.ee_pose[1], self.ee_pose[2]]], dtype=torch.float32).to(self.device)
        self.velocity = action.cpu().numpy().squeeze(0)[:3] - self.ee_pose

    def step(self):
        cur_tp = time.time()
        dur = cur_tp - self.last_tp
        self.last_tp = cur_tp
        dur = min(dur, 0.033)  # Cap the duration to prevent large jumps


        step_result = (not self.pause and (self.velocity != np.array([0.0, 0.0, 0.0])).any()) or self.done

        if self.done:
            self.done_count += 1
            if self.success:
                self.success_count += 1
            self.reset()

        if self.joystick is not None:
            pygame.event.pump()  # Process event queue
            if self.joystick.get_button(0):
                self.pause = False
            if self.joystick.get_button(1):
                exit(0)
            if self.joystick.get_button(2):
                self.done = True
                self.success = False
            if self.joystick.get_button(3):
                self.done = True
                self.success = True

        if step_result:
            self.num_steps += 1
            self.franka_client.set_pos(self.targ_pose + self.velocity * dur)
            
        return step_result
    
    def reset(self):
        self.franka_client.reset()
        self.ee_pose, self.targ_pose = self.franka_client.get_pos()
        self.last_tp = time.time()
        self.pause = True
        self.done = False
        self.success = False
        self.num_steps = 0
        self.velocity = np.array([0.0, 0.0, 0.0])

    def close(self):
        self.pipeline.stop()
        self.franka_client.close()