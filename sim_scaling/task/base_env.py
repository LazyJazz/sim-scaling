import argparse
import os
import numpy as np
import torch
import sim_scaling.task.launch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.rigid_object import RigidObjectCfg, RigidObject, RigidObjectData
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import *
from isaaclab.utils.dict import convert_dict_to_backend
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip
from isaaclab.assets.articulation import ArticulationCfg, Articulation
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import CameraCfg, Camera, TiledCameraCfg, TiledCamera


class BaseEnv:
    def __init__(self, seed=0, num_envs=1, env_spacing=4.0, step_limit=2000, **kargs):
        self.seed = seed
        self.step_limit = step_limit

        app_launcher = sim_scaling.task.launch.get_app_launcher()
        args = sim_scaling.task.launch.get_launch_args()

        self.app = app_launcher.app
        self.sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args.device)
        self.sim = sim_utils.SimulationContext(self.sim_cfg)
        self.device = self.sim.device

        self.scene = InteractiveScene(self.scene_setup(num_envs, env_spacing))
        self.robot = self.scene["robot"]
        self.robot: Articulation

        self.camera = self.scene["camera"]
        self.camera: TiledCamera

        self.sim.reset()

        self.head_offset = torch.tensor([-0.001, -0.001, -0.103], device=self.sim.device)
        self.targ_pose = torch.tensor([0.5, 0.0, 0.125, 0.7071, 0.0, 0.0, 0.7071], device=self.sim.device)
        self.targ_pose[3:] /= torch.norm(self.targ_pose[3:], dim=-1, keepdim=True)
        self.targ_marker_pose = torch.tensor([0.5, 0.0, 0.0751, 0.7071, 0.0, 0.0, 0.7071], device=self.sim.device)

        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.scene.num_envs, device=self.sim.device)

        self.ik_commands = torch.zeros(self.scene.num_envs, self.diff_ik_controller.action_dim, device=self.sim.device)
        self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=["fr3_joint.*"], body_names=["fr3_hand"])
        self.robot_entity_cfg.resolve(self.scene)
        self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1

        self.done = torch.ones(self.scene.num_envs, dtype=torch.bool, device=self.sim.device)
        self.success = torch.zeros(self.scene.num_envs, dtype=torch.bool, device=self.sim.device)
        self.num_steps = torch.zeros(self.scene.num_envs, dtype=torch.int32, device=self.sim.device)
        self.env_seed = [-1] * self.scene.num_envs
        
        self.success_record = {}
        self.success_count = 0
        self.done_count = 0

    def scene_setup(self, num_envs=1, env_spacing=4.0):
        cfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
        cfg.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Franka",
            spawn=sim_utils.UrdfFileCfg(
                asset_path="assets/franka_fr3/fr3_franka_hand.urdf",
                fix_base=True,
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    target_type="position",
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=400.0, damping=40.0
                    )),
                merge_fixed_joints=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True
                )
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.1),
                joint_pos={
                    "fr3_joint1": 0.0,
                    "fr3_joint2": -0.569,
                    "fr3_joint3": 0.0,
                    "fr3_joint4": -2.810,
                    "fr3_joint5": 0.0,
                    "fr3_joint6": 3.037,
                    "fr3_joint7": 0.741,
                    "fr3_finger_joint.*": 0.04,
                },
            ),
            actuators={
                "fr3_shoulder": ImplicitActuatorCfg(
                    joint_names_expr=["fr3_joint[1-4]"],
                    effort_limit_sim=87.0,
                    stiffness=400.0,
                    damping=80.0,
                ),
                "fr3_forearm": ImplicitActuatorCfg(
                    joint_names_expr=["fr3_joint[5-7]"],
                    effort_limit_sim=12.0,
                    stiffness=400.0,
                    damping=80.0,
                ),
                "fr3_hand": ImplicitActuatorCfg(
                    joint_names_expr=["fr3_finger_joint.*"],
                    effort_limit_sim=200.0,
                    stiffness=2e3,
                    damping=1e2,
                ),
            },
            soft_joint_pos_limit_factor=1.0,
        )
        cfg.table = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.CuboidCfg(
                size=(2.0, 2.0, 0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_linear_velocity=0.0,
                    max_angular_velocity=0.0
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.02, 0.1, 0.02),
                    roughness=1.0
                )
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.05))
        )

        cfg.ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(size=(1000.0, 1000.0)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )

        cfg.dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=300.0, color=(0.75, 0.75, 0.75))
        )

        cfg.spherical_light = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Spherical_Light",
            spawn=sim_utils.SphereLightCfg(intensity=3000000.0, color=(1.0, 1.0, 1.0), radius=0.03),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-2.0, -1.0, 3.0))
        )

        cfg.camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            width=160,
            height=160,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=45.0, focus_distance=400.0, horizontal_aperture=20.0, clipping_range=(0.1, 1.0e5)
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(2.0, 0.0, 1.5),
                rot=(0.64597, 0.28761, 0.28761, 0.64597),
                convention="opengl",
            )
        )
        return cfg

    def reset(self, env_ids=None, seed=None):
        if seed is not None:
            self.seed = seed
        if env_ids is None:
            env_ids = torch.arange(self.scene.num_envs, device=self.sim.device)

        for env_id in env_ids:
            self.reset_env(env_id, self.seed)
            self.seed += 1

        self.sim.render(mode=sim_utils.SimulationContext.RenderMode.FULL_RENDERING)

    def reset_env(self, env_id, seed):
        joint_pos = self.robot.data.default_joint_pos[env_id].clone()
        joint_vel = self.robot.data.default_joint_vel[env_id].clone()
        self.robot.write_joint_state_to_sim(joint_pos[None], joint_vel[None], env_ids=env_id.unsqueeze(0))
        self.robot.reset()

        self.ik_commands[env_id] = torch.tensor([0.5, 0, 0.4, 0.0, 1.0, 0.0, 0.0], device=self.sim.device)

        self.diff_ik_controller.reset()
        self.diff_ik_controller.set_command(self.ik_commands)

        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        ee_pose_w = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
        root_pose_w = self.robot.data.root_pose_w
        # compute frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        # compute the joint commands
        joint_pos_des = self.diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, self.robot.data.default_joint_pos[:, self.robot_entity_cfg.joint_ids])
        joint_pos[self.robot_entity_cfg.joint_ids] = joint_pos_des[env_id]
        joint_vel[self.robot_entity_cfg.joint_ids] = 0.0
        self.robot.write_joint_state_to_sim(joint_pos[None], joint_vel[None], env_ids=env_id.unsqueeze(0))
        self.robot.set_joint_position_target(joint_pos_des[env_id.unsqueeze(0)], joint_ids=self.robot_entity_cfg.joint_ids, env_ids=env_id.unsqueeze(0))

        self.done[env_id] = False
        self.success[env_id] = False
        self.num_steps[env_id] = 0
        self.env_seed[env_id] = seed

        generator = np.random.default_rng(seed)
        return generator
    
    def step(self):
        self.diff_ik_controller.set_command(self.ik_commands)
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        ee_pose_w = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
        root_pose_w = self.robot.data.root_pose_w
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        # compute frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        # compute the joint commands
        joint_pos_des = self.diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)
        self.scene.write_data_to_sim()

        self.app.update()
        self.sim.step(render=False)
        self.scene.update(self.sim.get_physics_dt())
        self.sim.render(mode=sim_utils.SimulationContext.RenderMode.FULL_RENDERING)
        self.num_steps += 1

        reset_env_ids = torch.nonzero(self.done).squeeze(-1)
        if len(reset_env_ids) > 0:
            for reset_env_id in reset_env_ids:
                print(f"Env {reset_env_id.item()} done. Success: {self.success[reset_env_id].item()}, Num Steps: {self.num_steps[reset_env_id].item()}, Seed: {self.env_seed[reset_env_id]}")
                if self.success[reset_env_id]:
                    self.success_count += 1
                self.done_count += 1
                self.success_record[int(reset_env_id.item())] = {
                    "success": self.success[reset_env_id].item(),
                    "num_steps": self.num_steps[reset_env_id].item(),
                }

            self.reset(env_ids=reset_env_ids)

        fail = self.num_steps >= self.step_limit
        self.done = self.done | fail

    def close(self):
        # 0) Make prints appear even if shutdown is abrupt
        import sys
        def log(msg):
            print(msg, flush=True)

        log("Shutting down...")

        # 1) Stop timeline (prevents sensors/render from continuing to tick)
        try:
            import omni.timeline
            omni.timeline.get_timeline_interface().stop()
            log("Timeline stopped")
        except Exception as e:
            log(f"Timeline stop skipped: {e}")

        # 2) Clear/destroy your sim/env objects (do this BEFORE app.close)
        try:
            # Prefer explicit env/sensor destroy calls if you have them
            # e.g., self.env.close() / camera.destroy() / render_product.destroy()
            self.sim.clear_instance()
            log("Simulation instance cleared")
        except Exception as e:
            log(f"sim.clear_instance failed: {e}")

        self.scene = None
        log("Scene reference cleared (Python-side)")

        # 3) Let Kit process destruction for a few frames
        try:
            for _ in range(5):
                self.app.update()
            log("App updated (cleanup frames)")
        except Exception as e:
            log(f"App update skipped/failed: {e}")

        # 4) Close the app LAST (often terminates the process)
        log("Calling app.close()")
        self.app.close()

    def get_observations(self):

        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        ee_pose_w = ee_pose_w.clone()
        ee_pose_w[:, 0:3] += self.head_offset
        ee_pose_w[:, 0:3] -= self.scene.env_origins

        obs = {
            "rgb": self.camera.data.output['rgb'].clone(),
            "head_pose": ee_pose_w.clone(),
            "done": self.done.clone(),
            "success": self.success.clone(),
            "num_steps": self.num_steps.clone(),
        }
        return obs
        
    def set_action(self, action):
        self.ik_commands = action.clone()

    def get_success_rate(self):
        return self.success_count / max(1, self.done_count)
    
    def get_success_record(self):
        return self.success_record