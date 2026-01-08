import numpy as np
import torch
import sim_scaling.task.base_env
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

class PushTEnv(sim_scaling.task.base_env.BaseEnv):
    def __init__(self, linear_damping=None, **kargs):
        super().__init__(**kargs)
        self.t_shape = self.scene["t_shape"]
        self.t_marker = self.scene["t_marker"]
        self.t_shape: RigidObject
        self.t_marker: RigidObject

        self.linear_damping = linear_damping

    def scene_setup(self, num_envs=1, env_spacing=4):
        cfg = super().scene_setup(num_envs, env_spacing)

        cfg.t_shape = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/T_shape",
            spawn=sim_utils.UsdFileCfg(
                usd_path="assets/t-shape.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.6, 0.1, 0.6)
                )
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.5, 0.0, 0.15)
            )
        )

        cfg.t_marker = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/T_marker",
            spawn=sim_utils.UsdFileCfg(
                usd_path="assets/t-shape.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=False
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.7, 0.1, 0.1)
                )
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.5, 0.0, 0.0751),
                rot=(0.707, 0.0, 0.0, 0.707)
            )
        )

        return cfg
    

    def reset_env(self, env_id, seed):
        generator = super().reset_env(env_id, seed)
        def rand_pos_quat(generator):
            generator: np.random.Generator
            poffset = generator.uniform(-0.1, 0.1, size=(2,))
            quat_offset = generator.uniform(-1, 1, size=(2,))
            quat_offset /= np.linalg.norm(quat_offset)

            pos=np.array([0.55, 0.0, 0.125]) + np.array([poffset[0] * 0.5, poffset[1], 0.0])
            quat=np.array([quat_offset[0], 0.0, 0.0, quat_offset[1]])

            return np.concatenate([pos, quat])
        pos_quat = rand_pos_quat(generator)

        pos_quat = torch.tensor(pos_quat, device=self.sim.device, dtype=torch.float32)

        pos_quat[0:3] += self.scene.env_origins[env_id]
        self.t_shape.write_root_pose_to_sim(pos_quat[None], env_ids=env_id.unsqueeze(0))

        if self.linear_damping is not None:
            t_shape_prim = self.stage.GetPrimAtPath(f'/World/envs/env_{env_id.item()}/T_shape/T_shape_usd/mesh')
            lin_damp = self.linear_damping
            t_shape_prim.GetAttribute('physxRigidBody:linearDamping').Set(lin_damp)

        return generator
    
    def get_observations(self):
        obs = super().get_observations()
        t_pose = self.t_shape.data.root_pose_w.clone()
        t_pose[:, 0:3] -= self.scene.env_origins
        obs["t_pose"] = t_pose.clone()
        obs["targ_pose"] = self.targ_pose.clone()

        return obs
    

    def step(self):
        super().step()
        def quat_geodesic_angle(q1, q2):
            q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
            q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)
            dot = torch.abs(torch.sum(q1 * q2, dim=-1))
            return 2.0 * torch.acos(torch.clamp(dot, 0.0, 1.0))

        dpos = (self.t_shape.data.root_pose_w[:, :3] - self.scene.env_origins) - self.targ_pose[:3]
        dpos = torch.norm(dpos, dim=-1)
        dquat = quat_geodesic_angle(self.t_shape.data.root_pose_w[:, 3:7], self.targ_pose[3:7])  # * quat_conjugate
        # check whether dpos < 0.005 and dquat < 0.05, in tensor
        success = (dpos < 0.005) & (dquat < 0.05)
        self.done = self.done | success
        self.success = self.success | success