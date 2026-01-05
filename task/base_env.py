import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Launch Isaac Lab tasks with specified environment.")
parser.add_argument("--renderer", type=str, choices=["RayTracedLighting", "PathTracing"], default="RayTracedLighting", help="Renderer to use.")
parser.add_argument("--samples-per-pixel-per-frame", type=int, default=4, help="Number of samples per pixel per frame.")
parser.add_argument("--use-denoiser", action="store_true", help="Whether to use denoiser.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)


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
    def __init__(self):
        self.app = app_launcher.app
        self.sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args.device)
        self.sim = sim_utils.SimulationContext(self.sim_cfg)

        self.scene = InteractiveScene(self.scene_setup())
        self.sim.reset()

    def scene_setup(self, num_envs=1, env_spacing=4.0):
        cfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
        cfg.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Franka",
            spawn=sim_utils.UrdfFileCfg(
                asset_path="franka_fr3/fr3_franka_hand.urdf",
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
        return cfg

    def reset(self, idx=None):
        pass
    
    def step(self, action=None):
        self.sim.step(render=False)
        self.scene.update(self.sim.get_physics_dt())
        self.sim.render(mode=sim_utils.SimulationContext.RenderMode.FULL_RENDERING)
        pass