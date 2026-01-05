import task.base_env
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

class PushTEnv(task.base_env.BaseEnv):
    def __init__(self):
        super().__init__()

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

    def reset(self, idx=None):
        return "PushTEnv reset called"
    