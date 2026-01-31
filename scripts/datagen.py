import argparse
import os
from omegaconf import OmegaConf


# launch_app:
#   renderer: RayTracedLighting
#   # renderer: PathTracing
#   # samples_per_pixel_per_frame: 4
#   # use_denoiser: true

# env:
#   __target__: sim_scaling.task.pusht.PushTEnv
#   args:
#     seed: 0
#     num_envs: 16
#     env_spacing: 40.0
#     step_limit: 2000

# policy:
#   __target__: sim_scaling.policy.pusht_motion_planning.PushTMotionPlanningPolicy
#   args: {}

# manager:
#   __target__: sim_scaling.manager.datagen_manager.DataGenManager
#   args:
#     path: ./data/pusht_rt100.zarr
#     succ_traj: 100

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--traj', type=int, default=100, help='Number of trajectories to generate')
    parser.add_argument('--num-envs', type=int, default=None, help='Number of environments to use')
    parser.add_argument('--pt', action='store_true', help='Whether to generate validation data')
    parser.add_argument('--val', action='store_true', help='Whether to generate validation data')
    parser.add_argument('--damp', action='store_true', help='Whether to generate validation data')
    args = parser.parse_args()

    cfg = OmegaConf.create()

    suffix = ""

    if args.pt:
        suffix += "pt"
        cfg['launch_app'] = {
            'renderer': 'PathTracing',
            'samples_per_pixel_per_frame': 4,
            'use_denoiser': True
        }
    else:
        suffix += "rt"
        cfg['launch_app'] = {
            'renderer': 'RayTracedLighting'
        }

    num_envs = args.num_envs
    if num_envs is None:
        num_envs = 2
        while num_envs * 8 < args.traj and num_envs < 256:
            num_envs *= 2
    
    suffix += f"{args.traj}"

    cfg['env'] = {
        "__target__": 'sim_scaling.task.pusht.PushTEnv',
        'args': {
            'seed': 0,
            'num_envs': num_envs,
            'env_spacing': 40.0,
            'step_limit': 2000
        }
    }

    if args.damp:
        suffix += "d"
        cfg['env']['args']['linear_damping'] = 90.0
        cfg['env']['args']['gravity'] = 19.62

    if args.val:
        suffix += "_val"
        cfg['env']['args']['seed'] = 1000000000


    cfg['policy'] = {
        "__target__": 'sim_scaling.policy.pusht_motion_planning.PushTMotionPlanningPolicy',
        'args': {}
    }

    cfg['manager'] = {
        "__target__": 'sim_scaling.manager.datagen_manager.DataGenManager',
        'args': {
            'path': f'./data/pusht_{suffix}.zarr',
            'succ_traj': args.traj,
        }
    }

    path = f"conf/datagen_{suffix}.yaml"
    OmegaConf.save(cfg, path)
    print(f"Saved config to {path}")


if __name__ == "__main__":
    main()