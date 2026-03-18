import argparse
import os
from omegaconf import OmegaConf

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint name for evaluation')
    parser.add_argument('--traj', type=int, default=1000, help='Number of trajectories to evaluate')
    parser.add_argument('--num-envs', type=int, default=256, help='Number of environments to use')
    parser.add_argument('--pt', action='store_true', help='Whether to generate validation data')
    parser.add_argument('--damp', action='store_true', help='Whether to generate validation data')
    parser.add_argument('--adamp', action='store_true', help='Whether to generate validation data')
    parser.add_argument('--vis-rand', action='store_true', help='Whether to generate validation data')
    parser.add_argument('--marker', action='store_true', help='Whether to include marker in the scene')
    parser.add_argument('--run', action='store_true', help='Whether to run evaluation after generating config')
    parser.add_argument('--all', action='store_true', help='Generate configs for all settings')
    args = parser.parse_args()

    if args.all:
        run_suffix = '--run' if args.run else ''
        run_suffix += ' --vis-rand' if args.vis_rand else ''
        run_suffix += ' --pt' if args.pt else ''
        run_suffix += ' --marker' if args.marker else ''
        os.system(f"python scripts/eval.py --ckpt {args.ckpt} {run_suffix}")
        os.system(f"python scripts/eval.py --ckpt {args.ckpt} --damp {run_suffix}")
        os.system(f"python scripts/eval.py --ckpt {args.ckpt} --adamp {run_suffix}")
        return

    cfg = OmegaConf.create()

    suffix = ""

    num_envs = args.num_envs

    cfg['env'] = {
        "__target__": 'sim_scaling.task.pusht.PushTEnv',
        'args': {
            'seed': 2000000000,
            'num_envs': num_envs,
            'env_spacing': 40.0,
            'step_limit': 3000,
            'eval_mode': True
        }
    }

    if args.pt:
        suffix += "pt"
        cfg['launch_app'] = {
            'renderer': 'PathTracing',
            'samples_per_pixel_per_frame': 4,
            'use_denoiser': True
        }
        cfg['env']['args']['light_intensity'] = 100.0
    else:
        suffix += "rt"
        cfg['launch_app'] = {
            'renderer': 'RayTracedLighting'
        }

    
    suffix += f"{args.traj}"

    if args.marker:
        cfg['env']['args']['marker'] = True
        suffix += "m"

    if args.damp:
        suffix += "d"
        cfg['env']['args']['linear_damping'] = 90.0
        cfg['env']['args']['gravity'] = 19.62
    
    if args.adamp:
        suffix += "l"
        cfg['env']['args']['angular_damping'] = 300.0
        
    if args.vis_rand:
        suffix += "v"
        cfg['env']['args']['visual_random'] = True


# policy:
#   __target__: sim_scaling.policy.diffusion_policy.DiffusionPolicy
#   args:
#     ckpt_path: ckpt/cotrain_100000d_100/latest.pt
#     num_inference_timesteps: 10

# manager:
#   __target__: sim_scaling.manager.eval_manager.EvalManager
#   args:
#     num_eval_envs: 1000
#     save_path: result/eval_cotrain_100000d_100

    cfg['policy'] = {
        "__target__": 'sim_scaling.policy.diffusion_policy.DiffusionPolicy',
        'args': {
            'ckpt_path': f"ckpt/{args.ckpt}/latest.pt",
            'num_inference_timesteps': 10
        }
    }

    cfg['manager'] = {
        "__target__": 'sim_scaling.manager.eval_manager.EvalManager',
        'args': {
            'num_eval_envs': args.traj,
            'save_path': f"results/{args.ckpt}/{suffix}"
        }
    }

    path = f"conf/eval_{args.ckpt}_on_{suffix}.yaml"
    OmegaConf.save(cfg, path)
    print(f"Saved config to {path}")

    if args.run:
        command = f"python workspace.py --config {path} --headless"
        os.system(command)


if __name__ == "__main__":
    main()