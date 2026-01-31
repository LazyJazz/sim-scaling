import argparse
import os
from omegaconf import OmegaConf


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, default=None, help='Path to dataset 1')
    parser.add_argument('--val', type=str, required=True, default=None, help='Path to validation dataset 1')
    parser.add_argument('--mix-ratio', type=float, default=0.75, help='Mixing ratio for datasets')
    parser.add_argument('--training-step', type=int, default=1600000, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')

    args = parser.parse_args()

    cfg = OmegaConf.create()

    suffix = f"{args.dataset1}_{args.dataset2}"
    if args.mix_ratio != 0.75:
        suffix += f"_{args.mix_ratio}mix"
    if args.training_step != 1600000:
        suffix += f"_{args.training_step}step"

    cfg['dataset'] = {
        "__target__": 'sim_scaling.util.dataset.ImageDataset',
        "args": {
            "path": f"data/pusht_{args.dataset}.zarr"
        }
    }

    cfg['val_dataset'] = {
        "__target__": 'sim_scaling.util.dataset.ImageDataset',
        "args": {
            "path": f"data/pusht_{args.val}.zarr"
        }
    }

    cfg['training'] = {
        'lr': 1e-4,
        'lr_min': 0.0,
        'batch_size': args.batch_size,
        'num_workers': 32,
        'total_steps': args.training_step,
        'checkpoint_path': f'ckpt/cotrain_{suffix}',
        'checkpoint_interval': 10000,
        'wandb': {
            'project': 'cotrain_scaling_law',
            'run_name': f'train_cotrain_{suffix}'
        }
    }

    cfg['policy'] = {
        "__target__": 'sim_scaling.policy.diffusion.policy.DiffusionPolicy',
        'args': {
            'n_obs': 2,
            'n_actions': 8,
            'num_train_timesteps': 100,
            'num_inference_timesteps': 10,
            'pretrained': True
        }
    }

    path = f"conf/train_cotrain_{suffix}.yaml"
    OmegaConf.save(cfg, path)

    command = f"python train.py --config {path}"
    os.system(command)


if __name__ == "__main__":
    main()