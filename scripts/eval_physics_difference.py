import argparse
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default="rt4000"
    )
    args, _ = parser.parse_known_args()
    
    template_path = "conf/eval_temp.yaml"
    cfg = OmegaConf.load(template_path)
    cfg['policy']['args']['ckpt_path'] = f"ckpt/{args.name}/latest.pt"

    #save dir
    save_dir = f"conf/eval_{args.name}"

    # makedir if not exist
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(11):
        ratio = i * 0.1
        cfg['env']['args']['gravity'] = 9.81 * (1 + ratio)
        cfg['env']['args']['linear_damping'] = 90.0 * ratio
        cfg['manager']['args']['save_path'] = f"result/eval_{args.name}_physics_difference/{i}"

        save_path = os.path.join(save_dir, f"{i}.yaml")
        OmegaConf.save(cfg, save_path)
        print(f"Saved config to {save_path}")

        # run python workspace.py --headless --config {save_path} sequentially
        os.system(f"python workspace.py --headless --config {save_path}")


if __name__ == "__main__":
    main()