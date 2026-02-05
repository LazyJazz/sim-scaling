import hydra
from omegaconf import OmegaConf
import argparse
import asyncio

class Workspace:
    def __init__(self, cfg: OmegaConf):
        OmegaConf.resolve(cfg)
        self.cfg = cfg
        if cfg.launch_app is not None:
            import sim_scaling.task.launch
            sim_scaling.task.launch.launch_app(**cfg.launch_app)

        env_cls = hydra.utils.get_class(cfg.env.__target__)
        self.env = env_cls(**cfg.env.args)
        self.env.reset()

        policy_cls = hydra.utils.get_class(cfg.policy.__target__)
        self.policy = policy_cls(device=self.env.device, **cfg.policy.args)

        manager_cls = hydra.utils.get_class(cfg.manager.__target__)
        self.manager = manager_cls(env=self.env, policy=self.policy, **cfg.manager.args)

    def run(self):
        while not self.manager.should_terminate():
            obs = self.env.get_observations()
            action = self.policy.get_action(obs)
            self.env.set_action(action)
            self.env.step()
            self.manager.step(obs, action)

            if self.manager.__repr__() != "":
                print(f"{self.manager}")
        self.manager.close()
        self.env.close()
            
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    # Load from conf/default.yaml to cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="conf/default.yaml", help="Path to the config file.")
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)