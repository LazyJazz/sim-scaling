import hydra
from omegaconf import OmegaConf
import argparse

class Workspace:
    def __init__(self, cfg: OmegaConf):
        OmegaConf.resolve(cfg)
        self.cfg = cfg
        import sim_scaling.task.launch
        sim_scaling.task.launch.launch_app(**cfg.launch_app)

        import sim_scaling.task.base_env
        env_cls = hydra.utils.get_class(cfg.env.__target__)
        self.env = env_cls(**cfg.env.args)
        self.env: sim_scaling.task.base_env.BaseEnv
        self.env.reset()

        import sim_scaling.policy.base_policy
        policy_cls = hydra.utils.get_class(cfg.policy.__target__)
        self.policy = policy_cls(device=self.env.device, **cfg.policy.args)
        self.policy: sim_scaling.policy.base_policy.BasePolicy

        import sim_scaling.manager.base_manager
        manager_cls = hydra.utils.get_class(cfg.manager.__target__)
        self.manager = manager_cls(env=self.env, policy=self.policy, **cfg.manager.args)
        self.manager: sim_scaling.manager.base_manager.BaseManager

    def run(self):
        while not self.manager.should_terminate():
            obs = self.env.get_observations()
            action = self.policy.get_action(obs)
            self.env.set_action(action)
            self.env.step()
            self.manager.step(obs, action)

            if self.manager.__repr__() != "":
                print(f"{self.manager}")
        
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