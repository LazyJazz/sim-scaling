import hydra
from omegaconf import OmegaConf
import classes
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

            print(f"Step {self.manager.iter}: Success Rate = {self.env.get_success_rate():.3f}")
        
        self.env.close()
            
@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()