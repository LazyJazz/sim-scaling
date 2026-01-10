import hydra
from omegaconf import OmegaConf
import argparse
import torch
import tqdm
from sim_scaling.policy.diffusion.policy import DiffusionPolicy

class TrainWorkspace:
    def __init__(self, cfg, lr=1e-4, lr_min=0.0, device:str = 'cuda', **kargs):
        self.cfg = cfg
        self.device = torch.device(device)

        policy_cls = hydra.utils.get_class(cfg.policy.__target__)
        self.policy = policy_cls(device=self.device, **cfg.policy.args)

        dataset_cls = hydra.utils.get_class(cfg.dataset.__target__)
        self.dataset = dataset_cls(**cfg.dataset.args)
        self.dataset_loader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=cfg.training.batch_size, 
            shuffle=True, 
            num_workers=cfg.training.num_workers, 
            pin_memory=True
        )

        val_dataset_cls = hydra.utils.get_class(cfg.val_dataset.__target__)
        self.val_dataset = val_dataset_cls(**cfg.val_dataset.args)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.training.total_steps, eta_min=lr_min)

        self.step_count = 0


    def run(self):
        # Implement the training loop here
        epoch_idx = 0
        while self.step_count < self.cfg.training.total_steps:
            with tqdm.tqdm(self.dataset_loader, desc=f"Training epoch {epoch_idx}", leave=False, mininterval=1.0) as tepoch:
                for batch_idx, (img, head_pose, action) in enumerate(tepoch):
                    img = img.to(self.device).permute(0, 1, 4, 2, 3).float()
                    head_pose = head_pose.to(self.device).float()
                    action = action.to(self.device).float()

                    loss = self.policy.compute_loss(img, head_pose, action)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()


                    tepoch.set_postfix(loss=loss.item())

                    self.step_count += 1
                    if self.step_count % self.cfg.training.checkpoint_interval == 0:
                        print(f"Step {self.step_count}: Saving checkpoint...")
            epoch_idx += 1
        
        # Finalize training
        print("Training completed.")

def main(cfg: OmegaConf, device: str = 'cuda'):
    trainer = TrainWorkspace(cfg, **cfg.training, device=device)
    trainer.run()

if __name__ == "__main__":
    # Load from conf/default.yaml to cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="conf/train_default.yaml", help="Path to the config file.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the training on.")
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    main(cfg, device=args.device)
