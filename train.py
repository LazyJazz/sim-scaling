import hydra
from omegaconf import OmegaConf
import argparse
import torch
import tqdm
from sim_scaling.policy.diffusion.policy import DiffusionPolicy
import os
import json
import wandb

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
        self.val_dataset_loader = torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True
        )

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.training.total_steps, eta_min=lr_min)

        self.step_count = 0

        # makedir for checkpoint 
        if not os.path.exists(cfg.training.checkpoint_path):
            os.makedirs(cfg.training.checkpoint_path)

        self.best_val_loss = float('inf')
        self.best_val_loss_step = 0

        self.wandb_runner = wandb.init(
            project=cfg.training.wandb.project,
            name=cfg.training.wandb.run_name)


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

                    wandb_log = {
                        "train/loss": loss.item(),
                        "train/lr": self.lr_scheduler.get_last_lr()[0],
                        "step": self.step_count
                    }
                    self.wandb_runner.log(wandb_log, step=self.step_count)

                    if self.step_count % self.cfg.training.checkpoint_interval == 0:
                        self.policy.eval()
                        val_losses = []
                        with torch.no_grad():
                            with tqdm.tqdm(self.val_dataset_loader, desc=f"Validation at step {self.step_count}", leave=False, mininterval=1.0) as vepoch:
                                for val_batch_idx, (val_img, val_head_pose, val_action) in enumerate(vepoch):
                                    val_img = val_img.to(self.device).permute(0, 1, 4, 2, 3).float()
                                    val_head_pose = val_head_pose.to(self.device).float()
                                    val_action = val_action.to(self.device).float()

                                    batch_val_loss = self.policy.compute_loss(val_img, val_head_pose, val_action)
                                    val_losses.append(batch_val_loss.item())
                        val_loss = sum(val_losses) / len(val_losses)
                        print(f"\nValidation loss at step {self.step_count}: {val_loss:.6f}")
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.best_val_loss_step = self.step_count
                            checkpoint_path = os.path.join(self.cfg.training.checkpoint_path, f"best_val.pt")
                            torch.save(self.policy.state_dict(), checkpoint_path)
                            print(f"Saved best model checkpoint at step {self.step_count} with loss {val_loss:.6f}")
                        checkpoint_path = os.path.join(self.cfg.training.checkpoint_path, f"latest.pt")
                        torch.save(self.policy.state_dict(), checkpoint_path)
                        # save lr_scheduler and optimizer state
                        checkpoint_path = os.path.join(self.cfg.training.checkpoint_path, f"latest_opt.pt")
                        torch.save(self.optimizer.state_dict(), checkpoint_path)

                        checkpoint_path = os.path.join(self.cfg.training.checkpoint_path, f"latest_lr.pt")
                        torch.save(self.lr_scheduler.state_dict(), checkpoint_path)

                        metadata = {
                            "step": self.step_count,
                            "val_loss": val_loss,
                            "best_val_loss": self.best_val_loss,
                            "best_val_loss_step": self.best_val_loss_step
                        }
                        checkpoint_path = os.path.join(self.cfg.training.checkpoint_path, f"metadata.pt")
                        with open(checkpoint_path, 'w') as f:
                            json.dump(metadata, f)

                        self.wandb_runner.log({
                            "val/loss": val_loss,
                            "val/best_val_loss": self.best_val_loss,
                            "val/best_val_loss_step": self.best_val_loss_step,
                            "step": self.step_count
                        }, step=self.step_count)

                        self.policy.train()
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
