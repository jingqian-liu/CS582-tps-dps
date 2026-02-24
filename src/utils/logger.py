import logging
import sys

import torch
import wandb

from .plot import Plot
from .metrics import Metric

class Logger:
    def __init__(self, args, mds):
        self.molecule = args.molecule
        self.save_dir = args.save_dir
        self.wandb = args.wandb
        self.plot = Plot(args, mds)
        self.metrics = Metric(args, mds)
        self.rmsd = float("inf")

        # Set up file logging
        self.logger = logging.getLogger(f"tps-dps-{args.molecule}")
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # File handler
        log_file = f"{self.save_dir}/training.log"
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Training started for {args.molecule}")
        self.logger.info(f"Saving results to {self.save_dir}")

    # MODIFIED: Added log_ri parameter
    def __call__(self, loss, loss_kl, loss_tps, loss_entropy, rollout, policy):
        metrics = self.metrics()
        
        # Save every 100 rollouts
        if rollout % 100 == 0:
            torch.save(policy.state_dict(), f"{self.save_dir}/policies/policy_rollout{rollout}.pt")




        log_msg = f"Rollout {rollout}: Loss = {loss:.6f}, KL Loss = {loss_kl:.6f}, TPS Loss = {loss_tps:.6f}, Entropy Loss = {loss_entropy:.6f}, RMSD = {metrics['rmsd']:.6f}, Best RMSD = {self.rmsd:.6f}"

            
        if metrics.get('thp') is not None:
            log_msg += f", THP = {metrics['thp']:.6f}"
        if metrics.get('ets') is not None:
            log_msg += f", ETS = {metrics['ets']:.6f}"


        self.logger.info(log_msg)

        if self.rmsd > metrics["rmsd"]:
            self.rmsd = metrics["rmsd"]
            torch.save(policy.state_dict(), f"{self.save_dir}/policy.pt")



        if self.wandb:
            if metrics["ets"] is not None:
                wandb.log({
                    "rmsd": metrics["rmsd"],
                    "thp": metrics["thp"],
                    "ets": metrics["ets"],
                    "loss": loss,
                    "KL loss": loss_kl,
                    "TPS loss": loss_tps,
                    "Entropy loss": loss_entropy,

                })
            else:
                wandb.log({
                    "rmsd": metrics["rmsd"],
                    "thp": metrics["thp"],
                    "loss": loss,
                    "KL loss": loss_kl,
                    "TPS loss": loss_tps,
                    "Entropy loss": loss_entropy,

                })
