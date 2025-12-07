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
        self.metric = Metric(args, mds)
        self.rmsd = float("inf")



        # Set up file logging
        self.logger = logging.getLogger(f"tps-dps-{args.molecule}")
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # File handler
        log_file = f"{self.save_dir}/training.log"
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info(f"Training started for {args.molecule}")
        self.logger.info(f"Saving results to {self.save_dir}")



    def __call__(self, loss, rollout, policy,avg_log_measure):
        metrics = self.metric()

        # Save every 100 rollouts
        if rollout % 10 == 0:
            torch.save(policy.state_dict(), f"{self.save_dir}/policies/policy_rollout_{rollout}.pt")


        # Log metrics
        log_msg = f"Rollout {rollout}: Loss = {loss:.6f}, Log Measure = {avg_log_measure:.6f},  RMSD = {metrics['rmsd']:.6f}, Best RMSD = {self.rmsd:.6f}"
        if metrics.get('thp') is not None:
            log_msg += f", THP = {metrics['thp']:.6f}"
        if metrics.get('ets') is not None:
            log_msg += f", ETS = {metrics['ets']:.6f}"

        self.logger.info(log_msg)


        # Add this in the Logger.__call__ method, right after metrics = self.metric()
        print(f"Rollout {rollout}: RMSD = {metrics['rmsd']}, Best RMSD = {self.rmsd}")

        if self.rmsd > metrics["rmsd"]:
            self.rmsd = metrics["rmsd"]
            torch.save(policy.state_dict(), f"{self.save_dir}/policy.pt")
        if self.wandb:
            if metrics["ets"] is not None:
                wandb.log({
                    "rmsd": metrics["rmsd"],
                    "thp": metrics["thp"],
                    "ets": metrics["ets"],
                    "loss": loss
                })
            else:
                wandb.log({
                    "rmsd": metrics["rmsd"],
                    "thp": metrics["thp"],
                    "loss": loss
                })
