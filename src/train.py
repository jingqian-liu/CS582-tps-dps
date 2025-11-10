import os
import argparse

import torch
import wandb

from dynamics.mds import MDs
from utils.logging import Logger
from dps import DiffusionPathSampler


def main():
    parser = argparse.ArgumentParser()
    # System Config
    parser.add_argument("--date", type=str)
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--molecule", default="aldp", type=str)
    parser.add_argument('--wandb', action='store_true', default=False)
    # Logger Config
    parser.add_argument("--save_dir", default="results", type=str)
    # Policy Config
    parser.add_argument("--bias", default="force", type=str)
    parser.add_argument("--stochastic_policy", default=False)
    parser.add_argument("--entropy_coef", default=0.01,type=float)
    # Sampling Config
    parser.add_argument("--start_state", default="c5", type=str)
    parser.add_argument("--end_state", default="c7ax", type=str)
    parser.add_argument("--num_steps", default=1000, type=int)
    parser.add_argument("--timestep", default=1, type=float)
    parser.add_argument("--sigma", default=0.1, type=float)
    parser.add_argument("--num_samples", default=16, type=int)
    parser.add_argument("--temperature", default=300, type=float)
    parser.add_argument("--friction", default=0.001, type=float)
    # Training Config
    parser.add_argument("--start_temperature", default=600, type=float)
    parser.add_argument("--end_temperature", default=300, type=float)
    parser.add_argument("--num_rollouts", default=1000, type=int)
    parser.add_argument("--trains_per_rollout", default=1000, type=int)
    parser.add_argument("--log_z_lr", default=1e-3, type=float)
    parser.add_argument("--policy_lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--buffer_size", default=1000, type=int)
    parser.add_argument("--max_grad_norm", default=1, type=int)
    parser.add_argument("--control_variate", default="global", type=str)
    args = parser.parse_args()
    args.training = True
    args.save_dir = f"results/{args.date}"
    for name in ["policies", "positions"]:
        if not os.path.exists(f"{args.save_dir}/{name}"):
            os.makedirs(f"{args.save_dir}/{name}")
    if args.wandb:
        wandb.init(project="tps-dps", config=args)
    torch.manual_seed(args.seed)
    mds = MDs(args)
    logger = Logger(args, mds)
    agent = DiffusionPathSampler(args, mds)
    temperatures = torch.linspace(
        args.start_temperature, args.end_temperature, args.num_rollouts
    )
    for rollout in range(args.num_rollouts):
        agent.sample(args, mds, temperatures[rollout])
        loss = agent.train(args, mds, temperatures[rollout])
        logger(loss, rollout, agent.policy)


if __name__ == "__main__":
    main()
