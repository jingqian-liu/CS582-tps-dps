import torch
from torch import nn
from torch.nn.functional import softplus

from utils.utils import kabsch


class BiasForce(nn.Module):
    def __init__(self, args, mds):
        super().__init__()
        self.bias = args.bias
        self.heavy_atoms = mds.heavy_atoms
        self.num_particles = mds.num_particles

        self.stochastic = args.stochastic_policy

        if self.bias == "force":
            self.output_dim = mds.num_particles * 3
        elif self.bias == "pot":
            self.output_dim = 1
        elif self.bias == "scale":
            self.output_dim = mds.num_particles


        if self.stochastic:
            self.output_dim = self.output_dim * 2

        self.input_dim = mds.num_particles * (3 + 1)
        print(self.output_dim)

        if args.molecule == "aldp" and self.stochastic == False:
            self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, self.output_dim),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, self.output_dim),
            )

        self.log_z = nn.Parameter(torch.tensor(0.0))
        self.to(args.device)

    def forward(self, pos, target):
        R, t = kabsch(pos[:, self.heavy_atoms], target[:, self.heavy_atoms])
        if self.bias == "pot":
            pos.requires_grad = True
        input_tensor = torch.matmul(pos, R.transpose(-2, -1)) + t
        dist = torch.norm(input_tensor - target, dim=-1, keepdim=True)
        input_tensor = torch.cat([input_tensor, dist], dim=-1)
        out = self.mlp(input_tensor.reshape(-1, self.input_dim))


        if self.stochastic:
            mean, log_std = torch.chunk(out, 2, dim = -1)
            log_std = torch.clamp(log_std, min = -10, max = 2)
            std = torch.exp(log_std)

            eps = torch.randn_like(mean)
            sampled = mean + eps * std

        else:
            sampled = out
            mean = sampled
            std = None


        if self.bias == "force":
            force = sampled.view(*pos.shape)
            force = torch.matmul(force, R)
            
            if self.stochastic:
                force_mean = mean.view(*pos.shape)
                force_mean = torch.matmul(force_mean, R)
                force_std = std.view(*pos.shape)
                force_std = torch.matmul(force_std, R)

            else:
                force_mean = force
                force_std = None

        elif self.bias == "pot":
            pot = sampled
            force = - torch.autograd.grad(pot.sum(), pos, create_graph=True)[0]
            
            if self.stochastic:
                pot_mean = mean
                force_mean = - torch.autograd.grad(pot_mean.sum(), pos, create_graph=True)[0]
                # Compute force_std as the gradient of std (since force = -∇pot and pot ~ N(mean, std))
                force_std = torch.abs(torch.autograd.grad(std.sum(), pos, create_graph=True)[0])
            else:
                force_mean = force
                force_std = None


        elif self.bias == "scale":
            target_aligned = torch.matmul(target - t, R)
            scale = softplus(sampled.view(*pos.shape[:-2], self.num_particles, 1))
            force = scale * (target_aligned - pos)
            
            if self.stochastic:
                scale_mean = softplus(mean.view(*pos.shape[:-2], self.num_particles, 1))
                force_mean = scale_mean * (target_aligned - pos)
                scale_std = std.view(*pos.shape[:-2], self.num_particles, 1)
                force_std = scale_std * torch.abs(target_aligned - pos)
            else:
                force_mean = force
                force_std = None


        return force, force_mean, force_std








