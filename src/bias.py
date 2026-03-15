import torch
from torch import nn
from torch.nn.functional import softplus

from utils.utils import kabsch

import torch
import torch.nn as nn

class BiasForceVAE(nn.Module):
    def __init__(self, args, mds):
        super().__init__()
        self.bias = args.bias
        self.heavy_atoms = mds.heavy_atoms
        self.num_particles = mds.num_particles

        if self.bias == "force":
            self.output_dim = mds.num_particles * 3
        elif self.bias == "pot":
            self.output_dim = 1
        elif self.bias == "scale":
            self.output_dim = mds.num_particles

        self.input_dim = mds.num_particles * (3 + 1)

        # Latent dimension
        self.latent_dim = args.latent_dim if hasattr(args, 'latent_dim') else 10

        if args.molecule == "aldp":
            # ALDP: input_dim -> input_dim -> input_dim -> input_dim/2 -> 5
            half_dim = self.input_dim // 2

            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(64, self.latent_dim)
            self.fc_logvar = nn.Linear(64, self.latent_dim)

            # Decoder for bias force (main task)
            self.decoder_bias = nn.Sequential(
                nn.Linear(self.latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.output_dim),
            )
            
            # Decoder for reconstruction (auxiliary task)
            self.decoder_recon = nn.Sequential(
                nn.Linear(self.latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, self.input_dim),
            )

        else:
            # For larger molecules
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(256, self.latent_dim)
            self.fc_logvar = nn.Linear(256, self.latent_dim)

            # Decoder for bias force (main task)
            self.decoder_bias = nn.Sequential(
                nn.Linear(self.latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, self.output_dim),
            )
            
            # Decoder for reconstruction (auxiliary task - simple)
            self.decoder_recon = nn.Sequential(
                nn.Linear(self.latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.input_dim),
            )

        self.log_z = nn.Parameter(torch.tensor(0.0))
        self.to(args.device)

    def encode(self, input_tensor):
        """
        Encode input to latent distribution parameters
        """
        h = self.encoder(input_tensor)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        This is used DURING TRAINING to add noise in latent space
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_bias(self, z):
        """
        Decode latent vector to bias force prediction (main task)
        """
        force = self.decoder_bias(z)
        return force
    
    def decode_recon(self, z):
        """
        Decode latent vector to reconstructed coordinates (auxiliary task)
        """
        recon = self.decoder_recon(z)
        return recon

    def forward(self, pos, target, sample=True):
        """
        Forward pass through VAE
        Args:
            pos: current positions
            target: target positions
            sample: whether to sample from latent distribution
                    True during training (adds noise)
                    False during inference (uses mean)
        
        Returns:
            force: predicted force
            mu: latent mean
            logvar: latent log variance
            recon: reconstructed input (for auxiliary loss)
        """

        # Kabsch alignment
        R, t = kabsch(pos[:, self.heavy_atoms], target[:, self.heavy_atoms])

        if self.bias == "pot":
            pos.requires_grad = True

        # Prepare input
        input_tensor = torch.matmul(pos, R.transpose(-2, -1)) + t
        dist = torch.norm(input_tensor - target, dim=-1, keepdim=True)
        input_tensor = torch.cat([input_tensor, dist], dim=-1)
        input_flat = input_tensor.reshape(-1, self.input_dim)

        # Encode to latent distribution
        mu, logvar = self.encode(input_flat)

        # Sample from latent distribution during training, use mean during inference
        if sample:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        # Decode to force (main task)
        force_mean = self.decode_bias(z)
        
        # Decode to reconstruction (auxiliary task)
        recon = self.decode_recon(z)

        # Apply bias type specific transformations
        if self.bias == "force":
            force = force_mean.view(*pos.shape)
            force = torch.matmul(force, R)
        elif self.bias == "pot":
            target_aligned = torch.matmul(target - t, R)
            scale = softplus(force_mean.view(*pos.shape[:-2], self.output_dim, 1))
            force = scale * (target_aligned - pos)
        elif self.bias == "scale":
            target_aligned = torch.matmul(target - t, R)
            scale = softplus(force_mean.view(*pos.shape[:-2], self.output_dim, 1))
            force = scale * (target_aligned - pos)

        #print(input_flat.size())

        return force, mu, logvar, recon, input_flat
