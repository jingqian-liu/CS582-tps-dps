import torch
from torch import nn
from torch.nn.functional import softplus

from utils.utils import kabsch

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
        self.latent_dim = args.latent_dim
        
        if args.molecule == "aldp":
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
            
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.output_dim),
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
            
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, self.output_dim),
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
    
    def decode(self, z):
        """
        Decode latent vector to force prediction
        """
        force = self.decoder(z)
        return force
    
    def forward(self, pos, target, sample=True):
        """
        Forward pass through VAE
        Args:
            pos: current positions
            target: target positions
            sample: whether to sample from latent distribution
                   - True during training (adds noise)
                   - False during inference (uses mean)
        Returns:
            force: predicted force
            mu: latent mean
            logvar: latent log variance
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
        
        # Decode to force
        force_mean = self.decode(z)
        
        # Apply bias type specific transformations
        if self.bias == "force":
            force = force_mean.view(*pos.shape)
            force = torch.matmul(force, R)
        elif self.bias == "pot":
            force = -torch.autograd.grad(force_mean.sum(), pos, create_graph=True)[0]
        elif self.bias == "scale":
            target_aligned = torch.matmul(target - t, R)
            scale = softplus(force_mean.view(*pos.shape[:-2], self.output_dim, 1))
            force = scale * (target_aligned - pos)
        
        return force, mu, logvar
