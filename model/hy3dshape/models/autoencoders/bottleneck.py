import torch
import torch.nn as nn
from .utils import DiagonalGaussianDistribution


class KLBottleneck(nn.Module):
    def __init__(self, dim, latent_dim, kl_weight=1e-3, **kwargs):
        super().__init__()
        self.kl_weight = kl_weight
        self.proj = nn.Linear(latent_dim, dim)

        self.mean_fc = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)
    
    def pre(self, x, sample_posterior=True):
        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)
        posterior = DiagonalGaussianDistribution(mean, logvar)
        z = posterior.sample() if sample_posterior else mean
        kl = posterior.kl()
        return {'x': z, 'kl_loss': self.kl_weight * kl}
    
    def post(self, x):
        return self.proj(x)

class NormalizedBottleneck(nn.Module):
    def __init__(self, dim, latent_dim, **kwargs):
        super().__init__()
        self.pre_bottleneck_proj = nn.Linear(dim, latent_dim)
        # self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-3)
        self.post_bottleneck_proj = nn.Linear(latent_dim, dim)
        self.gamma = nn.Parameter(torch.ones(latent_dim))
        self.beta = nn.Parameter(torch.zeros(latent_dim))
    
    def pre(self, x, **kwargs):
        z = self.norm(self.pre_bottleneck_proj(x))
        return {'x': z}
    
    def post(self, x):
        x = x * self.gamma + self.beta
        return self.post_bottleneck_proj(x)