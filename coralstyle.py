import torch
from torch import nn

class ConvAttnNetwork(nn.Module):
    def __init__(self, in_channels, out_channels=32, kernel_size=1):
        super().__init__()

        self.block1 = nn.Sequential([
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.Relu(inplace=True),
        ])
        self.block2 = nn.Sequential([
            nn.Conv2d(out_channels, 1, kernel_size),
            nn.Sigmoid(),
        ])

    def forward(self, feature):
        return self.block2(self.block1(feature))

class MLP(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()

        self.mlp = nn.Sequential([
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        ])

    def forward(self, latent):
        return self.mlp(latent)

class BiEqual(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()

        self.block1 = nn.Sequential([
            MLP(latent_dim),
            nn.LeakyReLU(inplace=True),
        ])
        self.block2 = nn.Sequential([
            MLP(latent_dim),
            nn.LeakyReLU(inplace=True),
        ])

    def forward(self, latent):
        return self.block1(latent) - self.block2(latent)

class Mapper(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()

        self.mapper = nn.Sequential([
            BiEqual(latent_dim),
            BiEqual(latent_dim),
            BiEqual(latent_dim),
            BiEqual(latent_dim),
            MLP(latent_dim),
        ])

    def forward(self, latent):
        return self.mapper(latent)