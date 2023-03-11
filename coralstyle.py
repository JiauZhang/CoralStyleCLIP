import torch
from torch import nn

class ConvAttnNetwork(nn.Module):
    def __init__(self, in_channels, out_channels=32, kernel_size=1):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size),
            nn.Sigmoid(),
        )

    def forward(self, feature):
        return self.block2(self.block1(feature))

class MLP(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, latent):
        return self.mlp(latent)

class BiEqual(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()

        self.block1 = nn.Sequential(
            MLP(latent_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            MLP(latent_dim),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, latent):
        return self.block1(latent) - self.block2(latent)

class Mapper(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()

        self.mapper = nn.Sequential(
            BiEqual(latent_dim),
            BiEqual(latent_dim),
            BiEqual(latent_dim),
            BiEqual(latent_dim),
            MLP(latent_dim),
        )

    def forward(self, latent):
        return self.mapper(latent)

class CoralAttnNet(nn.Module):
    def __init__(self, stylegan2):
        super().__init__()

        self.stylegan2 = stylegan2
        self.conv_attn_nets = nn.ModuleList()
        self.mapper = Mapper()

        for i in range(2, self.stylegan2.log_size + 1):
            out_channel = self.stylegan2.channels[2 ** i]
            self.conv_attn_nets.append(ConvAttnNetwork(out_channel))

    def stylegan2_forward(self, w_plus, return_features=True):
        image, _, features = self.stylegan2(
            w_plus, input_is_latent=True,
            return_features=return_features,
        )
        return image, features

    def blend_layer(self, layer, f_l_1_star, w_plus1, w_plus2, mask, noise):
        f_l_bar_star = layer(f_l_1_star, w_plus2[:, 0], noise=noise[0])
        f_l_bar = layer(f_l_1_star, w_plus1[:, 0], noise=noise[0])
        f_l_star = mask * f_l_bar_star + (1 - mask) * f_l_bar
        return f_l_star

    def stylegan2_blended_forward(self, w_plus1, w_plus2, masks,
        noise=None, randomize_noise=True,
    ):
        stylegan2 = self.stylegan2
        if noise is None:
            if randomize_noise:
                noise = [None] * stylegan2.num_layers
            else:
                noise = [
                    getattr(stylegan2.noises, f"noise_{i}") for i in range(stylegan2.num_layers)
                ]
        f_l_1_star = stylegan2.input(w_plus1)
        f_l_star = self.blend_layer(stylegan2.conv1, f_l_1_star, w_plus1, w_plus2, masks[0], noise[0])
        skip = stylegan2.to_rgb1(f_l_star, w_plus2[:, 1])

        i = 1
        f_l_1_star = f_l_star
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            stylegan2.convs[::2], stylegan2.convs[1::2], noise[1::2], noise[2::2], stylegan2.to_rgbs
        ):
            f_l_star = self.blend_layer(conv1, f_l_1_star, w_plus1[:, i], w_plus2[:, i], masks[i], noise1)
            f_l_star = self.blend_layer(conv2, f_l_star, w_plus1[:, i+1], w_plus2[:, i+1], masks[i+1], noise2)
            skip = to_rgb(f_l_star, w_plus2[:, i+2], skip)
            f_l_1_star = f_l_star
            i += 2

        image = skip
        return image

    def predict_mask(self, features):
        masks = []
        for i in range(len(features)):
            masks.append(self.conv_attn_nets[i](features[i]))
        return masks

    def forward(self, w_plus):
        delta_w_plus = self.mapper(w_plus)
        w_plus2 = w_plus + delta_w_plus
        image, f_l = self.stylegan2_forward(w_plus)
        image_bar, _ = self.stylegan2_forward(w_plus2, return_features=False)
        masks = self.predict_mask(f_l)
        image_star = self.stylegan2_blended_forward(w_plus, w_plus2, masks)
        return image, image_star, image_bar, masks, delta_w_plus
