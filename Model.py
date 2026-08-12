"""Generator, Discriminator, and the building blocks used by both."""

import numpy as np
import torch
import torch.nn as nn


class MapNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512):
        super().__init__()
        layers = []
        in_dim = z_dim
        for _ in range(7):
            layers.append(nn.Linear(in_dim, 512))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(512, w_dim))
        self.map = nn.Sequential(*layers)

    def forward(self, z):
        z = z / torch.norm(z, dim=1, keepdim=True)
        return self.map(z)


class LearnedConstant(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.constant = nn.Parameter(torch.randn(1, channels, 4, 4))

    def forward(self, batch_size):
        return self.constant.repeat(batch_size, 1, 1, 1)


class AdaIN(nn.Module):
    def __init__(self, w_dim, num_channel):
        super().__init__()
        self.affine = nn.Linear(w_dim, 2 * num_channel)

    def forward(self, x, w):
        y = self.affine(w)
        c = x.shape[1]
        y_s = y[:, :c].view(-1, c, 1, 1)
        y_b = y[:, c:].view(-1, c, 1, 1)
        mn = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True)
        return y_s * (x - mn) / (std + 1e-8) + y_b


class NoiseInj(nn.Module):
    def __init__(self, num_channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, num_channel, 1, 1))

    def forward(self, x):
        batch_size, num_channel, height, width = x.shape
        noise = torch.randn(batch_size, 1, height, width, device=x.device)
        return x + noise * self.weight


class FirstBlock(nn.Module):
    def __init__(self, channels=512, w_dim=512):
        super().__init__()
        self.const = LearnedConstant(channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.inj = NoiseInj(channels)
        self.adain = AdaIN(w_dim, channels)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, w):
        batch_size = w.shape[0]
        x = self.const(batch_size)
        x = self.conv(x)
        x = self.inj(x)
        x = self.adain(x, w)
        x = self.lrelu(x)
        return x


class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim=512):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.inj1 = NoiseInj(out_channels)
        self.adain1 = AdaIN(w_dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.inj2 = NoiseInj(out_channels)
        self.adain2 = AdaIN(w_dim, out_channels)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        x = self.upsample(x)
        x = self.lrelu(self.adain1(self.inj1(self.conv1(x)), w))
        x = self.lrelu(self.adain2(self.inj2(self.conv2(x)), w))
        return x


CHANNELS = [512, 512, 256, 256, 128, 64]  # 4x4, 8x8, 16x16, 32x32, 64x64


def mix_styles(w1, w2, num_blocks, crossover=None):
    if crossover is None:
        p = np.random.randint(1, num_blocks)
    else:
        p = crossover
    return [w1 if i < p else w2 for i in range(num_blocks)]


class Generator(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, channels=CHANNELS):
        super().__init__()
        self.mapping = MapNetwork(z_dim, w_dim)
        self.first_block = FirstBlock(channels[0], w_dim)

        self.synth_blocks = nn.ModuleList()
        for i in range(1, len(channels)):
            self.synth_blocks.append(
                SynthesisBlock(channels[i - 1], channels[i], w_dim)
            )

        self.to_rgb = nn.Conv2d(channels[-1], 3, kernel_size=1)
        self.num_blocks = len(channels)

    def forward(self, z, mixing_prob=0.9):
        batch_size = z.shape[0]

        use_mixing = np.random.rand() < mixing_prob
        if use_mixing:
            z2 = torch.randn_like(z)
            w1 = self.mapping(z)
            w2 = self.mapping(z2)
            w_list = mix_styles(w1, w2, self.num_blocks)
        else:
            w = self.mapping(z)
            w_list = [w] * self.num_blocks

        x = self.first_block(w_list[0])
        for i, block in enumerate(self.synth_blocks):
            x = block(x, w_list[i + 1])

        img = torch.tanh(self.to_rgb(x))  # output in [-1, 1]
        return img


class Discriminator(nn.Module):
    def __init__(self, channels=CHANNELS):
        super().__init__()
        rev_channels = channels[::-1]
        layers = [nn.Conv2d(3, rev_channels[0], kernel_size=1), nn.LeakyReLU(0.2)]

        for i in range(len(rev_channels) - 1):
            layers.append(nn.Conv2d(rev_channels[i], rev_channels[i + 1], 3, padding=1))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.AvgPool2d(2))

        self.features = nn.Sequential(*layers)
        final_res = 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(rev_channels[-1] * final_res * final_res, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)