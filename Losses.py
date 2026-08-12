"""Adversarial losses with an R1 gradient penalty on the discriminator."""

import torch
import torch.nn as nn


def discriminator_loss(D, x_real, x_fake, gamma=2):
    x_real = x_real.clone().requires_grad_(True)

    d_real = D(x_real)
    d_fake = D(x_fake.detach())

    grad_real = torch.autograd.grad(
        outputs=d_real.sum(), inputs=x_real, create_graph=True
    )[0]
    grad_penalty_real = (grad_real.view(grad_real.size(0), -1).pow(2).sum(1)).mean()

    criterion = nn.BCELoss()
    loss_fake = criterion(d_fake, torch.zeros_like(d_fake))
    loss_real = criterion(d_real, torch.ones_like(d_real))

    d_loss = loss_fake + loss_real + gamma * grad_penalty_real / 2
    return d_loss


def generator_loss(D, x_fake):
    d_fake = D(x_fake)
    criterion = nn.BCELoss()
    g_loss = criterion(d_fake, torch.ones_like(d_fake))
    return g_loss