"""Checkpointing, sample-image saving, and training-curve plotting."""

import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils

from .Model import Discriminator, Generator


def save_sample_images(G, z_dim, epoch, device, num_images=16):
    G.eval()
    with torch.no_grad():
        z = torch.randn(num_images, z_dim, device=device)
        fake_images = G(z)
        fake_images = (fake_images + 1) / 2
    G.train()

    grid = vutils.make_grid(fake_images.cpu(), nrow=4, padding=2)

    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.title(f"Generated samples - epoch {epoch+1}")
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
    plt.close()


def save_checkpoint(G, D, opt_g, opt_d, epoch, history, path="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'opt_g_state_dict': opt_g.state_dict(),
        'opt_d_state_dict': opt_d.state_dict(),
        'history': history,
    }, path)
    print(f"Checkpoint saved to {path} (epoch {epoch+1})")


def load_checkpoint(path, z_dim=512, lr_g=2e-4, lr_d=2e-4, device='cuda'):
    checkpoint = torch.load(path, map_location=device)

    G = Generator(z_dim=z_dim).to(device)
    D = Discriminator().to(device)
    G.load_state_dict(checkpoint['G_state_dict'])
    D.load_state_dict(checkpoint['D_state_dict'])

    opt_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.0, 0.99))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.0, 0.99))
    opt_g.load_state_dict(checkpoint['opt_g_state_dict'])
    opt_d.load_state_dict(checkpoint['opt_d_state_dict'])

    history = checkpoint.get(
        'history',
        {'d_loss': [], 'g_loss': [], 'fid': [], 'fid_epoch': [], 'ppl': [], 'ppl_epoch': []}
    )
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from {path}, resuming at epoch {start_epoch}")
    return G, D, opt_g, opt_d, history, start_epoch


def plot_training_curves(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history['d_loss'], label='D loss', alpha=0.7)
    axes[0].plot(history['g_loss'], label='G loss', alpha=0.7)
    axes[0].set_xlabel('Training step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Discriminator / Generator Loss')
    axes[0].legend()

    axes[1].plot(history['fid_epoch'], history['fid'], marker='o', color='tab:red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('FID (lower = better)')
    axes[1].set_title('FID over training')

    axes[2].plot(history['ppl_epoch'], history['ppl'], marker='o', color='tab:blue')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Perceptual Path Length')
    axes[2].set_title('PPL over training (lower = more disentangled)')

    plt.tight_layout()
    plt.show()
