class PerceptualDistance(nn.Module):

    def __init__(self, device='cpu'):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False

        self.layer_indices = [3, 8, 15, 22]
        self.vgg = vgg
        self.device = device

        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.to(device)

    def get_features(self, x):
        x = (x + 1) / 2
        x = (x - self.mean) / self.std

        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layer_indices:
                features.append(x)
        return features

    def forward(self, img1, img2):
        feats1 = self.get_features(img1)
        feats2 = self.get_features(img2)

        dist = 0.0
        for f1, f2 in zip(feats1, feats2):
            dist = dist + (f1 - f2).pow(2).mean()
        return dist


def slerp(w1, w2, t):
    w1_norm = w1 / w1.norm(dim=1, keepdim=True)
    w2_norm = w2 / w2.norm(dim=1, keepdim=True)
    omega = torch.acos((w1_norm * w2_norm).sum(1, keepdim=True).clamp(-1, 1))
    sin_omega = torch.sin(omega) + 1e-8
    t = t.view(-1, 1)
    return (torch.sin((1 - t) * omega) / sin_omega) * w1 + (torch.sin(t * omega) / sin_omega) * w2


def lerp(w1, w2, t):
    t = t.view(-1, 1)
    return (1 - t) * w1 + t * w2


def generate_from_single_w(G, w):
    x = G.first_block(w)
    for block in G.synth_blocks:
        x = block(x, w)
    return torch.tanh(G.to_rgb(x))


def compute_ppl(G, perceptual_metric, z_dim=512, num_samples=100, epsilon=1e-4, device='cpu'):

    G.eval()
    total_distance = 0.0

    with torch.no_grad():
        for _ in range(num_samples):
            z1 = torch.randn(1, z_dim, device=device)
            z2 = torch.randn(1, z_dim, device=device)

            w1 = G.mapping(z1)
            w2 = G.mapping(z2)

            t = torch.rand(1, device=device)

            w_t = lerp(w1, w2, t)
            w_t_eps = lerp(w1, w2, t + epsilon)
            img_t = generate_from_single_w(G, w_t)
            img_t_eps = generate_from_single_w(G, w_t_eps)

            dist = perceptual_metric(img_t, img_t_eps)
            total_distance += dist.item() / (epsilon ** 2)

    G.train()
    return total_distance / num_samples

class InceptionFeatureExtractor(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        inception.fc = nn.Identity()
        inception.eval()
        for p in inception.parameters():
            p.requires_grad = False
        self.inception = inception
        self.device = device
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device)

    def forward(self, x):
        x = (x + 1) / 2
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        with torch.no_grad():
            features = self.inception(x)
        return features


def get_real_features(dataloader, extractor, num_samples, device):
    feats = []
    collected = 0
    for x_real, _ in dataloader:
        x_real = x_real.to(device)
        f = extractor(x_real)
        feats.append(f.cpu().numpy())
        collected += x_real.shape[0]
        if collected >= num_samples:
            break
    return np.concatenate(feats, axis=0)[:num_samples]


def get_fake_features(G, extractor, z_dim, num_samples, device, batch_size=32):
    G.eval()
    feats = []
    with torch.no_grad():
        collected = 0
        while collected < num_samples:
            bs = min(batch_size, num_samples - collected)
            z = torch.randn(bs, z_dim, device=device)
            x_fake = G(z, mixing_prob=0.0)
            f = extractor(x_fake)
            feats.append(f.cpu().numpy())
            collected += bs
    G.train()
    return np.concatenate(feats, axis=0)

def compute_fid(real_features, fake_features, eps=1e-6):

    mu_r, mu_f = real_features.mean(axis=0), fake_features.mean(axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_f = np.cov(fake_features, rowvar=False)

    diff = mu_r - mu_f

    covmean, _ = linalg.sqrtm(
        (sigma_r + eps * np.eye(sigma_r.shape[0])) @ (sigma_f + eps * np.eye(sigma_f.shape[0])),
        disp=False
    )
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean)
    return float(fid)

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