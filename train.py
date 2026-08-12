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

def train(
    data_root,
    z_dim=512,
    batch_size=32,
    num_epochs = 1,
    lr_g=2e-4,
    lr_d=2e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    G=None,
    D=None,
    opt_g=None,
    opt_d=None,
    history=None,
    start_epoch=0,
    checkpoint_path="checkpoint.pth",
    save_every=1,
    eval_every=1,
    fid_samples=200,
    ppl_samples=50
):
    if G is None:
        G = Generator(z_dim=z_dim).to(device)
    else:
        G = G.to(device)

    if D is None:
        D = Discriminator().to(device)
    else:
        D = D.to(device)

    if opt_g is None:
        opt_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.0, 0.99))
    if opt_d is None:
        opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.0, 0.99))

    if history is None:
        history = {'d_loss': [], 'g_loss': [], 'fid': [], 'fid_epoch': [], 'ppl': [], 'ppl_epoch': []}

    dataloader = get_dataloader(data_root, image_size=64, batch_size=batch_size)

    inception_extractor = InceptionFeatureExtractor(device=device)
    vgg_extractor = PerceptualDistance(device=device)

    real_features = get_real_features(dataloader, inception_extractor, fid_samples, device)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{start_epoch + num_epochs}")
        for i, (x_real, _) in enumerate(loop):
            x_real = x_real.to(device)
            bs = x_real.shape[0]

            # Discriminator step
            z = torch.randn(bs, z_dim, device=device)
            x_fake = G(z)

            opt_d.zero_grad()
            d_loss = discriminator_loss(D, x_real, x_fake, gamma=2*epoch)
            d_loss.backward()
            opt_d.step()

            # Generator step
            z = torch.randn(bs, z_dim, device=device)
            x_fake = G(z)

            opt_g.zero_grad()
            g_loss = generator_loss(D, x_fake)
            g_loss.backward()
            opt_g.step()

            history['d_loss'].append(d_loss.item())
            history['g_loss'].append(g_loss.item())

            loop.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())

        save_sample_images(G, z_dim, epoch, device)

        if (epoch + 1) % eval_every == 0:
            fake_features = get_fake_features(G, inception_extractor, z_dim, fid_samples, device)
            fid_score = compute_fid(real_features, fake_features)
            history['fid'].append(fid_score)
            history['fid_epoch'].append(epoch)

            ppl_score = compute_ppl(G, vgg_extractor, z_dim=z_dim, num_samples=ppl_samples, device=device)
            history['ppl'].append(ppl_score)
            history['ppl_epoch'].append(epoch)

            print(f"Epoch {epoch+1}: FID={fid_score:.2f}  PPL={ppl_score:.2f}")

        if (epoch + 1) % save_every == 0:
            save_checkpoint(G, D, opt_g, opt_d, epoch, history, path=checkpoint_path)

    return G, D, opt_g, opt_d, history