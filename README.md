# StyleGAN From Scratch

A from-scratch PyTorch implementation of StyleGAN. The mapping network, learned constant input, adaptive instance normalization (AdaIN), per-pixel noise injection, and style mixing are all implemented directly rather than adapted from an existing codebase. FID and Perceptual Path Length (PPL) are computed during training to track sample quality and latent-space disentanglement.

## Architecture

- **Mapping network** — 8 fully connected layers mapping `z` to the intermediate latent `w`
- **Generator** — a learned 4x4 constant input, followed by AdaIN-modulated synthesis blocks with per-pixel noise injection and progressive upsampling
- **Discriminator** — a convolutional stack trained with an R1 gradient penalty
- **Style mixing** — regularization by crossing over between two `w` codes at a random synthesis block during training
- **Metrics** — FID (via Inception-v3 features) and PPL (via a VGG16 perceptual distance), computed every `eval_every` epochs

## Project structure

```
stylegan-from-scratch/
├── run.py              # Entry point: downloads data and starts training
├── requirements.txt
└── src/
    ├── model.py         # Generator, Discriminator, and building blocks
    ├── losses.py        # Adversarial loss with R1 gradient penalty
    ├── dataset.py        # Dataset download + DataLoader
    ├── metrics.py         # FID, PPL, perceptual distance
    ├── utils.py            # Checkpointing, sample saving, plotting
    └── train.py             # Training loop
```

## Setup

```bash
git clone <repo-url>
cd stylegan-from-scratch
pip install -r requirements.txt
```

## Usage

```bash
python run.py
```

This downloads the anime face dataset (`manavkumarjalan93/anime-face-dataset` on Kaggle, via `kagglehub`), trains at 64x64 resolution, saves a sample grid and a checkpoint every epoch, and plots the loss/FID/PPL curves at the end.

Inside Google Colab, `run.py` mounts Google Drive automatically and checkpoints there; outside Colab it checkpoints to a local `checkpoints/` folder instead.

To resume from a saved checkpoint, uncomment the resume block near the top of `run.py`:

```python
G, D, opt_g, opt_d, history, start_epoch = load_checkpoint(CHECKPOINT_PATH, device="cuda")
G, D, opt_g, opt_d, history = train(
    data_root=data_root, G=G, D=D, opt_g=opt_g, opt_d=opt_d,
    history=history, start_epoch=start_epoch, num_epochs=3,
    checkpoint_path=CHECKPOINT_PATH,
)
```

## Known issues

- `train()` calls `get_dataloader(..., image_size=64)`, but `CHANNELS` in `src/model.py` has 6 stages, which makes the Generator/Discriminator's native resolution 128x128. Feeding 64x64 real images into the Discriminator will raise a shape-mismatch error. Fix by either dropping one entry from `CHANNELS` (e.g. `[512, 512, 256, 256, 128]` for 64x64) or removing `image_size=64` in `src/train.py` so `get_dataloader` defaults to 128x128.

## Notes

Built as a learning exercise implementing the StyleGAN paper's core components from scratch — not optimized for large-scale or high-resolution training.