"""
Entry point: downloads the dataset and runs training.

    python run.py

Automatically mounts Google Drive and checkpoints there when run inside
Google Colab; falls back to a local ./checkpoints folder otherwise.
"""

import os

from dataset import download_dataset
from train import train
from utils import load_checkpoint, plot_training_curves

try:
    from google.colab import drive
    drive.mount('/content/drive')
    CHECKPOINT_PATH = "/content/drive/MyDrive/stylegan_checkpoint.pth"
except ImportError:
    os.makedirs("checkpoints", exist_ok=True)
    CHECKPOINT_PATH = "checkpoints/stylegan_checkpoint.pth"


if __name__ == "__main__":
    data_root = download_dataset()

    G, D, opt_g, opt_d, history = train(
        data_root=data_root,
        batch_size=64,
        num_epochs=1,
        eval_every=1,
        checkpoint_path=CHECKPOINT_PATH,
        fid_samples=200,
        ppl_samples=50,
    )
    
    # To resume training from a saved checkpoint instead, use:
        # G, D, opt_g, opt_d, history, start_epoch = load_checkpoint(CHECKPOINT_PATH, device="cuda")
        # G, D, opt_g, opt_d, history = train(
        #     data_root=data_root, G=G, D=D, opt_g=opt_g, opt_d=opt_d,
        #     history=history, start_epoch=start_epoch, num_epochs=3,
        #     checkpoint_path=CHECKPOINT_PATH,
        # )
  
    plot_training_curves(history)
