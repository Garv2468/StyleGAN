
from google.colab import drive
drive.mount('/content/drive')

if __name__ == "__main__":

#   G, D, opt_g, opt_d, history, start_epoch = load_checkpoint("/content/drive/MyDrive/stylegan_checkpoint.pth", device = 'cuda')
#   G, D, opt_g, opt_d, history = train(
#     data_root=path, G=G, D=D, opt_g=opt_g, opt_d=opt_d,
#     history=history, start_epoch=start_epoch, num_epochs=3, checkpoint_path="/content/drive/MyDrive/stylegan_checkpoint.pth"
#  )
#   plot_training_curves(history)

  G, D, opt_g, opt_d, history = train(
    data_root=path,
    batch_size=64,
    num_epochs=1,
    eval_every=1,
    checkpoint_path = "/content/drive/MyDrive/stylegan_checkpoint.pth",
    fid_samples=200,
    ppl_samples=50
)

  plot_training_curves(history)