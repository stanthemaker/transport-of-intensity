import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
from scipy.io import savemat
import torch
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

from unet import UNet
from dataset import sourceDataset, targetDataset


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate test set with pretrained Deeplabv3 model."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory path for predition",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=int,
        default=0,
        help="device ID",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None, help="Model path to load"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    torch.cuda.empty_cache()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(device)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    model = UNet(1, 1)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    test_set = targetDataset(mode="test", path="../data/0226/train")
    print(len(test_set))
    namelist = test_set.images_list
    test_loader = DataLoader(test_set, shuffle=False, batch_size=1)

    MSEs = []
    for i, batch in enumerate(test_loader):
        x, y = batch

        x = x.to(device)
        phase_pred, _, _ = model(x)
        phase_pred = phase_pred.squeeze().detach().cpu().numpy()
        y = y.squeeze().numpy()

        name = namelist[i]
        name = os.path.basename(name).split(".")[0]
        save_path = os.path.join(args.output, f"{name}_ref.mat")
        savemat(save_path, {"phi_recon": phase_pred})
        save_path = os.path.join(args.output, f"{name}.png")
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        input = x.squeeze().detach().cpu().numpy()
        axes[0].imshow(input, cmap="gray")
        axes[0].set_title("input")
        cbar_gt = plt.colorbar(axes[0].imshow(input, cmap="gray"), ax=axes[0])
        cbar_gt.set_label("grayscale")

        axes[1].imshow(y, cmap="viridis")
        axes[1].set_title("Ground Truth")
        cbar_gt = plt.colorbar(axes[1].imshow(y, cmap="viridis"), ax=axes[1])
        cbar_gt.set_label("Radiance")

        y = np.maximum(y, 0)
        axes[2].imshow(y, cmap="viridis")
        axes[2].set_title("Clipped Ground Truth")
        cbar_gt = plt.colorbar(axes[2].imshow(y, cmap="viridis"), ax=axes[2])
        cbar_gt.set_label("Radiance")

        axes[3].imshow(phase_pred, cmap="viridis")
        axes[3].set_title("Prediction")
        cbar_pred = plt.colorbar(axes[3].imshow(phase_pred, cmap="viridis"), ax=axes[3])
        cbar_pred.set_label("Radiance")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
