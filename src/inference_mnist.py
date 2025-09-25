import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
from scipy.io import savemat
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

from unet import UNet
from dataset import mnistDataset


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

    device = f"cuda:{0}" if torch.cuda.is_available() else "cpu"
    print(device)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    model = UNet(1, 1)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    test_set = mnistDataset(mode="test")
    print(len(test_set))
    batch_size = 4
    x_batch, y_batch = test_set.get_random_batch(batch_size=batch_size)
    x_batch = x_batch.to(device)
    with torch.no_grad():
        phase_pred_batch, _, _ = model(x_batch)
    phase_pred_batch = phase_pred_batch.cpu().numpy()
    x_batch = x_batch.cpu().numpy()
    y_batch = y_batch.cpu().numpy()

    for i in range(batch_size):
        input_img = x_batch[i].squeeze()
        gt_img = y_batch[i].squeeze()
        pred_img = phase_pred_batch[i].squeeze()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        im0 = axes[0].imshow(input_img, cmap="gray")
        axes[0].set_title("Input")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(gt_img, cmap="viridis")
        axes[1].set_title("Ground Truth")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(pred_img, cmap="viridis")
        axes[2].set_title("Prediction")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = os.path.join(args.output, f"sample_{i}.png")
        plt.savefig(save_path)
        plt.close()
