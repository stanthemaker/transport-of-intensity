import os, argparse
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

    test_set = targetDataset(mode="train")
    print(len(test_set))
    test_loader = DataLoader(test_set, shuffle=False, batch_size=1)

    MSEs = []
    for i, batch in enumerate(test_loader):
        x, y = batch

        x = x.to(device)
        phase_pred, _, _ = model(x)
        phase_pred = phase_pred.squeeze().detach().cpu().numpy()
        y = y.squeeze().numpy()

        save_path = os.path.join(args.output, f"{i}.png")
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(y, cmap="viridis")
        axes[0].set_title("Ground Truth")
        cbar_gt = plt.colorbar(axes[0].imshow(y, cmap="viridis"), ax=axes[0])
        cbar_gt.set_label("Radiance")

        axes[1].imshow(phase_pred, cmap="viridis")
        axes[1].set_title("Prediction")
        cbar_pred = plt.colorbar(axes[1].imshow(phase_pred, cmap="viridis"), ax=axes[1])
        cbar_pred.set_label("Radiance")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    #     if mode != "experiment":
    #         save_path_gt = os.path.join(args.output, f"{i}_gt.png")
    #         save_image(y, save_path_gt)
    #         y = y.numpy()
    #         phase_pred = phase_pred.detach().cpu().numpy()
    #         mse = np.mean(np.square(phase_pred - y))
    #         print("MSE:", mse)
    #         MSEs.append(mse)

    # if mode != "experiment":
    #     print("avg MSE:", sum(MSEs) / len(MSEs))

    # for i, batch in enumerate(test_loader):
    #     if i == 20:
    #         break
    #     I_delta_z, phase_gt = batch
    #     I_delta_z = I_delta_z.to(device)
    #     phase_pred = model(I_delta_z)

    #     save_path_input = os.path.join(args.output, f"{i}_input.png")
    #     save_path_pred = os.path.join(args.output, f"{i}_pred.png")
    #     save_path_gt = os.path.join(args.output, f"{i}_gt.png")
    #     save_image(phase_pred, save_path_pred)
    #     save_image(phase_gt, save_path_gt)
    #     save_image(I_delta_z, save_path_input)

    #     phase_gt = phase_gt.numpy()
    #     phase_pred = phase_pred.detach().cpu().numpy()
    #     mse = np.mean(np.square(phase_pred - phase_gt))
    #     print("MSE:", mse)
    #     MSEs.append(mse)

    # print("avg MSE:", sum(MSEs) / len(MSEs))
