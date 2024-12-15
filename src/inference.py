import os, argparse
import torch
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

from unet import UNet
from dataset import mnistDataset, phaseDataset


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate test set with pretrained Deeplabv3 model."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory path for predition masks",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None, help="Model path to load"
    )
    return parser.parse_args()


def safe_mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    args = get_args()
    torch.cuda.empty_cache()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    model = UNet(1, 1)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    # model.double()

    valid_set = phaseDataset(is_train=False)
    print(len(valid_set))
    test_loader = DataLoader(valid_set, shuffle=False, batch_size=1)

    MSEs = []
    for i, batch in enumerate(test_loader):
        if i == 20:
            break
        I_delta_z, phase_gt = batch
        I_delta_z = I_delta_z.to(device)
        phase_pred = model(I_delta_z)

        save_path_input = os.path.join(args.output, f"{i}_input.png")
        save_path_pred = os.path.join(args.output, f"{i}_pred.png")
        save_path_gt = os.path.join(args.output, f"{i}_gt.png")
        save_image(phase_pred, save_path_pred)
        save_image(phase_gt, save_path_gt)
        save_image(I_delta_z, save_path_input)

        phase_gt = phase_gt.numpy()
        phase_pred = phase_pred.detach().cpu().numpy()
        mse = np.mean(np.square(phase_pred - phase_gt))
        print("MSE:", mse)
        MSEs.append(mse)

    print("avg MSE:", sum(MSEs) / len(MSEs))
