import os
import torch
import random
import numpy as np
import torchvision as tv
import matplotlib.pyplot as plt
from scipy.io import loadmat

from torchvision.transforms import functional as TF
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.utils import save_image
from torch.nn import functional as func

from utils import *

transform = tv.transforms.Compose(
    [
        tv.transforms.RandomHorizontalFlip(p=0.5),
        tv.transforms.RandomVerticalFlip(p=0.5),
        tv.transforms.RandomRotation(
            (0, 359),
            interpolation=tv.transforms.InterpolationMode.BILINEAR,
            expand=False,
        ),
    ]
)


class sourceDataset(Dataset):
    def __init__(self, mode):
        path = "../data"
        self.images_list = []
        self.mode = mode
        if mode == "train":
            path = os.path.join(path, "unlabelled")
            self.images_list = sorted(
                [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".tif")]
            )
        elif mode == "valid":
            path = os.path.join(path, "labelled")
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith("_img.tif"):
                        file_id = int(file[:5])
                        if file_id % 2 == 0:
                            self.images_list.append(os.path.join(root, file))

        elif mode == "test":
            path = os.path.join(path, "labelled")
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith("_img.tif"):
                        file_id = int(file[:5])
                        if file_id % 2 == 1:
                            self.images_list.append(os.path.join(root, file))

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        # x : input I(z=dz), y : groundtruth phi(z=0)
        img = Image.open(self.images_list[idx])
        img = np.maximum(img, 0)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        max_rad = 1 * torch.pi
        y = torch.from_numpy(img).unsqueeze(dim=0)
        if self.mode == "train":
            y = transform(y)
        y = y * max_rad

        u0 = torch.exp(1j * y)
        z = random.gauss(mu=10e-6, sigma=0)
        _lambda = random.gauss(mu=4e-7, sigma=0)
        u_delta_z = fresnel_prop_torch(u0, z, _lambda)
        I_delta_z = torch.abs(u_delta_z) ** 2

        x = I_delta_z.to(torch.float32)
        x = x / torch.mean(I_delta_z)
        noise = generate_noise(x)
        x = x + noise

        return x, y

    def get_random_batch(self, batch_size):
        indices = random.sample(range(len(self)), batch_size)
        batch = [self[i] for i in indices]
        x_batch, y_batch = zip(*batch)

        x_batch = torch.stack(x_batch)
        y_batch = torch.stack(y_batch)

        return x_batch, y_batch


class targetDataset(Dataset):
    def __init__(self, mode):
        path = "../data"
        self.images_list = []
        self.mode = mode
        if mode == "train":
            path = os.path.join(path, "0130/train")
            self.images_list = sorted(
                [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")]
            )
            self.gt_list = sorted(
                [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".mat")]
            )
        elif mode == "test":
            path = os.path.join(path, "0130/test")
            self.images_list = sorted(
                [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")]
            )

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if self.mode == "train":
            img = Image.open(self.images_list[idx])
            x = TF.to_tensor(img).to(torch.float32)

            # img = np.invert(img)
            # x = x.unsqueeze(dim=0)

            x = tv.transforms.Resize(
                size=[600, 600], interpolation=tv.transforms.InterpolationMode.BICUBIC
            )(x)
            x = x / torch.mean(x)

            y = loadmat(self.gt_list[idx])["phi_recon"]
            # y = np.maximum(y, 0)  # clipp negative values
            y = torch.from_numpy(y).to(torch.float32)
            y = y.unsqueeze(dim=0)
            y = tv.transforms.Resize(
                size=[600, 600], interpolation=tv.transforms.InterpolationMode.BICUBIC
            )(y)

            if np.random.rand() > 0.5:
                x = TF.hflip(x)
                y = TF.hflip(y)

            if np.random.rand() > 0.5:
                x = TF.vflip(x)
                y = TF.vflip(y)

            return x, y

        elif self.mode == "test":
            img = Image.open(self.images_list[idx])
            x = TF.to_tensor(img).to(torch.float32)
            # img = np.invert(img)
            # x = x.unsqueeze(dim=0)
            x = tv.transforms.Resize(
                size=[600, 600], interpolation=tv.transforms.InterpolationMode.BICUBIC
            )(x)
            x = x / torch.mean(x)

            return x

    def get_random_batch(self, batch_size):
        indices = random.sample(range(len(self)), batch_size)
        batch = [self[i] for i in indices]
        x_batch, y_batch = zip(*batch)

        x_batch = torch.stack(x_batch)
        y_batch = torch.stack(y_batch)

        return x_batch, y_batch


# dataset = sourceDataset(mode="train")
# print(len(dataset))
# x, y = dataset.__getitem__(0)
# x, y = dataset.get_random_batch(4)
# # plt.savefig("temp.png")
# print(x.shape)
# print(y.shape)
