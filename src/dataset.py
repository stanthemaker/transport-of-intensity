import os
import torch
import random
import numpy as np
import torchvision as tv
import matplotlib.pyplot as plt

from torchvision.transforms import functional as TF
from PIL import Image
from torch.utils.data import Dataset
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


class phaseDataset(Dataset):
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

        elif mode == "experiment":
            path = os.path.join(path, "1220")
            # for file in os.listdir(path):
            #     if file.endswith("_050.jpg")
            self.images_list = sorted(
                [
                    os.path.join(path, x)
                    for x in os.listdir(path)
                    if x.endswith("_050.jpg")
                ]
            )

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        # x : input I(z=dz), y : groundtruth phi(z=0)
        if self.mode == "experiment":
            img = Image.open(self.images_list[idx])
            img = np.invert(img)
            x = torch.from_numpy(img).to(torch.float32)
            x = x.unsqueeze(dim=0)
            x = tv.transforms.Resize(
                size=[600, 600], interpolation=tv.transforms.InterpolationMode.BICUBIC
            )(x)
            x = x / torch.mean(x)
            y = 0  # null gt

        else:
            img = Image.open(self.images_list[idx])
            img = np.maximum(img, 0)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            max_rad = 0.6 * torch.pi
            y = torch.from_numpy(img).unsqueeze(dim=0)
            y = transform(y)
            y = y * max_rad

            u0 = torch.exp(1j * y)
            z = random.gauss(mu=1e-5, sigma=0)
            _lambda = random.gauss(mu=4e-7, sigma=0)
            u_delta_z = fresnel_prop_torch(u0, z, _lambda)
            I_delta_z = torch.abs(u_delta_z) ** 2

            x = I_delta_z.to(torch.float32)
            # normalized by intensity
            x = x / torch.mean(I_delta_z)
            # x = func.normalize(x)
        # save_image(y, "y.png")
        # print(x.dtype, y.dtype)

        return x, y


# dataset = phaseDataset(mode="experiment")
# print(len(dataset))
# x, y = dataset.__getitem__(1)
# plt.hist(x.numpy().flat, bins=100)
# plt.savefig("temp.png")
# # print(x.shape)
# print(y.shape)
