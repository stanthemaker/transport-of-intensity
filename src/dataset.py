import os
import torch
import random
import numpy as np
import torchvision as tv

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


class mnistDataset(Dataset):
    def __init__(self, is_train):
        self.data = tv.datasets.MNIST(
            "../", train=is_train, transform=None, target_transform=None, download=True
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx][0]
        img = np.array(img, dtype=np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        y = torch.from_numpy(img).unsqueeze(dim=0)

        u0 = np.exp(1j * img * np.pi)
        u_delta_z = fresnel_prop(u0)
        I_delta_z = np.abs(u_delta_z) ** 2

        x = torch.from_numpy(I_delta_z).float()
        x = func.normalize(x)
        x = x.unsqueeze(dim=0)

        return x, y


class phaseDataset(Dataset):
    def __init__(self, mode):
        path = "../data"
        self.images_list = []
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
        y = torch.from_numpy(img).unsqueeze(dim=0)
        y = transform(y)

        u0 = torch.exp(1j * y * 0.6 * torch.pi)
        z = random.gauss(mu=1e-4, sigma=2e-5)
        _lambda = random.gauss(mu=5.5e-7, sigma=5e-8)
        u_delta_z = fresnel_prop_torch(u0, z, _lambda)
        I_delta_z = torch.abs(u_delta_z) ** 2

        x = I_delta_z.to(torch.float32)
        x = func.normalize(x)
        # save_image(y, "y.png")
        # print(x.dtype, y.dtype)

        return x, y


# dataset = phaseDataset(mode="test")
# print(len(dataset))
# # dataset = mnistDataset(is_train=True)
# x, y = dataset.__getitem__(1)
# print(x.shape)
