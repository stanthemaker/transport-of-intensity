import torch
import numpy as np
import scipy
from PIL import Image
from torch import linalg as la
import matplotlib.pyplot as plt
from torch.nn import functional as F
from scipy.fft import fft2, ifft2, fftshift, ifftshift

from ignite.engine.engine import Engine
from ignite.metrics import SSIM


def fresnel_prop_np(u0, z=1e-4, L=5.12e-4, wavelength=6.32e-7):
    """
    Fresnel propagation using the Transfer function method in Python.

    Parameters:
    u0         - Complex amplitude of the beam at the source plane
    L          - Side length of the simulation window of the source plane
    wavelength - Wavelength of light
    z          - Propagation distance

    Returns:
    u1         - Complex amplitude of the beam at the observation plane
    """
    # Input array size
    M = u0.shape[-1]
    # Sampling interval size
    dx = L / M
    # Frequency coordinates sampling
    fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / L, M)
    FX, FY = np.meshgrid(fx, fx)
    # Transfer function
    H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    H = fftshift(H)

    U0 = fft2(fftshift(u0))
    # Apply the transfer function
    U1 = H * U0
    # Inverse Fourier transform to obtain u2
    u1 = ifftshift(ifft2(U1))

    return u1


def fresnel_prop_torch(u0, z=10e-6, wavelength=6.32e-7):
    """
    Fresnel propagation using the Transfer function method with PyTorch.

    Parameters:
    u0         - Complex amplitude of the beam at the source plane (torch tensor, complex64 or complex128)
    L          - Side length of the field of view of image
    wavelength - Wavelength of light
    z          - Propagation distance

    Returns:
    u1         - Complex amplitude of the beam at the observation plane (torch tensor)
    """
    # Input array size
    M = u0.shape[-1]

    # Sampling interval size
    L = 5.12e-4
    dx = L / M

    # Frequency coordinates sampling
    fx = torch.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / L, M)
    FX, FY = torch.meshgrid(fx, fx, indexing="ij")
    FX = FX.T.to(torch.complex128)
    FY = FY.T.to(torch.complex128)

    # Transfer function
    H = torch.exp(-1j * torch.pi * wavelength * z * (FX**2 + FY**2))
    H = torch.fft.fftshift(H)

    U0 = torch.fft.fft2(torch.fft.fftshift(u0))
    U1 = H * U0
    u1 = torch.fft.ifftshift(torch.fft.ifft2(U1))

    return u1


def optimizer_scheduler(optimizer, p, init=1e-2):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = init / (1.0 + 10 * p) ** 0.75

    return optimizer


def mdd_loss(feat_src, feat_tgt):
    batch_size = feat_src.size(0)
    softmax_src = F.softmax(feat_src, dim=1)
    softmax_tgt = F.softmax(feat_tgt, dim=1)

    mddloss = la.norm(softmax_src - softmax_tgt, ord=2, dim=1).sum() / float(batch_size)
    return mddloss


def masked_MSE(pred, gt):
    mask = (gt >= 0).float()
    mse_loss = ((pred - gt) ** 2) * mask
    return mse_loss.sum() / mask.sum()


def eval_step(engine, batch):
    return batch


def normalize_01(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


class phaseSSIM:
    def __init__(self):
        self.evaluator = Engine(eval_step)
        metric = SSIM(data_range=1.0)
        metric.attach(self.evaluator, "ssim")

    def eval(self, pred, gt):
        pred_norm = normalize_01(pred)
        gt_norm = normalize_01(gt)

        # mse = masked_MSE(pred, gt)

        state = self.evaluator.run([[pred_norm, gt_norm]])
        ssim = state.metrics["ssim"]
        return torch.tensor((1 - ssim))


def plot(pred, gt, savepath):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(gt, cmap="viridis")
    axes[0].set_title("Ground Truth")
    cbar_gt = plt.colorbar(axes[0].imshow(gt, cmap="viridis"), ax=axes[0])
    cbar_gt.set_label("Radiance")

    axes[1].imshow(pred, cmap="viridis")
    axes[1].set_title("Prediction")
    cbar_pred = plt.colorbar(axes[1].imshow(pred, cmap="viridis"), ax=axes[1])
    cbar_pred.set_label("Radiance")

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def generate_noise(x, sigma=20):
    shape = x.shape[-2:]
    scale = x.mean().numpy() * 0.08

    noise = np.random.randn(*shape)
    noise = scipy.ndimage.gaussian_filter(noise, sigma=sigma)
    noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise)) * scale
    return torch.from_numpy(noise).float()
