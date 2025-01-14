import torch
from torch import linalg as la
from torch.nn import functional as F
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image


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


def fresnel_prop_torch(u0, z=1e-4, wavelength=6.32e-7):
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


def histogram_matching(source, template):
    """
    Adjust the pixel values of the source image to match the histogram of the template image.

    Parameters:
        source (np.ndarray): Source image whose histogram is to be matched.
        template (np.ndarray): Template image whose histogram will be matched to.

    Returns:
        np.ndarray: The transformed source image.
    """
    # Flatten images
    src_values, src_counts = np.unique(source.flatten(), return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.flatten(), return_counts=True)

    # Calculate CDF
    src_cdf = np.cumsum(src_counts).astype(np.float64) / source.size
    tmpl_cdf = np.cumsum(tmpl_counts).astype(np.float64) / template.size

    # Map source pixel values to template pixel values
    interp_values = np.interp(src_cdf, tmpl_cdf, tmpl_values)
    mapped = np.interp(source.flatten(), src_values, interp_values)

    return mapped.reshape(source.shape)


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
