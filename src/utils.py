import torch
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


# img = Image.open("/home/R13k47024/Stan/tie/data/labelled/PC3/00001_PC3_img.tif")
# img = np.maximum(img , 0)
# img = (img - np.min(img)) / (np.max(img) - np.min(img))
# u0 = np.exp(1j * img * 0.6 * np.pi)

# u1_np =fresnel_prop_np(u0)
# u1_torch = fresnel_prop_torch(torch.tensor(u0, dtype=torch.complex64))
# u1_np = torch.from_numpy(u1_np)
# print(u1_np)
# print(u1_torch)
# mse = torch.mean(torch.abs(u1_np - u1_torch) ** 2)
# print(mse)


# Load the .mat file (assumed to contain 'u2' calculated by MATLAB)
# data = loadmat('u2.mat')
# u_matlab = data['u1']  # Adjust the key if different

# img = io.imread('usaf.png')
# u0 = np.exp(1j * np.pi * img)

# # Parameters
# dz = 1e-4       # Propagation distance
# L = 5.12e-4     # Side length
# wavelength = 6.328e-7  # Wavelength

# # Perform propagation
# u_py = propTF(u0, L, wavelength, dz)

# # Compare the results
# difference = np.abs(u_py - u_matlab)

# # Display the difference as an image
# plt.imshow(difference, cmap='viridis')
# plt.colorbar()
# plt.title("Difference between MATLAB and Python u2")
# plt.savefig('tmp.png')

# # Optionally, compute mean absolute error
# mae = np.mean(difference)
# print(f"Mean Absolute Error between MATLAB and Python u2: {mae}")
