import torch
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image

img = Image.open("/home/R13k47024/Stan/tie/data/labelled/PC3/00001_PC3_img.tif")
img = np.maximum(img, 0)
img = (img - np.min(img)) / (np.max(img) - np.min(img))
img = img.astype(np.float64)

u0 = np.exp(1j * img * 0.6 * np.pi)
u0_t = torch.from_numpy(u0).to(torch.complex128)

M, _ = u0.shape

z = 1e-4
L = 5.12e-4
wavelength = 6.32e-7

dx = L / M
fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / L, M)
FX, FY = np.meshgrid(fx, fx)
H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
H = fftshift(H)

U0 = fftshift(u0)
U0 = fft2(U0)
U1 = H * U0
print(U1.dtype)
u1 = ifftshift(ifft2(U1))
# ---------- torch

fx_t = torch.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / L, M)
FX_t, FY_t = torch.meshgrid(fx_t, fx_t, indexing="ij")
FX_t = FX_t.T.to(torch.complex128)
FY_t = FY_t.T.to(torch.complex128)

H_t = torch.exp(-1j * torch.pi * wavelength * z * (FX_t**2 + FY_t**2))
H_t = torch.fft.fftshift(H_t)

U0_t = torch.fft.fftshift(u0_t)
U0_t = torch.fft.fft2(U0_t)
U1_t = H_t * U0_t
u1_t = torch.fft.ifftshift(torch.fft.ifft2(U1_t))
print("testing:", np.allclose(u1, u1_t.numpy()))
# torch.testing.assert_allclose(torch.from_numpy(U1), U1_t)
