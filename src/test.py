import torch
import piqa

# PSNR
# x = torch.rand(5, 3, 256, 256)
# y = torch.rand(5, 3, 256, 256)

# psnr = piqa.PSNR()
# l = psnr(x, y)

# SSIM
x = torch.rand(5, 1, 256, 256, requires_grad=True).cuda()
y = torch.rand(5, 1, 256, 256).cuda()
# kernel = gaussian_kernel(7).repeat(3, 1, 1)
ssim = piqa.SSIM(n_channels=1).cuda()
# print(type(ssim))
l = 1 - ssim(x, y)
# print(type(l))
l.backward()
