import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

path = "../data/0130/train"
# path = os.path.join(path, "0130/train")
# savepath = os.path.join(path, "0130/profile")

imgs= sorted(
    [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")]
)
maps= sorted(
    [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".mat")]
)
for img_path, map_path in zip(imgs, maps):
    filename = os.path.basename(map_path)
    savepath = os.path.join("../data/0130/profile", f"{filename}.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    gt = loadmat(map_path)["phi_recon"]
    img = Image.open(img_path)

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Brightfield")
    cbar_gt = plt.colorbar(axes[0].imshow(img, cmap="gray"), ax=axes[0])
    cbar_gt.set_label("Grayscale")

    # img = Image.open(imgs[i])
    axes[1].imshow(gt, cmap="viridis")
    axes[1].set_title("Ground Truth")
    cbar_gt = plt.colorbar(axes[1].imshow(gt, cmap="viridis"), ax=axes[1])
    cbar_gt.set_label("Radiance")

    gt = np.maximum(gt, 0)
    axes[2].imshow(gt, cmap="viridis")
    axes[2].set_title("Clipped Ground Truth")
    cbar_pred = plt.colorbar(axes[2].imshow(gt, cmap="viridis"), ax=axes[2])
    cbar_pred.set_label("Radiance")

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()