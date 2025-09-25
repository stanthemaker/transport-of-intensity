import torch
import numpy as np

from dataset import *

if __name__ == "__main__":
    # dataset = mnistDataset(mode="test")
    dataset = sourceDataset(mode="train")
    n = len(dataset)
    print(f"dataset size:{n}")
    x_sum = 0.0
    x_sq_sum = 0.0
    count = 0

    for i in range(n):
        x, y = dataset[i]

        x_np = x.cpu().numpy().flatten()
        x_sum += x_np.sum()
        x_sq_sum += (x_np**2).sum()
        count += x_np.size
    x_mean = x_sum / count
    x_std = np.sqrt(x_sq_sum / count - x_mean**2)

    print(f"x mean: {x_mean:.8f}, x std: {x_std:.8f}")
