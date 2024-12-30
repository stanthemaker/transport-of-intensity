import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint
import os
import cv2
from sklearn.manifold import TSNE
from utils import *

source_images = []
path = "../data"
source_path = os.path.join(path, "labelled")
for root, dirs, files in os.walk(source_path):
    for file in files:
        if file.endswith("_img.tif"):
            file_id = int(file[:5])
            if file_id % 2 == 1:
                img_path = os.path.join(root, file)
                img = Image.open(img_path)
                img = np.array(img)
                source_images.append(img)
        if len(source_images) > 100:
            break


X_source = np.stack(source_images, axis=0)
w = X_source.shape[1]
h = X_source.shape[2]
X_source = X_source.reshape(X_source.shape[0], -1)
Y_source = np.zeros(X_source.shape[0])
print(X_source.shape, Y_source.shape)

target_images = []
target_path = os.path.join(path, "1220")
for file in os.listdir(target_path):
    if file.endswith("_100.jpg"):
        img_path = os.path.join(target_path, file)
        img = Image.open(img_path)
        img = np.array(img)
        img = cv2.resize(
            img,
            dsize=(w, h),
            interpolation=cv2.INTER_CUBIC,
        )
        img = np.invert(img)
        target_images.append(img)

X_target = np.stack(target_images, axis=0)
X_target = X_target.reshape(X_target.shape[0], -1)
Y_target = np.ones(X_target.shape[0])
print(X_target.shape, Y_target.shape)

X = np.concatenate((X_source, X_target), axis=0)
Y = np.concatenate((Y_source, Y_target), axis=0)

tsne = TSNE(n_components=2, perplexity=12, init="random", learning_rate="auto")
X_embedded_source = tsne.fit_transform(X)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(
    X_embedded_source[:, 0],
    X_embedded_source[:, 1],
    c=Y,
    alpha=0.8,
)

ax.set_title("t-SNE on MNIST")
ax.set_xlabel("dimension 1")
ax.set_ylabel("dimension 2")
ax.grid(True)
fig.tight_layout()
fig.savefig("visualize.png")
