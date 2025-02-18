import os, argparse
import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from datetime import datetime

from torch.utils.data import DataLoader
from torchvision.transforms import functional as tf

# Customized packages.
from utils import *
from unet import UNet
from dataset import sourceDataset


def get_args():
    parser = argparse.ArgumentParser(
        description="UNet-based model for intensity-phase transformation"
    )
    parser.add_argument(
        "--device",
        "-d",
        type=int,
        default=0,
        help="device ID",
    )

    return parser.parse_args()


args = get_args()
exp_name = "denoise_qpi"
dt_string = datetime.now().strftime("%m%d_%H%M_")
exp_name = dt_string + exp_name
config = {
    "Optimizer": "Adam",
    "batch_size": 8,
    "lr": 2e-5,
    "n_epochs": 100,
    "patience": 15,
    "exp_name": dt_string + exp_name,
}
output_dir = f"../output/{exp_name}"
log_file = f"../output/{exp_name}/training_record.log"
ckpt_file = f"../output/{exp_name}/model.ckpt"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
myseed = 7777
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
print(device)

train_set = sourceDataset(mode="train")
valid_set = sourceDataset(mode="valid")
print(len(train_set))
print(len(valid_set))
train_loader = DataLoader(train_set, shuffle=True, batch_size=config["batch_size"])
valid_loader = DataLoader(valid_set, shuffle=True, batch_size=config["batch_size"])


model = UNet(in_dim=1, out_dim=1)
model = model.to(device)

loss_func = nn.MSELoss()
# loss_func = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5)

stale = 0
best_loss = 100
n_epochs = config["n_epochs"]

# ---------- Training ----------
for epoch in tqdm(range(n_epochs + 1)):

    model.train()
    train_loss = []
    train_accs = []

    for i, batch in enumerate(train_loader):
        p = float(i + epoch * len(train_loader)) / n_epochs / len(train_loader)
        optimizer = optimizer_scheduler(optimizer=optimizer, p=p)
        optimizer.zero_grad()

        I_delta_z, phase_gt = batch
        I_delta_z = I_delta_z.to(device)
        phase_gt = phase_gt.to(device)

        phase_pred, _, _ = model(I_delta_z)
        loss = loss_func(phase_pred, phase_gt)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())

    train_loss = sum(train_loss) / len(train_loss)
    with open(log_file, "a") as f:
        f.write(
            f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ], loss = {train_loss:.6f}\n"
        )
    # ---------- inferernce ----------
    if epoch % 10 == 0:
        model.eval()
        batch_size = 2
        x, y = valid_set.get_random_batch(batch_size)
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            phase_pred, _, _ = model(x)

        phase_pred = phase_pred.squeeze().detach().cpu().numpy()
        y = y.squeeze().detach().cpu().numpy()

        for i in range(batch_size):
            save_path = os.path.join(output_dir, f"{epoch}_{i}.png")
            plot(phase_pred[i], y[i], save_path)

    # ---------- Validation ----------
    model.eval()
    valid_loss = []
    for i, batch in enumerate(valid_loader):
        I_delta_z, phase_gt = batch
        I_delta_z = I_delta_z.to(device)
        phase_gt = phase_gt.to(device)
        with torch.no_grad():
            phase_pred, _, _ = model(I_delta_z)
        loss = loss_func(phase_pred, phase_gt)

        valid_loss.append(loss.detach().cpu().numpy())

    valid_loss = sum(valid_loss) / len(valid_loss)
    note = ""
    if valid_loss < best_loss:
        best_loss = valid_loss
        note = "-->best"
        stale = 0
        torch.save(model.state_dict(), ckpt_file)
    else:
        stale += 1

    with open(log_file, "a") as f:
        f.write(
            f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.6f} {note}\n"
        )

    if stale > config["patience"]:
        print(f"No improvment {stale} consecutive epochs, early stop in {epoch} epochs")
        break
