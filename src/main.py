import os, argparse
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision.transforms import functional as tf

# Customized packages.
from utils import optimizer_scheduler
from unet import UNet
from dataset import phaseDataset


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate test set with pretrained Deeplabv3 model."
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
exp_name = "qpi"
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
log_file = f"../log/{exp_name}.log"
if os.path.exists(log_file):
    os.remove(log_file)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
myseed = 1314520
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
print(device)

train_set = phaseDataset(mode="train")
valid_set = phaseDataset(mode="valid")
print(len(train_set))
print(len(valid_set))
train_loader = DataLoader(train_set, shuffle=True, batch_size=config["batch_size"])
valid_loader = DataLoader(valid_set, shuffle=True, batch_size=config["batch_size"])


model = UNet(in_dim=1, out_dim=1)
model = model.to(device)

# ckpt_path = "../input/dlcvhw1/1008_04_52_deeplabv3.ckpt"
# model.load_state_dict(torch.load(ckpt_path))
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5)

stale = 0
best_loss = 100
n_epochs = config["n_epochs"]

# ---------- Training ----------
for epoch in tqdm(range(n_epochs)):

    model.train()
    train_loss = []
    train_accs = []

    for i, batch in enumerate(train_loader):
        # A batch consists of image data and corresponding labels.
        p = float(i + epoch * len(train_loader)) / n_epochs / len(train_loader)
        optimizer = optimizer_scheduler(optimizer=optimizer, p=p)
        optimizer.zero_grad()

        I_delta_z, phase_gt = batch
        I_delta_z = I_delta_z.to(device)
        phase_gt = phase_gt.to(device)

        phase_pred = model(I_delta_z)
        loss = loss_func(phase_pred, phase_gt)

        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())

    train_loss = sum(train_loss) / len(train_loss)
    with open(log_file, "a") as f:
        f.write(
            f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ], loss = {train_loss:.6f}\n"
        )

    # ---------- Validation ----------
    model.eval()
    valid_loss = []
    for i, batch in enumerate(valid_loader):
        I_delta_z, phase_gt = batch
        I_delta_z = I_delta_z.to(device)
        phase_gt = phase_gt.to(device)

        phase_pred = model(I_delta_z)
        loss = loss_func(phase_pred, phase_gt)

        valid_loss.append(loss.detach().cpu().numpy())

    valid_loss = sum(valid_loss) / len(valid_loss)
    note = ""
    if valid_loss < best_loss:
        best_loss = valid_loss
        note = "-->best"
        stale = 0
        torch.save(model.state_dict(), f"../ckpt/{exp_name}.ckpt")
    else:
        stale += 1

    with open(log_file, "a") as f:
        f.write(
            f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.6f} {note}\n"
        )

    if stale > config["patience"]:
        print(f"No improvment {stale} consecutive epochs, early stop in {epoch} epochs")
        break
