import os, argparse
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision.transforms import functional as tf

# Customized packages.
from utils import *
from unet import UNet
from dataset import phaseDataset, Subset


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
    parser.add_argument(
        "--model", "-m", type=str, default=None, help="Model path to load"
    )

    return parser.parse_args()


args = get_args()
exp_name = "mdd"
dt_string = datetime.now().strftime("%m%d_%H%M_")
exp_name = dt_string + exp_name
config = {
    "Optimizer": "Adam",
    "batch_size": 4,
    "lr": 2e-5,
    "n_epochs": 2000,
    "patience": 100,
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

source_train_set = phaseDataset(mode="train")
target_train_set = phaseDataset(mode="experiment")
# target_train_set = Subset(target_train_set, list(range(8)))
print(len(source_train_set))
print(len(target_train_set))
source_train_loader = DataLoader(
    source_train_set, shuffle=True, batch_size=config["batch_size"]
)
target_train_loader = DataLoader(
    target_train_set, shuffle=True, batch_size=config["batch_size"]
)


model = UNet(in_dim=1, out_dim=1, domain_adapt=True)
model = model.to(device)

# ckpt_path = "../input/dlcvhw1/1008_04_52_deeplabv3.ckpt"
model.load_state_dict(torch.load(args.model))
MSE_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5)

best_loss = 1e9
lambda_mdd = 1e-7
n_epochs = config["n_epochs"]

# ---------- Training ----------
for epoch in tqdm(range(n_epochs), position=0, leave=True):

    model.train()
    len_dataloader = min(len(source_train_loader), len(target_train_loader))
    data_source_iter = iter(source_train_loader)
    data_target_iter = iter(target_train_loader)
    loss_dict = {
        "mseloss_src": [],
        "mseloss_tgt": [],
        "mddloss_enc": [],
        "mddloss_dec": [],
    }

    for i in range(len_dataloader):
        # p = float(i + epoch * len_dataloader) / n_epochs / len_dataloader
        # optimizer = optimizer_scheduler(optimizer=optimizer, p=p, init=1e-4)
        optimizer.zero_grad()

        source_data = next(data_source_iter)
        I_src, Phi_gt_src = source_data
        I_src = I_src.to(device)
        Phi_gt_src = Phi_gt_src.to(device)

        target_data = next(data_target_iter)
        I_tgt, Phi_gt_tgt = target_data
        I_tgt = I_tgt.to(device)
        Phi_gt_tgt = Phi_gt_tgt.to(device)

        Phi_pred_src, f1_src, f2_src = model(I_src)
        Phi_pred_tgt, f1_tgt, f2_tgt = model(I_tgt)

        mseloss_src = MSE_loss(Phi_pred_src, Phi_gt_src)
        mseloss_tgt = MSE_loss(Phi_pred_tgt, Phi_gt_tgt)

        mddloss_enc = mdd_loss(f1_src, f1_tgt) * lambda_mdd
        mddloss_dec = mdd_loss(f2_src, f2_tgt) * lambda_mdd

        loss = mseloss_src + mseloss_tgt + mddloss_enc + mddloss_dec
        loss.backward()
        optimizer.step()
        # Store losses
        loss_dict["mseloss_src"].append(mseloss_src.detach().cpu().numpy())
        loss_dict["mseloss_tgt"].append(mseloss_tgt.detach().cpu().numpy())
        loss_dict["mddloss_enc"].append(mddloss_enc.detach().cpu().numpy())
        loss_dict["mddloss_dec"].append(mddloss_dec.detach().cpu().numpy())

    avg_losses = {key: sum(values) / len(values) for key, values in loss_dict.items()}
    note = ""
    loss = (
        avg_losses["mseloss_src"]
        + avg_losses["mseloss_tgt"]
        + avg_losses["mddloss_enc"]
        + avg_losses["mddloss_dec"]
    )
    if loss < best_loss:
        best_loss = loss
        note = "-->best"
        torch.save(model.state_dict(), f"../ckpt/{exp_name}.ckpt")

    with open(log_file, "a") as f:
        f.write(
            f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ], "
            f"MSEloss_src = {avg_losses['mseloss_src']:.6f}, "
            f"MSEloss_tgt = {avg_losses['mseloss_tgt']:.6f}, "
            f"MDDloss_enc = {avg_losses['mddloss_enc']:.6f}, "
            f"MDDloss_dec = {avg_losses['mddloss_dec']:.6f} {note}\n"
        )

    # ---------- Validation ----------
    # model.eval()
    # valid_loss = []
    # for i, batch in enumerate(target_train_loader):
    #     I_delta_z, phase_gt = batch
    #     I_delta_z = I_delta_z.to(device)
    #     phase_gt = phase_gt.to(device)

    #     phase_pred = model(I_delta_z)
    #     loss = MSE_loss(phase_pred, phase_gt)

    #     valid_loss.append(loss.detach().cpu().numpy())

    # valid_loss = sum(valid_loss) / len(valid_loss)
    # note = ""
    # valid_loss = 1e9
    # if valid_loss < best_loss:
    #     best_loss = valid_loss
    #     note = "-->best"
    #     stale = 0
    #     torch.save(model.state_dict(), f"../ckpt/{exp_name}.ckpt")
    # else:
    #     stale += 1

    # with open(log_file, "a") as f:
    #     f.write(
    #         f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.6f} {note}\n"
    #     )

    # if stale > config["patience"]:
    #     print(f"No improvment {stale} consecutive epochs, early stop in {epoch} epochs")
    #     break
