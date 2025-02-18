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
from dataset import sourceDataset, targetDataset


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
    parser.add_argument(
        "--model", "-m", type=str, default=None, help="Model path to load"
    )

    return parser.parse_args()


args = get_args()
exp_name = "ssim"
dt_string = datetime.now().strftime("%m%d_%H%M_")
exp_name = dt_string + exp_name
config = {
    "Optimizer": "Adam",
    "batch_size": 4,
    "lr": 2e-5,
    "n_epochs": 1500,
    "patience": 50,
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

source_train_set = sourceDataset(mode="train")
target_train_set = targetDataset(mode="train")
print(len(source_train_set))
print(len(target_train_set))
source_train_loader = DataLoader(
    source_train_set, shuffle=True, batch_size=config["batch_size"]
)
target_train_loader = DataLoader(
    target_train_set, shuffle=True, batch_size=config["batch_size"]
)


model = UNet(in_dim=1, out_dim=1)
model = model.to(device)
model.load_state_dict(torch.load(args.model))

MSE_loss = nn.MSELoss()
targetLOSS = phaseSSIM()
# MAE_loss = nn.L1Loss()
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
        key: []
        for key in [
            "mseloss_src",
            "mseloss_tgt",
            "ssim_tgt",
            "mddloss_enc",
            "mddloss_dec",
        ]
    }
    for i in range(len_dataloader):
        # p = float(i + epoch * len_dataloader) / n_epochs / len_dataloader
        # optimizer = optimizer_scheduler(optimizer=optimizer, p=p, init=1e-4)

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
        ssim_tgt = targetLOSS.eval(Phi_pred_tgt, Phi_gt_tgt)

        mddloss_enc = mdd_loss(f1_src, f1_tgt) * lambda_mdd
        mddloss_dec = mdd_loss(f2_src, f2_tgt) * lambda_mdd

        loss = mseloss_src + mseloss_tgt + ssim_tgt + mddloss_enc + mddloss_dec
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for key, value in zip(
            loss_dict.keys(),
            [mseloss_src, mseloss_tgt, ssim_tgt, mddloss_enc, mddloss_dec],
        ):
            loss_dict[key].append(value.detach().cpu().numpy())

    # ---------- log ----------
    avg_losses = {key: sum(values) / len(values) for key, values in loss_dict.items()}
    total_loss = sum(avg_losses.values())
    note = ""
    if total_loss < best_loss:
        best_loss = loss
        note = "-->best"
        torch.save(model.state_dict(), ckpt_file)

    loss_str = ", ".join([f"{key} = {value:.6f}" for key, value in avg_losses.items()])
    with open(log_file, "a") as f:
        f.write(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ], {loss_str} {note}\n")

    # ---------- inferernce ----------
    if epoch % 100 == 0:
        model.eval()
        batch_size = 2
        x, y = target_train_set.get_random_batch(batch_size)
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            phase_pred, _, _ = model(x)

        phase_pred = phase_pred.squeeze().detach().cpu().numpy()
        y = y.squeeze().detach().cpu().numpy()

        for i in range(batch_size):
            save_path = os.path.join(output_dir, f"{epoch}_{i}.png")
            plot(phase_pred[i], y[i], save_path)
