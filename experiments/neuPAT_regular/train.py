import sys

sys.path.append("./")
from src.net.model import NeuPAT_torch
import torch
from glob import glob
from tqdm import tqdm
import json
import os
import numpy as np


import torch
import numpy as np


def xyz2model(xs, ys, zs, sizes, freqs):
    inputs = torch.stack([xs, ys, zs], dim=1)
    rs = torch.sqrt(inputs[:, 0] ** 2 + inputs[:, 1] ** 2 + inputs[:, 2] ** 2)
    phi = torch.acos(inputs[:, 2] / rs) / np.pi
    theta = (torch.atan2(inputs[:, 1], inputs[:, 0]) + np.pi) / (2 * np.pi)
    rs = rs / bbox_size - 1
    x_batch = torch.stack([rs, theta, phi, sizes, freqs], dim=1)
    # print("rs", rs.min(), rs.max())
    # print("phi", phi.min(), phi.max())
    # print("theta", theta.min(), theta.max())
    # print("x_batch", x_batch.min(), x_batch.max())
    pred = model(x_batch)
    return torch.complex(pred[:, 0], pred[:, 1])


def calculate_helmholtz_loss(xs, ys, zs, ks, sizes, freqs):
    h = 1e-4
    laplacian = (
        xyz2model(xs + h, ys, zs, sizes, freqs)
        + xyz2model(xs - h, ys, zs, sizes, freqs)
        + xyz2model(xs, ys + h, zs, sizes, freqs)
        + xyz2model(xs, ys - h, zs, sizes, freqs)
        + xyz2model(xs, ys, zs + h, sizes, freqs)
        + xyz2model(xs, ys, zs - h, sizes, freqs)
        - 6 * xyz2model(xs, ys, zs, sizes, freqs)
    ) / h**2
    rhs = ks**2 * xyz2model(xs, ys, zs, sizes, freqs)
    mask = torch.isnan(laplacian)
    laplacian = laplacian[~mask]
    rhs = rhs[~mask]
    loss = (laplacian - rhs).abs().mean()
    return loss * 0.001


data_dir = "dataset/NeuPAT_new/regular/baseline"
data = torch.load(f"{data_dir}/../data/0.pt")
bbox_size = data["bbox_size"].item()
x = data["x"].cuda()
y = data["y"].cuda()

mask = torch.isnan(y).any(dim=-1)
x = x[~mask]
y = y[~mask]

xx = data["xx"].cuda()

x = x.reshape(-1, x.shape[-1])
y = y.reshape(-1, y.shape[-1])
xx = xx.reshape(-1, xx.shape[-1])

y = y / (y.abs().max() + 1e-6) * 4

xs_train = x[: int(len(x) * 0.8)]
ys_train = y[: int(len(y) * 0.8)]
xs_test = x[int(len(x) * 0.8) :]
ys_test = y[int(len(y) * 0.8) :]
del x, y

with open(f"{data_dir}/net.json", "r") as file:
    net_config = json.load(file)

train_params = net_config.get("train", {})
batch_size = train_params.get("batch_size")

from torch.utils.tensorboard import SummaryWriter

logs = glob(f"{data_dir}/events*")
for log in logs:
    os.remove(log)
writer = SummaryWriter(log_dir=data_dir)


model = NeuPAT_torch(2, net_config).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=train_params.get("lr"))
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=train_params.get("step_size"), gamma=train_params.get("gamma")
)

# torch.autograd.set_detect_anomaly(True)
max_epochs = train_params.get("max_epochs")
test_step = train_params.get("test_step")
for epoch_idx in tqdm(range(max_epochs)):
    # Create batches manually
    indices = torch.randperm(xs_train.size(0), device="cuda")
    xx_start_idx = torch.randint(0, len(xx) - len(xs_train), (1,)).item()
    xx_epoch = xx[xx_start_idx : xx_start_idx + len(xs_train)]

    loss_train = []
    loss_reg = []
    for batch_idx in tqdm(range(0, len(xs_train), batch_size)):
        x_batch = xs_train[indices[batch_idx : batch_idx + batch_size]]
        y_batch = ys_train[indices[batch_idx : batch_idx + batch_size]]
        xx_batch = xx_epoch[batch_idx : batch_idx + batch_size]
        # Forward and backward passes
        y_pred = model(x_batch)
        loss1 = torch.nn.functional.mse_loss(y_pred, y_batch)

        xs = xx_batch[:, 0]
        ys = xx_batch[:, 1]
        zs = xx_batch[:, 2]
        sizes = xx_batch[:, 3]
        freqs = xx_batch[:, 4]
        ks = xx_batch[:, -1]
        loss_reg = calculate_helmholtz_loss(xs, ys, zs, ks, sizes, freqs)
        loss = loss_train + loss_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())
    loss_train = sum(loss_train) / len(loss_train)
    print(f"epoch {epoch_idx}: train loss {loss_train}")
    writer.add_scalar("loss_train", loss_train, epoch_idx)

    if epoch_idx % test_step == 0:
        loss_test = []
        for batch_idx in tqdm(range(0, len(xs_test), batch_size)):
            x_batch = xs_test[batch_idx : batch_idx + batch_size]
            y_batch = ys_test[batch_idx : batch_idx + batch_size]
            y_pred = model(x_batch)
            loss = torch.nn.functional.mse_loss(y_pred, y_batch)
            loss_test.append(loss.item())
        loss_test = sum(loss_test) / len(loss_test)
        writer.add_scalar("loss_test", loss_test, epoch_idx)
        print(f"epoch {epoch_idx}: test loss {loss_test}")

    scheduler.step()
    if epoch_idx % 100 == 0:
        torch.save(model.state_dict(), f"{data_dir}/model.pt")
