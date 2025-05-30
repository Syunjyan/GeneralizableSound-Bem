import sys

sys.path.append("./")
from src.net.model import NeuPAT
import torch
from glob import glob
import torch
from tqdm import tqdm
import json
import os

data_dir = "dataset/NeuPAT_new/combine_new/baseline"

for i in range(100):
    os.system(f"python experiments/neuPAT_combine/generate.py {i}")

data_points_lst = glob(f"{data_dir}/../data/*.pt")
xs = []
ys = []
for data_points in data_points_lst:
    data = torch.load(data_points)
    xs.append(data["x"])
    ys.append(data["y"])

xs = torch.cat(xs, dim=0).reshape(-1, xs[0].shape[-1])
ys = torch.cat(ys, dim=0).reshape(-1, ys[0].shape[-1])
ys = ((ys + 10e-6) / 10e-6).log10()

print(xs.shape, ys.shape)

xs_train = xs[: int(len(xs) * 0.8)]
ys_train = ys[: int(len(ys) * 0.8)]
xs_test = xs[int(len(xs) * 0.8) :]
ys_test = ys[int(len(ys) * 0.8) :]
del xs, ys

with open(f"{data_dir}/net.json", "r") as file:
    config_data = json.load(file)

train_params = config_data.get("train", {})
batch_size = train_params.get("batch_size")

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=data_dir)

model = NeuPAT(ys_train.shape[-1], config_data).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=train_params.get("lr"))
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=train_params.get("step_size"), gamma=train_params.get("gamma")
)


# Custom shuffle function that operates on the GPU
def shuffle_tensors(x, y):
    indices = torch.randperm(x.size(0), device=x.device)
    return x[indices], y[indices]


max_epochs = train_params.get("max_epochs")
test_step = train_params.get("test_step")
for epoch_idx in tqdm(range(max_epochs)):
    # Create batches manually
    if epoch_idx % 10 == 0:
        torch.cuda.empty_cache()
        xs_train, ys_train = shuffle_tensors(xs_train, ys_train)

    loss_train = []
    for batch_idx in range(0, len(xs_train), batch_size):
        x_batch = xs_train[batch_idx : batch_idx + batch_size].cuda()
        y_batch = ys_train[batch_idx : batch_idx + batch_size].cuda()
        y_pred = model(x_batch)
        loss = torch.nn.functional.mse_loss(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())

    loss = sum(loss_train) / len(loss_train)
    print(f"train loss: {loss}")
    scheduler.step()
    writer.add_scalar("loss_train", loss, epoch_idx)

    if epoch_idx % test_step == 0:
        loss_test = []
        for batch_idx in range(0, len(xs_test), batch_size):
            x_batch = xs_test[batch_idx : batch_idx + batch_size].cuda()
            y_batch = ys_test[batch_idx : batch_idx + batch_size].cuda()
            y_pred = model(x_batch)
            loss = torch.nn.functional.mse_loss(y_pred, y_batch)
            loss_test.append(loss.item())
        loss = sum(loss_test) / len(loss_test)
        print(f"test loss: {loss}")
        writer.add_scalar("loss_test", loss, epoch_idx)


torch.save(model.state_dict(), f"{data_dir}/model.pt")
