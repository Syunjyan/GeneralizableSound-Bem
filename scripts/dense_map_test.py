import sys

sys.path.append("./")
from src.net.dense_map import DenseMap
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from tqdm import tqdm


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.grid = DenseMap(map_num=map_num)
        self.mlp = nn.Sequential(
            nn.Linear(self.grid.output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.grid(x).float().reshape(batch_size, map_num, self.grid.output_dim)
        x = self.mlp(x)
        return x


map_num = 4
num_epochs = 1000
batch_size = 2**14
img = Image.open("dataset/albert.jpg")
img = torch.from_numpy(np.array(img) / 255.0).float()
imgs = [
    img[:1024, :1024],
    img[1024:2048, :1024],
    img[:1024, 1024:2048],
    img[1024:2048, 1024:2048],
]
for i in range(map_num):
    Image.fromarray((imgs[i].numpy() * 255).astype(np.uint8)).save(f"output/gt_{i}.jpg")


model = Model().cuda()
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
tqdm_iter = tqdm(range(num_epochs))

for epoch in tqdm_iter:
    x = torch.rand(batch_size, 2)
    coords = x * (1024 - 1)
    xi = coords.long()
    p0 = imgs[0][xi[:, 0], xi[:, 1]]
    p1 = imgs[1][xi[:, 0], xi[:, 1]]
    p2 = imgs[2][xi[:, 0], xi[:, 1]]
    p3 = imgs[3][xi[:, 0], xi[:, 1]]

    x = x.cuda()
    y = torch.stack([p0, p1, p2, p3], dim=1).unsqueeze(-1).cuda()

    pred = model(x)
    loss = loss_fn(pred, y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    tqdm_iter.set_description(f"Loss: {loss.item():.4f}")

model.eval()

res = (1024, 1024)
x = (np.arange(res[0], dtype=np.float32) + 0.5) / res[0]
y = (np.arange(res[1], dtype=np.float32) + 0.5) / res[1]
x, y = np.meshgrid(x, y)
input = torch.tensor(np.stack([x, y], axis=-1).reshape(-1, 2)).cuda()
pred = model(input).reshape(1024, 1024, map_num).cpu().detach().numpy()

for i in range(map_num):
    img = pred[:, :, i].T
    img = Image.fromarray((img * 255).astype(np.uint8))
    img.save(f"output/pred_{i}.jpg")
