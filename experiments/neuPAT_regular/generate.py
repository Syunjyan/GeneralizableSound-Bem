import sys

sys.path.append("./")

from src.scene import Scene
import torch
from tqdm import tqdm
from src.utils import Timer

data_dir = "dataset/NeuPAT_new/regular"


scene = Scene(f"{data_dir}/config.json")
x = torch.zeros(
    scene.src_sample_num,
    scene.trg_sample_num,
    3 + scene.move_num + 1,
    dtype=torch.float32,
)
y = torch.zeros(scene.src_sample_num, scene.trg_sample_num, 2, dtype=torch.float32)

xx = torch.zeros(
    scene.src_sample_num * 32,
    scene.trg_sample_num,
    3 + scene.move_num + 1 + 1,
    dtype=torch.float32,
)

for src_idx in tqdm(range(scene.src_sample_num)):
    scene.sample()
    scene.solve()
    x[src_idx, :, :3] = scene.trg_factor
    x[src_idx, :, 3] = scene.move_factors
    x[src_idx, :, 4] = scene.freq_factor
    y[src_idx, :, 0] = scene.potential.real
    y[src_idx, :, 1] = scene.potential.imag

    if src_idx == 0:
        scene.show()

for src_idx in tqdm(range(scene.src_sample_num * 32)):
    scene.sample()
    xx[src_idx, :, 0] = scene.xs
    xx[src_idx, :, 1] = scene.ys
    xx[src_idx, :, 2] = scene.zs
    xx[src_idx, :, 3] = scene.move_factors
    xx[src_idx, :, 4] = scene.freq_factor
    xx[src_idx, :, 5] = scene.k

torch.save(
    {"x": x, "y": y, "xx": xx, "bbox_size": scene.bbox_size},
    f"{data_dir}/data/{sys.argv[1]}.pt",
)
