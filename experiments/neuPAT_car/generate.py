import sys

sys.path.append("./")

from src.scene import Scene
import torch
from tqdm import tqdm
from src.utils import Timer

data_dir = "dataset/NeuPAT_new/car"


scene = Scene(f"{data_dir}/config.json")
x = torch.zeros(
    scene.src_sample_num,
    scene.trg_sample_num,
    3 + scene.obj_list_num + 1,
    dtype=torch.float32,
)
y = torch.zeros(scene.src_sample_num, scene.trg_sample_num, 1, dtype=torch.float32)

for src_idx in tqdm(range(scene.src_sample_num)):
    scene.sample()
    scene.solve()
    x[src_idx, :, :3] = scene.trg_factor
    x[src_idx, :, 3 : 3 + scene.obj_list_num] = scene.obj_list_factors
    x[src_idx, :, -1] = scene.freq_factor
    y[src_idx] = scene.potential.unsqueeze(-1)

    if src_idx == 0:
        scene.show()

torch.save({"x": x, "y": y}, f"{data_dir}/data/{sys.argv[1]}.pt")
