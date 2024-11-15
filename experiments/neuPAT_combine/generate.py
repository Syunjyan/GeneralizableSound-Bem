import sys

sys.path.append("./")

from src.scene import Scene
import torch
from tqdm import tqdm

data_dir = "dataset/combine_new"


scene = Scene(f"{data_dir}/config.json")
x = torch.zeros(
    scene.src_sample_num,
    scene.trg_sample_num,
    3 + scene.rot_num + scene.move_num + 1 + 1,
    dtype=torch.float32,
)
# 3 for trg sample position, rot_num for rotation, move_num for move, 1 for scale (resize), 1 for freq
print(x.shape)
y = torch.zeros(scene.src_sample_num, scene.trg_sample_num, 1, dtype=torch.float32)

for src_idx in tqdm(range(scene.src_sample_num)):
    scene.sample()
    scene.solve()
    x[src_idx, :, :3] = scene.trg_factor
    x[src_idx, :, 3 : 3 + scene.rot_num] = scene.rot_factors
    x[src_idx, :, 3 + scene.rot_num : 3 + scene.rot_num + scene.move_num] = (
        scene.move_factors
    )
    x[src_idx, :, -2] = scene.resize_factor
    x[src_idx, :, -1] = scene.freq_factor
    y[src_idx] = scene.potential.abs().unsqueeze(-1)

    # if src_idx == 0:
    #     scene.show()


torch.save({"x": x, "y": y}, f"{data_dir}/data/{sys.argv[1]}.pt")
