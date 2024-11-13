import sys

sys.path.append("./")

from src.scene import EditableModalSound
from src.utils import Visualizer
from tqdm import tqdm
from time import time
import torch

data_dir = "dataset/NeuPAT_new/regular"

modal_sound = EditableModalSound(data_dir, uniform=False)

term_num = 5

x = torch.zeros(
    modal_sound.sample_num,
    modal_sound.point_num_per_sample,
    3 + 1 + 1,
    dtype=torch.float32,
)
# 3+1+1 is | 3: (r, theta, phi) point position | 1: size | 1: freq |
y = torch.zeros(
    modal_sound.sample_num,
    modal_sound.point_num_per_sample,
    modal_sound.mode_num,
    dtype=torch.float32,
)

for sample_idx in tqdm(range(modal_sound.sample_num)):
    x[sample_idx] = modal_sound.sample()
    y[sample_idx] = modal_sound.solve().T

    if sample_idx == 0:
        modal_sound.show(0)

torch.save({"x": x, "y": y}, f"{data_dir}/data/{sys.argv[1]}.pt")
