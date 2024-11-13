import sys

sys.path.append("./")
from src.utils import Visualizer
from src.modalobj.model import get_spherical_surface_points, complex_ssim, StaticObj
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from time import time
from src.bem.solver import BEM_Solver
import torch

data_dir = "dataset/NeuPAT_new/audio"

import json
import numpy as np

with open(f"{data_dir}/config.json", "r") as file:
    config_data = json.load(file)

vibration_objects = config_data.get("vibration_obj", [])
for obj in vibration_objects:
    mesh = obj.get("mesh")
    size = obj.get("size")
    obj_vib = StaticObj(f"{data_dir}/{mesh}", scale=size)

static_objects = config_data.get("static_obj", [])
for obj in static_objects:
    mesh = obj.get("mesh")
    size = obj.get("size")
    obj_static = StaticObj(f"{data_dir}/{mesh}", scale=size)

vertices_vib = torch.from_numpy(obj_vib.vertices).cuda().to(torch.float32)
triangles_vib = torch.from_numpy(obj_vib.triangles).cuda().to(torch.int32)
neumann_vib = torch.ones(len(vertices_vib), 1, dtype=torch.complex64).cuda()

vertices_static = torch.from_numpy(obj_static.vertices).cuda().to(torch.float32)
triangles_static = torch.from_numpy(obj_static.triangles).cuda().to(torch.int32)
neumann_static = torch.zeros(len(vertices_static), 1, dtype=torch.complex64).cuda()

neumann_vib[vertices_vib[:, 1] > -0.04] = 0

neumann = torch.cat([neumann_vib, neumann_static], dim=0).squeeze(-1)
triangles = torch.cat([triangles_vib, triangles_static + len(vertices_vib)], dim=0)

src_pos_min = torch.tensor(
    config_data.get("solver", {}).get("src_pos_min"), device="cuda", dtype=torch.float32
)
src_pos_max = torch.tensor(
    config_data.get("solver", {}).get("src_pos_max"), device="cuda", dtype=torch.float32
)
bbox_center = torch.tensor(
    config_data.get("solver", {}).get("bbox_center"), device="cuda", dtype=torch.float32
)
bbox_r = config_data.get("solver", {}).get("bbox_r")
r_max = config_data.get("solver", {}).get("r_max")
print("src_pos_min:", src_pos_min)
print("src_pos_max:", src_pos_max)
print("bbox_center:", bbox_center)
print("bbox_r:", bbox_r)
print("r_max:", r_max)

freq_min = config_data.get("solver", {}).get("freq_min", 100)
freq_max = config_data.get("solver", {}).get("freq_max", 10000)
freq_min_log = np.log10(freq_min)
freq_max_log = np.log10(freq_max)
print("freq_min:", freq_min)
print("freq_max:", freq_max)
trg_sample_num = config_data.get("solver", {}).get("trg_sample_num", 1000)
src_sample_num = config_data.get("solver", {}).get("src_sample_num", 1000)
print("trg_sample_num:", trg_sample_num)
print("src_sample_num:", src_sample_num)

x = torch.zeros(src_sample_num, trg_sample_num, 5, dtype=torch.float32)
y = torch.zeros(src_sample_num, trg_sample_num, 1, dtype=torch.float32)

for sample_idx in tqdm(range(src_sample_num)):
    src_pos_base = torch.rand(3, device="cuda", dtype=torch.float32)
    displacement = src_pos_base * (src_pos_max - src_pos_min) + src_pos_min
    vertices_vib_updated = vertices_vib + displacement
    vertices = torch.cat([vertices_vib_updated, vertices_static], dim=0)
    sample_points_base = torch.rand(
        trg_sample_num, 3, device="cuda", dtype=torch.float32
    )
    rs = (sample_points_base[:, 0] * (r_max - 1) + 1) * bbox_r
    theta = sample_points_base[:, 1] * 2 * np.pi - np.pi
    phi = sample_points_base[:, 2] * np.pi

    xs = rs * torch.sin(phi) * torch.cos(theta) + bbox_center[0]
    ys = rs * torch.sin(phi) * torch.sin(theta) + bbox_center[1]
    zs = rs * torch.cos(phi) + bbox_center[2]

    trg_points = torch.stack([xs, ys, zs], dim=-1)

    freq_base = torch.rand(1, device="cuda", dtype=torch.float32)
    freq_log = freq_base * (freq_max_log - freq_min_log) + freq_min_log
    freq = 10**freq_log
    k = (2 * np.pi * freq / 343.2).item()

    x[sample_idx, :, :3] = sample_points_base.cpu()
    x[sample_idx, :, 3] = src_pos_base[1].cpu()
    x[sample_idx, :, 4] = freq_base.cpu()

    solver = BEM_Solver(vertices, triangles)
    dirichlet = solver.neumann2dirichlet(k, neumann)
    potential = solver.boundary2potential(k, neumann, dirichlet, trg_points)
    y[sample_idx, :, 0] = potential.abs().cpu()
    # if sample_idx == 0:
    #     Visualizer().add_mesh(vertices, triangles, neumann.abs()).add_points(
    #         trg_points, y[sample_idx]
    #     ).show()
    #     break


torch.save({"x": x, "y": y}, f"{data_dir}/data/{sys.argv[1]}.pt")
