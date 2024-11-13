import sys

sys.path.append("./")

from src.mcs.mcs import (
    ImportanceSampler,
    MonteCarloWeight,
    get_weights_boundary_ks_base,
    get_weights_potential_ks_base,
)
from src.modalobj.model import get_spherical_surface_points
from src.utils import plot_point_cloud, plot_mesh, CombinedFig
from src.solver import BiCGSTAB_batch
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from time import time
import configparser
import meshio
import torch
from src.ffat_solve import monte_carlo_solve, bem_solve

data_dir = sys.argv[1]

import json
import numpy as np


with open(f"{data_dir}/config.json", "r") as file:
    config_data = json.load(file)

data = torch.load(f"{data_dir}/data.pt")

vertices_static = data["vertices_static"].cuda()
triangles_static = data["triangles_static"].cuda()
vertices_rot = data["vertices_rot"].cuda()
triangles_rot = data["triangles_rot"].cuda()
triangles = torch.cat([triangles_static, triangles_rot + len(vertices_static)], dim=0)
neumann_tri = data["neumann_tri"].cuda().reshape(1, -1)
rot_pos = torch.tensor(
    config_data.get("solver", {}).get("rot_pos"), device="cuda", dtype=torch.float32
)
rot_axis = config_data.get("solver", {}).get("rot_axis")
rot_max_degree = config_data.get("solver", {}).get("rot_max_degree")
src_sample_num = config_data.get("solver", {}).get("src_sample_num", 1000)

mesh_static_size = (vertices_static.max(dim=0)[0] - vertices_static.min(dim=0)[0]).max()
mesh_rot_size = (vertices_rot.max(dim=0)[0] - vertices_rot.min(dim=0)[0]).max()

if mesh_static_size > mesh_rot_size:
    ffat_base_mesh = vertices_static
else:
    ffat_base_mesh = vertices_rot

vertices_rot -= rot_pos

freq_min = config_data.get("solver", {}).get("freq_min", 100)
freq_max = config_data.get("solver", {}).get("freq_max", 10000)
freq_min_log = np.log10(freq_min)
freq_max_log = np.log10(freq_max)
print("freq_min:", freq_min)
print("freq_max:", freq_max)
r_min = config_data.get("solver", {}).get("r_min", 3)
r_max = config_data.get("solver", {}).get("r_max", 4)


def calculate_ffat_map():
    return monte_carlo_solve(
        vertices,
        triangles,
        neumann_tri,
        ks_batch,
        trg_points,
        sample_num,
        plot=False,
    )


import scipy.spatial.transform as transform


def rotate_vertices(vs, rot_axis, rot_degree):
    vs = transform.Rotation.from_euler(rot_axis, rot_degree.item(), degrees=True).apply(
        vs.cpu().numpy()
    )
    return torch.from_numpy(vs).cuda()


sample_num = 8000
x = torch.zeros(src_sample_num, 64 * 32, 5, dtype=torch.float32)
y = torch.zeros(src_sample_num, 64 * 32, 1, dtype=torch.float32)

xs = torch.linspace(0, 1, 64, device="cuda", dtype=torch.float32)
ys = torch.linspace(0, 1, 32, device="cuda", dtype=torch.float32)
gridx, gridy = torch.meshgrid(xs, ys)
check_correct = False


rot_k = torch.rand(1).cuda()
vertices_rot_updated = rotate_vertices(vertices_rot, rot_axis, rot_k * rot_max_degree)
vertices_rot_updated += rot_pos
vertices = torch.cat([vertices_static, vertices_rot_updated], dim=0).float().cuda()
r_scale = torch.ones(1).cuda() * 0.5
trg_pos = torch.zeros(64, 32, 3, device="cuda", dtype=torch.float32)
r = (r_scale * (r_max - r_min) + r_min).item()
trg_pos[:, :, 0] = r_scale
trg_pos[:, :, 1] = gridx
trg_pos[:, :, 2] = gridy
trg_pos = trg_pos.reshape(-1, 3)
trg_points = get_spherical_surface_points(ffat_base_mesh, r).reshape(64, 32, 3)

freq_pos = torch.rand(1, device="cuda", dtype=torch.float32)
freq_log = freq_pos * (freq_max_log - freq_min_log) + freq_min_log
freq = 10**freq_log
k = 2 * np.pi * freq / 343.2

ks_batch = torch.tensor([-k], dtype=torch.float32).cuda()


CombinedFig().add_mesh(
    vertices, triangles, neumann_tri[0].abs(), opacity=1.0
).add_points(trg_points[0, 0]).show()
