import sys

sys.path.append("./")
from src.mcs.mcs import (
    ImportanceSampler,
    MonteCarloWeight,
    get_weights_boundary_ks_base,
    get_weights_potential_ks_base,
)
from src.utils import plot_point_cloud, plot_mesh, CombinedFig
from src.solver import BiCGSTAB_batch
from src.ffat_solve import bem_solve, monte_carlo_solve
from src.modalobj.model import get_spherical_surface_points, complex_ssim, SNR
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from time import time
import configparser
import meshio
import torch

data_dir = "dataset/NeuPAT/shape"

import json
import numpy as np

with open(f"{data_dir}/config.json", "r") as file:
    config_data = json.load(file)

data = torch.load(f"{data_dir}/data.pt")
vib_vertices = data["vib_vertices"].cuda()
vib_triangles = data["vib_triangles"].cuda()
vertices_lst = data["vertices_lst"]
triangles_lst = data["triangles_lst"]

freq_min = config_data.get("solver", {}).get("freq_min", 100)
freq_max = config_data.get("solver", {}).get("freq_max", 10000)
freq_min_log = np.log10(freq_min)
freq_max_log = np.log10(freq_max)
src_sample_num = config_data.get("solver", {}).get("src_sample_num", 1000)


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


check_correct = False
sample_num = 7000
snrs = []
ssims = []
xs = torch.linspace(0, 1, 64, device="cuda", dtype=torch.float32)
ys = torch.linspace(0, 1, 32, device="cuda", dtype=torch.float32)
gridx, gridy = torch.meshgrid(xs, ys)

x = torch.zeros(len(vertices_lst), src_sample_num, 64 * 32, 5, dtype=torch.float32)
y = torch.zeros(len(vertices_lst), src_sample_num, 64 * 32, 1, dtype=torch.float32)

for obj_id in range(len(vertices_lst)):
    for src_i in tqdm(range(src_sample_num)):
        while True:
            vertices_shape = vertices_lst[obj_id]
            triangles_shape = triangles_lst[obj_id]
            vertices = torch.cat([vib_vertices, vertices_shape], dim=0).float()
            triangles = torch.cat(
                [vib_triangles, triangles_shape + len(vib_vertices)], dim=0
            ).int()
            neumann_vib = torch.ones(
                len(vib_triangles), device="cuda", dtype=torch.complex64
            )
            neumann_shape = torch.zeros(
                len(triangles_shape), device="cuda", dtype=torch.complex64
            )
            neumann_tri = torch.cat([neumann_vib, neumann_shape], dim=0)
            neumann_tri = neumann_tri.reshape(1, -1)
            r_min = 1.5
            r_max = 3
            if check_correct:
                r_scale = torch.zeros(1).cuda()
            else:
                r_scale = torch.rand(1).cuda()
            trg_pos = torch.zeros(64, 32, 3, device="cuda", dtype=torch.float32)
            r = (r_scale * (r_max - r_min) + r_min).item()
            trg_pos[:, :, 0] = r_scale
            trg_pos[:, :, 1] = gridx
            trg_pos[:, :, 2] = gridy
            trg_pos = trg_pos.reshape(-1, 3)
            trg_points = get_spherical_surface_points(vertices, r)

            if check_correct:
                freq_pos = torch.tensor(
                    [src_i / src_sample_num], device="cuda", dtype=torch.float32
                )
            else:
                freq_pos = torch.rand(1, device="cuda", dtype=torch.float32)

            freq_log = freq_pos * (freq_max_log - freq_min_log) + freq_min_log
            freq = 10**freq_log
            k = 2 * np.pi * freq / 343.2

            ks_batch = torch.tensor([-k], dtype=torch.float32).cuda()
            ffat_map, success = calculate_ffat_map()
            if success:
                ffat_map = np.abs(ffat_map)
                break
        if check_correct:
            import matplotlib.pyplot as plt

            plt.imshow(ffat_map.reshape(64, 32))
            plt.colorbar()
            os.makedirs(f"{data_dir}/{obj_id}", exist_ok=True)
            plt.savefig(f"{data_dir}/{obj_id}/{src_i}.png")
            plt.close()

        x[obj_id, src_i, :, 0] = obj_id / len(vertices_lst)
        x[obj_id, src_i, :, 1] = freq_pos
        x[obj_id, src_i, :, 2:] = trg_pos
        y[obj_id, src_i] = torch.from_numpy(ffat_map).T


if check_correct:
    print("snr:", np.mean(snrs))
    print("ssim:", np.mean(ssims))
else:
    torch.save({"x": x, "y": y}, f"{data_dir}/data/{sys.argv[1]}.pt")
