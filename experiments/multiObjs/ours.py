import sys

sys.path.append("./")

import torch
from src.mcs.mcs import ImportanceSampler, MonteCarloWeight
from src.timer import Timer
from src.modalobj.model import (
    solve_points_dirichlet,
    MultipoleModel,
    MeshObj,
    BEMModel,
    SNR,
    complex_ssim,
)
import numpy as np
from src.utils import plot_mesh, plot_point_cloud, CombinedFig
from src.solver import BiCGSTAB, BiCGSTAB_batch, BiCGSTAB_batch2
import os
from glob import glob
from tqdm import tqdm
from numba import njit
import matplotlib.pyplot as plt
import configparser
import meshio

data_dir = sys.argv[1]
import json

with open(f"{data_dir}/config.json", "r") as file:
    config_data = json.load(file)

data = np.load(f"{data_dir}/bem.npz")
vertices = data["vertices"]
triangles = data["triangles"]
neumann = data["neumann"]
ks = data["wave_number"]
points = data["points"]
ffat_map_bem = data["ffat_map"]

mode_num = len(ks)
vertices = torch.from_numpy(vertices).cuda().to(torch.float32)
triangles = torch.from_numpy(triangles).cuda().to(torch.int32)
importance = torch.ones(len(triangles), dtype=torch.float32).cuda()

neumann_tri = torch.from_numpy(neumann).unsqueeze(-1).cuda()
ks = torch.from_numpy(-ks).cuda().to(torch.float32)
points = torch.from_numpy(points).cuda().to(torch.float32)

for i in range(2):
    timer = Timer(log_output=True)
    sampler = ImportanceSampler(vertices, triangles, importance, 100000)
    sampler.update()
    r = config_data.get("solver", {}).get("radius", 0.001)
    print("radius:", r)
    sampler.poisson_disk_resample(r, 4)
    timer.log("sample points: ", sampler.num_samples, record=True)

    timer_solver = Timer()
    G0_constructor = MonteCarloWeight(sampler.points, sampler)
    G1_constructor = MonteCarloWeight(sampler.points, sampler, deriv=True)
    G0_batch = G0_constructor.get_weights_boundary_ks(ks)
    G1_batch = G1_constructor.get_weights_boundary_ks(ks)
    neumann = neumann_tri[:, sampler.points_index, :]

    print(G0_batch.shape, G1_batch.shape, neumann.shape)
    b_batch = torch.bmm(G0_batch, neumann).permute(1, 2, 0)
    print(b_batch.shape)
    timer.log("construct G and b", record=True)

    solver = BiCGSTAB_batch(
        lambda x: (torch.bmm(G1_batch, x.permute(2, 0, 1)).permute(1, 2, 0) - x)
    )
    timer.log("construct A", record=True)

    tol = config_data.get("solver", {}).get("tol", 1e-6)
    nsteps = config_data.get("solver", {}).get("nsteps", 100)
    dirichlet = solver.solve(b_batch, tol=tol, nsteps=nsteps).permute(2, 0, 1)

    timer.log("solve equation", record=True)
    timer_solver.log("solve equation", record=True)
    G0_constructor = MonteCarloWeight(points, sampler)
    G1_constructor = MonteCarloWeight(points, sampler, deriv=True)
    G0 = G0_constructor.get_weights_potential_ks(ks)
    G1 = G1_constructor.get_weights_potential_ks(ks)
    ffat_map = G1 @ dirichlet - G0 @ neumann
    timer.log("solve ffat map", record=True)

    ffat_map_ours = ffat_map.reshape(mode_num, -1).cpu().numpy()
    cost_time = timer.record_time
    cost_time_solver = timer_solver.record_time

SNRs = np.zeros((mode_num, 1))
ssims = np.zeros((mode_num, 1))


for i in range(mode_num):
    points_dirichlet = ffat_map_ours[i]
    points_dirichlet_gt = ffat_map_bem[i]
    SNRs[i] = SNR(points_dirichlet_gt, points_dirichlet)
    ssims[i] = complex_ssim(points_dirichlet_gt, points_dirichlet)
    # if i == 1:
    #     CombinedFig().add_mesh(vertices, triangles).add_points(
    #         points, points_dirichlet.imag
    #     ).show()
    #     CombinedFig().add_mesh(vertices, triangles).add_points(
    #         points, points_dirichlet_gt.imag
    #     ).show()
    #     CombinedFig().add_mesh(vertices, triangles).add_points(
    #         points, points_dirichlet.real
    #     ).show()
    #     CombinedFig().add_mesh(vertices, triangles).add_points(
    #         points, points_dirichlet_gt.real
    #     ).show()

print(SNRs)
print(ssims)

np.savez_compressed(
    f"{data_dir}/ours.npz",
    ffat_map=ffat_map_ours,
    cost_time=cost_time,
    cost_time_solver=cost_time_solver,
)
