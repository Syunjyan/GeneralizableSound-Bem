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
from src.ffat_solve import monte_carlo_solve
import os
from glob import glob
from tqdm import tqdm
from numba import njit
import matplotlib.pyplot as plt
import configparser
import meshio

data_dir = sys.argv[1]
config = configparser.ConfigParser()
config.read(f"{data_dir}/config.ini")
data = np.load(f"{data_dir}/bem.npz")
vertices = data["vertices"]
triangles = data["triangles"]
neumann = data["neumann"]
ks = data["wave_number"]
points = data["points"]

mode_num = len(ks)
vertices = torch.from_numpy(vertices).cuda().to(torch.float32)
triangles = torch.from_numpy(triangles).cuda().to(torch.int32)

neumann_tri = torch.from_numpy(neumann).cuda()
ks = torch.from_numpy(-ks).cuda().to(torch.float32)
points = torch.from_numpy(points).cuda().to(torch.float32)
monte_carlo_solve(vertices, triangles, neumann_tri, ks, points, 0)
for n in [0, 1000, 2000, 4000]:
    ffat_map, cost_time = monte_carlo_solve(
        vertices,
        triangles,
        neumann_tri,
        ks,
        points,
        n,
        nsteps=100,
        check_converge=False,
        return_cost_time=True,
    )
    np.savez_compressed(
        f"{data_dir}/ours_{n}.npz",
        ffat_map=ffat_map,
        cost_time=cost_time,
    )
