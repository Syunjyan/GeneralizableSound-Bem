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


def compute_mesh_area(vertices, triangles):
    vertices = vertices.float()
    triangles = triangles.long()
    edge1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    edge2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    cross_product = torch.cross(edge1, edge2, dim=1)
    triangle_areas = 0.5 * torch.norm(cross_product, dim=1)
    total_area = torch.sum(triangle_areas)
    return total_area.item()


data_dir = sys.argv[1]

config = configparser.ConfigParser()
config.read(f"{data_dir}/config.ini")

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
mesh_area = compute_mesh_area(vertices, triangles)
importance = torch.ones(len(triangles), dtype=torch.float32).cuda()

neumann_tri_base = torch.from_numpy(neumann).unsqueeze(-1).cuda()
ks_base = torch.from_numpy(-ks).cuda().to(torch.float32)
points = torch.from_numpy(points).cuda().to(torch.float32)


def run(r, ks, neumann_tri):
    timer = Timer(log_output=True)
    if r == 0:
        sampler = ImportanceSampler(vertices, triangles, importance, 1000)
        sampler.update()
    else:
        sampler = ImportanceSampler(vertices, triangles, importance, 50000)
        sampler.update()
        sampler.poisson_disk_resample(r, 4)
    timer.log("sample points: ", sampler.num_samples, record=True)

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

    tol = config.getfloat("solver", "tol")
    nsteps = config.getint("solver", "nsteps")
    dirichlet, convergence = solver.solve(b_batch, tol=tol, nsteps=nsteps)
    dirichlet = dirichlet.permute(2, 0, 1)

    timer.log("solve equation", record=True)
    return timer.record_time


our_maps = []
snrs = []
ssims = []
n = 2000
r = (mesh_area / (2 * n)) ** 0.5
run(r, ks_base, neumann_tri_base)
t1 = run(r, ks_base, neumann_tri_base)
t2 = 0
for i in range(len(ks_base)):
    ks = [ks_base[i]]
    t2 += run(r, ks, neumann_tri_base[i : i + 1, :, :])

print("parallel time: ", t1)
print("sequential time: ", t2)
