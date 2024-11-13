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

mesh = meshio.read(f"{data_dir}/mesh.obj")
vertices = mesh.points
triangles = mesh.cells_dict["triangle"]

x0 = [0, 0, 0]
freq = 0
points = np.loadtxt(f"{data_dir}/points.txt").astype(np.float32).reshape(-1, 3)
points = torch.from_numpy(points).cuda().to(torch.float32)

vertices = torch.from_numpy(vertices).cuda().to(torch.float32)
triangles = torch.from_numpy(triangles).cuda().to(torch.int32)
importance = torch.ones(len(triangles), dtype=torch.float32).cuda()

sampler = ImportanceSampler(vertices, triangles, importance, 100000)
sampler.update()
sampler.poisson_disk_resample(0.02, 5)
model = MultipoleModel(x0, [1, 0, 0], -10, 0)

neumann = (
    model.solve_neumann(sampler.points, sampler.points_normals)
    .unsqueeze(0)
    .unsqueeze(-1)
)

ks = torch.tensor([-10]).cuda().to(torch.float32)

G0_constructor = MonteCarloWeight(sampler.points, sampler)
G1_constructor = MonteCarloWeight(sampler.points, sampler, deriv=True)
G0_batch = G0_constructor.get_weights_boundary_ks(ks)
G1_batch = G1_constructor.get_weights_boundary_ks(ks)

print(G0_batch.shape, G1_batch.shape, neumann.shape)
b_batch = torch.bmm(G0_batch, neumann).permute(1, 2, 0)
print(b_batch.shape)

solver = BiCGSTAB_batch(
    lambda x: (torch.bmm(G1_batch, x.permute(2, 0, 1)).permute(1, 2, 0) - x)
)
dirichlet, convergence = solver.solve(b_batch, tol=1e-6, nsteps=2000)
dirichlet = dirichlet.permute(2, 0, 1)

ffat_map = torch.zeros(256, 1, 256, 1, dtype=torch.complex64).cuda()
points = points.reshape(256, 256, 3)
for i in range(256):
    points_batch = points[i]
    G0_constructor = MonteCarloWeight(points_batch, sampler)
    G1_constructor = MonteCarloWeight(points_batch, sampler, deriv=True)
    G0 = G0_constructor.get_weights_potential_ks(ks)
    G1 = G1_constructor.get_weights_potential_ks(ks)
    ffat_map[i] = G1 @ dirichlet - G0 @ neumann


np.save(f"{data_dir}/helmholtz_ours.npy", ffat_map.cpu().numpy())
