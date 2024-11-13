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

data_dir = sys.argv[1]

config = configparser.ConfigParser()
config.read(f"{data_dir}/config.ini")
obj = MeshObj(f"{data_dir}/mesh.obj")
obj.center[0] += config.getfloat("mesh", "offset_x")
obj.center[1] += config.getfloat("mesh", "offset_y")
obj.center[2] += config.getfloat("mesh", "offset_z")

x0 = obj.center
freqs = [2000]
points = obj.spherical_surface_points(2)
Ms = [1]
x1 = x0 + np.array([0, 0.05, 0])
x2 = x0 + np.array([0, -0.05, 0])
x3 = x0 + np.array([0, 0, 0])

ffat_map_bem = np.zeros((len(Ms), len(freqs), len(points)), dtype=np.complex64)
ffat_map_gt = np.zeros((len(Ms), len(freqs), len(points)), dtype=np.complex64)
neumann = np.zeros((len(Ms), len(freqs), len(obj.triangles_center)), dtype=np.complex64)
dirichlet_bem = np.zeros((len(Ms), len(freqs), len(obj.vertices)), dtype=np.complex64)
SNRs = np.zeros((len(Ms), len(freqs)))


mode_num = len(freqs) * len(Ms)
timer = Timer()
for i, M in enumerate(Ms):
    for j, freq in enumerate(freqs):
        k = freq * 2 * np.pi / 343.2
        CombinedFig().add_mesh(obj.vertices, obj.triangles).add_points(
            [x1, x2, x3]
        ).show()
        model1 = MultipoleModel(x1, [1, 0, 0], -k, M)
        model2 = MultipoleModel(x2, [0, 1, 0], -k, M)
        model3 = MultipoleModel(x3, [0, 0, 1], -k, M)
        neumann_coeff = (
            model1.solve_neumann(obj.triangles_center, obj.triangles_normal)
            + model2.solve_neumann(obj.triangles_center, obj.triangles_normal)
            + model3.solve_neumann(obj.triangles_center, obj.triangles_normal)
        )
        bem = BEMModel(obj.vertices, obj.triangles, -k)
        bem.boundary_equation_solve(neumann_coeff)
        points_dirichlet = bem.potential_solve(points)
        points_dirichlet_gt = (
            model1.solve_dirichlet(points).cpu().numpy()
            + model2.solve_dirichlet(points).cpu().numpy()
            + model3.solve_dirichlet(points).cpu().numpy()
        )
        ffat_map_bem[i, j] = points_dirichlet
        ffat_map_gt[i, j] = points_dirichlet_gt
        neumann[i, j] = neumann_coeff.cpu().numpy()
        SNRs[i, j] = SNR(points_dirichlet_gt, points_dirichlet)

np.savez_compressed(
    f"{data_dir}/gt.npz",
    ffat_map=ffat_map_gt,
)

np.savez_compressed(
    f"{data_dir}/bem.npz",
    ffat_map=ffat_map_bem,
    cost_time=timer.get_time(),
    neumann=neumann,
    dirichlet=dirichlet_bem,
    SNR=SNRs,
)

print(SNRs)
