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
SNRs = np.zeros((len(Ms), len(freqs)))

vertices = torch.from_numpy(obj.vertices).cuda().to(torch.float32)
triangles = torch.from_numpy(obj.triangles).cuda().to(torch.int32)
importance = torch.ones(len(triangles), dtype=torch.float32).cuda()

for i in range(2):
    timer = Timer(log_output=True)
    sampler = ImportanceSampler(vertices, triangles, importance, 100000)
    sampler.update()
    r = config.getfloat("sampling", "radius")
    sampler.poisson_disk_resample(r, 5)
    timer.log("sample points: ", sampler.num_samples, record=True)

    neumann_data = []
    ks = []
    mode_num = len(freqs) * len(Ms)
    cost_time = 0
    for i, M in enumerate(Ms):
        for j, freq in enumerate(freqs):
            k = freq * 2 * np.pi / 343.2
            ks.append(-k)
            model1 = MultipoleModel(x1, [1, 0, 0], -k, M)
            model2 = MultipoleModel(x2, [0, 1, 0], -k, M)
            model3 = MultipoleModel(x3, [0, 0, 1], -k, M)
            neumann_coeff = (
                model1.solve_neumann(sampler.points, sampler.points_normals)
                + model2.solve_neumann(sampler.points, sampler.points_normals)
                + model3.solve_neumann(sampler.points, sampler.points_normals)
            )
            neumann_data.append(neumann_coeff)

    neumann = torch.stack(neumann_data, dim=0).unsqueeze(-1)
    ks = torch.tensor(ks).cuda().to(torch.float32)

    timer.log("prepare data", record=False)
    timer_solver = Timer()
    G0_constructor = MonteCarloWeight(sampler.points, sampler)
    G1_constructor = MonteCarloWeight(sampler.points, sampler, deriv=True)
    G0_batch = G0_constructor.get_weights_boundary_ks(ks)
    G1_batch = G1_constructor.get_weights_boundary_ks(ks)

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
    # dirichlet = (torch.linalg.inv(G1_batch[0]) @ b_batch.squeeze(-1)).reshape(1, -1, 1)
    timer.log("solve equation", record=True)
    timer_solver.log("solve equation", record=True)
    G0_constructor = MonteCarloWeight(points, sampler)
    G1_constructor = MonteCarloWeight(points, sampler, deriv=True)
    G0 = G0_constructor.get_weights_potential_ks(ks)
    G1 = G1_constructor.get_weights_potential_ks(ks)
    ffat_map = G1 @ dirichlet - G0 @ neumann
    timer.log("solve ffat map", record=True)

    ffat_map_ours = ffat_map.reshape(len(Ms), len(freqs), -1).cpu().numpy()
    cost_time = timer.record_time
    cost_time_solver = timer_solver.record_time

SNRs = np.zeros((len(Ms), len(freqs)))
for i, M in enumerate(Ms):
    for j, freq in enumerate(freqs):
        points_dirichlet = ffat_map_ours[i, j]
        k = freq * 2 * np.pi / 343.2
        points_dirichlet_gt = (
            model1.solve_dirichlet(points).cpu().numpy()
            + model2.solve_dirichlet(points).cpu().numpy()
            + model3.solve_dirichlet(points).cpu().numpy()
        )
        SNRs[i, j] = SNR(points_dirichlet_gt, points_dirichlet)
        # if i == 1 and j == 0:
        #     CombinedFig().add_mesh(obj.vertices, obj.triangles).add_points(
        #         points, points_dirichlet.imag
        #     ).show()
        #     CombinedFig().add_mesh(obj.vertices, obj.triangles).add_points(
        #         points, points_dirichlet_gt.imag
        #     ).show()
        #     CombinedFig().add_mesh(obj.vertices, obj.triangles).add_points(
        #         points, points_dirichlet.real
        #     ).show()
        #     CombinedFig().add_mesh(obj.vertices, obj.triangles).add_points(
        #         points, points_dirichlet_gt.real
        #     ).show()
        #     CombinedFig().add_mesh(obj.vertices, obj.triangles).add_points(
        #         sampler.points, neumann.reshape(-1, 1).real
        #     ).add_points(x0, point_size=10, showscale=False).show()
        #     CombinedFig().add_mesh(obj.vertices, obj.triangles).add_points(
        #         sampler.points, neumann.reshape(-1, 1).imag
        #     ).add_points(x0, point_size=10, showscale=False).show()

print(SNRs)

np.savez_compressed(
    f"{data_dir}/ours.npz",
    ffat_map=ffat_map_ours,
    cost_time=cost_time,
    cost_time_solver=cost_time_solver,
)
