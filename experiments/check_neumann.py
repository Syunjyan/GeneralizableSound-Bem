import sys

sys.path.append("./")

import torch
from src.mcs.mcs import ImportanceSampler, MonteCarloWeight
from src.timer import Timer
from src.modalobj.model import solve_points_dirichlet
import numpy as np
from src.utils import plot_mesh, plot_point_cloud, crop_center, combine_images
from src.solver import BiCGSTAB, BiCGSTAB_batch, BiCGSTAB_batch2
import os
from glob import glob
from tqdm import tqdm
from numba import njit
import matplotlib.pyplot as plt


eigen_path = sys.argv[1]


@njit()
def unit_sphere_surface_points(res):
    # r = 0.5
    points = np.zeros((2 * res, res, 3))
    phi_spacing = 2 * np.pi / (res * 2 - 1)
    theta_spacing = np.pi / (res - 1)
    for i in range(2 * res):
        for j in range(res):
            phi = phi_spacing * i
            theta = theta_spacing * j
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points[i, j] = [x, y, z]
    return points * 0.5


def save_ffat_maps(ffat_map, img_path):
    if isinstance(ffat_map, torch.Tensor):
        ffat_map = ffat_map.cpu().numpy()

    rows = 4
    cols = ffat_map.shape[0] // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    for i in range(rows):
        for j in range(cols):
            img = axes[i, j].imshow(ffat_map[i * cols + j])
            fig.colorbar(img, ax=axes[i, j])  # Add color bar for each subplot

    plt.savefig(img_path)


data = np.load(eigen_path)
vertices, triangles = data["vertices"], data["triangles"]
neumann_bem, dirichlet_bem = data["neumann"], data["dirichlet"]

wave_number = data["wave_number"]
bem_cost_time = data["bem_cost_time"]
mode_num = len(wave_number)
vertices = torch.from_numpy(vertices).cuda().to(torch.float32)
triangles = torch.from_numpy(triangles).cuda().to(torch.int32)
ks = torch.from_numpy(wave_number).cuda().to(torch.float32)
neumann = torch.from_numpy(neumann_bem).cuda().to(torch.complex64).T.unsqueeze(-1)
# print(neumann.shape)  # (batch_size, n, 1)

timer = Timer(log_output=True)
importance = torch.ones(len(triangles), dtype=torch.float32).cuda()
sampler = ImportanceSampler(vertices, triangles, importance, 100000)
sampler.update()
sampler.poisson_disk_resample(0.004, 4)
timer.log("sample points: ", sampler.num_samples, record=True)

G0_constructor = MonteCarloWeight(sampler.points, sampler)
G1_constructor = MonteCarloWeight(sampler.points, sampler, deriv=True)
G0_batch = G0_constructor.get_weights_boundary_ks(ks)
G1_batch = G1_constructor.get_weights_boundary_ks(ks)
neumann = neumann[:, sampler.points_index, :]
b_batch = torch.bmm(G0_batch, neumann).permute(1, 2, 0)
timer.log("construct G and b", record=True)

solver = BiCGSTAB_batch(
    lambda x: (torch.bmm(G1_batch, x.permute(2, 0, 1)).permute(1, 2, 0) - x)
)
timer.log("construct A", record=True)

dirichlet = solver.solve(b_batch, tol=1e-6, nsteps=70).permute(2, 0, 1)
timer.log("solve", record=True)

cost_time = timer.record_time

print("bem cost time: ", bem_cost_time)
print("ours cost time: ", cost_time)

dirichlet = dirichlet.permute(1, 0, 2).squeeze(-1)
print(dirichlet.shape)
print(dirichlet_bem.shape)

plot_point_cloud(vertices, triangles, sampler.points, dirichlet.real).show()

plot_point_cloud(vertices, triangles, vertices, dirichlet_bem.real).show()

plot_point_cloud(vertices, triangles, sampler.points, dirichlet.imag).show()

plot_point_cloud(vertices, triangles, vertices, dirichlet_bem.imag).show()
