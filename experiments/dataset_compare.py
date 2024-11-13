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


data_dir = "dataset/ABC/data"
eigen_dir = os.path.join(data_dir, "eigen")
eigen_list = glob(os.path.join(eigen_dir, "*.npz"))


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


def process(eigen_path, out_path):
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

    # timer = Timer(log_output=True)
    # importance = torch.ones(len(triangles), dtype=torch.float32).cuda()
    # sampler = ImportanceSampler(vertices, triangles, importance, 100000)
    # sampler.update()
    # sampler.poisson_disk_resample(0.004, 4)
    # timer.log("sample points: ", sampler.num_samples, record=True)

    # G0_constructor = MonteCarloWeight(sampler.points, sampler)
    # G1_constructor = MonteCarloWeight(sampler.points, sampler, deriv=True)
    # G0_batch = G0_constructor.get_weights_boundary_ks(ks)
    # G1_batch = G1_constructor.get_weights_boundary_ks(ks)
    # neumann = neumann[:, sampler.points_index, :]
    # b_batch = torch.bmm(G0_batch, neumann).permute(1, 2, 0)
    # timer.log("construct G and b", record=True)

    # solver = BiCGSTAB_batch(
    #     lambda x: (torch.bmm(G1_batch, x.permute(2, 0, 1)).permute(1, 2, 0) - x)
    # )
    # timer.log("construct A", record=True)

    # dirichlet = solver.solve(b_batch, tol=1e-6, nsteps=20).permute(2, 0, 1)
    # timer.log("solve", record=True)

    # cost_time = timer.record_time

    # print("bem cost time: ", bem_cost_time)
    # print("ours cost time: ", cost_time)

    image_size = 32
    length = 0.15
    points_ = unit_sphere_surface_points(image_size) * length
    points_list = [points_ * 1.25, points_ * 1.25**2, points_ * 1.25**3]

    # ffat_maps_ours = []
    # for points in points_list:
    #     points = torch.from_numpy(points).cuda().float().reshape(-1, 3)
    #     points_batch = 256
    #     idx = 0
    #     ffat_map = torch.zeros((mode_num, image_size * 2 * image_size)).cuda()
    #     while idx < len(points):
    #         stride = min(points_batch, len(points) - idx)
    #         sub_points = points[idx : idx + stride]
    #         G0_constructor = MonteCarloWeight(sub_points, sampler)
    #         G1_constructor = MonteCarloWeight(sub_points, sampler, deriv=True)
    #         G0 = G0_constructor.get_weights_potential_ks(ks)
    #         G1 = G1_constructor.get_weights_potential_ks(ks)
    #         # print(G0.shape, G1.shape, dirichlet.shape, neumann.shape)
    #         RHS = G1 @ dirichlet - G0 @ neumann
    #         ffat_map[:, idx : idx + stride] = RHS.reshape(mode_num, -1).abs()
    #         idx += stride
    #     ffat_map = ffat_map.reshape(mode_num, image_size * 2, image_size)
    #     ffat_maps_ours.append(ffat_map.cpu().numpy())
    #     # print(ffat_map.shape)

    ffat_maps_bem = []
    vertices = vertices.cpu().numpy()
    triangles = triangles.cpu().numpy()
    for points in points_list:
        points = torch.from_numpy(points).cuda().float().reshape(-1, 3)
        ffat_map = np.zeros((mode_num, image_size * 2, image_size))
        for i in range(mode_num):
            mode_ffat_map = solve_points_dirichlet(
                vertices,
                triangles,
                neumann_bem[:, i],
                dirichlet_bem[:, i],
                ks[i].item(),
                points,
            ).reshape(image_size * 2, image_size)
            ffat_map[i] = np.abs(mode_ffat_map)
        ffat_maps_bem.append(ffat_map)

    ns_data = np.load(eigen_path.replace("eigen", "NeuralSound"))
    ns_ffat_map = ns_data["ffat_map"][:, 0, :, :]
    ffat_maps_neuralsound = []
    for points in points_list:
        r = (points**2).sum(-1) ** 0.5
        ffat_map = ns_ffat_map / r[0]
        ffat_maps_neuralsound.append(ffat_map)

    np.savez_compressed(
        out_path,
        # ours=ffat_maps_ours,
        bem=ffat_maps_bem,
        neuralsound=ffat_maps_neuralsound,
        # ours_time=cost_time,
        bem_time=bem_cost_time,
        neuralsound_time=ns_data["cost_time"],
    )


warm_start = False
for eigen_path in tqdm(eigen_list):
    out_path = eigen_path.replace("eigen", "compare")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not warm_start:
        process(eigen_path, out_path)
        warm_start = True
        process(eigen_path, out_path)
    else:
        if os.path.exists(out_path):
            continue
        process(eigen_path, out_path)
