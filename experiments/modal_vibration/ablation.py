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
mesh_size = config.getfloat("mesh", "size")
bem_time = data["cost_time"]
NeuralSound_data = np.load(data_dir + "/NeuralSound.npz")
r = (points**2).sum(-1) ** 0.5
NeuralSound_ffat = (
    NeuralSound_data["ffat_map"].reshape(-1, 64, 32)
    / r[0]
    / 1.225
    * (mesh_size / 0.15) ** (5 / 2)
)
NeuralSound_time = (
    NeuralSound_data["cost_time"] + np.load(data_dir + "/voxel.npz")["cost_time"]
)
NeuralSoundSNR = []
NeuralSoundSSIM = []
for i in range(8):
    NeuralSoundSNR.append(SNR(ffat_map_bem[i], NeuralSound_ffat[i].reshape(-1)))
    NeuralSoundSSIM.append(
        complex_ssim(ffat_map_bem[i], NeuralSound_ffat[i].reshape(-1))
    )

mode_num = len(ks)
vertices = torch.from_numpy(vertices).cuda().to(torch.float32)
triangles = torch.from_numpy(triangles).cuda().to(torch.int32)
mesh_area = compute_mesh_area(vertices, triangles)
importance = torch.ones(len(triangles), dtype=torch.float32).cuda()

neumann_tri = torch.from_numpy(neumann).unsqueeze(-1).cuda()
ks = torch.from_numpy(-ks).cuda().to(torch.float32)
points = torch.from_numpy(points).cuda().to(torch.float32)


def run(r):
    timer = Timer(log_output=True)
    print("r:", r)
    if r is None:
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
    G0_constructor = MonteCarloWeight(points, sampler)
    G1_constructor = MonteCarloWeight(points, sampler, deriv=True)
    G0 = G0_constructor.get_weights_potential_ks(ks)
    G1 = G1_constructor.get_weights_potential_ks(ks)
    ffat_map = G1 @ dirichlet - G0 @ neumann
    timer.log("solve ffat map", record=True)

    ffat_map_ours = ffat_map.reshape(mode_num, -1).cpu().numpy()
    cost_time = timer.record_time

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

    print(SNRs.mean())
    print(ssims.mean())

    return (
        np.abs(ffat_map_ours[0]).reshape(64, 32),
        SNRs.mean(),
        ssims.mean(),
        cost_time,
        solver.log_rerr,
    )


our_maps = []
snrs = []
ssims = []
cost_times = []
rerrs = []
run(1000)
for n in [0, 1000, 2000, 4000]:
    if n == 0:
        r = None
    else:
        r = (mesh_area / (2 * n)) ** 0.5
    maps, snr, ssim, cost_time, rerr = run(r)
    our_maps.append(maps)
    snrs.append(snr)
    ssims.append(ssim)
    cost_times.append(cost_time)
    rerrs.append(rerr)
titles = ["Random 1K", "Poisson 1K", "Poisson 2K", "Poisson 4K"]

cost_time_dict = {
    "NeuralSound": NeuralSound_time,
    "Random 1K": cost_times[0],
    "Poisson 1K": cost_times[1],
    "Poisson 2K": cost_times[2],
    "Poisson 4K": cost_times[3],
    "BEM": float(bem_time),
}
import json

# save cost time dict to json
with open(f"{data_dir}/cost_time.json", "w") as f:
    json.dump(cost_time_dict, f)

from matplotlib.font_manager import FontProperties
from matplotlib.image import imread
from matplotlib.gridspec import GridSpec

# Assuming the other parts of your script are already defined (ffat_map_bem, our_maps, snrs, ssims, data_dir)

# Load specific font
font_path = (
    "/home/jxt/.local/share/fonts/LinBiolinum_R.ttf"  # Replace with your font file path
)
font_bold_path = "/home/jxt/.local/share/fonts/2LinBiolinum_RB.ttf"  # Replace with your font file path

my_font = FontProperties(fname=font_path)
my_font_bold = FontProperties(fname=font_bold_path)

# Set color scale limits
vmin, vmax = np.abs(ffat_map_bem[0]).min(), np.abs(ffat_map_bem[0]).max()
title_pad = 20
font_size = 25
# Create figure
fig = plt.figure(figsize=(28, 7))
gs = GridSpec(1, 8, width_ratios=[1.2, 1, 1, 1, 1, 1, 1, 3])
# Load the image from the specified path
mesh_render_img = imread(f"{data_dir}/mesh_render.png")
# Calculate the indices to crop 1/5 from the sides
left_index = mesh_render_img.shape[1] // 5
right_index = -left_index if left_index != 0 else mesh_render_img.shape[1]

# Crop the image
cropped_img = mesh_render_img[:, left_index:right_index]
# Plot the image from the path in the first subplot
ax = plt.subplot(gs[0])
ax.imshow(cropped_img)
ax.text(
    0.5,
    -0.1,
    "SNR | SSIM",
    transform=ax.transAxes,
    ha="center",
    fontproperties=my_font,
    fontsize=font_size,
)
if len(sys.argv) > 2:
    plt.title("Mesh", fontproperties=my_font, fontsize=font_size, pad=title_pad)
ax.axis("off")

# Plot the first image
ax = plt.subplot(gs[1])

ax.imshow(np.abs(ffat_map_bem[0]).reshape(64, 32), cmap="viridis", vmin=vmin, vmax=vmax)
ax.text(
    16,
    70,
    f"Inf | 1.0",
    ha="center",
    fontproperties=my_font,
    fontsize=font_size,
)

if len(sys.argv) > 2:
    plt.title("BEM", fontproperties=my_font, fontsize=font_size, pad=title_pad)
plt.axis("off")

ax = plt.subplot(gs[2])
ax.imshow(NeuralSound_ffat[0], cmap="viridis", vmin=vmin, vmax=vmax)
ax.text(
    16,
    70,
    f"{np.mean(NeuralSoundSNR):.2f} | {np.mean(NeuralSoundSSIM):.2f}",
    ha="center",
    fontproperties=my_font,
    fontsize=font_size,
)
if len(sys.argv) > 2:
    plt.title("NeuralSound", fontproperties=my_font, fontsize=font_size, pad=title_pad)
plt.axis("off")

# Plot other images and add titles
for i in range(4):
    ax = plt.subplot(gs[i + 3])
    ax.imshow(np.abs(our_maps[i]).reshape(64, 32), cmap="viridis", vmin=vmin, vmax=vmax)
    if i == 3:
        ax.text(
            16,
            70,
            f"{snrs[i]:.2f} | {ssims[i]:.2f}",
            ha="center",
            fontproperties=my_font_bold,
            fontsize=font_size,
            fontweight="bold",
        )
    else:
        ax.text(
            16,
            70,
            f"{snrs[i]:.2f} | {ssims[i]:.2f}",
            ha="center",
            fontproperties=my_font,
            fontsize=font_size,
        )
    if len(sys.argv) > 2:
        plt.title(titles[i], fontproperties=my_font, fontsize=font_size, pad=title_pad)
    plt.axis("off")
    # Add text below the image


import seaborn as sns
from matplotlib.ticker import MaxNLocator

sns.set()
ax = plt.subplot(gs[7])
ax.set_position([0.75, 0.29, 0.20, 0.465])
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# plot the error
for i in range(0, 4):
    print(rerrs[i])
    plt.plot(rerrs[i], label=titles[i])
plt.legend(prop=my_font, fontsize=font_size * 2)
plt.xlabel("Iteration", fontproperties=my_font, fontsize=font_size)
plt.ylabel("Relative Error", fontproperties=my_font, fontsize=font_size)
plt.xticks(fontproperties=my_font, fontsize=font_size)
plt.yticks(fontproperties=my_font, fontsize=font_size)
plt.yscale("log")
if len(sys.argv) > 2:
    plt.title(
        "Convergence Curve", fontproperties=my_font, fontsize=font_size, pad=title_pad
    )
# plt.tight_layout()
plt.savefig(f"{data_dir}/ablation.png")
