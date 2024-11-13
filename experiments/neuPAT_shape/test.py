import sys

sys.path.append("./")
from src.net.model import NeuPAT
import torch
from glob import glob
import torch
from tqdm import tqdm
import json
from src.timer import Timer
import os
from src.modalobj.model import get_spherical_surface_points, BEMModel
from src.mcs.mcs import (
    ImportanceSampler,
    MonteCarloWeight,
    get_weights_boundary_ks_base,
    get_weights_potential_ks_base,
)
from src.solver import BiCGSTAB_batch
from src.utils import plot_point_cloud, plot_mesh, CombinedFig
from src.ffat_solve import monte_carlo_solve, bem_solve
from src.audio import calculate_bin_frequencies

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

with open(f"{data_dir}/baseline/net.json", "r") as file:
    train_config_data = json.load(file)

train_params = train_config_data.get("train", {})
model = NeuPAT(
    1,
    train_config_data.get("encoding_config"),
    train_config_data.get("network_config"),
).cuda()

model.load_state_dict(torch.load(f"{data_dir}/baseline/model.pt"))
model.eval()
torch.set_grad_enabled(False)

xs = torch.linspace(0, 1, 64, device="cuda", dtype=torch.float32)
ys = torch.linspace(0, 1, 32, device="cuda", dtype=torch.float32)
gridx, gridy = torch.meshgrid(xs, ys)

trg_x, trg_y = 0, 0

freq_bins = calculate_bin_frequencies(256)
print("freq_bins:", freq_bins)

nc_cost_time = 0
r_min = 2
r_max = 4
trg_pos = torch.zeros(64, 32, 3, device="cuda", dtype=torch.float32)
r_scale = torch.ones(1).cuda() * 0.5
r = (r_scale * (r_max - r_min) + r_min).item()
trg_pos[:, :, 0] = r_scale
trg_pos[:, :, 1] = gridx
trg_pos[:, :, 2] = gridy
trg_pos = trg_pos[trg_x, trg_y].reshape(3)

freq_pos = torch.zeros(len(freq_bins), device="cuda", dtype=torch.float32)
for freq_i in tqdm(range(len(freq_bins))):
    freq_bin = freq_bins[freq_i]
    if freq_bin < freq_min or freq_bin > freq_max:
        continue
    freq_pos[freq_i] = (np.log10(freq_bin) - freq_min_log) / (
        freq_max_log - freq_min_log
    )

mesh_num = len(vertices_lst)
src_pos = torch.zeros(mesh_num, 1, device="cuda", dtype=torch.float32)
for src_pos_i in tqdm(range(0, mesh_num)):
    src_pos[src_pos_i] = src_pos_i / mesh_num


def run_neual_cache():
    x = torch.zeros(len(src_pos), len(freq_pos), 5, dtype=torch.float32, device="cuda")
    x[:, :, 0] = src_pos
    x[:, :, 1] = freq_pos.reshape(1, -1)
    x[:, :, 2:5] = trg_pos.reshape(1, 1, 3)

    nc_spec = model(x.reshape(-1, 5)).reshape(len(src_pos), len(freq_pos))
    timer = Timer()
    nc_spec = model(x.reshape(-1, 5)).reshape(len(src_pos), len(freq_pos))
    nc_spec = nc_spec.cpu().numpy()
    nc_cost_time = timer.get_time()
    return nc_spec, nc_cost_time


nc_spec, nc_cost_time = run_neual_cache()

from matplotlib import pyplot as plt
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
title_pad = 20
font_size = 20

x_start = 0
x_end = 1

fig = plt.figure(figsize=(18, 3))
gs = GridSpec(1, 3, width_ratios=[1, 1, 3])
ax = fig.add_subplot(gs[0])
img = imread(f"{data_dir}/start.png")
ax.imshow(img)
ax.text(
    0.5,
    -0.2,
    "Start State",
    transform=ax.transAxes,
    ha="center",
    fontproperties=my_font,
    fontsize=font_size,
)
ax.axis("off")
ax = fig.add_subplot(gs[1])
img = imread(f"{data_dir}/end.png")
ax.imshow(img)
ax.text(
    0.5,
    -0.2,
    "End State",
    transform=ax.transAxes,
    ha="center",
    fontproperties=my_font,
    fontsize=font_size,
)
ax.axis("off")
ax = fig.add_subplot(gs[2])
img = nc_spec[:, ::-1].T
v_max = np.max(np.abs(img)) * 0.9
v_min = np.min(np.abs(img)) * 1.1
ax.imshow(
    np.abs(img),
    vmin=v_min,
    vmax=v_max,
    cmap="viridis",
    extent=[x_start, x_end, freq_bins[0], freq_bins[-1]],
    aspect="auto",
)  # Stretching x-axis with extent and interpolation

num_y_ticks = 8
num_x_ticks = 8
# Set ticks
y_ticks = np.linspace(freq_bins[0], freq_bins[-1], num_y_ticks)
ax.set_yticks(y_ticks)
x_ticks = np.linspace(x_start, x_end, num_x_ticks)
ax.set_xticks(x_ticks)

ax.set_xlabel("Shape Coeff", fontproperties=my_font, fontsize=font_size)
ax.set_ylabel("Frequency (Hz)", fontproperties=my_font, fontsize=font_size)

plt.savefig(f"{data_dir}/nc_spec.png", bbox_inches="tight", dpi=300)
