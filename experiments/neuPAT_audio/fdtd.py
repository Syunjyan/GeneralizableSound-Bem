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
    FDTDSimulator,
    get_bound_info,
)
from src.solver import BiCGSTAB_batch
from src.utils import plot_point_cloud, plot_mesh, CombinedFig
from src.ffat_solve import monte_carlo_solve, bem_solve
from src.audio import calculate_bin_frequencies
import matplotlib.pyplot as plt

data_dir = "dataset/NeuPAT/audio/large_mlp"


def get_output_dir(freq, src_pos_y):
    dir_name = f"{data_dir}/{src_pos_y:.1f}_{freq:.0f}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


import json
import numpy as np


with open(f"{data_dir}/../config.json", "r") as file:
    config_data = json.load(file)

sample_data = torch.load(f"{data_dir}/../sample_points.pt")
neumann = sample_data["neumann"].cuda()
cdf = sample_data["cdf"].item()
importance = sample_data["importance"].cuda()
vertices_vib = sample_data["vertices_vib"].cuda()
triangles_vib = sample_data["triangles_vib"].cuda()
vertices_static = sample_data["vertices_static"].cuda()
triangles_static = sample_data["triangles_static"].cuda()
neumann_tri_static = torch.zeros(len(triangles_static), 1, dtype=torch.float32).cuda()
neumann_tri_vib = torch.ones(len(triangles_vib), 1, dtype=torch.float32).cuda()

triangle_y = vertices_vib[triangles_vib][:, :, 1].mean(1)
neumann_tri_vib[triangle_y > -0.04] = 0

neumann_tri = torch.cat([neumann_tri_vib, neumann_tri_static], dim=0).reshape(-1)

triangles = torch.cat([triangles_vib, triangles_static + len(vertices_vib)], dim=0)

src_pos_min = torch.tensor(
    config_data.get("solver", {}).get("src_pos_min"), device="cuda", dtype=torch.float32
)
src_pos_max = torch.tensor(
    config_data.get("solver", {}).get("src_pos_max"), device="cuda", dtype=torch.float32
)
freq_min = config_data.get("solver", {}).get("freq_min", 100)
freq_max = config_data.get("solver", {}).get("freq_max", 10000)
freq_min_log = np.log10(freq_min)
freq_max_log = np.log10(freq_max)

with open(f"{data_dir}/net.json", "r") as file:
    train_config_data = json.load(file)

train_params = train_config_data.get("train", {})
model = NeuPAT(
    1,
    train_config_data.get("encoding_config"),
    train_config_data.get("network_config"),
).cuda()
model.load_state_dict(torch.load(f"{data_dir}/model.pt"))
model.eval()
torch.set_grad_enabled(False)

first = True
xs = torch.linspace(0, 1, 64, device="cuda", dtype=torch.float32)
ys = torch.linspace(0, 1, 32, device="cuda", dtype=torch.float32)
gridx, gridy = torch.meshgrid(xs, ys)

trg_x, trg_y = 0, 0

freq_bins = calculate_bin_frequencies()
print("freq_bins:", freq_bins)

check_correctness = False
move_step_num = 10
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
trg_points = get_spherical_surface_points(vertices_static, r).reshape(64, 32, 3)
trg_points = trg_points[trg_x, trg_y].reshape(-1, 3)

freq_pos = torch.zeros(len(freq_bins), device="cuda", dtype=torch.float32)
freqs = torch.zeros(len(freq_bins), device="cuda", dtype=torch.float32)
for freq_i in tqdm(range(len(freq_bins))):
    freq_bin = freq_bins[freq_i]
    if freq_bin < freq_min or freq_bin > freq_max:
        continue
    freq_pos[freq_i] = (np.log10(freq_bin) - freq_min_log) / (
        freq_max_log - freq_min_log
    )
    freqs[freq_i] = freq_bins[freq_i]

src_pos = torch.zeros(move_step_num, 1, device="cuda", dtype=torch.float32)
for src_pos_i in tqdm(range(0, move_step_num)):
    src_pos_y = src_pos_i / move_step_num
    src_pos[src_pos_i] = src_pos_y


def run_neual_cache():
    x = torch.zeros(len(src_pos), len(freq_pos), 5, dtype=torch.float32, device="cuda")
    x[:, :, 0] = src_pos
    x[:, :, 1:4] = trg_pos
    x[:, :, 4] = freq_pos.reshape(1, -1)
    nc_value = model(x.reshape(-1, 5)).reshape(len(src_pos), len(freq_pos))
    timer = Timer()
    nc_spec = model(x.reshape(-1, 5)).reshape(len(src_pos), len(freq_pos))
    nc_spec = nc_spec.cpu().numpy()
    nc_cost_time = timer.get_time()
    return nc_spec, nc_cost_time


nc_spec, nc_cost_time = run_neual_cache()
print(nc_cost_time)
import torchaudio

get_spectrogram = torchaudio.transforms.Spectrogram(n_fft=128, power=2.0).cuda()

import librosa

signal_resampleds = []

resolution = int(sys.argv[1])


def run_fdtd():
    min_bound, max_bound, bound_size = get_bound_info(vertices_static, padding=4)
    fdtd = FDTDSimulator(
        min_bound, max_bound, bound_size, resolution, trg_points.reshape(-1)
    )
    batch_size = 2000
    # print(0.1 / fdtd.dt /   batch_size)
    ts = torch.arange(batch_size, device="cuda", dtype=torch.float32) * fdtd.dt
    neumann_signal = torch.zeros(batch_size, device="cuda", dtype=torch.float32)
    for freq in freq_bins:
        neumann_signal += (
            torch.sin(2 * np.pi * freq * ts + torch.rand(1).item() * 2 * np.pi)
            + torch.sin(2 * np.pi * freq * ts + torch.rand(1).item() * 2 * np.pi)
            + torch.sin(2 * np.pi * freq * ts + torch.rand(1).item() * 2 * np.pi)
        ) / 3
    # plt.plot(neumann_signal.cpu().numpy())
    # plt.savefig(f"{data_dir}/neumann_signal.png")
    # plt.close()
    neumann_surf = neumann_tri.reshape(-1, 1) * neumann_signal.reshape(1, -1)
    signal = fdtd.update(vertices, triangles, neumann_surf)
    # print(signal.max(), signal.min())
    # plt.plot(signal.cpu().numpy())
    # plt.savefig(f"{data_dir}/signal.png")
    # plt.close()
    signal_resampled = librosa.resample(
        signal.cpu().numpy(), orig_sr=fdtd.sample_rate, target_sr=16000
    )
    # plt.plot(signal_resampled)
    # plt.savefig(f"{data_dir}/signal_resampled.png")
    # plt.close()
    signal_resampled = torch.from_numpy(signal_resampled).cuda()
    signal_resampleds.append(signal_resampled)


fdtd_spec = np.zeros((move_step_num, len(freq_bins)))
fdtd_cost_time = 0
for src_pos_i in tqdm(range(0, move_step_num)):
    src_pos_y = src_pos_i / move_step_num
    src_pos = torch.zeros(3, device="cuda", dtype=torch.float32)
    src_pos[1] = src_pos_y
    displacement = src_pos * (src_pos_max - src_pos_min) + src_pos_min
    vertices_vib_updated = vertices_vib + displacement
    vertices = torch.cat([vertices_vib_updated, vertices_static], dim=0)
    timer = Timer()
    run_fdtd()
    # spec = run_fdtd()
    # fdtd_spec[src_pos_i] = spec
    fdtd_cost_time += timer.get_time()

# plt.subplot(211)
# plt.imshow(nc_spec)
# plt.subplot(212)
# plt.imshow(fdtd_spec)
# plt.savefig(f"{data_dir}/fdtd_nc_spec_comp.png")
# plt.close()
torch.save(
    {
        "cost_time": fdtd_cost_time,
        "signal_resampleds": signal_resampleds,
    },
    f"{data_dir}/fdtd_{resolution}.pt",
)
