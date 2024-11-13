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
from scipy.spatial.transform import Rotation as R

import json
import numpy as np


def global_to_local(X, ori, global_position):
    rotation = R.from_quat(ori)
    rotation_matrix = rotation.as_matrix()
    inv_rotation_matrix = np.transpose(rotation_matrix)
    inv_translation = -np.dot(inv_rotation_matrix, X)
    local_position = np.dot(inv_rotation_matrix, global_position) + inv_translation
    return local_position


data_dir = "dataset/NeuPAT/audio_cam/large_mlp"


with open(f"{data_dir}/../config.json", "r") as file:
    config_data = json.load(file)

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

xs = torch.linspace(0, 1, 64, device="cuda", dtype=torch.float32)
ys = torch.linspace(0, 1, 32, device="cuda", dtype=torch.float32)
gridx, gridy = torch.meshgrid(xs, ys)

animation_data = np.load(f"{data_dir}/../animation.npz")
move_lst = animation_data["move"]
cam_lst = animation_data["cam"]

n_fft = 1024
freq_bins = calculate_bin_frequencies(n_fft)

nc_cost_time = 0
r_min = 2
r_max = 4
trg_pos = torch.zeros(64, 32, 3, device="cuda", dtype=torch.float32)
trg_pos[:, :, 0] = 1
trg_pos[:, :, 1] = gridx
trg_pos[:, :, 2] = gridy
trg_pos_0 = trg_pos[0, 0].reshape(3)
trg_pos_1 = trg_pos[15, 15].reshape(3)
cam_lst = torch.from_numpy(cam_lst).cuda()
trg_pos = trg_pos_0 + (trg_pos_1 - trg_pos_0) * cam_lst.reshape(-1, 1, 1)

freq_pos = torch.zeros(len(freq_bins), device="cuda", dtype=torch.float32)
for freq_i in tqdm(range(len(freq_bins))):
    freq_bin = freq_bins[freq_i]
    if freq_bin < freq_min or freq_bin > freq_max:
        continue
    freq_pos[freq_i] = (np.log10(freq_bin) - freq_min_log) / (
        freq_max_log - freq_min_log
    )

src_num = len(move_lst)
src_pos = torch.from_numpy(move_lst).cuda().unsqueeze(1)
print("src_pos:", src_pos.shape)
print("freq_pos:", freq_pos.shape)
print("trg_pos:", trg_pos.shape)


def run_neual_cache():
    x = torch.zeros(len(src_pos), len(freq_pos), 5, dtype=torch.float32, device="cuda")
    x[:, :, 0] = src_pos
    x[:, :, 1:4] = trg_pos
    x[:, :, 4] = freq_pos.reshape(1, -1)
    nc_spec = model(x.reshape(-1, 5)).reshape(len(src_pos), len(freq_pos))
    timer = Timer()
    nc_spec = model(x.reshape(-1, 5)).reshape(len(src_pos), len(freq_pos))
    nc_spec = nc_spec.cpu().numpy()
    nc_cost_time = timer.get_time()
    return nc_spec, nc_cost_time


nc_spec, nc_cost_time = run_neual_cache()
nc_spec, nc_cost_time = run_neual_cache()
print(nc_cost_time)
nc_spec = (10**nc_spec.T) * 10e-6 - 10e-6

from src.audio import apply_spec_mask_to_audio
from scipy.io import wavfile

audio = apply_spec_mask_to_audio(sys.argv[-1], nc_spec, len(move_lst), n_fft=n_fft)

wavfile.write(f"{data_dir}/../audio_{sys.argv[-1]}.wav", 16000, audio)
