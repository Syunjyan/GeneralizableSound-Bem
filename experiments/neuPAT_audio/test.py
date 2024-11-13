import sys

sys.path.append("./")
from src.net.model import NeuPAT
import torch
from glob import glob
import torch
from tqdm import tqdm
import json
from src.utils import Timer
import os
from src.modalobj.model import get_spherical_surface_points, StaticObj
from src.utils import Visualizer, calculate_bin_frequencies
from src.bem.solver import BEM_Solver


def get_output_dir(freq, src_pos_y):
    dir_name = f"{data_dir}/test/{src_pos_y:.1f}_{freq:.0f}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def bem_process(vertices, triangles, k, neumann, trg_points):
    timer = Timer()
    solver = BEM_Solver(vertices, triangles)
    dirichlet = solver.neumann2dirichlet(k, neumann)
    potential = solver.boundary2potential(k, neumann, dirichlet, trg_points)
    cost_time = timer.get_time_cost()
    np.savez(
        f"{get_output_dir(freq, src_pos_y)}/bem.npz",
        vertices=vertices.cpu().numpy(),
        triangles=triangles.cpu().numpy(),
        neumann=neumann.cpu().numpy(),
        wave_number=[-k],
        ffat_map=potential.abs().cpu().numpy(),
        cost_time=cost_time,
        points=trg_points.cpu().numpy(),
    )


def neuPAT_process(src_base, trg_base, freq_base):
    x = torch.zeros(len(trg_points), 5, dtype=torch.float32, device="cuda")
    x[:, :3] = trg_base
    x[:, 3] = src_base[1]
    x[:, 4] = freq_base
    timer = Timer()
    potential = model(x).T
    cost_time = timer.get_time_cost()
    np.savez(
        f"{get_output_dir(freq, src_pos_y)}/neuPAT.npz",
        ffat_map=potential.cpu().numpy(),
        cost_time=cost_time,
    )


data_dir = "dataset/NeuPAT_new/audio"

import json
import numpy as np

with open(f"{data_dir}/config.json", "r") as file:
    config_data = json.load(file)

vibration_objects = config_data.get("vibration_obj", [])
for obj in vibration_objects:
    mesh = obj.get("mesh")
    size = obj.get("size")
    obj_vib = StaticObj(f"{data_dir}/{mesh}", scale=size)

static_objects = config_data.get("static_obj", [])
for obj in static_objects:
    mesh = obj.get("mesh")
    size = obj.get("size")
    obj_static = StaticObj(f"{data_dir}/{mesh}", scale=size)

vertices_vib = torch.from_numpy(obj_vib.vertices).cuda().to(torch.float32)
triangles_vib = torch.from_numpy(obj_vib.triangles).cuda().to(torch.int32)
neumann_vib = torch.ones(len(vertices_vib), 1, dtype=torch.complex64).cuda()

vertices_static = torch.from_numpy(obj_static.vertices).cuda().to(torch.float32)
triangles_static = torch.from_numpy(obj_static.triangles).cuda().to(torch.int32)
neumann_static = torch.zeros(len(vertices_static), 1, dtype=torch.complex64).cuda()

neumann_vib[vertices_vib[:, 1] > -0.04] = 0

neumann = torch.cat([neumann_vib, neumann_static], dim=0).squeeze(-1)
triangles = torch.cat([triangles_vib, triangles_static + len(vertices_vib)], dim=0)

src_pos_min = torch.tensor(
    config_data.get("solver", {}).get("src_pos_min"), device="cuda", dtype=torch.float32
)
src_pos_max = torch.tensor(
    config_data.get("solver", {}).get("src_pos_max"), device="cuda", dtype=torch.float32
)
trg_pos_min = torch.tensor(
    config_data.get("solver", {}).get("trg_pos_min"), device="cuda", dtype=torch.float32
)
trg_pos_max = torch.tensor(
    config_data.get("solver", {}).get("trg_pos_max"), device="cuda", dtype=torch.float32
)
print("src_pos_min:", src_pos_min)
print("src_pos_max:", src_pos_max)
print("trg_pos_min:", trg_pos_min)
print("trg_pos_max:", trg_pos_max)
freq_min = config_data.get("solver", {}).get("freq_min", 100)
freq_max = config_data.get("solver", {}).get("freq_max", 10000)
freq_min_log = np.log10(freq_min)
freq_max_log = np.log10(freq_max)
print("freq_min:", freq_min)
print("freq_max:", freq_max)
trg_sample_num = config_data.get("solver", {}).get("trg_sample_num", 1000)
src_sample_num = config_data.get("solver", {}).get("src_sample_num", 1000)
print("trg_sample_num:", trg_sample_num)
print("src_sample_num:", src_sample_num)

net_dir = sys.argv[1]
with open(f"{net_dir}/net.json", "r") as file:
    train_config_data = json.load(file)

train_params = train_config_data.get("train", {})
model = NeuPAT(
    1,
    train_config_data.get("encoding_config"),
    train_config_data.get("network_config"),
).cuda()
model.load_state_dict(torch.load(f"{net_dir}/model.pt"))
model.eval()
torch.set_grad_enabled(False)

first = True
freq_bins = calculate_bin_frequencies()
print("freq_bins:", freq_bins)
for src_pos_y in range(0, 2):
    src_pos_y = src_pos_y / 10
    print("src_pos_y:", src_pos_y)
    for freq_bin in freq_bins:
        if freq_bin < freq_min or freq_bin > freq_max:
            continue
        freq_base = (np.log10(freq_bin) - freq_min_log) / (freq_max_log - freq_min_log)
        src_base = torch.zeros(3, device="cuda", dtype=torch.float32)
        src_base[1] = src_pos_y
        displacement = src_base * (src_pos_max - src_pos_min) + src_pos_min
        vertices_vib_updated = vertices_vib + displacement
        vertices = torch.cat([vertices_vib_updated, vertices_static], dim=0)
        trg_points = get_spherical_surface_points(vertices_static, 1.5)
        trg_base = (trg_points - trg_pos_min) / (trg_pos_max - trg_pos_min)

        freq_log = freq_base * (freq_max_log - freq_min_log) + freq_min_log
        freq = 10**freq_log
        k = (2 * np.pi * freq / 343.2).item()
        if first:
            bem_process(vertices, triangles, k, neumann, trg_points)
            neuPAT_process(src_base, trg_base, freq_base)
            first = False

        bem_process(vertices, triangles, k, neumann, trg_points)
        neuPAT_process(src_base, trg_base, freq_base)
