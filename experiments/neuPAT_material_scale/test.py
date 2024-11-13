import sys

sys.path.append("./")
from src.net.model import NeuPAT
import torch
from glob import glob
import torch
from tqdm import tqdm
import json
import os
from src.utils import Timer, Visualizer
import meshio

data_dir = "dataset/NeuPAT_new/scale/baseline"


def get_output_dir(size_k, freq_k):
    dir_name = f"{data_dir}/../test/{size_k:.2f}_{freq_k:.2f}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    return dir_name


def bem_process():
    timer = Timer()
    y = modal_sound.solve()
    cost_time = timer.get_time_cost()
    np.savez(
        f"{get_output_dir(sizeK_base, freqK_base)}/bem.npz",
        vertices=modal_sound.vertices.cpu().numpy(),
        triangles=modal_sound.triangles.cpu().numpy(),
        neumann=modal_sound.neumann_vtx.cpu().numpy(),
        wave_number=-modal_sound.ks.cpu().numpy(),
        ffat_map=y.numpy(),
        cost_time=cost_time,
        points=modal_sound.trg_points.cpu().numpy(),
    )
    return y


def calculate_ffat_map_neuPAT(x_in):
    x_in = torch.cat([x_in, (x_in[..., -1] * x_in[..., -2]).unsqueeze(-1)], dim=-1)
    timer = Timer()
    ffat_map = model(x_in).T
    cost_time = timer.get_time_cost()
    np.savez(
        f"{get_output_dir(sizeK_base, freqK_base)}/neuPAT.npz",
        ffat_map=ffat_map.cpu().numpy(),
        cost_time=cost_time,
    )
    return ffat_map


import json
import numpy as np
from src.scene import EditableModalSound

modal_sound = EditableModalSound(data_dir + "/../", uniform=True)

with open(f"{data_dir}/net.json", "r") as file:
    net_config = json.load(file)

model = NeuPAT(modal_sound.mode_num, net_config).cuda()
model.load_state_dict(torch.load(f"{data_dir}/model.pt"))
model.eval()
torch.set_grad_enabled(False)

first = True

for sizeK_base, freqK_base in [
    (0.5, 0.4),
    (0.5, 0.75),
    (0.5, 1.0),
    (0.7, 1.0),
    (1.0, 1.0),
]:
    x = modal_sound.sample(freqK_base=freqK_base, sizeK_base=sizeK_base).cuda()
    if first:
        y = bem_process()
        y_pred = calculate_ffat_map_neuPAT(x)
        first = False
    y = bem_process()
    y_pred = calculate_ffat_map_neuPAT(x)
    meshio.write_points_cells(
        os.path.join(f"{get_output_dir(sizeK_base, freqK_base)}/mesh_surf.obj"),
        modal_sound.vertices.cpu().numpy(),
        [("triangle", modal_sound.triangles.cpu().numpy())],
    )
    y = ((y + 10e-6) / 10e-6).log10().cuda()
    print("y", y.shape, "y_pred", y_pred.shape)
    print("loss", torch.nn.functional.mse_loss(y_pred, y))
