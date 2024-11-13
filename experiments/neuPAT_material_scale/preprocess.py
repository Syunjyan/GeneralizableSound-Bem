import sys

sys.path.append("./")

from src.modalobj.model import VibrationObj, MatSet, Material
from src.bem.solver import map_triangle2vertex
from src.utils import Visualizer
import numpy as np
import torch

data_dir = "dataset/NeuPAT_new/scale"

import json

with open(f"{data_dir}/config.json", "r") as file:
    config_data = json.load(file)

obj = config_data["vibration_obj"]
mesh = obj.get("mesh")
size = obj.get("size")
material = obj.get("material")
mode_num = obj.get("mode_num")
obj_vib = VibrationObj(f"{data_dir}/{mesh}")
obj_vib.normalize(size)
obj_vib.modal_analysis(k=mode_num, material=Material(getattr(MatSet, material)))

vertices_vib = torch.from_numpy(obj_vib.surf_vertices).cuda().to(torch.float32)
triangles_vib = torch.from_numpy(obj_vib.surf_triangles).cuda().to(torch.int32)

neumann_vtx = torch.zeros(mode_num, len(vertices_vib), dtype=torch.complex64).cuda()
wave_number = []
for i in range(mode_num):
    neumann_tri = (
        torch.from_numpy(obj_vib.get_triangle_neumann(i)).cuda().to(torch.complex64)
    )
    neumann_vtx[i] = map_triangle2vertex(vertices_vib, triangles_vib, neumann_tri)
    wave_number.append(-obj_vib.get_wave_number(i))

import meshio
import os

meshio.write_points_cells(
    os.path.join(data_dir, "mesh_surf.obj"),
    vertices_vib.cpu().numpy(),
    [("triangle", triangles_vib.cpu().numpy())],
)

ks = torch.tensor(wave_number).to(torch.float32).cuda()
print(len(ks))
torch.save(
    {
        "vertices": vertices_vib,
        "triangles": triangles_vib,
        "neumann_vtx": neumann_vtx,
        "ks": ks,
        "eigenvalues": obj_vib.eigenvalues,
        "modes": obj_vib.modes,
    },
    f"{data_dir}/modal_data.pt",
)
