import sys

sys.path.append("./")

from src.modalobj.model import (
    ModalSoundObj,
    MatSet,
    Material,
    BEMModel,
    MeshObj,
    get_spherical_surface_points,
)
from src.mcs.mcs import ImportanceSampler, MonteCarloWeight
from src.utils import plot_point_cloud, plot_mesh, CombinedFig
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from time import time
import configparser
import meshio
import torch

data_dir = sys.argv[1]

import json

with open(f"{data_dir}/config.json", "r") as file:
    config_data = json.load(file)

vibration_objects = config_data.get("vibration_obj", [])
vib_obj = vibration_objects[0]
vib_mesh = vib_obj.get("mesh")
vib_size = vib_obj.get("size")
vib_position = vib_obj.get("position")
vib_mesh = MeshObj(f"{data_dir}/{vib_mesh}", scale=vib_size)
vib_mesh.vertices = vib_mesh.vertices + vib_position

static_objects = config_data.get("static_obj", [])
if len(static_objects) > 0:
    static_obj = static_objects[0]
    static_mesh = static_obj.get("mesh")
    static_size = static_obj.get("size")
    static_position = static_obj.get("position")
    static_mesh = MeshObj(f"{data_dir}/{static_mesh}", scale=static_size)
    static_mesh.vertices = static_mesh.vertices + static_position

rot_objects = config_data.get("rot_obj", [])
rot_obj = rot_objects[0]
rot_mesh = rot_obj.get("mesh")
rot_size = rot_obj.get("size")
rot_position = rot_obj.get("position")
rot_mesh = MeshObj(f"{data_dir}/{rot_mesh}", scale=rot_size)
rot_mesh.vertices = rot_mesh.vertices + rot_position

if len(static_objects) > 0:
    vertices_static = torch.from_numpy(static_mesh.vertices).cuda().to(torch.float32)
    triangles_static = torch.from_numpy(static_mesh.triangles).cuda().to(torch.int32)
    vertices_vib = torch.from_numpy(vib_mesh.vertices).cuda().to(torch.float32)
    triangles_vib = torch.from_numpy(vib_mesh.triangles).cuda().to(torch.int32)
    vertices_static = torch.cat([vertices_vib, vertices_static], dim=0)
    triangles_static = torch.cat(
        [triangles_vib, triangles_static + len(vertices_vib)], dim=0
    )
    neumann_static = torch.zeros(len(triangles_static), dtype=torch.complex64).cuda()
    neumann_static[: len(triangles_vib)] = 1
else:
    vertices_static = torch.from_numpy(vib_mesh.vertices).cuda().to(torch.float32)
    triangles_static = torch.from_numpy(vib_mesh.triangles).cuda().to(torch.int32)
    neumann_static = torch.ones(len(triangles_static), dtype=torch.complex64).cuda()

vertices_rot = torch.from_numpy(rot_mesh.vertices).cuda().to(torch.float32)
triangles_rot = torch.from_numpy(rot_mesh.triangles).cuda().to(torch.int32)

neumann_rot = torch.zeros(len(triangles_rot), dtype=torch.complex64).cuda()
neumann_tri = torch.cat([neumann_static, neumann_rot], dim=0)

vertices = torch.cat([vertices_static, vertices_rot], dim=0)
triangles = torch.cat([triangles_static, triangles_rot + len(vertices_static)], dim=0)

print(vertices.shape, triangles.shape, neumann_tri.shape)
CombinedFig().add_mesh(vertices, triangles, neumann_tri.real, opacity=1.0).show()

torch.save(
    {
        "vertices_static": vertices_static,
        "triangles_static": triangles_static,
        "vertices_rot": vertices_rot,
        "triangles_rot": triangles_rot,
        "neumann_tri": neumann_tri,
    },
    f"{data_dir}/data.pt",
)
