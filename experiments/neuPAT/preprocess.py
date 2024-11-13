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

data_dir = "dataset/NeuPAT/bowl"

import json

with open(f"{data_dir}/config.json", "r") as file:
    config_data = json.load(file)


def precompute_sample_points():
    vibration_objects = config_data.get("vibration_obj", [])
    for obj in vibration_objects:
        mesh = obj.get("mesh")
        size = obj.get("size")
        material = obj.get("material")
        mode_num = obj.get("mode_num")
        obj_vib = ModalSoundObj(f"{data_dir}/{mesh}")
        obj_vib.normalize(size)
        obj_vib.modal_analysis(k=mode_num, material=Material(getattr(MatSet, material)))
        print(obj_vib.get_frequencies())

    static_objects = config_data.get("static_obj", [])
    for obj in static_objects:
        mesh = obj.get("mesh")
        size = obj.get("size")
        position = obj.get("position")
        obj_static = MeshObj(f"{data_dir}/{mesh}", scale=size)
        mesh_bem = mesh.replace(".obj", "_bem.obj")
        obj_static_bem = MeshObj(f"{data_dir}/{mesh_bem}", scale=size)
        obj_static.vertices = obj_static.vertices + position
        obj_static_bem.vertices = obj_static_bem.vertices + position

    vertices_vib = torch.from_numpy(obj_vib.surf_vertices).cuda().to(torch.float32)
    triangles_vib = torch.from_numpy(obj_vib.surf_triangles).cuda().to(torch.int32)

    neumann_vib_tri = np.zeros((mode_num, len(triangles_vib)), dtype=np.complex64)
    wave_number = []
    for i in range(mode_num):
        neumann_vib_tri[i] = obj_vib.get_triangle_neumann(i)
        wave_number.append(obj_vib.get_wave_number(i))

    neumann_vib = torch.from_numpy(neumann_vib_tri).cuda()

    vertices_static = torch.from_numpy(obj_static.vertices).cuda().to(torch.float32)
    triangles_static = torch.from_numpy(obj_static.triangles).cuda().to(torch.int32)

    neumann_static = torch.zeros(
        mode_num, len(triangles_static), dtype=torch.complex64
    ).cuda()

    vertices_bem = torch.from_numpy(obj_static_bem.vertices).cuda().to(torch.float32)
    triangles_bem = torch.from_numpy(obj_static_bem.triangles).cuda().to(torch.int32)

    neumann_static_bem = torch.zeros(
        mode_num, len(triangles_bem), dtype=torch.complex64
    ).cuda()
    ks = torch.from_numpy(-np.array(wave_number)).to(torch.float32).cuda()

    neumann_tri = torch.cat([neumann_vib, neumann_static], dim=1).cuda()
    vertices = torch.cat([vertices_vib, vertices_static], dim=0).cuda()
    triangles = torch.cat(
        [triangles_vib, triangles_static + len(vertices_vib)], dim=0
    ).cuda()
    CombinedFig().add_mesh(vertices, triangles, neumann_tri[0].real, opacity=1.0).show()

    neumann_tri_bem = torch.cat([neumann_vib, neumann_static_bem], dim=1).cuda()
    vertices = torch.cat([vertices_vib, vertices_bem], dim=0).cuda()
    triangles = torch.cat(
        [triangles_vib, triangles_bem + len(vertices_vib)], dim=0
    ).cuda()
    CombinedFig().add_mesh(
        vertices, triangles, neumann_tri_bem[0].real, opacity=1.0
    ).show()

    torch.save(
        {
            "vertices_vib": vertices_vib,
            "triangles_vib": triangles_vib,
            "vertices_static": vertices_static,
            "triangles_static": triangles_static,
            "neumann_tri": neumann_tri,
            "ks": ks,
            "vertices_bem": vertices_bem,
            "triangles_bem": triangles_bem,
            "neumann_tri_bem": neumann_tri_bem,
        },
        f"{data_dir}/modal_data.pt",
    )


precompute_sample_points()
