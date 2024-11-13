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
from src.utils import plot_point_cloud, plot_mesh, CombinedFig
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from time import time
import configparser
import meshio

data_dir = sys.argv[1]

import json

with open(f"{data_dir}/config.json", "r") as file:
    config_data = json.load(file)

vibration_objects = config_data.get("vibration_obj", [])
for obj in vibration_objects:
    mesh = obj.get("mesh")
    size = obj.get("size")
    position = obj.get("position")
    material = obj.get("material")
    mode_num = obj.get("mode_number")
    obj_vib = ModalSoundObj(f"{data_dir}/{mesh}")
    obj_vib.normalize(size)
    obj_vib.modal_analysis(k=mode_num, material=Material(getattr(MatSet, material)))
    print(obj_vib.get_frequencies())
    obj_vib.surf_vertices = obj_vib.surf_vertices + position

static_objects = config_data.get("static_obj", [])
for obj in static_objects:
    mesh = obj.get("mesh")
    size = obj.get("size")
    position = obj.get("position")
    obj_static = MeshObj(f"{data_dir}/{mesh}", scale=size)
    obj_static.vertices = obj_static.vertices + position

vertices = np.concatenate([obj_vib.surf_vertices, obj_static.vertices], axis=0)
static_triangles = obj_static.triangles + len(obj_vib.surf_vertices)
triangles = np.concatenate([obj_vib.surf_triangles, static_triangles], axis=0)
vib_triangles_mask = np.zeros(len(triangles), dtype=bool)
vib_triangles_mask[: len(obj_vib.surf_triangles)] = True

points = get_spherical_surface_points(vertices, 2)

start_time = time()

ffat_map_bem = np.zeros((mode_num, len(points)), dtype=np.complex64)
neumann = np.zeros((mode_num, len(triangles)), dtype=np.complex64)
dirichlet = np.zeros((mode_num, len(vertices)), dtype=np.complex64)
wave_number = []

for i in range(mode_num):
    k = -obj_vib.get_wave_number(i)
    neumann_coeff = np.zeros(len(triangles), dtype=np.complex64)
    neumann_coeff[vib_triangles_mask] = obj_vib.get_triangle_neumann(i)
    CombinedFig().add_mesh(
        vertices, triangles, np.abs(neumann_coeff), opacity=1.0
    ).show()
    bem = BEMModel(vertices, triangles, k)
    bem.boundary_equation_solve(neumann_coeff)
    dirichlet_coeff = bem.get_dirichlet_coeff()
    points_dirichlet = bem.potential_solve(points)
    ffat_map_bem[i] = points_dirichlet
    neumann[i] = neumann_coeff
    dirichlet[i] = dirichlet_coeff
    wave_number.append(obj_vib.get_wave_number(i))

cost_time = time() - start_time
out_path = os.path.join(data_dir, "bem.npz")

np.savez_compressed(
    out_path,
    points=points.cpu().numpy(),
    ffat_map=ffat_map_bem,
    neumann=neumann,
    dirichlet=dirichlet,
    wave_number=wave_number,
    vertices=vertices,
    triangles=triangles,
    cost_time=cost_time,
)
