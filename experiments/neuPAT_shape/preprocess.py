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
import mcubes
import open3d as o3d


def get_sdf(mesh, resolution=32):
    vertices = mesh.vertices.astype(np.float32)
    triangles = mesh.triangles.astype(np.uint32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(vertices, triangles)
    min_bound = vertices.min(0)
    max_bound = vertices.max(0)
    center = (min_bound + max_bound) / 2
    size = (max_bound - min_bound).max()
    size = size * 1.2
    min_bound = center - size / 2
    max_bound = center + size / 2
    xyz_range = np.linspace(min_bound, max_bound, num=resolution)
    query_points = np.stack(np.meshgrid(*xyz_range.T), axis=-1).astype(np.float32)
    signed_distance = scene.compute_signed_distance(query_points)
    return (
        signed_distance.numpy().reshape(resolution, resolution, resolution),
        query_points,
    )


data_dir = "dataset/NeuPAT/shape"

import json

with open(f"{data_dir}/config.json", "r") as file:
    config_data = json.load(file)
vibration_objects = config_data.get("vibration_obj", [])
vib_obj = vibration_objects[0]
mesh = vib_obj.get("mesh")
size = vib_obj.get("size")
vib_obj = MeshObj(f"{data_dir}/{mesh}", scale=size)

start_objects = config_data.get("start_obj", [])
start_obj = start_objects[0]
mesh = start_obj.get("mesh")
size_start = start_obj.get("size")
start_obj = MeshObj(f"{data_dir}/{mesh}", scale=size_start)
sdf_start, query_points = get_sdf(start_obj)

end_objects = config_data.get("end_obj", [])
end_obj = end_objects[0]
mesh = end_obj.get("mesh")
size_end = end_obj.get("size")
end_obj = MeshObj(f"{data_dir}/{mesh}", scale=size_end)
sdf_end, query_points = get_sdf(end_obj)

step_num = 100
vertices = []
triangles = []
vib_vertices = vib_obj.vertices
vib_vertices[:, 0] -= 0.03
vib_triangles = vib_obj.triangles
for t in range(100):
    t = t / 100
    sdf = sdf_start * (1 - t) + sdf_end * t
    sdf[sdf < 0] *= -1
    mask = np.logical_and(
        query_points[:, :, :, 2] > 0,
        (
            query_points[:, :, :, 0] ** 2 + query_points[:, :, :, 1] ** 2
            < (size_start * 0.08) ** 2
        ),
    )
    sdf[mask] = 0.01
    vertices_single, triangles_single = mcubes.marching_cubes(sdf, 0.006)
    bbox_min = vertices_single.min(axis=0)
    bbox_max = vertices_single.max(axis=0)
    vertices_single = (
        (vertices_single - (bbox_max + bbox_min) / 2)
        / (bbox_max - bbox_min).max()
        * size_start
    )
    print(vertices_single.shape)
    meshio.write_points_cells(
        f"{data_dir}/meshes/{t}.obj",
        vertices_single,
        [("triangle", triangles_single)],
        file_format="obj",
    )
    vertices.append(torch.from_numpy(vertices_single).cuda())
    triangles.append(torch.from_numpy(triangles_single.astype(np.int32)).cuda())

    vs = np.concatenate([vertices_single, vib_vertices], axis=0)
    ts = np.concatenate(
        [triangles_single, vib_triangles + len(vertices_single)], axis=0
    )

torch.save(
    {
        "vertices_lst": vertices,
        "triangles_lst": triangles,
        "vib_vertices": torch.from_numpy(vib_vertices).cuda(),
        "vib_triangles": torch.from_numpy(vib_triangles).cuda(),
    },
    f"{data_dir}/data.pt",
)
