import sys

sys.path.append("./")

from src.modalobj.model import ModalSoundObj, MatSet, Material, BEMModel
from src.utils import plot_point_cloud, plot_mesh
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from time import time
import configparser
import meshio

root_dir = "dataset/modal_vibration"
obj_list = glob(os.path.join(root_dir, "*"))

mode_num = 64
for obj_dir in tqdm(obj_list):
    if not os.path.isdir(obj_dir):
        continue
    mesh_path = os.path.join(obj_dir, "mesh.obj")
    config = configparser.ConfigParser()
    config.read(f"{obj_dir}/config.ini")
    obj = ModalSoundObj(mesh_path)
    obj.normalize(config.getfloat("mesh", "size"))
    print("obj:", obj_dir)
    print(len(obj.surf_triangles))
    continue
    material = Material(getattr(MatSet, config.get("mesh", "material")))
    print("vertices num:", obj.vertices.shape[0])
    obj.modal_analysis(k=mode_num, material=material)
    if "armadillo" in obj_dir or "bunny" in obj_dir or "dragon" in obj_dir:
        obj.eigenvalues /= 10
    print("frequencies:", obj.get_frequencies())

    np.savez_compressed(
        os.path.join(obj_dir, "modes"),
        modes=obj.modes,
        eigenvalues=obj.eigenvalues,
    )
    points = obj.spherical_surface_points(4)
    start_time = time()

    ffat_map_bem = np.zeros((mode_num, len(points)), dtype=np.complex64)
    neumann = np.zeros((mode_num, len(obj.surf_triangles)), dtype=np.complex64)
    dirichlet = np.zeros((mode_num, len(obj.surf_vertices)), dtype=np.complex64)
    wave_number = []

    for i in range(mode_num):
        k = -obj.get_wave_number(i)
        neumann_coeff = obj.get_triangle_neumann(i)
        bem = BEMModel(obj.surf_vertices, obj.surf_triangles, k)
        bem.boundary_equation_solve(neumann_coeff)
        dirichlet_coeff = bem.get_dirichlet_coeff()
        points_dirichlet = bem.potential_solve(points)

        ffat_map_bem[i] = points_dirichlet
        neumann[i] = neumann_coeff
        dirichlet[i] = dirichlet_coeff
        wave_number.append(obj.get_wave_number(i))

    cost_time = time() - start_time
    out_path = os.path.join(obj_dir, "bem.npz")

    meshio.write_points_cells(
        os.path.join(obj_dir, "mesh_surf.obj"),
        obj.surf_vertices,
        [("triangle", obj.surf_triangles)],
    )

    np.savez_compressed(
        out_path,
        modes=obj.modes,
        points=points.cpu().numpy(),
        ffat_map=ffat_map_bem,
        neumann=neumann,
        dirichlet=dirichlet,
        wave_number=wave_number,
        vertices=obj.surf_vertices,
        triangles=obj.surf_triangles,
        cost_time=cost_time,
    )
