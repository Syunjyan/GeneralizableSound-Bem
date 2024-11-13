import sys

sys.path.append("./")

from src.modalobj.model import SoundObj, MatSet, Material
from src.utils import plot_point_cloud, plot_mesh
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from time import time

root_dir = "dataset/ABC"

mesh_dir = os.path.join(root_dir, "mesh")
eigen_dir = os.path.join(root_dir, "data/eigen")

mesh_list = glob(os.path.join(mesh_dir, "*.obj"))
mode_num = 16

for mesh_path in tqdm(mesh_list):
    out_path = os.path.join(
        eigen_dir, os.path.basename(mesh_path).replace(".sf.obj", ".npz")
    )
    if os.path.exists(out_path):
        continue
    obj = SoundObj(mesh_path)
    if len(obj.vertices) > 3000:
        continue
    obj.normalize(0.15)
    material = Material(MatSet.Plastic)
    obj.modal_analysis(k=mode_num, material=material)

    start_time = time()
    neumann = []
    dirichlet = []
    wave_number = []
    residuals = []
    for i in range(mode_num):
        neumann.append(obj.get_triangle_neumann(i))
        wave_number.append(obj.get_wave_number(i))
        bm, residual = obj.solve_by_BEM(i)
        residuals.append(residual)
        dirichlet.append(bm.get_dirichlet_coeff())
    cost_time = time() - start_time
    neumann = np.array(neumann).T
    dirichlet = np.array(dirichlet).T
    np.savez_compressed(
        out_path,
        neumann=neumann,
        dirichlet=dirichlet,
        wave_number=wave_number,
        modes=obj.modes,
        vertices=obj.surf_vertices,
        triangles=obj.surf_triangles,
        bem_cost_time=cost_time,
        bem_residual=residuals,
    )
