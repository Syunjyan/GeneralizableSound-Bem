import sys

sys.path.append("./")

import torch
from src.mcs.mcs import ImportanceSampler, MonteCarloWeight
from src.timer import Timer
from src.modalobj.model import (
    solve_points_dirichlet,
    MultipoleModel,
    MeshObj,
    BEMModel,
    SNR,
)
import numpy as np
from src.utils import plot_mesh, plot_point_cloud, CombinedFig
from src.solver import BiCGSTAB, BiCGSTAB_batch, BiCGSTAB_batch2
import os
from glob import glob
from tqdm import tqdm
from numba import njit
import matplotlib.pyplot as plt
import configparser
import meshio

data_dir = "dataset/sample"
mesh = meshio.read(f"{data_dir}/mesh.obj")
vertices = torch.tensor(mesh.points).cuda().to(torch.float32)
triangles = torch.tensor(mesh.cells_dict["triangle"]).cuda().to(torch.int32)
importance = torch.ones(len(triangles), dtype=torch.float32).cuda()

sampler = ImportanceSampler(vertices, triangles, importance, 100000)
sampler.update()
sampler.poisson_disk_resample(0.008, 8)
print(sampler.points)
sampler2 = ImportanceSampler(vertices, triangles, importance, len(sampler.points))
sampler2.update()
np.savetxt(f"{data_dir}/points.txt", sampler.points.cpu().numpy())
np.savetxt(f"{data_dir}/points2.txt", sampler2.points.cpu().numpy())
