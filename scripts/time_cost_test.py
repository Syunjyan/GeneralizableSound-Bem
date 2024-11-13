"""
Compare the boundary matrices computed by Bempp and the CUDA BEM solver.

!!! Note:   Hypersingular operator is not tested here because the implementation 
            of the hypersingular operator in Bempp is different from the CUDA BEM solver.
"""

import sys

sys.path.append("./")
from src.bem.solver import BEM_Solver
import bempp.api
import torch
import numpy as np
from src.utils import Timer
import meshio

torch.set_printoptions(precision=5)
bempp.api.enable_console_logging("debug")
bempp.api.BOUNDARY_OPERATOR_DEVICE_TYPE = "gpu"
bempp.api.POTENTIAL_OPERATOR_DEVICE_TYPE = "gpu"
mesh = meshio.read("dataset/meshes/sphere.obj")
vertices = torch.from_numpy(mesh.points.astype("float32")).cuda()
triangles = torch.from_numpy(mesh.cells_dict["triangle"].astype("int32")).cuda()
print("vertices:", vertices.shape)
print("triangles:", triangles.shape)
cuda_bem = BEM_Solver(vertices, triangles)  # warm up


wave_number = 1
t = Timer()
cuda_bem = BEM_Solver(vertices, triangles)
t.log("preprocess")
cuda_bem.assemble_boundary_matrix(wave_number, "single")
t.log("single layer")
cuda_bem.assemble_boundary_matrix(wave_number, "double")
t.log("double layer")
cuda_bem.assemble_boundary_matrix(wave_number, "adjointdouble")
t.log("adjoint double layer")
cuda_bem.assemble_boundary_matrix(wave_number, "hypersingular")
t.log("hypersingular")
cuda_bem.assemble_boundary_matrix(wave_number, "bm_lhs")
t.log("lhs")
cuda_bem.assemble_boundary_matrix(wave_number, "bm_rhs")
t.log("rhs")
cuda_bem.assemble_boundary_matrix(wave_number, "single", approx=True)
t.log("single layer approx")
cuda_bem.assemble_boundary_matrix(wave_number, "double", approx=True)
t.log("double layer approx")
cuda_bem.assemble_boundary_matrix(wave_number, "adjointdouble", approx=True)
t.log("adjoint double layer approx")
cuda_bem.assemble_boundary_matrix(wave_number, "hypersingular", approx=True)
t.log("hypersingular approx")
cuda_bem.assemble_boundary_matrix(wave_number, "bm_lhs", approx=True)
t.log("lhs approx")
cuda_bem.assemble_boundary_matrix(wave_number, "bm_rhs", approx=True)
t.log("rhs approx")


t = Timer()
grid = bempp.api.Grid(vertices.cpu().numpy().T, triangles.cpu().numpy().T)
space = bempp.api.function_space(grid, "P", 1)
slp = bempp.api.operators.boundary.helmholtz.single_layer(
    space,
    space,
    space,
    wave_number,
    device_interface="opencl",
    precision="single",
)
A = slp.weak_form().A
dlp = bempp.api.operators.boundary.helmholtz.double_layer(
    space,
    space,
    space,
    wave_number,
    device_interface="opencl",
    precision="single",
)
B = dlp.weak_form().A
adlp = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(
    space,
    space,
    space,
    wave_number,
    device_interface="opencl",
    precision="single",
)
C = adlp.weak_form().A
hyp = bempp.api.operators.boundary.helmholtz.hypersingular(
    space,
    space,
    space,
    wave_number,
    device_interface="opencl",
    precision="single",
)
D = hyp.weak_form().A
t.log("time cost for bempp")
