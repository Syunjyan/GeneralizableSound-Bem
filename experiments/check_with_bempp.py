"""
Compare the boundary matrices computed by Bempp and the CUDA BEM solver.

Origin by Jinxutong
Edit by Houxinyun
"""

import sys

sys.path.append("./")
from src.bem.solver import MCBEM_Solver
from src.bem.vallinaBem import BEM_Solver
import bempp.api
import torch
import numpy as np

torch.set_printoptions(precision=5)

bempp.api.BOUNDARY_OPERATOR_DEVICE_TYPE = "gpu"
bempp.api.POTENTIAL_OPERATOR_DEVICE_TYPE = "gpu"
bempp.api.PLOT_BACKEND = "gmsh"
grid = bempp.api.shapes.regular_sphere(0)
grid.plot()
vertices = torch.from_numpy(grid.vertices.T.astype("float32")).cuda()
print(vertices)
triangles = torch.from_numpy(grid.elements.T.astype("int32")).cuda()
print(triangles)
space = bempp.api.function_space(grid, "P", 1)

check_points = torch.tensor(
    [
        [0.0, 0.0, 2.0],
        [0.0, 2.0, 0.0],
        [2.0, 0.0, 0.0],
    ]
)
check_points = check_points.float().cuda()
vallina_bem = BEM_Solver(vertices, triangles)
cuda_bem = MCBEM_Solver(vertices, triangles)

identity_matrix_bempp = vallina_bem.identity_matrix()
identity_matrix_cuda = cuda_bem.identity_matrix()

rerr = torch.norm(identity_matrix_bempp - identity_matrix_cuda) / torch.norm(
    identity_matrix_bempp
)
if rerr > 1e-4:
    print(f"Relative error identity: {rerr}")
    print("identity_matrix_cuda:")
    print(identity_matrix_cuda)
    print("identity_matrix_bempp:")
    print(identity_matrix_bempp)


for wave_number in [1, 10, 100]:
    print(f"Wave number: {wave_number}")
    beta = 1j / wave_number
    single_matrix_bempp = vallina_bem.single_layer_potential(wave_number)
    single_matrix_cuda = cuda_bem.assemble_boundary_matrix(wave_number, "single")
    rerr = torch.norm(single_matrix_bempp - single_matrix_cuda) / torch.norm(
        single_matrix_bempp
    )
    if rerr > 1e-4:
        print(f"Relative error single layer: {rerr}")
        print("single_matrix_cuda:")
        print(single_matrix_cuda)
        print("single_matrix_bempp:")
        print(single_matrix_bempp)

    double_matrix_bempp = vallina_bem.double_layer_potential(wave_number)
    double_matrix_cuda = cuda_bem.assemble_boundary_matrix(wave_number, "double")

    rerr = torch.norm(double_matrix_bempp - double_matrix_cuda) / torch.norm(
        double_matrix_bempp
    )
    if rerr > 1e-4:
        print(f"Relative error double layer: {rerr}")
        print("double_matrix_cuda:")
        print(double_matrix_cuda)
        print("double_matrix_bempp:")
        print(double_matrix_bempp)

    LHS_bempp = vallina_bem.assemble_boundary_matrix(wave_number, "bm_lhs")
    LHS_cuda = cuda_bem.assemble_boundary_matrix(wave_number, "bm_lhs")
    rerr = torch.norm(LHS_bempp - LHS_cuda) / torch.norm(LHS_bempp)
    if rerr > 1e-4:
        print(f"Relative error LHS: {rerr}")
        print("LHS_cuda:")
        print(LHS_cuda)
        print("LHS_bempp:")
        print(LHS_bempp)

    RHS_bempp = vallina_bem.assemble_boundary_matrix(wave_number, "bm_rhs")
    RHS_cuda = cuda_bem.assemble_boundary_matrix(wave_number, "bm_rhs")
    rerr = torch.norm(RHS_bempp - RHS_cuda) / torch.norm(RHS_bempp)
    if rerr > 1e-4:
        print(f"Relative error RHS: {rerr}")
        print("RHS_cuda:")
        print(RHS_cuda)
        print("RHS_bempp:")
        print(RHS_bempp)

    grid_fun = bempp.api.GridFunction(
        space, coefficients=np.random.rand(space.global_dof_count)
    )
    # in code (needed)
    double_potential_cuda = cuda_bem.double_temp(
        wave_number,
        check_points
    )
    dlp = bempp.api.operators.potential.helmholtz.double_layer(
        space,
        check_points.T.cpu().numpy(),
        wave_number,
        device_interface="opencl",
        precision="single",
    )
    double_potential_bempp = torch.from_numpy(dlp * grid_fun).cuda()
    rerr = torch.norm(double_potential_bempp - double_potential_cuda) / torch.norm(
        double_potential_cuda
    )

    if rerr > 1e-4:
        print(f"Relative error double potential: {rerr}")
        print("double_potential_cuda:")
        print(double_potential_cuda)
        print("double_potential_bempp:")
        print(double_potential_bempp)
