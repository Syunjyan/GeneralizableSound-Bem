import bempp.api
import numpy as np
import sys
import torch

sys.path.append("./")
from src.bem.solver import BEM_Solver
from src.utils import Timer

bempp.api.enable_console_logging("debug")
bempp.api.BOUNDARY_OPERATOR_DEVICE_TYPE = "gpu"
bempp.api.POTENTIAL_OPERATOR_DEVICE_TYPE = "gpu"

x0 = np.array([0.0, 0.0, 0.0])
CBIE_bempp = []
HBIE_bempp = []
CBIE_cuda = []
HBIE_cuda = []
HBIE_cuda_approx = []

for idx in range(20):
    wave_number = 100.6 + 0.01 * idx
    # wave_number = 5 + 5 * idx
    k = -wave_number

    @bempp.api.complex_callable
    def get_dirichlet(x, n, domain_index, result):
        r_vec = x - x0
        r = np.dot(r_vec, r_vec) ** 0.5
        result[0] = np.exp(1j * k * r) / (4 * np.pi * r)

    @bempp.api.complex_callable
    def get_neumann(x, n, domain_index, result):
        r_vec = x - x0
        r = np.dot(r_vec, r_vec) ** 0.5
        ikr = 1j * k * r
        result[0] = (
            -np.exp(ikr) / (4 * np.pi * r * r * r) * (1 - ikr) * ((x - x0) * n).sum()
        )

    wavelength = 2 * np.pi / abs(k)
    h = wavelength / 6
    grid = bempp.api.shapes.ellipsoid(0.15, 0.05, 0.05, h=h * 4)
    space = bempp.api.function_space(grid, "P", 1)

    identity = bempp.api.operators.boundary.sparse.identity(
        space, space, space, device_interface="opencl", precision="single"
    )
    slp = bempp.api.operators.boundary.helmholtz.single_layer(
        space, space, space, k, device_interface="opencl", precision="single"
    )
    dlp = bempp.api.operators.boundary.helmholtz.double_layer(
        space, space, space, k, device_interface="opencl", precision="single"
    )
    hyp = bempp.api.operators.boundary.helmholtz.hypersingular(
        space, space, space, k, device_interface="opencl", precision="single"
    )
    adlp = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(
        space, space, space, k, device_interface="opencl", precision="single"
    )

    neumann_fun = bempp.api.GridFunction(space, fun=get_neumann)
    dirichlet_fun_gt = bempp.api.GridFunction(space, fun=get_dirichlet)
    neumann_coeff = (
        torch.from_numpy(neumann_fun.coefficients).cuda().to(torch.complex64)
    )
    dirichlet_coeff = (
        torch.from_numpy(dirichlet_fun_gt.coefficients).cuda().to(torch.complex64)
    )

    t = Timer()
    t_asm = Timer()
    LHS = 0.5 * identity - dlp
    RHS = -slp
    RHS = RHS * neumann_fun
    bempp_tc1 = t_asm.get_time_cost()
    dirichlet_fun, info = bempp.api.linalg.gmres(
        LHS, RHS, tol=1e-5, use_strong_form=True, maxiter=1000
    )
    tc = t.get_time_cost()

    cuda_bem = BEM_Solver(grid.vertices.T, grid.elements.T.astype("int32"))

    rerr = np.linalg.norm(
        dirichlet_fun.coefficients - dirichlet_fun_gt.coefficients
    ) / np.linalg.norm(dirichlet_fun_gt.coefficients)

    CBIE_bempp.append([wave_number, rerr, tc])

    t = Timer()
    dirichlet_coeff_cuda = cuda_bem.CBIE(k, neumann_coeff)
    tc = t.get_time_cost()
    rerr = torch.norm(dirichlet_coeff_cuda - dirichlet_coeff) / torch.norm(
        dirichlet_coeff
    )
    CBIE_cuda.append([wave_number, rerr.item(), tc])

    t = Timer()
    beta = 1j / k
    LHS = 0.5 * identity - dlp + beta * hyp
    RHS = (-slp - beta * (adlp + 0.5 * identity)) * neumann_fun

    dirichlet_fun, info = bempp.api.linalg.gmres(
        LHS, RHS, tol=1e-5, use_strong_form=True, maxiter=1000
    )
    tc = t.get_time_cost()
    rerr = np.linalg.norm(
        dirichlet_fun.coefficients - dirichlet_fun_gt.coefficients
    ) / np.linalg.norm(dirichlet_fun_gt.coefficients)

    HBIE_bempp.append([wave_number, rerr, tc + bempp_tc1])

    t = Timer()
    dirichlet_coeff_cuda = cuda_bem.HBIE(k, neumann_coeff)
    tc = t.get_time_cost()
    rerr = torch.norm(dirichlet_coeff_cuda - dirichlet_coeff) / torch.norm(
        dirichlet_coeff
    )

    HBIE_cuda.append([wave_number, rerr.item(), tc])

    t = Timer()
    dirichlet_coeff_cuda = cuda_bem.neumann2dirichlet(k, neumann_coeff)
    tc = t.get_time_cost()
    rerr = torch.norm(dirichlet_coeff_cuda - dirichlet_coeff) / torch.norm(
        dirichlet_coeff
    )
    HBIE_cuda_approx.append([wave_number, rerr.item(), tc])

    np.savetxt("output/CBIE_bempp.txt", CBIE_bempp)
    np.savetxt("output/HBIE_bempp.txt", HBIE_bempp)
    np.savetxt("output/CBIE_cuda.txt", CBIE_cuda)
    np.savetxt("output/HBIE_cuda.txt", HBIE_cuda)
    np.savetxt("output/HBIE_cuda_approx.txt", HBIE_cuda_approx)
