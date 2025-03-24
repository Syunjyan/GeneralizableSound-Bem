from torch.utils.cpp_extension import load as load_cuda
import os
from glob import glob
import torch
from .BiCGSTAB import BiCGSTAB
import numpy as np


def load_cuda_imp(Debug=False, Verbose=False):
    src_dir = os.path.dirname(os.path.abspath(__file__)) + "/cuda"
    os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(src_dir, "build")
    cflags = ["--extended-lambda", "--expt-relaxed-constexpr"]
    if Debug:
        cflags += ["-G", "-g", "-O0"]
        cflags += ["-DDEBUG"]
    else:
        cflags += ["-O3"]
        cflags += ["-DNDEBUG"]

    cuda_files = glob(src_dir + "/*.cu")
    include_paths = [src_dir + "/include"]
    return load_cuda(
        name="CUDA_MODULE",
        sources=cuda_files,
        extra_include_paths=include_paths,
        extra_cuda_cflags=cflags,
        verbose=Verbose,
    )


cuda_imp = load_cuda_imp(Debug=False, Verbose=False)


def check_tensor(tensor, dtype):
    assert tensor.dtype == dtype
    assert tensor.is_cuda
    assert tensor.is_contiguous()


def preprocess(vertices, triangles):
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    reference_gradient = torch.tensor(
        [[-1, 1, 0], [-1, 0, 1]], dtype=torch.float32, device=vertices.device
    )
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    normals = normals / torch.norm(normals, dim=1, keepdim=True)
    jacobian = torch.stack([v1 - v0, v2 - v0], dim=2)
    jac_transpose_jac = torch.bmm(jacobian.transpose(1, 2), jacobian)
    jac_transpose_jac_inv = torch.inverse(jac_transpose_jac)
    jac_inv_transpose = torch.bmm(jacobian, jac_transpose_jac_inv)
    surface_gradients = torch.matmul(jac_inv_transpose, reference_gradient)
    surface_gradients_transpose = surface_gradients.transpose(1, 2)
    surface_curls_trans = torch.linalg.cross(
        normals.unsqueeze(1), surface_gradients_transpose
    )
    return normals, surface_curls_trans


def solve_linear_equation(A_func, b, x=None, nsteps=500, tol=1e-10, atol=1e-16):
    if callable(A_func):
        solver = BiCGSTAB(A_func)
    else:
        solver = BiCGSTAB(lambda x: A_func @ x)
    return solver.solve(b, x=x, nsteps=nsteps, tol=tol, atol=atol)


class MCBEM_Solver:
    def __init__(self, vertices, triangles):
        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).cuda().float()
        if isinstance(triangles, np.ndarray):
            triangles = torch.from_numpy(triangles).cuda().int()

        check_tensor(vertices, torch.float32)
        check_tensor(triangles, torch.int32)
        self.vertices = vertices
        self.triangles = triangles
        self.device = vertices.device
        self.normals, self.surface_curls_trans = preprocess(vertices, triangles)

    def assemble_boundary_matrix(self, wavenumber, layer_type, approx=False):
        return getattr(
            cuda_imp, layer_type + "_boundary_matrix" + ("_approx" if approx else "")
        )(
            self.vertices,
            self.triangles,
            self.normals,
            self.surface_curls_trans,
            wavenumber,
        )

    def identity_matrix(self):
        return cuda_imp.identity_matrix(self.vertices, self.triangles)

    def CBIE(self, k, neumann):
        identity = self.identity_matrix()
        slp = self.assemble_boundary_matrix(k, "single")
        dlp = self.assemble_boundary_matrix(k, "double")

        LHS = 0.5 * identity - dlp
        RHS = -slp @ neumann
        return solve_linear_equation(LHS, RHS)

    def HBIE(self, k, neumann):
        identity = self.identity_matrix()
        slp = self.assemble_boundary_matrix(k, "single")
        dlp = self.assemble_boundary_matrix(k, "double")
        adlp = self.assemble_boundary_matrix(k, "adjointdouble")
        hyp = self.assemble_boundary_matrix(k, "hypersingular")

        beta = 1j / k
        LHS = 0.5 * identity - dlp + beta * hyp
        RHS = (-slp - beta * (adlp + 0.5 * identity)) @ neumann
        return solve_linear_equation(LHS, RHS)

    def neumann2dirichlet(self, k, neumann):
        identity = self.identity_matrix()
        beta = 1j / k
        LHS = self.assemble_boundary_matrix(k, "bm_lhs", approx=True) + 0.5 * identity
        RHS = (
            self.assemble_boundary_matrix(k, "bm_rhs", approx=True)
            - beta * 0.5 * identity
        ) @ neumann
        return solve_linear_equation(LHS, RHS)

    def single_potential(self, k, neumann, points):
        return (
            cuda_imp.single_boundary_potential(self.vertices, self.triangles, points, k)
            @ neumann
        )

    def double_potential(self, k, dirichlet, points):
        return (
            cuda_imp.double_boundary_potential(self.vertices, self.triangles, points, k)
            @ dirichlet
        )

    def double_temp(self, k, points):
        return cuda_imp.double_boundary_potential(
            self.vertices, self.triangles, points, k)
            

    def boundary2potential(self, k, neumann, dirichlet, points):
        slp = cuda_imp.single_boundary_potential(
            self.vertices, self.triangles, points, k
        )
        dlp = cuda_imp.double_boundary_potential(
            self.vertices, self.triangles, points, k
        )
        return -slp @ neumann + dlp @ dirichlet


def map_triangle2vertex(vertices, triangles, features):
    check_tensor(vertices, torch.float32)
    check_tensor(triangles, torch.int32)
    check_tensor(features, torch.complex64)
    return cuda_imp.triangle2vertex(vertices, triangles, features)


def get_potential_of_sources(x0, x1, n0, n1, wave_number, degree=0, grad=False):
    check_tensor(x0, torch.float32)
    check_tensor(x1, torch.float32)
    check_tensor(n0, torch.float32)
    check_tensor(n1, torch.float32)
    if degree == 0:
        if grad:
            return cuda_imp.double_boundary_potential(x0, x1, n0, n1, wave_number)
        else:
            return cuda_imp.single_boundary_potential(x0, x1, n0, n1, wave_number)
    elif degree == 1:
        if grad:
            return cuda_imp.hypersingular_boundary_potential(
                x0, x1, n0, n1, wave_number
            )
        else:
            return cuda_imp.adjointdouble_boundary_potential(
                x0, x1, n0, n1, wave_number
            )
    else:
        raise ValueError("degree must be 0 or 1")
