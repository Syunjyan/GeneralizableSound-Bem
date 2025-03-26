# created by houxinyun on 2025-03-17
import os
from glob import glob
import torch
from .BiCGSTAB import BiCGSTAB
import numpy as np

import bempp.api

bempp.api.BOUNDARY_OPERATOR_DEVICE_TYPE = "gpu"
bempp.api.POTENTIAL_OPERATOR_DEVICE_TYPE = "gpu"
bempp.api.PLOT_BACKEND = "paraview"

# from bempp.core import opencl_kernels
# opencl_kernels.device_type='gpu'
# bempp.api.BOUNDARY_OPERATOR_DEVICE_TYPE = "gpu"


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


class BEM:
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
        self.grid = bempp.api.Grid(self.vertices.T.cpu(), self.triangles.T.cpu())
        self.space = bempp.api.function_space(self.grid, "P", 1)


    def assemble_boundary_matrix(self, wavenumber, layer_type, approx=False):
        space = self.space

        beta = 1j / wavenumber
        if layer_type == "bm_lhs":
            hyp = bempp.api.operators.boundary.helmholtz.hypersingular(
                space,
                space,
                space,
                wavenumber,
                device_interface="opencl",
                precision="single",
            )
            hyp = hyp.weak_form().A

            hyp_matrix_bempp = torch.from_numpy(hyp).cuda()
            double_matrix_bempp = bempp.api.operators.boundary.helmholtz.double_layer(
                space, space, space, wavenumber
            )
            double_matrix_bempp = torch.from_numpy(double_matrix_bempp.weak_form().A).cuda()
            operator = LHS_bempp = -double_matrix_bempp + beta * hyp_matrix_bempp
        elif layer_type == "bm_rhs":

            adlp = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(
                space,
                space,
                space,
                wavenumber,
                device_interface="opencl",
                precision="single",
            )
            adlp_matrix_bempp = torch.from_numpy(adlp.weak_form().A).cuda()
            single_matrix_bempp = bempp.api.operators.boundary.helmholtz.single_layer(
                space, space, space, wavenumber
            )
            single_matrix_bempp = torch.from_numpy(single_matrix_bempp.weak_form().A).cuda()
            operator = RHS_bempp = -single_matrix_bempp - beta * adlp_matrix_bempp
        else:
            raise ValueError(f"Unknown layer_type: {layer_type}")

        if approx:
            # 近似处理逻辑
            pass

        return operator.to(torch.complex64)

    def identity_matrix(self):
        space = self.space
        identity = bempp.api.operators.boundary.sparse.identity(
            space, space, space, device_interface="opencl", precision="single"
        )
        return torch.from_numpy(identity.weak_form().A.todense()).cuda()

    def neumann2dirichlet(self, k, neumann):
        identity = self.identity_matrix().to(torch.complex64)
        beta = 1j / k
        LHS = self.assemble_boundary_matrix(k, "bm_lhs", approx=True) + 0.5 * identity
        RHS = (
            self.assemble_boundary_matrix(k, "bm_rhs", approx=True).to(torch.complex64)
            - beta * 0.5 * identity
        ) @ neumann.to(torch.complex64)
        return solve_linear_equation(LHS, RHS)
    
    def single_layer_potential(self, k):
        space = self.space
        slp = bempp.api.operators.boundary.helmholtz.single_layer(
            space,
            space,
            space,
            k,
            device_interface="opencl",
            precision="single",
        )
        return torch.from_numpy(slp.weak_form().A).cuda()

    def double_layer_potential(self, k):
        space = self.space
        dlp = bempp.api.operators.boundary.helmholtz.double_layer(
            space,
            space,
            space,
            k,
            device_interface="opencl",
            precision="single",
        )
        return torch.from_numpy(dlp.weak_form().A).cuda()
    
    # def single_potential(self, k, points):
    #     space = self.space
    #     slp = bempp.api.operators.potential.helmholtz.single_layer(
    #         space,
    #         points.T.cpu().numpy(),
    #         k,
    #         device_interface="opencl",
    #         precision="single",
    #     )
    #     return torch.from_numpy(slp * self.grid_fun).cuda()
    
    # def double_potential(self, k, points):
    #     space = self.space
    #     dlp = bempp.api.operators.potential.helmholtz.double_layer(
    #         space,
    #         points.T.cpu().numpy(),
    #         k,
    #         device_interface="opencl",
    #         precision="single",
    #     )
    #     return torch.from_numpy(dlp * self.grid_fun).cuda()

    def boundary2potential(self, k, neumann, dirichlet, points):

        neumann = bempp.api.GridFunction(
            self.space, coefficients=neumann.cpu().numpy()
        )
        dirichlet = bempp.api.GridFunction(
            self.space, coefficients=dirichlet.cpu().numpy()
        )
        slp = bempp.api.operators.potential.helmholtz.single_layer(
            self.space,
            points.T.cpu().numpy(),
            k,
            device_interface="opencl",
            precision="single",
        ).evaluate(neumann)
        dlp = bempp.api.operators.potential.helmholtz.double_layer(
            self.space,
            points.T.cpu().numpy(),
            k,
            device_interface="opencl",
            precision="single",
        ).evaluate(dirichlet)
        slp = torch.from_numpy(slp).cuda().to(torch.complex64)
        dlp = torch.from_numpy(dlp).cuda().to(torch.complex64)
        p = -slp + dlp
        return p
        # return -self.single_potential(k, points) + self.double_potential(k, points)

