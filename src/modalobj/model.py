import meshio
import numpy as np
from .mesh_process import tetra_from_mesh, update_triangle_normals
from scipy.spatial import KDTree
# from .fem import FEMmodel, LOBPCG_solver, Material, MatSet
import torch
from numba import njit
from skimage.metrics import structural_similarity as ssim
# from ..bem.solver import map_triangle2vertex


def SNR(ground_truth, prediction):
    ground_truth = np.abs(ground_truth)
    prediction = np.abs(prediction)
    return 10 * np.log10(
        (ground_truth**2).mean() / ((ground_truth - prediction) ** 2).mean()
    )


def complex_ssim(x, y):
    x = np.abs(x)
    y = np.abs(y)
    max_val = x.max()
    min_val = x.min()
    return ssim(x, y, data_range=max_val - min_val)


@njit()
def unit_sphere_surface_points(res):
    # r = 0.5
    points = np.zeros((2 * res, res, 3))
    phi_spacing = 2 * np.pi / (res * 2 - 1)
    theta_spacing = np.pi / (res - 1)
    for i in range(2 * res):
        for j in range(res):
            phi = phi_spacing * i
            theta = theta_spacing * j
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points[i, j] = [x, y, z]
    return points * 0.5


def update_normals(vertices, triangles):
    """
    vertices: (n, 3)
    triangles: (m, 3)
    """
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    return normals


def get_mesh_center(vertices):
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    return (bbox_max + bbox_min) / 2


def get_mesh_size(vertices):
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    return (bbox_max - bbox_min).max()


def get_spherical_surface_points(vertices, scale=2):
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
    points = unit_sphere_surface_points(32)
    points = points.reshape(-1, 3)
    points = points * get_mesh_size(vertices) * scale + get_mesh_center(vertices)
    points = torch.tensor(points).float().cuda()
    return points


def normalize_vertices(vertices, scale=1.0):
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    # print("size:", (bbox_max - bbox_min).max())  # debug
    vertices = (vertices - (bbox_max + bbox_min) / 2) / (bbox_max - bbox_min).max()
    vertices = vertices * scale
    return vertices


class StaticObj:
    def __init__(self, mesh_path, scale=0.15, normalize=True):
        mesh = meshio.read(mesh_path)
        self.vertices = mesh.points
        # bbox_min = self.vertices.min(axis=0)
        # bbox_max = self.vertices.max(axis=0)
        # print(mesh_path, "size:", (bbox_max - bbox_min).max())  # debug
        if normalize:
            self.vertices = normalize_vertices(self.vertices, scale)
        self.triangles = mesh.cells_dict["triangle"]
        self.bbox_min = self.vertices.min(axis=0)
        self.bbox_max = self.vertices.max(axis=0)
        self.center = (self.bbox_max + self.bbox_min) / 2
        self.size = (self.bbox_max - self.bbox_min).max()


# class VibrationObj:
#     def __init__(self, mesh_path):
#         (
#             self.vertices,
#             self.tets,
#             self.surf_vertices,
#             self.surf_triangles,
#         ) = tetra_from_mesh(mesh_path)
#         self.bbox_min = self.vertices.min(axis=0)
#         self.bbox_max = self.vertices.max(axis=0)
#         self.surf_normals = update_triangle_normals(
#             self.surf_vertices, self.surf_triangles
#         )

#     def normalize(self, scale=1.0):
#         self.vertices = (self.vertices - (self.bbox_max + self.bbox_min) / 2) / (
#             self.bbox_max - self.bbox_min
#         ).max()
#         self.vertices = self.vertices * scale
#         self.surf_vertices = (
#             self.surf_vertices - (self.bbox_max + self.bbox_min) / 2
#         ) / (self.bbox_max - self.bbox_min).max()
#         self.surf_vertices = self.surf_vertices * scale
#         self.bbox_min = self.vertices.min(axis=0)
#         self.bbox_max = self.vertices.max(axis=0)
#         self.size = (self.bbox_max - self.bbox_min).max()
#         self.center = (self.bbox_max + self.bbox_min) / 2

#     def modal_analysis(self, k=32, material=Material(MatSet.Plastic)):
#         self.fem_model = FEMmodel(self.vertices, self.tets, material)
#         eigenvalues, eigenvectors = LOBPCG_solver(
#             self.fem_model.stiffness_matrix, self.fem_model.mass_matrix, k
#         )
#         eigenvectors = eigenvectors.reshape(-1, 3, k)
#         kd_tree = KDTree(self.vertices)
#         _, surf_points_index = kd_tree.query(self.surf_vertices)
#         surf_eigenvecs = eigenvectors[surf_points_index]
#         self.modes = surf_eigenvecs.reshape(-1, 3, k)
#         self.eigenvalues = eigenvalues

#     def get_triangle_neumann(self, mode_id):
#         vertex_modes = self.modes[:, :, mode_id]
#         triangle_neumann = vertex_modes[self.surf_triangles].mean(axis=1)
#         triangle_neumann = (triangle_neumann * self.surf_normals).sum(axis=1)
#         return triangle_neumann

#     def get_frequencies(self):
#         return self.eigenvalues**0.5 / (2 * np.pi)

#     def get_frequency(self, mode_id):
#         return self.eigenvalues[mode_id] ** 0.5 / (2 * np.pi)

#     def get_omega(self, mode_id):
#         return self.eigenvalues[mode_id] ** 0.5

#     def get_wave_number(self, mode_id):
#         return self.get_omega(mode_id) / 343.2
