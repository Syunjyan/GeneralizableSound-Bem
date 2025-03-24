import sys

sys.path.append("./")

import trimesh
import numpy as np

container: trimesh.Trimesh = trimesh.load("dataset/trumpet/static.obj")
plane: trimesh.Trimesh = trimesh.load("dataset/trumpet/rot.obj")

# normalize
container.vertices = container.vertices - (container.vertices.max(0) + container.vertices.min(0)) / 2  
plane.vertices = plane.vertices - (plane.vertices.max(0) + plane.vertices.min(0)) / 2

# scale
container.vertices = container.vertices / (container.vertices.max() - container.vertices.min()) * 0.2 
plane.vertices = plane.vertices / (plane.vertices.max() - plane.vertices.min()) * 0.2

import os
if not os.path.exists("dataset/trumpet/my_obstacles/"):
    os.mkdir("dataset/trumpet/my_obstacles/")

container.apply_translation([0.05, 0, 0])

for degree in range(0, 91, 10):
    plane_rotated = plane.copy()
    plane_rotated.apply_transform(trimesh.transformations.rotation_matrix(-degree*np.pi/180, [0, 0, 1], [0.05, 0.1, 0]))
    plane_rotated.apply_translation([-0.10, 0, 0])
    scene = trimesh.Scene([plane_rotated, container])
    # save to meshes/container_xxx.obj
    scene.export(f"dataset/trumpet/my_obstacles/trumpet_{degree:03d}.obj")
