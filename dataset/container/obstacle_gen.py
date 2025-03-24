import sys

sys.path.append("./")

import trimesh
import numpy as np

container: trimesh.Trimesh = trimesh.load("dataset/container/container.obj")
plane: trimesh.Trimesh = trimesh.load("dataset/container/plane.obj")

# normalize
container.vertices = container.vertices - (container.vertices.max(0) + container.vertices.min(0)) / 2  
plane.vertices = plane.vertices - (plane.vertices.max(0) + plane.vertices.min(0)) / 2

# scale
container.vertices = container.vertices / (container.vertices.max() - container.vertices.min()) * 0.6 # 0.6m
plane.vertices = plane.vertices / (plane.vertices.max() - plane.vertices.min()) * 0.4 # 0.4m

import os
if not os.path.exists("dataset/container/my_obstacles/"):
    os.mkdir("dataset/container/my_obstacles/")

# container开口向z正方向，于是旋转后组装

for degree in range(2, 183, 15):
    plane_rotated = plane.copy()
    plane_rotated.apply_transform(trimesh.transformations.rotation_matrix(-degree*np.pi/180, [0, 1, 0], [-0.2, 0, 0]))
    plane_rotated.apply_translation([0, 0, 0.32])
    scene = trimesh.Scene([plane_rotated, container])
    # 将scene绕过原点的x轴旋转90度
    scene.apply_transform(trimesh.transformations.rotation_matrix(-90*np.pi/180, [1, 0, 0], [0, 0, 0]))
    # save to meshes/container_xxx.obj
    scene.export(f"dataset/container/my_obstacles/container_{degree:03d}.obj")
