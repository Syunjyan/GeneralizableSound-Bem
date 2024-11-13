import meshio
import sys
from glob import glob


data_dir = sys.argv[1]

obj_lst = glob(f"{data_dir}/*")

for obj_dir in obj_lst:
    print(obj_dir)
    mesh = meshio.read(f"{obj_dir}/mesh.obj")
    # normalize vertices to [0, 1]
    vertices = mesh.points
    bbox_min = vertices.min(0)
    bbox_max = vertices.max(0)
    center = (bbox_min + bbox_max) / 2
    vertices -= center
    vertices /= (bbox_max - bbox_min).max()
    print(vertices.min(0), vertices.max(0))
    # switch y and z axis
    # vertices[:, [1, 2]] = vertices[:, [2, 1]]
    # vertices[:, 2] *= -1
    # vertices[:, 2] += 1
    mesh.points = vertices
    meshio.write(f"{obj_dir}/mesh_norm.obj", mesh)
