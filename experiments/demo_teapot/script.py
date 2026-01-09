import trimesh
import numpy as np

data_dir = "dataset/teapot"



def transition_matrix(angle, axis, translation, point=None):
    """
    计算旋转矩阵和平移矩阵的组合
    :param angle: 旋转角度
    :param axis: 旋转轴
    :param translation: 平移向量
    :return: 4x4的变换矩阵
    """
    rotation = trimesh.transformations.rotation_matrix(angle, axis, point)
    translation = trimesh.transformations.translation_matrix(translation)
    return np.dot(translation, rotation)


container = trimesh.load_mesh(f"{data_dir}/hushen.obj")
plane = trimesh.load_mesh(f"{data_dir}/gaizi.obj")
# normalize
container = container.apply_scale(0.26 / np.max(container.extents))
container = container.apply_translation([0, -0.09, 0])
plane = plane.apply_scale(0.13 / np.max(plane.extents))

# container_center = (container.bounds[0] + container.bounds[1]) / 2
# container = container.apply_translation(-container_center)

plane_center = (plane.bounds[0] + plane.bounds[1]) / 2
plane_trans = np.array([0, 0.102, 0])
plane = plane.apply_translation(-plane_center)
# rescale
# CONTAINER_SCALE = 0.15
# PLANE_SCALE = 0.11
# container = container.apply_scale(CONTAINER_SCALE)
# plane = plane.apply_scale(PLANE_SCALE)
# plane_translate_z = CONTAINER_SCALE / 2 + 0.01
# align
container_scale = np.max(container.extents)
plane_scale = np.max(plane.extents)

theta = 0
plane_matrix = transition_matrix(-theta, [0,0,1], plane_trans, point=[plane_scale/2, 0, 0])
plane.apply_transform(plane_matrix)

obstacle = trimesh.util.concatenate([container, plane])
obstacle.export(f"{data_dir}/obstacle.obj")

container.export(f"{data_dir}/n_teapot.obj")
plane.export(f"{data_dir}/n_gaizi.obj")