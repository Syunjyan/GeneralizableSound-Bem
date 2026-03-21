'''
2025-04-28
1.优先验证 数据集正确性（动画脚本）
2. 暂时固定声源位置，实验采样密度
3. 实验新采样方法（声源位置&场景位置同时随机）
'''
import sys

sys.path.append("./")

import torch
import os, sys
from tqdm import tqdm

from kwave_solver import KWaveSolver

import numpy as np
import trimesh


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

def concat_obstacle(data_dir, solver: KWaveSolver, seed = None):
    '''
    障碍物的组装，平移、旋转采样在这里完成。
    '''
    container = trimesh.load_mesh(f"{data_dir}/hushen.obj")
    plane = trimesh.load_mesh(f"{data_dir}/gaizi.obj")
    # normalize
    container = container.apply_scale(0.26 / np.max(container.extents))
    container = container.apply_translation([0, -0.09, 0])
    # plane = plane.apply_scale(0.11 / np.max(plane.extents))

    # container_center = (container.bounds[0] + container.bounds[1]) / 2
    # container = container.apply_translation(-container_center)

    plane_trans = np.array([0, 0.075, 0])
    plane = plane.apply_translation(-plane_trans)

    if seed is None:
        seed = np.random.randint(1000000000)
    np.random.seed(seed)
    # randomly rotate
    theta = np.random.rand()*np.pi

    plane_matrix = transition_matrix(-theta, [0,0,1], plane_trans, point=[plane.bounds[1,0]+0.01, 0, 0])

    plane.apply_transform(plane_matrix)

    # concate
    solver.add_object(container)
    solver.add_object(plane)

    return trimesh.util.concatenate([container, plane])


# 自定义sample过程，做指定方向的旋转，不做放缩，不做平移
# 希望能在每次srcidx变化时，scene中的obstacle也能变化
def custom(data_dir, data_name, src_sample_num = None, trg_sample_num = None , 
                             show_scene:bool=False,
                             split_mode:str = 'train',
                             gpu_id = 0
                             ):
    
    # 避开Scene对container的正则化

    # 是否跳过重复数据
    skip = False

        
    for src_idx in tqdm(range(src_sample_num), position=gpu_id+1, desc=f"gpu_{gpu_id}", leave=False):
        
        solver = KWaveSolver(resolution=0.005, pml_size=5, time_end=0.03)

        random_int = np.random.randint(1000000000)
        seed = random_int % 1000000000
        if skip:
            if os.path.exists(f"{data_dir}/data/{split_mode}_mesh/{data_name}_{src_idx}.obj"):
                # print(f"data exists, gpu_id: {gpu_id} skips.")
                continue
        
        x = np.zeros(
                (trg_sample_num, 3),
                dtype=np.float32,
            )
        y = np.zeros((trg_sample_num, 65), dtype=np.float32)
        # new
        mesh = concat_obstacle(data_dir, solver)
        
        if src_idx == 0: # debug
            mesh.export(f"{data_dir}/debug_mesh_{gpu_id}.obj")

        r = np.random.randn(trg_sample_num) * 0.5 + np.max(np.sum(mesh.vertices**2, axis=1)**0.5)
        theta = np.random.randn(trg_sample_num) * 2 * np.pi
        phi = np.random.randn(trg_sample_num) * np.pi

        x[:, 0] = r * np.sin(phi) * np.cos(theta)
        x[:, 1] = r * np.sin(phi) * np.sin(theta)
        x[:, 2] = r * np.cos(phi)

        y = solver.solve_spec(x, device_num = 0, show_sim_log = False)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        
        
        if split_mode == 'train':
            torch.save({"x": x, "y": y}, f"{data_dir}/data/train_data/{data_name}_{src_idx}.pt")
            mesh.export(f"{data_dir}/data/train_mesh/{data_name}_{src_idx}.obj")
        else: # 测试集
            torch.save({"x": x, "y": y}, f"{data_dir}/data/val_data/{data_name}_{src_idx}.pt")
            mesh.export(f"{data_dir}/data/val_mesh/{data_name}_{src_idx}.obj")


if __name__ == "__main__":

    data_dir = sys.argv[1]
    tag = sys.argv[2]
    gpu_id = int(sys.argv[3])
    mode = 'train'

    src_num = None
    if len(sys.argv) > 4:
        src_num = int(sys.argv[4])
        mode = sys.argv[5]
    
    if mode != 'train':
        src_num = 1 # only 1 source in test and val

    # if os.path.exists(f"{data_dir}/data/{mode}_mesh/out_{tag}_0.obj"):
    #     print(f"data exists, gpu_id: {gpu_id} skips.")
    #     exit()

    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # print(f"started a process. gpu_id: {gpu_id}")

    custom(data_dir, f"out_{tag}", 
                                  src_sample_num=src_num,
                                  trg_sample_num=2000,
                                  show_scene=False,
                                  split_mode=mode,
                                  gpu_id=gpu_id)

