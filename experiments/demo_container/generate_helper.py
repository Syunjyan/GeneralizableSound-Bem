import sys

sys.path.append("./")

from src.scene import Scene
import torch
import os, sys
from tqdm import tqdm


import numpy as np
import trimesh

def transition_matrix(angle=0, axis=[1,0,0], translation=[0,0,0], point=None):
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


def normalize_mesh(mesh: trimesh.Trimesh, scale=1):
    mesh.vertices = mesh.vertices - (mesh.vertices.max(0) + mesh.vertices.min(0)) / 2
    mesh.vertices = mesh.vertices / (mesh.vertices.max(0) - mesh.vertices.min(0)).max() * scale
    return mesh


def concat_obstacle(data_dir, seed = None, plane_move = True, scene_move = True):
    '''
    障碍物的组装，平移、旋转采样在这里完成。
    '''
    container = trimesh.load_mesh(f"{data_dir}/container.obj")
    plane = trimesh.load_mesh(f"{data_dir}/plane.obj")
    # normalize
    container = normalize_mesh(container, scale=0.15)
    plane = normalize_mesh(plane, scale=0.11)
    
    container_translate = np.array([0,0,0.075])
    container = container.apply_translation(container_translate)

    plane_translate = np.array([0,0,0.15])
    plane_rot_axis = 'y'
    plane_rot_pos = np.array([0.055,0,0])

    sound_source_pos = np.array([0,0,0.05])


    if seed is None:
        seed = np.random.randint(1000000000)
    np.random.seed(seed)
    scene_resize = False
    if scene_move:
        # 30% resize数据, 30%普通数据(声源移动旋转),40%盖子开关数据
        number = np.random.rand()
        if number < 0.3:
            scene_resize = True
            plane_move = False
        elif number < 0.6:
            scene_resize = False
            plane_move = False
        else:
            scene_resize = False
            plane_move = True

    # random resize
    if scene_resize:
        resize_x = np.random.rand()+1.0
        resize_y = resize_x
        resize_z = 1
        container.apply_scale([resize_x, resize_y, resize_z])
        plane.apply_scale([resize_x, resize_y, resize_z])
        container_translate = container_translate * np.array([resize_x, resize_y, resize_z])
        plane_translate = plane_translate * np.array([resize_x, resize_y, resize_z])
        plane_rot_pos = plane_rot_pos * np.array([resize_x, resize_y, resize_z])

    # randomly rotate plane
    if plane_move:# and not scene_resize:
        theta = np.random.rand()*np.pi
        plane_matrix = transition_matrix(theta, [0,1,0], plane_translate, plane_rot_pos)
        plane.apply_transform(plane_matrix)
    else :
        theta = np.pi
        plane_matrix = transition_matrix(theta, [0,1,0], plane_translate, plane_rot_pos)
        plane.apply_transform(plane_matrix)
    
    # concate
    obstacle = trimesh.util.concatenate([container, plane])
    
    obstacle.apply_translation(-sound_source_pos)

    # source rotate and move
    if scene_move:
        if scene_resize or plane_move:
            move_z = 0
        else:
            move_z = (np.random.rand())*0.2
            obstacle.apply_translation([0, 0, -move_z])

        theta = (np.random.rand())*2*np.pi
        scene_matrix = transition_matrix(theta, [0,1,0], [0,0,0])
        obstacle.apply_transform(scene_matrix)

    # return vert and tri
    return obstacle.vertices, obstacle.faces

# 自定义sample过程
def custom(data_dir, data_name, src_sample_num = None, trg_sample_num = None , 
                             show_scene:bool=False,
                             split_mode:str = 'train',
                             sound_src:str = 'source.obj',
                             gpu_id = 0
                             ):
    
    
    # 避开Scene对container的正则化
    scene = Scene(f"{data_dir}/config.json", normalize=False)

    # 是否跳过重复数据
    skip = False

    if src_sample_num is None:
        src_sample_num = scene.src_sample_num
    if trg_sample_num is None:
        trg_sample_num = scene.trg_sample_num
        
    for src_idx in tqdm(range(src_sample_num), position=gpu_id+1, desc=f"gpu_{gpu_id}", leave=False):
        random_int = np.random.randint(1000000000)
        seed = random_int % 1000000000
        if skip:
            if os.path.exists(f"{data_dir}/data/{split_mode}_mesh/{data_name}_{src_idx}.obj"):
                # print(f"data exists, gpu_id: {gpu_id} skips.")
                continue
        
        x = torch.zeros(
                trg_sample_num, 3,
                dtype=torch.float32,
            )
        y = torch.zeros(trg_sample_num, 65, dtype=torch.float32)

        # new
        for obj in scene.objs:
            if obj.name != sound_src:
                obj.vertices, obj.triangles = concat_obstacle(data_dir)
                obj.vertices_base = torch.tensor(obj.vertices).cuda().to(torch.float32)
                obj.neumann = torch.zeros(
                    len(obj.vertices_base), dtype=torch.complex64, device="cuda"
                )
                obj.triangles = torch.tensor(obj.triangles, dtype=torch.int32, device='cuda')
                obj.vertices = torch.tensor(obj.vertices, dtype=torch.float32, device='cuda')
        
        # if src_idx == 0: # debug
        #     show_debug(gpu_id, scene, 5)
        #     show_debug(gpu_id, scene, 10)
        #     show_debug(gpu_id, scene, 20)
        #     show_debug(gpu_id, scene, 40)
        #     mesh = trimesh.Trimesh(scene.vertices.detach().cpu().numpy(), scene.triangles.detach().cpu().numpy())
        #     mesh.export(f"{data_dir}/debug_mesh_{gpu_id}.obj")

        for freq_idx in tqdm(range(65), position=gpu_id+1, desc=f"gpu_{gpu_id}, src {src_idx}/{src_sample_num}", leave=False):

            scene.enclose_sample(seed=seed, freq_idx=freq_idx, max_freq_idx=65, sound_source=sound_src,
                            _transition=False, _resize=False, _rotate = False)
            scene.solve()

            x[:, :3] = scene.trg_points
         
            y[:, freq_idx] = scene.potential.abs()#.unsqueeze(-1)
        
        
        if split_mode == 'train':
            torch.save({"x": x, "y": y}, f"{data_dir}/data/train_data/{data_name}_{src_idx}.pt")
            # 以obj格式存储几何形状
            mesh = trimesh.Trimesh(scene.vertices.detach().cpu().numpy(), scene.triangles.detach().cpu().numpy())
            mesh.export(f"{data_dir}/data/train_mesh/{data_name}_{src_idx}.obj")
        else: # 测试集
            torch.save({"x": x, "y": y}, f"{data_dir}/data/val_data/{data_name}_{src_idx}.pt")
            # 以obj格式存储几何形状
            mesh = trimesh.Trimesh(scene.vertices.detach().cpu().numpy(), scene.triangles.detach().cpu().numpy())
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
                                  show_scene=False,
                                  split_mode=mode,
                                  sound_src="source.obj",
                                  gpu_id=gpu_id)

