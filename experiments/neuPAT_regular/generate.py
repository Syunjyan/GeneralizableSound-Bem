# 生成与NeuralSound对比场景，cup-phone
import sys

sys.path.append("./")

from src.scene import Scene
import torch
from tqdm import tqdm
import numpy as np
import trimesh 

data_dir = "dataset/neuralsound_comp"

def custom_generate(data_dir, data_name, src_sample_num = None, trg_sample_num = None , 
                             show_scene:bool=False,
                             split_mode:str = 'train',
                             sound_src:str = 'phone.obj',
                             gpu_id = 0
                             ):
    scene = Scene(f"{data_dir}/config.json")

    if src_sample_num is None:
        src_sample_num = scene.src_sample_num
    if trg_sample_num is None:
        trg_sample_num = scene.trg_sample_num

    # 手动摆放，采样
    src_sample_num=4
        
    for src_idx in tqdm(range(src_sample_num), position=gpu_id, desc=f"gpu_{gpu_id}", leave=False):
        random_int = np.random.randint(1000000000)
        seed = random_int % 1000000000
        x = torch.zeros(
                trg_sample_num, 3,
                dtype=torch.float32,
            )
        y = torch.zeros(trg_sample_num, 65, dtype=torch.float32)
        # for freq_idx in range(65):
        for freq_idx in tqdm(range(65), position=gpu_id, desc=f"gpu_{gpu_id}, src {src_idx}/{src_sample_num}", leave=False):
            
            # 点声源
            scene.enclose_sample(seed=seed, freq_idx=freq_idx, max_freq_idx=65, sound_source=sound_src, 
                                 _transition_vec = np.array([0, -src_idx*0.06, 0]))

            scene.solve()

            x[:, :3] = scene.trg_points
         #   x[:, :3] = scene.trg_points#scene.trg_factor
         #   x[:, -1] = scene.freq_factor
         
            y[:, freq_idx] = scene.potential.abs()#.unsqueeze(-1)

        
        torch.save({"x": x, "y": y}, f"{data_dir}/neuralsound/train_data/{data_name}_{src_idx}.pt")
        # 以obj格式存储几何形状
        mesh = trimesh.Trimesh(scene.vertices.detach().cpu().numpy(), scene.triangles.detach().cpu().numpy())
        mesh.export(f"{data_dir}/neuralsound/train_mesh/{data_name}_{src_idx}.obj")



import os
if not os.path.exists(f"{data_dir}/neuralsound/train_mesh"):
    os.makedirs(f"{data_dir}/neuralsound/train_mesh")
if not os.path.exists(f"{data_dir}/neuralsound/train_data"):
    os.makedirs(f"{data_dir}/neuralsound/train_data")

custom_generate(data_dir, "cup_phone", src_sample_num=4, split_mode='train', sound_src='phone.obj', gpu_id=0)