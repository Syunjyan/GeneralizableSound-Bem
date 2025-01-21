import sys

sys.path.append("./")

from src.scene import Scene
import torch
import os, sys
from tqdm import tqdm

import numpy as np
import trimesh

# 自定义sample过程，不做旋转，不做放缩（效果好再做），只做平移且仅将cup向-y移动
def custom(data_dir, data_name, src_sample_num = None, trg_sample_num = None , 
                             show_scene:bool=False,
                             split_mode:str = 'train',
                             sound_src:str = 'phone.obj',
                             gpu_id = 0
                             ):
    scene = Scene(f"{data_dir}/config.json")

    skip = True

    if src_sample_num is None:
        src_sample_num = scene.src_sample_num
    if trg_sample_num is None:
        trg_sample_num = scene.trg_sample_num
        
    for src_idx in tqdm(range(src_sample_num), position=gpu_id+1, desc=f"gpu_{gpu_id}", leave=False):
        random_int = np.random.randint(1000000000)
        seed = random_int % 1000000000
        if skip:
            if os.path.exists(f"{data_dir}/e_data/{split_mode}_mesh/{data_name}_{src_idx}.obj"):
                # print(f"data exists, gpu_id: {gpu_id} skips.")
                continue
        x = torch.zeros(
                trg_sample_num, 3,
                dtype=torch.float32,
            )
        y = torch.zeros(trg_sample_num, 65, dtype=torch.float32)
        
        for freq_idx in tqdm(range(65), position=gpu_id+1, desc=f"gpu_{gpu_id}, src {src_idx}/{src_sample_num}", leave=False):
            
            # 自随机一个小于0.3的向量，使cup向-y移动
            _transition_vec = np.zeros(3)
            _transition_vec[1] = -np.random.rand() * 0.3
            scene.enclose_sample(seed=seed, freq_idx=freq_idx, max_freq_idx=65, sound_source=sound_src,
                                 _resize=False, _rotate = False, _transition_vec = _transition_vec)

            scene.solve()

            x[:, :3] = scene.trg_points
         
            y[:, freq_idx] = scene.potential.abs()#.unsqueeze(-1)

        
        if split_mode == 'train':
            torch.save({"x": x, "y": y}, f"{data_dir}/e_data/train_data/{data_name}_{src_idx}.pt")
            # 以obj格式存储几何形状
            mesh = trimesh.Trimesh(scene.vertices.detach().cpu().numpy(), scene.triangles.detach().cpu().numpy())
            mesh.export(f"{data_dir}/e_data/train_mesh/{data_name}_{src_idx}.obj")
        else: # 测试集
            torch.save({"x": x, "y": y}, f"{data_dir}/e_data/val_data/{data_name}_{src_idx}.pt")
            # 以obj格式存储几何形状
            mesh = trimesh.Trimesh(scene.vertices.detach().cpu().numpy(), scene.triangles.detach().cpu().numpy())
            mesh.export(f"{data_dir}/e_data/val_mesh/{data_name}_{src_idx}.obj")



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
                    sound_src="phone.obj",
                    gpu_id=gpu_id
                    )

