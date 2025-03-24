import sys

sys.path.append("./")

from src.scene import Scene, generate_sample_enclosed
import torch
import os, sys
from tqdm import tqdm


import numpy as np
import trimesh

# 自定义sample过程，不做旋转，不做放缩，不做平移,
def custom(data_dir, data_name, src_sample_num = None, trg_sample_num = None , 
                             show_scene:bool=False,
                             split_mode:str = 'train',
                             sound_src:str = 'ball.obj',
                             gpu_id = 0
                             ):
    
    # 想办法避开Scene对container的正则化
    scene = Scene(f"{data_dir}/config.json", normalize=False)

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
        
        for freq_idx in tqdm(range(65), position=gpu_id+1, desc=f"gpu_{gpu_id}, src {src_idx}/{src_sample_num}", leave=False):
            
            scene.enclose_sample(seed=seed, freq_idx=freq_idx, max_freq_idx=65, sound_source=sound_src,
                                  _transition = False,
                                  _rotate = False,
                                  _resize = False)

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
                                  sound_src="ball.obj",
                                  gpu_id=gpu_id
                                  )

