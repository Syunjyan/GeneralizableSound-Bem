import sys

sys.path.append("./")

from src.scene import Scene,generate_sample_scene, generate_sample_scene_simpler
import torch
import os, sys
from tqdm import tqdm

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

    if os.path.exists(f"{data_dir}/data/{mode}_mesh/out_{tag}_0.obj"):
        print(f"data exists, gpu_id: {gpu_id} skips.")
        exit()
        
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"started a process. gpu_id: {gpu_id}")

    generate_sample_scene_simpler(data_dir, f"out_{tag}", 
                                  src_sample_num=src_num,
                                  show_scene=False,
                                  sound_src='phone.obj',
                                  split_mode=mode,
                                  gpu_id=gpu_id)

