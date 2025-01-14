import sys

sys.path.append("./")

from src.scene import generate_sample_enclosed
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

    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # print(f"started a process. gpu_id: {gpu_id}")

    generate_sample_enclosed(data_dir, f"out_{tag}", 
                                  src_sample_num=src_num,
                                  show_scene=False,
                                  split_mode=mode,
                                  sound_src='phone.obj',
                                  gpu_id=gpu_id)

