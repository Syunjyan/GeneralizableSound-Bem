import sys

sys.path.append("./")

from src.scene import Scene,generate_sample_scene, generate_sample_scene_simpler
import torch
import os, sys
from tqdm import tqdm

if __name__ == "__main__":

    data_dir = sys.argv[1]
    tag = sys.argv[2]
    gpu_id = sys.argv[3]

    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"started a process. gpu_id: {gpu_id}")

    generate_sample_scene_simpler(data_dir, f"out_{tag}", show_scene=False)

