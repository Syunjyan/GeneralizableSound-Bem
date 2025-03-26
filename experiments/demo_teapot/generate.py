# 生成teapot场景数据，生成半包围形式训练数据
import sys

sys.path.append("./")

from src.scene import Scene, generate_sample_scene_simpler
import torch
import os, sys
from tqdm import tqdm
import multiprocessing

import argparse

parser = argparse.ArgumentParser(description="Generate data using Point soundsource.")

parser.add_argument("--data_dir", "-d", type=str, default="dataset/teapot", help="Directory of the dataset, for example, dataset/fix")
# parser.add_argument("--type", "-t", type=str, default="obstacle", help="Generate type: obstacle or enclosed")
parser.add_argument("--gpu_set", "-g", type=int, default=0, help="GPU set.")

args = parser.parse_args()

# 2024.12.12 
# 选取不同的障碍物，生成数据。
data_dir = args.data_dir

def detect_available_gpu(max_num_gpu=4, power_threshold=130):
    """
    返回一个数组，数组中的元素为可用的 gpu 的编号。
    凡是功耗小于 50W 的 gpu，都认为是可用的。
    :param max_num_gpu: 最多返回的 gpu 的数量
    """
    gpu_info = os.popen("nvidia-smi --query-gpu=power.draw --format=csv").readlines()
    available_gpu = []
    if max_num_gpu == 0 or max_num_gpu == None:
        max_num_gpu = 100000
    for i, power in enumerate(gpu_info[1:]):
        power = float(power[:-2])
        if power < power_threshold:
            available_gpu.append(i)
        if len(available_gpu) == max_num_gpu:
            break
    return available_gpu


obstacles_name_list = os.listdir(os.path.join(data_dir, "my_obstacles"))
# 过滤掉非 obj 物体
obstacles_name_list = [obstacles_name for obstacles_name in obstacles_name_list if obstacles_name.endswith(".obj")]

os.environ["TORCH_CUDA_ARCH_LIST"]="8.6"

# available_gpus = detect_available_gpu()
available_gpus = [0]
if args.gpu_set == 0:
    available_gpus = [0, 1, 2, 3]
else:
    available_gpus = [4, 5, 6, 7]

print(f"available_gpus: {available_gpus}")


TRAIN_SRC_DATASIZE = 12
VAL_SRC_DATASIZE = 4

VAL_UNIQUE_OBSTACLES = 0 # 训练集中没有的障碍物数量

# type = args.type

# 2025.1.14


os.makedirs(f"{data_dir}/e_data/train_mesh", exist_ok=True)
os.makedirs(f"{data_dir}/e_data/train_data", exist_ok=True)
os.makedirs(f"{data_dir}/e_data/val_mesh", exist_ok=True)
os.makedirs(f"{data_dir}/e_data/val_data", exist_ok=True)
# 划分训练集和测试集，其中测试集包含部分训练集中没有的mesh。

# 划分训练、测试集obstacles
train_obstacles = obstacles_name_list[:len(obstacles_name_list)-VAL_UNIQUE_OBSTACLES]

val_obstacles = obstacles_name_list

# 训练集


# 多进程，每个进程调用一次 python generate_helper.py data_dir tag
def generate_data(data_dir, tag, gpu_id, src_num, mode):
    print("check command: ", f"export CUDA_VISIBLE_DEVICES={gpu_id}; python experiments/demo_teapot/generate_helper.py {data_dir} {tag} {gpu_id} {src_num} {mode}")
    os.system(f"export CUDA_VISIBLE_DEVICES={gpu_id}; python experiments/demo_teapot/generate_helper.py {data_dir} {tag} {gpu_id} {src_num} {mode}")

for i, obstacles_name in enumerate(tqdm(train_obstacles, desc="Processing train obstacles")):
    
    # print(f"obstacles_name: {obstacles_name}, {i+1}/{len(train_obstacles)}")
    # 复制该物体到 data_dir，并将其重命名为 obstacle.obj。若已存在，则覆盖。
    os.system(f"cp {os.path.join(data_dir, 'my_obstacles', obstacles_name)} {os.path.join(data_dir, 'obstacle.obj')}")
    
    pool = multiprocessing.Pool(processes=len(available_gpus))
    for gpu_id in available_gpus:
        tag = str(gpu_id) + "_" + str(i)
        pool.apply_async(generate_data, args=(data_dir, tag, gpu_id, TRAIN_SRC_DATASIZE//len(available_gpus), "train"))
        
    pool.close()
    pool.join()


# 测试集
for i, obstacles_name in enumerate(tqdm(val_obstacles, desc="Processing val obstacles")):
    
    # print(f"obstacles_name: {obstacles_name}, {i+1}/{len(val_obstacles)}")
    # 复制该物体到 data_dir，并将其重命名为 obstacle.obj。若已存在，则覆盖。
    os.system(f"cp {os.path.join(data_dir, 'my_obstacles', obstacles_name)} {os.path.join(data_dir, 'obstacle.obj')}")
    
    
    pool = multiprocessing.Pool(processes=len(available_gpus))
    for gpu_id in available_gpus:
        tag = str(gpu_id) + "_" + str(i)
        pool.apply_async(generate_data, args=(data_dir, tag, gpu_id, 1, "val"))
        
    pool.close()
    pool.join()
