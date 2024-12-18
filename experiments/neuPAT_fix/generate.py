import sys

sys.path.append("./")

from src.scene import Scene,genarate_sample_scene, generate_sample_scene_simpler
import torch
import os, sys
from tqdm import tqdm

# 2024.12.12 
# 选取不同的障碍物，生成数据。
data_dir = "dataset/fix"
# 新建文件夹
os.makedirs(f"{data_dir}/data/train_mesh", exist_ok=True)
os.makedirs(f"{data_dir}/data/train_data", exist_ok=True)


obstacles_name_list = os.listdir(os.path.join(data_dir, "my_obstacles"))
# 过滤掉非 obj 物体
obstacles_name_list = [obstacles_name for obstacles_name in obstacles_name_list if obstacles_name.endswith(".obj")]
for i, obstacles_name in enumerate(obstacles_name_list):
    print(f"obstacles_name: {obstacles_name}, {i+1}/{len(obstacles_name_list)}")
    # 复制该物体到 data_dir，并将其重命名为 obstacle.obj。若已存在，则覆盖。
    os.system(f"cp {os.path.join(data_dir, 'my_obstacles', obstacles_name)} {os.path.join(data_dir, 'obstacle.obj')}")
    # 生成数据
    generate_sample_scene_simpler(data_dir, f"out_{i}", show_scene=False)


'''
import sys

sys.path.append("./")

from src.scene import Scene
import torch
from tqdm import tqdm

data_dir = "dataset/NeuPAT_new/fix"

scene = Scene(f"{data_dir}/config.json")

x = torch.zeros(
    scene.src_sample_num,
    scene.trg_sample_num,
    3 + 1,
    dtype=torch.float32,
)
# 3 for trg sample position, 1 for freq

y = torch.zeros(scene.src_sample_num, scene.trg_sample_num, 1, dtype=torch.float32)

for src_idx in tqdm(range(scene.src_sample_num)):
    scene.sample()
    scene.solve()
    x[src_idx, :, :3] = scene.trg_factor
    x[src_idx, :, -1] = scene.freq_factor
    y[src_idx] = scene.potential.abs().unsqueeze(-1)

    if src_idx == 0:
        scene.show()


torch.save({"x": x, "y": y}, f"{data_dir}/data/{sys.argv[1]}.pt")

'''