import sys

sys.path.append("./")

from src.scene import Scene,generate_sample_scene, generate_sample_scene_simpler
import torch
from tqdm import tqdm

data_dir = "dataset/embed_1"
generate_sample_scene(data_dir, sys.argv[1], show_scene=True)
