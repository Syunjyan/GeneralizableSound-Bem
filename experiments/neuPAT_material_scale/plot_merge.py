from matplotlib.image import imread, imsave
import sys
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import subprocess
import os

data_dir = "dataset/NeuPAT_new/scale/test"
first = True
case_list = glob(f"{data_dir}/*")

for case_dir in case_list:
    if not os.path.isdir(case_dir):
        continue
    cmds = ["python", "experiments/neuPAT_material_scale/plot.py", case_dir]
    if first:
        cmds.append("first")
        first = False
    result = subprocess.run(
        cmds,
        capture_output=True,
        text=True,
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    if result.returncode == 0:
        print("Success!")
    else:
        print("Failed!")


img_lst = glob(f"{data_dir}/*/scale_comparison.png")

images = []
First = True
for img_path in img_lst:
    img = imread(img_path)
    if First:
        First = False
        up_index = -img.shape[0] // 6
        down_index = img.shape[0] // 6
    else:
        up_index = -img.shape[0] // 6
        down_index = img.shape[0] // 4
    left_index = img.shape[1] // 8
    right_index = -img.shape[1] // 30
    img = img[down_index:up_index, left_index:right_index]
    images.append(img)

vstacked_images = np.vstack(images)
imsave(f"{data_dir}/scale_comparison.png", vstacked_images)
