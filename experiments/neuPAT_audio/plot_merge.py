from matplotlib.image import imread, imsave
import sys
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import subprocess
import os


def crop_img(img, a, b, c, d):
    down_index = int(img.shape[0] * a)
    up_index = -int(img.shape[0] * b)
    left_index = int(img.shape[1] * c)
    right_index = -int(img.shape[1] * d)
    return img[down_index:up_index, left_index:right_index]


data_dir = "dataset/NeuPAT/audio/large_mlp"
case_list = [0.0, 0.3, 0.6, 0.9]
imgs = []
first = True
for case in case_list:
    case_dir1 = f"{data_dir}/{case:.1f}_{2000}"
    case_dir2 = f"{data_dir}/{case:.1f}_{7000}"
    cmds = ["python", "experiments/neuPAT_audio/plot2.py", case_dir1]
    if first:
        cmds.append("first")
    subprocess.run(
        cmds,
        capture_output=True,
        text=True,
    )
    print(cmds)
    cmds = ["python", "experiments/neuPAT_audio/plot.py", case_dir2]
    if first:
        cmds.append("first")
    subprocess.run(
        cmds,
        capture_output=True,
        text=True,
    )
    print(cmds)
    img1 = imread(f"{case_dir1}/scale_comparison.png")
    img2 = imread(f"{case_dir2}/scale_comparison.png")
    img1 = crop_img(img1, 0.15 if first else 0.22, 0.15 if first else 0.18, 0.1, 0.42)
    img2 = crop_img(img2, 0.15 if first else 0.22, 0.15 if first else 0.18, 0.25, 0.42)
    img = np.hstack([img1, img2])
    imgs.append(img)
    first = False

vstacked_images = np.vstack(imgs)
imsave(f"{data_dir}/audio_comparison.png", vstacked_images)
