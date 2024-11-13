import sys

sys.path.append("./")
from src.timer import Timer
from src.modalobj.model import (
    SNR,
    complex_ssim,
)
import numpy as np
import sys
import matplotlib.pyplot as plt
import configparser
import os
from glob import glob
from matplotlib.image import imread

data_dir = "dataset/sample"

img_poisson = imread(f"{data_dir}/sample.png")
img_random = imread(f"{data_dir}/rsample.png")


def crop_image(img, a, b, c, d):
    up_index = -img.shape[0] // a
    down_index = img.shape[0] // b
    left_index = img.shape[1] // c
    right_index = -img.shape[1] // d
    img = img[down_index:up_index, left_index:right_index]
    return img


img_poisson = crop_image(img_poisson, 5, 8, 7, 7)
img_random = crop_image(img_random, 5, 8, 7, 7)

from matplotlib.font_manager import FontProperties

from matplotlib.gridspec import GridSpec

# Assuming the other parts of your script are already defined (ffat_map_bem, our_maps, snrs, ssims, data_dir)

# Load specific font
font_path = "dataset/fonts/LinBiolinum_R.ttf"  # Replace with your font file path
font_bold_path = "dataset/fonts/LinBiolinum_RB.ttf"  # Replace with your font file path
font_size = 35
my_font = FontProperties(fname=font_path)
my_font_bold = FontProperties(fname=font_bold_path)

fig = plt.figure(figsize=(14, 7))
gs = GridSpec(1, 2, width_ratios=[1, 1])

ax = plt.subplot(gs[0])
ax.imshow(img_random)
ax.text(
    0.5,
    0.0,
    "Random Sample",
    transform=ax.transAxes,
    ha="center",
    fontproperties=my_font,
    fontsize=font_size,
)
plt.axis("off")

ax = plt.subplot(gs[1])
ax.imshow(img_poisson)
ax.text(
    0.5,
    0.0,
    "Poisson Disk Sample",
    transform=ax.transAxes,
    ha="center",
    fontproperties=my_font,
    fontsize=font_size,
)
plt.axis("off")
plt.tight_layout()
plt.show()
plt.savefig(f"{data_dir}/compare.png")
