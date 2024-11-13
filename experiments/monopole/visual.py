import sys

sys.path.append("./")

from src.modalobj.model import (
    SNR,
    complex_ssim,
)
import numpy as np

import matplotlib.pyplot as plt
from glob import glob


obj_path = sys.argv[1]
gt_data = np.load(obj_path + "/gt.npz")
bem_data = np.load(obj_path + "/bem.npz")
ours_data = np.load(obj_path + "/ours.npz")

gt_ffat = gt_data["ffat_map"].reshape(64, 32)
bem_ffat = bem_data["ffat_map"].reshape(*gt_ffat.shape)
ours_ffat = ours_data["ffat_map"].reshape(*gt_ffat.shape)

bem_time = bem_data["cost_time"]
ours_time = ours_data["cost_time"]


from matplotlib.font_manager import FontProperties
from matplotlib.image import imread
from matplotlib.gridspec import GridSpec

# Assuming the other parts of your script are already defined (ffat_map_bem, our_maps, snrs, ssims, data_dir)

# Load specific font
font_path = (
    "/home/jxt/.local/share/fonts/LinBiolinum_R.ttf"  # Replace with your font file path
)
my_font = FontProperties(fname=font_path)
font_size = 27
# Set color scale limits
vmin, vmax = np.abs(gt_ffat).min(), np.abs(gt_ffat).max()


# Create figure
fig = plt.figure(figsize=(15, 6 if len(sys.argv) > 2 else 5.5))
gs = GridSpec(1, 4, width_ratios=[2, 1, 1, 1])
# Load the image from the specified path
mesh1 = imread(f"{obj_path}/mesh_edge.png")

left_index = mesh1.shape[1] // 7
right_index = -left_index if left_index != 0 else mesh1.shape[1]
up_index = -mesh1.shape[0] // 7
down_index = 0
# Crop the image
cropped_img = mesh1[down_index:up_index, left_index:right_index]
# Plot the image from the path in the first subplot
ax = plt.subplot(gs[0])
ax.imshow(cropped_img)
ax.text(
    0.5,
    -0.1,
    "SNR | SSIM | Cost Time (s)",
    transform=ax.transAxes,
    ha="center",
    fontproperties=my_font,
    fontsize=font_size,
)
if len(sys.argv) > 2:
    plt.title("Input Boundary Mesh", fontproperties=my_font, fontsize=font_size)
ax.axis("off")

# Plot the first image
ax = plt.subplot(gs[1])

ax.imshow(np.abs(gt_ffat).reshape(64, 32), cmap="viridis", vmin=vmin, vmax=vmax)
ax.text(
    16,
    70,
    f"Inf | 1.0 | 0.0",
    ha="center",
    fontproperties=my_font,
    fontsize=font_size,
)

if len(sys.argv) > 2:
    plt.title("Ground Truth", fontproperties=my_font, fontsize=font_size)
plt.axis("off")

ax = plt.subplot(gs[2])

ax.imshow(np.abs(bem_ffat).reshape(64, 32), cmap="viridis", vmin=vmin, vmax=vmax)
ax.text(
    16,
    70,
    f"{SNR(gt_ffat, bem_ffat):.2f} | {complex_ssim(gt_ffat, bem_ffat):.2f} | {bem_time:.2f}",
    ha="center",
    fontproperties=my_font,
    fontsize=font_size,
)

if len(sys.argv) > 2:
    plt.title("BEM", fontproperties=my_font, fontsize=font_size)
plt.axis("off")

ax = plt.subplot(gs[3])

ax.imshow(np.abs(ours_ffat).reshape(64, 32), cmap="viridis", vmin=vmin, vmax=vmax)
ax.text(
    16,
    70,
    f"{SNR(gt_ffat, ours_ffat):.2f} | {complex_ssim(gt_ffat, ours_ffat):.2f} | {ours_time:.2f}",
    ha="center",
    fontproperties=my_font,
    fontsize=font_size,
)

if len(sys.argv) > 2:
    plt.title("Ours", fontproperties=my_font, fontsize=font_size)
plt.axis("off")
# Add text below the image

plt.tight_layout()
plt.savefig(f"{obj_path}/ablation.png")
