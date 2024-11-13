import sys

sys.path.append("./")

from src.modalobj.model import SNR, complex_ssim
import numpy as np

import matplotlib.pyplot as plt
from glob import glob


obj_path = "dataset/wob"
poisson_gt = np.loadtxt(f"{obj_path}/poisson_gt.txt").reshape(256, 256)
poisson_wob = np.loadtxt(f"{obj_path}/poisson_wob.txt").reshape(256, 256)
poisson_ours = np.load(f"{obj_path}/poisson_ours.npy").reshape(256, 256)

helmholtz_gt = np.loadtxt(f"{obj_path}/helmholtz_gt.txt").reshape(256, 256)
helmholtz_wob = np.loadtxt(f"{obj_path}/helmholtz_wob.txt").reshape(256, 256)
helmholtz_ours = np.load(f"{obj_path}/helmholtz_ours.npy").reshape(256, 256)


from matplotlib.font_manager import FontProperties
from matplotlib.image import imread
from matplotlib.gridspec import GridSpec

# Assuming the other parts of your script are already defined (ffat_map_bem, our_maps, snrs, ssims, data_dir)

# Load specific font
font_path = (
    "/home/jxt/.local/share/fonts/LinBiolinum_R.ttf"  # Replace with your font file path
)
my_font = FontProperties(fname=font_path)
font_size = 25

# Create figure
fig = plt.figure(figsize=(12, 7))

vmin, vmax = np.abs(poisson_gt).min(), np.abs(poisson_gt).max()
ax = plt.subplot(2, 3, 1)
ax.imshow(poisson_gt, cmap="viridis", vmin=vmin, vmax=vmax)
ax.text(
    -0.12,
    0.3,
    "Poisson",
    transform=ax.transAxes,
    ha="center",
    fontproperties=my_font,
    fontsize=font_size,
    rotation=90,
)
ax.text(
    0.5,
    -0.1,
    "SNR | SSIM | Cost Time (s)",
    transform=ax.transAxes,
    ha="center",
    fontproperties=my_font,
    fontsize=font_size - 5,
)
plt.title("Ground Truth", fontproperties=my_font, fontsize=font_size)
ax.axis("off")


ax = plt.subplot(2, 3, 2)
ax.imshow(poisson_wob, cmap="viridis", vmin=vmin, vmax=vmax)
ax.text(
    0.5,
    -0.1,
    f"{SNR(poisson_gt, poisson_wob):.2f} | {complex_ssim(poisson_gt, poisson_wob):.2f} | 49.14",
    transform=ax.transAxes,
    ha="center",
    fontproperties=my_font,
    fontsize=font_size - 5,
)
plt.title("WoB", fontproperties=my_font, fontsize=font_size)
ax.axis("off")
ax = plt.subplot(2, 3, 3)
ax.imshow(poisson_ours, cmap="viridis", vmin=vmin, vmax=vmax)
plt.title("Ours", fontproperties=my_font, fontsize=font_size)
ax.text(
    0.5,
    -0.12,
    f"{SNR(poisson_gt, poisson_ours):.2f} | {complex_ssim(poisson_gt, poisson_ours):.2f} | 0.05",
    transform=ax.transAxes,
    ha="center",
    fontproperties=my_font,
    fontsize=font_size - 5,
)
ax.axis("off")

vmin, vmax = np.abs(helmholtz_gt).min(), np.abs(helmholtz_gt).max()
ax = plt.subplot(2, 3, 4)
ax.imshow(helmholtz_gt, cmap="viridis", vmin=vmin, vmax=vmax)
ax.text(
    -0.12,
    0.2,
    "Helmholtz",
    transform=ax.transAxes,
    ha="center",
    fontproperties=my_font,
    fontsize=font_size,
    rotation=90,
)
ax.text(
    0.5,
    -0.12,
    "SNR | SSIM | Cost Time (s)",
    transform=ax.transAxes,
    ha="center",
    fontproperties=my_font,
    fontsize=font_size - 5,
)
ax.axis("off")
ax = plt.subplot(2, 3, 5)
ax.imshow(helmholtz_wob, cmap="viridis", vmin=vmin, vmax=vmax)
ax.text(
    0.5,
    -0.12,
    f"{SNR(helmholtz_gt.real, helmholtz_wob.real):.2f} | {complex_ssim(helmholtz_gt.real, helmholtz_wob.real):.2f} | 35.86",
    transform=ax.transAxes,
    ha="center",
    fontproperties=my_font,
    fontsize=font_size - 5,
)
ax.axis("off")
ax = plt.subplot(2, 3, 6)
ax.imshow(helmholtz_ours.real, cmap="viridis", vmin=vmin, vmax=vmax)
ax.text(
    0.5,
    -0.12,
    f"{SNR(helmholtz_gt.real, helmholtz_ours.real):.2f} | {complex_ssim(helmholtz_gt.real, helmholtz_ours.real):.2f} | 0.05",
    transform=ax.transAxes,
    ha="center",
    fontproperties=my_font,
    fontsize=font_size - 5,
)
ax.axis("off")

plt.tight_layout()
plt.savefig(f"{obj_path}/wob_ablation.png")
