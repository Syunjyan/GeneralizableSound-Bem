import numpy as np
import matplotlib.pyplot as plt


img = np.load("dataset/teaser/values.npy")

plt.imshow(img.T, vmax=0.0025, vmin=-0.0025, cmap="bwr")
plt.axis("off")
plt.savefig("dataset/teaser/fdtd.png")
