import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data_dir = "dataset/wob"
mean1 = np.loadtxt(f"{data_dir}/weights_mean_poisson.txt")
mean2 = np.loadtxt(f"{data_dir}/weights_mean_helmholtz.txt")

var1 = np.loadtxt(f"{data_dir}/weights_var_poisson.txt")
var2 = np.loadtxt(f"{data_dir}/weights_var_helmholtz.txt")


from matplotlib.font_manager import FontProperties
from matplotlib.image import imread
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as font_manager

# Assuming the other parts of your script are already defined (ffat_map_bem, our_maps, snrs, ssims, data_dir)

# Load specific font

sns.set_theme()
sns.set_context("notebook", font_scale=1.6)
font_path = (
    "/home/jxt/.local/share/fonts/LinBiolinum_R.ttf"  # Replace with your font file path
)
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = "Linux Biolinum"


n_samples = 100

# Generating data based on the given means and variances
data1 = np.random.normal(loc=mean1, scale=np.sqrt(var1), size=(n_samples, len(mean1)))
data2 = np.random.normal(loc=mean2, scale=np.sqrt(var2), size=(n_samples, len(mean2)))

categories = ["0", "1", "2", "3", "4"]
# Creating DataFrames
df1 = pd.DataFrame(data1, columns=categories)
df2 = pd.DataFrame(data2, columns=categories)

# Plotting in subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Left subplot for data1
sns.pointplot(data=df1, errorbar="sd", capsize=0.2, ax=axes[0])
axes[0].set_title("Poisson", fontsize=30)
axes[0].set_xlabel("Recursion Depth")
axes[0].set_ylabel("Weight Value")

# Right subplot for data2
sns.pointplot(data=df2, errorbar="sd", capsize=0.2, ax=axes[1])
axes[1].set_title("Helmholtz", fontsize=30)
axes[1].set_xlabel("Recursion Depth")

plt.tight_layout()
plt.savefig(f"{data_dir}/wob_weights.png")
