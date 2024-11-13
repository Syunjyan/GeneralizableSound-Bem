from matplotlib.image import imread, imsave
import sys
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import subprocess
import os

data_dir = sys.argv[1]
obj_lst = glob(f"{data_dir}/oloid*")
# reverse the order of the list
obj_lst.reverse()
first = True
for obj_dir in obj_lst:
    # skip files that are not directories
    if not os.path.isdir(obj_dir):
        continue
    print(obj_dir)
    cmds = ["python", "experiments/monopole/visual.py", obj_dir]
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


img_lst = glob(f"{data_dir}/*/ablation.png")
img_lst.reverse()

images = []
for img_path in img_lst:
    images.append(imread(img_path))

vstacked_images = np.vstack(images)
imsave(f"{data_dir}/BEM_compare.png", vstacked_images)
