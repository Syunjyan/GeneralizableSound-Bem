from glob import glob
import os


root_dir = "dataset/NeuPAT_new/scale/test/"
sub_dirs = glob(root_dir + "*/")

for sub_dir in sub_dirs:
    os.system(f"blender --background --python BlenderToolbox/scale_edit.py {sub_dir}")
