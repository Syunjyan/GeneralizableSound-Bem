import sys

data_dir = sys.argv[1]

import subprocess

subprocess.run(["python", "experiments/neuPAT_audio/test.py", data_dir])
subprocess.run(["python", "NeuralSound/tetra2hex.py", data_dir + "/../test"])
subprocess.run(["python", "NeuralSound/test.py", "--dataset", data_dir + "/../test"])

from glob import glob

sub_dir_list = glob(f"{data_dir}/../test/*")
import os

for sub_dir in sub_dir_list:
    if os.path.isdir(sub_dir):
        subprocess.run(["python", "experiments/neuPAT_audio/plot.py", sub_dir])
