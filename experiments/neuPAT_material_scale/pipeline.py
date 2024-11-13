import sys
import subprocess

subprocess.run(["python", "experiments/neuPAT_material_scale/test.py"])
subprocess.run(["python", "NeuralSound/tetra2hex.py", "dataset/NeuPAT_new/scale/test"])
subprocess.run(
    ["python", "NeuralSound/test.py", "--dataset", "dataset/NeuPAT_new/scale/test"]
)
subprocess.run(["python", "experiments/neuPAT_material_scale/plot_merge.py"])
