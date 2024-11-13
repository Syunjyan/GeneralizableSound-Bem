import sys

sys.path.append("./")

from src.modalobj.model import MatSet, Material
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from time import time
from numba import njit
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from scipy.io.wavfile import write


def global_to_local(X, ori, global_position):
    rotation = R.from_quat(ori)
    rotation_matrix = rotation.as_matrix()
    inv_rotation_matrix = np.transpose(rotation_matrix)
    inv_translation = -np.dot(inv_rotation_matrix, X)
    local_position = np.dot(inv_rotation_matrix, global_position) + inv_translation
    return local_position


@njit()
def _IIR(f, e, theta, r, wwd):
    signal = np.zeros_like(f)
    for idx in range(len(signal)):
        if idx < 3:
            continue
        signal[idx] = (
            2 * e * np.cos(theta) * signal[idx - 1]
            - e**2 * signal[idx - 2]
            + 2
            * f[idx - 1]
            * (e * np.cos(theta + r) - e**2 * np.cos(2 * theta + r))
            / (3 * wwd)
        )
    return signal


def IIR(f, val, alpha, beta, h):
    d = 0.5 * (alpha + beta * val)
    e = np.exp(-d * h)
    wd = (val - d**2) ** 0.5
    theta = wd * h
    w = val**0.5
    r = np.arcsin(d / w)
    return _IIR(f, e, theta, r, w * wd)


obj_dir = sys.argv[1]
sample_tag = sys.argv[2]
sample_data = np.load(os.path.join(obj_dir, f"click/{sample_tag}.npz"))
obj_pos = sample_data["obj_pos"]
obj_ori = sample_data["obj_ori"]
camera_pos = sample_data["camera_pos"]
impact_idx = sample_data["impact_idx"]
import torch


def get_mesh_size(vertices):
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    return (bbox_max - bbox_min).max()


mode_num = 60
bem_base_data = torch.load(os.path.join(obj_dir, "../../modal_data.pt"))
modes = bem_base_data["modes"][:, :, :mode_num]
print("modes", modes.shape)
bem_data = np.load(os.path.join(obj_dir, "bem.npz"))
wave_number = bem_data["wave_number"]
eigenvalues = (wave_number * 343.2) ** 2
omegas = wave_number * 343.2
mesh_size = get_mesh_size(bem_data["vertices"])
points = bem_data["points"]
print("mesh_size", mesh_size)

print("eigenvalues", eigenvalues.shape)

if "0.75" in obj_dir:
    mat = MatSet.Wood
elif "0.40" in obj_dir:
    mat = MatSet.Steel
else:
    mat = MatSet.Ceramic

material = Material(mat)

audio_time = 2
audio_sample_rate = 44100
res = 32
phi_spacing = 2 * np.pi / (res * 2 - 1)
theta_spacing = np.pi / (res - 1)

frame_num = audio_time * audio_sample_rate
modes_f = np.zeros((mode_num, audio_time * audio_sample_rate), dtype=np.float32)
camera_local_position = global_to_local(obj_pos, obj_ori, camera_pos)

force_frame_num = 20
global_mode_f = modes[impact_idx].T @ np.array([1, 1, 1])
global_mode_f = global_mode_f * omegas**2
for i in range(force_frame_num):
    modes_f[:, i] += global_mode_f * np.sin(i / force_frame_num * np.pi)

r = np.linalg.norm(camera_local_position)
theta = np.arctan2(camera_local_position[1], camera_local_position[0])
phi = np.arccos(camera_local_position[2] / r)
x_i = int(theta / phi_spacing)
y_i = int(phi / theta_spacing)

signal_lst = np.zeros((mode_num, frame_num), dtype=np.float32)
for mode_idx in tqdm(range(mode_num)):
    signal_lst[mode_idx] = IIR(
        modes_f[mode_idx],
        eigenvalues[mode_idx],
        material.alpha,
        material.beta,
        1 / audio_sample_rate,
    )

audios = []
max_audio = 0
for method in [
    "bem",
    "NeuralSound",
    "neuPAT",
]:
    ffat_map = np.load(os.path.join(obj_dir, f"{method}.npz"))["ffat_map"].reshape(
        -1, 2 * res, res
    )
    print("ffat_map", ffat_map.shape)
    if method == "NeuralSound":
        r = (points**2).sum(-1) ** 0.5
        ffat_map = ffat_map / r[0] / 1.225 * (mesh_size / 0.15) ** (5 / 2)

    if method == "neuPAT":
        ffat_map = (10**ffat_map) * 10e-6 - 10e-6

    ffat_value = np.abs(ffat_map[:, x_i, y_i])
    audio = np.zeros(frame_num, dtype=np.float32)
    for mode_idx in tqdm(range(mode_num)):
        audio += signal_lst[mode_idx] * ffat_value[mode_idx]
    audios.append(audio)
    max_audio = max(max_audio, np.max(np.abs(audio)))

for i, method in enumerate(["bem", "NeuralSound", "neuPAT"]):
    audio = audios[i]
    audio /= max_audio
    audio = np.int16(audio * 32767)
    write(os.path.join(obj_dir, f"click/{sample_tag}_{method}.wav"), 44100, audio)
