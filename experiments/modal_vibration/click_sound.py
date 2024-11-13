import sys

sys.path.append("./")

from src.modalobj.model import ModalSoundObj, MatSet, Material, BEMModel
from src.utils import CombinedFig
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from time import time
import configparser
import meshio
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


from scipy.spatial import cKDTree


obj_dir = sys.argv[1]
sample_tag = sys.argv[2]
sample_data = np.load(os.path.join(obj_dir, f"click/{sample_tag}.npz"))
obj_pos = sample_data["obj_pos"]
obj_ori = sample_data["obj_ori"]
camera_pos = sample_data["camera_pos"]
impact_idx = sample_data["impact_idx"]

mode_num = 64
config = configparser.ConfigParser()
config.read(f"{obj_dir}/config.ini")
material = Material(getattr(MatSet, config.get("mesh", "material")))
modes_data = np.load(os.path.join(obj_dir, "modes.npz"))
modes = modes_data["modes"]
eigenvalues = modes_data["eigenvalues"]

audio_time = 2
audio_sample_rate = 20000
res = 32
phi_spacing = 2 * np.pi / (res * 2 - 1)
theta_spacing = np.pi / (res - 1)

frame_num = audio_time * audio_sample_rate
modes_f = np.zeros((mode_num, audio_time * audio_sample_rate), dtype=np.float32)
camera_local_position = global_to_local(obj_pos, obj_ori, camera_pos)

force_frame_num = 20
global_mode_f = modes[impact_idx].T @ np.array([1, 1, 1])
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
for method in [
    "bem",
    "ours_4000",
]:
    ffat_map = np.load(os.path.join(obj_dir, f"{method}.npz"))["ffat_map"].reshape(
        -1, 2 * res, res
    )
    ffat_value = np.abs(ffat_map[:, x_i, y_i])
    audio = np.zeros(frame_num, dtype=np.float32)
    for mode_idx in tqdm(range(mode_num)):
        audio += signal_lst[mode_idx] * ffat_value[mode_idx]
    max_audio = np.max(np.abs(audio))
    audio /= max_audio
    audio = np.int16(audio * 32767)
    write(os.path.join(obj_dir, f"click/{sample_tag}_{method}.wav"), 20000, audio)
