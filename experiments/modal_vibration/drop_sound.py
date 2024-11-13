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


def calculate_snr(predicted_signal, ground_truth_signal):
    # Mean square value of the ground truth signal
    signal_power = np.mean(ground_truth_signal**2)

    # Mean square value of the difference (noise)
    noise_power = np.mean((predicted_signal - ground_truth_signal) ** 2)

    # Calculate SNR
    snr = 10 * np.log10(signal_power / noise_power)

    return snr


def global_to_local(X, ori, global_position):
    rotation = R.from_quat(ori)
    rotation_matrix = rotation.as_matrix()
    inv_rotation_matrix = np.transpose(rotation_matrix)
    inv_translation = -np.dot(inv_rotation_matrix, X)
    local_position = np.dot(inv_rotation_matrix, global_position) + inv_translation
    return local_position


from scipy.ndimage import gaussian_filter1d


def smooth_curve_gaussian(data, sigma):
    return gaussian_filter1d(data, sigma)


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

camera_pos = np.array([0.5, -0.5, 0.7])
mode_num = 64
obj_dir = sys.argv[1]
mesh = meshio.read(os.path.join(obj_dir, "mesh_surf.obj"))
vertices = mesh.points
tree = cKDTree(vertices)
triangles = mesh.cells_dict["triangle"]
contact_data = np.load(os.path.join(obj_dir, "contact.npz"))
contact_pos = contact_data["pos"]
contact_normal = contact_data["normal"]
contact_force = contact_data["force"]
motion_data = np.load(os.path.join(obj_dir, "motion.npz"))
motion_pos = motion_data["pos"]  # xyz
motion_ori = motion_data["ori"]  # xyzw
time_step = motion_data["step"]
config = configparser.ConfigParser()
config.read(f"{obj_dir}/config.ini")
material = Material(getattr(MatSet, config.get("mesh", "material")))
modes_data = np.load(os.path.join(obj_dir, "modes.npz"))
modes = modes_data["modes"]
eigenvalues = modes_data["eigenvalues"]
print("vertices num:", vertices.shape[0])
print("modes shape:", modes.shape)
frame_num = motion_pos.shape[0]
res = 32
phi_spacing = 2 * np.pi / (res * 2 - 1)
theta_spacing = np.pi / (res - 1)
xs, ys, rs = [], [], []
modes_f = np.zeros((mode_num, frame_num), dtype=np.float32)
for frame_idx in tqdm(range(frame_num)):
    pos = motion_pos[frame_idx]
    ori = motion_ori[frame_idx]
    camera_local_position = global_to_local(pos, ori, camera_pos)
    contact_pos_lst = contact_pos[frame_idx]
    for contact_idx in range(contact_pos_lst.shape[0]):
        if contact_force[frame_idx, contact_idx] < 1:
            continue
        contact_pos_global = contact_pos_lst[contact_idx]
        contact_local_position = global_to_local(pos, ori, contact_pos_global)
        contact_normal_global = contact_normal[frame_idx, contact_idx]
        contact_normal_local = np.dot(
            R.from_quat(ori).as_matrix(), contact_normal_global
        )
        distance, index = tree.query(contact_local_position)
        modes_f[:, frame_idx] += contact_force[frame_idx, contact_idx] * (
            modes[index].T @ contact_normal_local
        )

    r = np.linalg.norm(camera_local_position)
    theta = np.arctan2(camera_local_position[1], camera_local_position[0])
    phi = np.arccos(camera_local_position[2] / r)
    x_i = int(theta / phi_spacing)
    y_i = int(phi / theta_spacing)
    xs.append(x_i)
    ys.append(y_i)
    rs.append(r)

signal_lst = np.zeros((mode_num, frame_num), dtype=np.float32)
for mode_idx in tqdm(range(mode_num)):
    signal_lst[mode_idx] = IIR(
        modes_f[mode_idx],
        eigenvalues[mode_idx],
        material.alpha,
        material.beta,
        time_step,
    )
print(obj_dir)
points = np.load(f"{obj_dir}/bem.npz")["points"]
for method in [
    "bem",
    "NeuralSound",
    "ours_0",
    "ours_1000",
    "ours_2000",
    "ours_4000",
]:
    ffat_map = np.load(os.path.join(obj_dir, f"{method}.npz"))["ffat_map"].reshape(
        -1, 2 * res, res
    )
    if method == "NeuralSound":
        mesh_size = config.getfloat("mesh", "size")
        r = (points**2).sum(-1) ** 0.5
        ffat_map = ffat_map / r[0] / 1.225 * (mesh_size / 0.15) ** (5 / 2)

    ffat_values = np.zeros((mode_num, frame_num), dtype=np.float32)
    for mode_idx in tqdm(range(mode_num)):
        ffat_values[mode_idx] = smooth_curve_gaussian(
            np.abs(ffat_map[mode_idx, xs, ys]), 500
        )
    audio = np.zeros(frame_num, dtype=np.float32)
    for mode_idx in tqdm(range(mode_num)):
        audio += signal_lst[mode_idx] * ffat_values[mode_idx]

    # if method == "bem":
    max_audio = np.max(np.abs(audio))
    audio /= max_audio
    if method == "bem":
        bem_audio = audio
        snr = None
    else:
        snr = calculate_snr(audio, bem_audio)
    audio = np.int16(audio * 32767)
    write(os.path.join(obj_dir, f"{method}.wav"), 20000, audio)
    print(method, snr)
    np.save(os.path.join(obj_dir, f"{method}_snr"), snr)
