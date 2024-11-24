import sys
sys.path.append("./")

import torch
from glob import glob
from tqdm import tqdm
import json
from src.utils import calculate_bin_frequencies, Visualizer, apply_spec_mask_to_audio
from scipy.spatial.transform import Rotation as R
from src.scene import Scene
import numpy as np
from scipy.io import wavfile

from scene import *

def global_to_local(X, ori, global_position):
    rotation = R.from_quat(ori)
    rotation_matrix = rotation.as_matrix()
    inv_rotation_matrix = np.transpose(rotation_matrix)
    inv_translation = -np.dot(inv_rotation_matrix, X)
    local_position = np.dot(inv_rotation_matrix, global_position) + inv_translation
    return local_position

def net_eval():
    x = torch.zeros(src_num, len(freq_pos), 4, dtype=torch.float32, device="cuda")
    x[:, :, :3] = x_data
    x[:, :, -1] = freq_pos.reshape(1, -1)
    nc_spec = model(x.reshape(-1, x.shape[-1])).reshape(src_num, len(freq_pos))
    return nc_spec

def scene_eval(scene: Scene, x_data, freq_pos):
    '''
    **Warning!** This function dose not support dynamic scene parameters.

    If 
    '''
    # TODO

    # x_data: trg_points + rotation + move + resize + freq

    trg_points = x_data[:, :3]

    freq_pos = freq_pos.reshape(1, -1)

    assert scene.rot_num + scene.move_num + scene.resize_num == 0, "Dynamic scene parameters are not supported."

    scene.trg_points = trg_points


    scene.freq_factor = freq_pos

    scene.solve()

    return scene.potential.abs().unsqueeze(-1)


def process_sound(data_dir, output_audio_path, n_fft=512, sampling_rate=16000):

    scene = Scene(f"{data_dir}/../config.json")

    # with open(f"{data_dir}/net.json", "r") as file:
    #     train_config_data = json.load(file)

    # train_params = train_config_data.get("train", {})

    # model = NeuPAT(1, train_config_data).cuda()
    # model.load_state_dict(torch.load(f"{data_dir}/model.pt"))
    # model.eval()

    torch.set_grad_enabled(False)

    xs = torch.linspace(0, 1, 64, device="cuda", dtype=torch.float32)
    ys = torch.linspace(0, 1, 32, device="cuda", dtype=torch.float32)
    gridx, gridy = torch.meshgrid(xs, ys)

    animation_data = np.load(f"{data_dir}/../animation_data.npz")
    x_data = animation_data["x"]
    fps = animation_data["fps"]

    freq_bins = calculate_bin_frequencies(n_fft, sampling_rate=sampling_rate)

    freq_pos = torch.zeros(len(freq_bins), device="cuda", dtype=torch.float32)
    for freq_i in tqdm(range(len(freq_bins))):
        freq_bin = freq_bins[freq_i]
        if freq_bin < scene.freq_min or freq_bin > scene.freq_max:
            continue
        freq_pos[freq_i] = (np.log10(freq_bin) - scene.freq_min_log) / (
            scene.freq_max_log - scene.freq_min_log
        )

    src_num = len(x_data)
    x_data = torch.from_numpy(x_data).cuda().unsqueeze(1)
    trg_pos = torch.zeros(3, device="cuda", dtype=torch.float32)
    trg_pos[0] = 0.5
    trg_pos[1] = 0.1
    trg_pos[2] = 0.2

    rs = (trg_pos[0] + 1) * scene.bbox_size
    theta = trg_pos[1] * 2 * np.pi - np.pi
    phi = trg_pos[2] * np.pi
    xs = rs * torch.sin(phi) * torch.cos(theta)
    ys = rs * torch.sin(phi) * torch.sin(theta)
    zs = rs * torch.cos(phi)
    trg_points = torch.stack([xs, ys, zs], dim=-1) + scene.bbox_center

    print("x_data:", x_data.shape)
    print("freq_pos:", freq_pos.shape)
    print("trg_pos:", trg_pos.shape)
    scene.sample()

    

    nc_spec = net_eval()
    nc_spec = (10**nc_spec.T) * 10e-6 - 10e-6

    nc_spec = nc_spec.cpu().numpy()

    audio = apply_spec_mask_to_audio(
        1, nc_spec, src_num, n_fft=n_fft, animation_frame_rate=fps
    )

    wavfile.write(output_audio_path, 16000, audio)

# Example usage:
# process_sound("dataset/NeuPAT_new/fix/baseline", "output_audio.wav")
