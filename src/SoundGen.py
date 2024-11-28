import sys
sys.path.append("./")

import torch
from glob import glob
from tqdm import tqdm
import json
from src.utils import calculate_bin_frequencies, Visualizer, apply_spec_mask_to_audio
from scipy.spatial.transform import Rotation as R
from src.scene import *
import numpy as np
from scipy.io import wavfile
import os
import hashlib
import inspect
import seaborn as sns

def global_to_local(X, ori, global_position):
    rotation = R.from_quat(ori)
    rotation_matrix = rotation.as_matrix()
    inv_rotation_matrix = np.transpose(rotation_matrix)
    inv_translation = -np.dot(inv_rotation_matrix, X)
    local_position = np.dot(inv_rotation_matrix, global_position) + inv_translation
    return local_position

# def net_eval():
#     x = torch.zeros(src_num, len(freq_pos), 4, dtype=torch.float32, device="cuda")
#     x[:, :, :3] = x_data
#     x[:, :, -1] = freq_pos.reshape(1, -1)
#     nc_spec = model(x.reshape(-1, x.shape[-1])).reshape(src_num, len(freq_pos))
#     return nc_spec

def scene_eval(scene: Scene, x_data, freq_pos,
                cache = False):
    '''
    **Warning!** This function dose not support dynamic scene parameters.

    For stable scene, we sample the scene for `freq_num` times and solve the trg_points in animation data.
    '''
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # print("Current directory:", current_dir)
    cache_dir = os.path.join(current_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Create a unique hash for the input data
    input_hash = hashlib.md5(x_data.cpu().numpy().tobytes() + freq_pos.cpu().numpy().tobytes()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{input_hash}.npz")
    if cache:
        print("Caching enabled.")

        # Check if the result is already cached
        if os.path.exists(cache_file):
            print("Cache found. Loading from cache.")
            nc_spec = np.load(cache_file)["nc_spec"]
            nc_spec = torch.from_numpy(nc_spec).cuda()

            return nc_spec
        else:
            print("Cache not found. Solving scene.")

    # x_data: trg_points + rotation + move + resize + freq
    trg_num = x_data.shape[0]
    trg_points = x_data[:, :3]
    freq_pos = freq_pos.reshape(-1)

    assert scene.rot_num + scene.move_num + scene.resize == 0, "Dynamic scene parameters are not supported."

    nc_spec = None
    for freq_fac in tqdm(freq_pos, desc=f"Solving scene in {len(freq_pos)} freqs"):
        scene.setting(freq_factor=freq_fac)
        scene.solve(trg_points)
        
        if torch.isnan(scene.potential).any():
            nan_indices = torch.isnan(scene.potential).nonzero(as_tuple=True)
            print(f"Warning: NaN in scene.potential at indices {torch.unique(nan_indices[0])}")
            print(f"example nan x_data: {x_data[torch.unique(nan_indices[0])]}")
        if nc_spec is None:
            nc_spec = scene.potential.abs().unsqueeze(-1)
        else:
            nc_spec = torch.cat([nc_spec, scene.potential.abs().unsqueeze(-1)], dim=-1)

    nc_spec = nc_spec.reshape(trg_num, -1)
    # Save the result to the cache
    np.savez(cache_file, nc_spec=nc_spec.cpu().numpy(), )
    return nc_spec


def generate_sound_static(data_dir, audio_name, n_fft=128, sampling_rate=16000, 
                          cache=False, save_fig=False):
    """
    Generate sound from animation data and a static scene configuration.
    Parameters:
    data_dir (str): Directory containing the configuration and animation data files.
    audio_name (str): Name of the output audio file.
    n_fft (int, optional): Number of FFT components. Default is 128.
    sampling_rate (int, optional): Sampling rate of the audio. Default is 16000.
    cache (bool, optional): Whether to cache intermediate results. Default is False.
    save_fig (bool, optional): Whether to save the spectrogram heatmap as a figure. Default is False.
    Returns:
    None
    Notes:
    This function does not support dynamic scenes.
    """
    '''
    **Warning!** This function dose not support dynamic scene.
    '''
    scene = Scene(f"{data_dir}/config.json")

    torch.set_grad_enabled(False)

    animation_data = np.load(f"{data_dir}/animation_data.npz")
    x_data = animation_data["x"]
    fps = animation_data["fps"]

    freq_bins = calculate_bin_frequencies(n_fft, sampling_rate=sampling_rate)

    freq_pos = torch.zeros(len(freq_bins), device="cuda", dtype=torch.float32)
    for freq_i in tqdm(range(len(freq_bins)), desc="Calculating freq_factor"):
        freq_bin = freq_bins[freq_i]
        if freq_bin < scene.freq_min:
            continue
        if freq_bin > scene.freq_max:
            freq_pos[freq_i] = 1
            continue
        freq_pos[freq_i] = (np.log10(freq_bin) - scene.freq_min_log) / (
            scene.freq_max_log - scene.freq_min_log
        )

    src_num = len(x_data)
    # x_data = torch.from_numpy(x_data).cuda()
    x_data = torch.from_numpy(x_data).to(dtype=torch.float32).cuda()
    print("x_data dtype:", x_data.dtype)
    print("x_data:", x_data.shape)
    print("freq_pos:", freq_pos.shape)
    

    nc_spec = scene_eval(scene, x_data, freq_pos, cache=cache)
    print("shape of nc_spec:", nc_spec.shape)

    nc_spec = nc_spec.cpu().numpy().T
    import matplotlib.pyplot as plt

    if save_fig:
        plt.figure(figsize=(10, 8))
        sns.heatmap(nc_spec, cmap="viridis")
        plt.title("Heatmap of nc_spec")
        plt.xlabel("Frequency Bins")
        plt.ylabel("Source Number")
        plt.savefig(f"dataset/spectrogram/nc_spec_heatmap.png")
        plt.close()

    audio = apply_spec_mask_to_audio(
        1, nc_spec, src_num, n_fft=n_fft, animation_frame_rate=fps, trg_sample_rate=sampling_rate,
        save_spec=True
    )

    wavfile.write(f"{data_dir}/{audio_name}", sampling_rate, audio)

# Example usage:
# generate_sound_stable("dataset/fix", "audio_1.wav")



def generate_sound_dynamic():
    # TODO
    pass