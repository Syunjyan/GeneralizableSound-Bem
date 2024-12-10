import plotly.graph_objects as go
import numpy as np
import plotly.graph_objs as go
import time
import torch


class Timer:
    def __init__(self):
        torch.cuda.synchronize()
        self.start_time = time.time()

    def log(self, prefix=""):
        cost_time = self.get_time_cost()
        print(f"{prefix} cost time: {cost_time}")
        return cost_time

    def get_time_cost(self):
        torch.cuda.synchronize()
        cost_time = time.time() - self.start_time
        self.start_time = time.time()
        return cost_time


def torch_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, list):
        return np.array(tensor)
    else:
        return tensor


class Visualizer:
    def __init__(self):
        self.fig = go.Figure()

    def add_mesh(
        self, vertices, triangles, data=None, opacity=1.0, cmax=None, cmin=None
    ):
        vertices, triangles, data = [
            torch_to_numpy(x) for x in [vertices, triangles, data]
        ]
        # Add traces, one for each slider step
        if data is None:
            self.fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=triangles[:, 0],
                    j=triangles[:, 1],
                    k=triangles[:, 2],
                    opacity=opacity,
                    visible=True,  # Mesh is always visible
                    name="",  # Don't show legend for mesh
                    showlegend=False,
                    showscale=False,
                )
            )
        else:
            if cmax is None or cmin is None:
                cmax = data.max()
                cmin = data.min()
            print("cmin = ", cmin, "cmax = ", cmax)
            self.fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=triangles[:, 0],
                    j=triangles[:, 1],
                    k=triangles[:, 2],
                    colorscale="Viridis",
                    intensity=data,
                    intensitymode=(
                        "cell" if data.shape[0] == triangles.shape[0] else "vertex"
                    ),
                    name="",
                    opacity=opacity,
                )
            )
        return self

    def add_points(
        self,
        coords,
        data=None,
        point_size=5,
        showscale=True,
        cmax=None,
        cmin=None,
        opacity=1.0,
    ):
        coords = coords.reshape(-1, 3)
        coords, data = [torch_to_numpy(x) for x in [coords, data]]
        if data is None:
            data = np.ones(len(coords))
        else:
            data = data.reshape(-1)

        coords = coords.reshape(-1, 3)
        if cmax is None or cmin is None:
            cmax = data.max()
            cmin = data.min()
        print("cmin = ", cmin, "cmax = ", cmax)
        self.fig.add_trace(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=data,
                    colorscale="Viridis",  # choose a colorscale
                    opacity=opacity,
                    cmax=cmax,
                    cmin=cmin,
                    colorbar=dict(title="") if showscale else None,
                ),
                name="",
            )
        )
        return self

    def show(self, grid=True):
        self.fig.update_layout(
            {
                "scene": {
                    "xaxis": {"visible": grid},
                    "yaxis": {"visible": grid},
                    "zaxis": {"visible": grid},
                },
                "scene_aspectmode": "data",
            }
        )
        self.fig.show()


def normalize_mesh(vertices):
    vertices = vertices - vertices.mean(0)
    vertices = vertices / vertices.abs().max()
    return vertices


def get_triangle_centers(vertices, triangles):
    return vertices[triangles].mean(1)


def get_triangle_normals(vertices, triangles):
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    normals = normals / torch.norm(normals, dim=1, keepdim=True)
    return normals


import numpy as np


def calculate_bin_frequencies(n_fft=128, sampling_rate=16000):
    # For a one-sided spectrogram, there are n_fft//2 + 1 bins
    num_bins = n_fft // 2 + 1
    return [bin_number * sampling_rate / n_fft for bin_number in range(num_bins)]


import matplotlib.pyplot as plt

# plt.imshow(nc_spec)
# plt.show()


def apply_spec_mask_to_audio(
    audio_id,
    mask_spec,
    frame_num,
    animation_frame_rate=30,
    trg_sample_rate=16000,
    n_fft=256,
    save_spec=False,
    save_path='dataset/spectrogram'
):
    from scipy.io import wavfile
    import librosa
    import torchaudio
    import torch
    from scipy.interpolate import RegularGridInterpolator

    sample_rate, data = wavfile.read(f"dataset/audio/{audio_id}.wav")
    data = data[:, 0]
    data = (data / 32768).astype(np.float32)
    data = librosa.resample(data, orig_sr=sample_rate, target_sr=trg_sample_rate)
    sample_rate = trg_sample_rate
    require_audio_frame = int(frame_num / animation_frame_rate * sample_rate)
    if len(data) < require_audio_frame:
        data_repeat_num = require_audio_frame // len(data) + 1
        data = np.tile(data, data_repeat_num)
    data = data[: int(frame_num / animation_frame_rate * sample_rate)]
    torch_data = torch.from_numpy(data)
    spec_audio = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=n_fft // 2, power=None
    )(torch_data.unsqueeze(0)).squeeze(0)

    def plot(spec, idx, title):
        def plot_spec(ax, spec, title):
            ax.set_title(title)
            ax.imshow(librosa.amplitude_to_db(spec), origin="lower", aspect="auto")
            plt.axis("off")

        ax = plt.subplot(3, 1, idx)
        plot_spec(ax, torch.abs(spec), title)

    plot(spec_audio, 1, "original")

    spec_audio = spec_audio.cpu().numpy()
    x = np.linspace(0, 1, mask_spec.shape[1])
    y = np.linspace(0, 1, mask_spec.shape[0])
    new_x = np.linspace(0, 1, spec_audio.shape[1])
    new_y = np.linspace(0, 1, spec_audio.shape[0])
    interp_function = RegularGridInterpolator((y, x), mask_spec)
    new_y, new_x = np.meshgrid(new_y, new_x, indexing="ij")
    new_points = np.array([new_y.ravel(), new_x.ravel()]).T
    interped_nc_spec = interp_function(new_points).reshape(*spec_audio.shape)
    spec_audio = spec_audio * interped_nc_spec
    spec_audio = torch.from_numpy(spec_audio)
# print("interped_nc_spec:", interped_nc_spec)
    plot(torch.from_numpy(interped_nc_spec), 2, "mask")
    plot(spec_audio, 3, "masked")
    if save_spec:
        plt.savefig(f"{save_path}/{audio_id}_spectrogram.png")
    # plt.show()

    audio = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft)(
        spec_audio.unsqueeze(0)
    ).squeeze(0)
    audio = audio.numpy()
    audio = audio / np.abs(audio).max() * 0.9
    audio = (audio * 32768).astype(np.int16)
    return audio

def apply_spec_mask_to_audio_stereo(
    audio_id,
    mask_spec,
    frame_num,
    animation_frame_rate=30,
    trg_sample_rate=16000,
    n_fft=256,
    save_spec=False,
    save_path='dataset/spectrogram'
):
    from scipy.io import wavfile
    import librosa
    import torchaudio
    import torch
    from scipy.interpolate import RegularGridInterpolator

    sample_rate, data = wavfile.read(f"dataset/audio/{audio_id}.wav")
    data = (data / 32768).astype(np.float32)
    data = librosa.resample(data, orig_sr=sample_rate, target_sr=trg_sample_rate, axis=0)
    sample_rate = trg_sample_rate
    require_audio_frame = int(frame_num / animation_frame_rate * sample_rate)
    if data.shape[0] < require_audio_frame:
        data_repeat_num = require_audio_frame // data.shape[0] + 1
        data = np.tile(data, (data_repeat_num, 1))
    data = data[: int(frame_num / animation_frame_rate * sample_rate), :]
    torch_data = torch.from_numpy(data.T)
    spec_audio = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=n_fft // 2, power=None
    )(torch_data)

    def plot(spec, idx, title):
        def plot_spec(ax, spec, title):
            ax.set_title(title)
            ax.imshow(librosa.amplitude_to_db(spec), origin="lower", aspect="auto")
            plt.axis("off")

        ax = plt.subplot(3, 1, idx)
        plot_spec(ax, torch.abs(spec), title)

    plot(spec_audio[0], 1, "original left")
    plot(spec_audio[1], 2, "original right")

    spec_audio = spec_audio.cpu().numpy()
    x = np.linspace(0, 1, mask_spec.shape[1])
    y = np.linspace(0, 1, mask_spec.shape[0])
    new_x = np.linspace(0, 1, spec_audio.shape[2])
    new_y = np.linspace(0, 1, spec_audio.shape[1])
    interp_function = RegularGridInterpolator((y, x), mask_spec)
    new_y, new_x = np.meshgrid(new_y, new_x, indexing="ij")
    new_points = np.array([new_y.ravel(), new_x.ravel()]).T
    interped_nc_spec = interp_function(new_points).reshape(spec_audio.shape[1:])
    spec_audio = spec_audio * interped_nc_spec
    spec_audio = torch.from_numpy(spec_audio)
    plot(torch.from_numpy(interped_nc_spec), 3, "mask")

    if save_spec:
        plt.savefig(f"{save_path}/{audio_id}_spectrogram.png")
    # plt.show()

    audio = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft)(
        spec_audio
    ).squeeze(1)
    audio = audio.numpy()
    audio = audio / np.abs(audio).max(axis=1, keepdims=True) * 0.9
    audio = (audio * 32768).astype(np.int16).T
    return audio