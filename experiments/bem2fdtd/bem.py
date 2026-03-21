import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# Ensure imports work when this script is launched from project root.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))

from kwave_solver import KWaveSolver
from src.scene import Scene


def load_config(config_path: Path):
	with open(config_path, "r", encoding="utf-8") as f:
		return json.load(f)


def validate_scene_config(config):
	obj_items = {obj["mesh"]: obj for obj in config.get("obj", [])}
	if "obstacle.obj" not in obj_items:
		raise ValueError("config.json must include obstacle.obj")
	if "ball.obj" not in obj_items:
		raise ValueError("config.json must include ball.obj as vibrating source")

	obstacle = obj_items["obstacle.obj"]
	ball = obj_items["ball.obj"]

	if float(obstacle.get("size", -1.0)) != 0.2:
		raise ValueError("obstacle.obj size must be 0.2")
	if float(ball.get("size", -1.0)) != 0.02:
		raise ValueError("ball.obj size must be 0.02")
	if ball.get("vibration") != "w":
		raise ValueError("ball.obj vibration must be 'w'")


def choose_fdtd_resolution(config, grid_size=(128, 128, 128), pml_size=10):
	"""Pick a resolution that is fine enough for geometry detail and still fits in valid region."""
	obj_items = {obj["mesh"]: obj for obj in config.get("obj", [])}
	obstacle_size = float(obj_items["obstacle.obj"]["size"])

	# Quality target: at least ~64 voxels across the obstacle's max extent.
	res_from_quality = obstacle_size / 64.0

	# Fit target: keep obstacle within ~75% of valid region (excluding PML).
	valid_n = min(grid_size) - 2 * pml_size
	if valid_n <= 0:
		raise ValueError("Invalid grid/pml setup: valid simulation region is empty.")
	res_from_fit = obstacle_size / (0.75 * valid_n)

	resolution = max(res_from_quality, res_from_fit)

	# Practical clamp for this project scale.
	resolution = float(np.clip(resolution, 0.0025, 0.0050))
	return resolution


def validate_fdtd_fit(config, resolution, grid_size=(128, 128, 128), pml_size=10):
	obj_items = {obj["mesh"]: obj for obj in config.get("obj", [])}
	obstacle_size = float(obj_items["obstacle.obj"]["size"])

	valid_len = (min(grid_size) - 2 * pml_size) * resolution
	if obstacle_size >= valid_len:
		raise ValueError(
			f"Obstacle size {obstacle_size} m does not fit valid FDTD region {valid_len:.6f} m."
		)


def run_fdtd_spec(config, data_dir: Path, receiver_world, target_fs: int = 16000, n_fft: int = 128):
	grid_size = (128, 128, 128)
	pml_size = 10
	resolution = 0.003
	validate_fdtd_fit(config, resolution, grid_size=grid_size, pml_size=pml_size)

	# Make sure k-Wave has a writable cache directory.
	tmp_dir = ROOT / "dataset" / "bem2fdtd" / "kwave_tmp"
	tmp_dir.mkdir(parents=True, exist_ok=True)
	os.environ["TMPDIR"] = str(tmp_dir)

	solver = KWaveSolver(
		grid_size=grid_size,
		resolution=resolution,
		pml_size=pml_size,
		time_end=0.03,
		filter_source=False,
	)
	for obj in config["obj"]:
		# FDTD uses an explicit grid-center source term, so do not add source geometry.
		if obj.get("mesh") == "ball.obj":
			continue
		mesh_path = data_dir / obj["mesh"]
		if not mesh_path.exists():
			raise FileNotFoundError(f"Mesh not found: {mesh_path}")
		solver.add_object(
			str(mesh_path),
			scale=float(obj["size"]),
			position=tuple(obj.get("position", [0.0, 0.0, 0.0])),
			normalize=True,
			centralize=True,
		)

	if len(solver.scene.objects) == 0:
		raise ValueError("FDTD scene has no obstacle objects. Expected obstacle.obj in config.")

	spec = solver.solve_spec(
		trg_points=[receiver_world],
		show_sim_log=False,
		target_fs=target_fs,
		n_fft=n_fft,
	)[0]
	return np.asarray(spec, dtype=np.float64)


def run_bem_spec(config_path: Path, receiver_world):
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is required by src.scene.Scene/BEM solver, but CUDA is not available.")

	scene = Scene(str(config_path), normalize=True)
	freq_bins_hz = np.asarray(scene.freq_bins, dtype=np.float64)
	receiver = torch.tensor(receiver_world, dtype=torch.float32, device="cuda").reshape(1, 3)
	spec = np.zeros(len(freq_bins_hz), dtype=np.float64)

	for freq_idx in range(len(freq_bins_hz)):
		scene.enclose_sample(
			seed=0,
			freq_idx=freq_idx,
			max_freq_idx=len(freq_bins_hz),
			sound_source="ball.obj",
			_transition=False,
			_rotate=False,
			_resize=False,
		)
		scene.solve(man_trg_point=receiver)
		spec[freq_idx] = float(scene.potential.abs().reshape(-1)[0].item())

	return freq_bins_hz, spec


def save_plot_and_metrics(out_dir: Path, freq_bins_hz, bem_spec, fdtd_spec):
	eps = 1e-12
	bem_log = np.log10(np.clip(bem_spec, eps, None))
	fdtd_log = np.log10(np.clip(fdtd_spec, eps, None))

	fig = plt.figure(figsize=(9, 5))
	ax = fig.add_subplot(111)
	ax.plot(freq_bins_hz, bem_log, linewidth=2, label="BEM log10(spec)")
	ax.plot(freq_bins_hz, fdtd_log, linewidth=2, label="FDTD log10(spec)")
	ax.set_xlabel("Frequency (Hz)")
	ax.set_ylabel("log10(spec)")
	ax.set_title("BEM vs FDTD Log Spectrum (65 bins)")
	ax.grid(True, alpha=0.3)
	ax.legend()
	fig.tight_layout()

	fig_path = out_dir / "bem_fdtd_logspec.png"
	fig.savefig(fig_path, dpi=200)
	plt.close(fig)

	metrics = {
		"bem_mean_linear": float(np.mean(bem_spec)),
		"fdtd_mean_linear": float(np.mean(fdtd_spec)),
		"bem_mean_log10": float(np.mean(bem_log)),
		"fdtd_mean_log10": float(np.mean(fdtd_log)),
		"receiver": [0.4, 0.4, 0.4],
		"source_object": "ball.obj",
		"source_vibration": "w",
		"num_freq_bins": int(len(freq_bins_hz)),
	}

	npz_path = out_dir / "bem_fdtd_spec.npz"
	np.savez(
		npz_path,
		freq_hz=freq_bins_hz,
		bem_spec=bem_spec,
		fdtd_spec=fdtd_spec,
		bem_log_spec=bem_log,
		fdtd_log_spec=fdtd_log,
	)

	metrics_path = out_dir / "bem_fdtd_metrics.json"
	with open(metrics_path, "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)

	print("Saved figure:", fig_path)
	print("Saved spectra:", npz_path)
	print("Saved metrics:", metrics_path)
	print("Average loudness (linear):")
	print("  BEM :", metrics["bem_mean_linear"])
	print("  FDTD:", metrics["fdtd_mean_linear"])
	print("Average loudness (log10):")
	print("  BEM :", metrics["bem_mean_log10"])
	print("  FDTD:", metrics["fdtd_mean_log10"])


def main():
	out_dir = ROOT / "dataset" / "bem2fdtd"
	config_path = out_dir / "config.json"
	receiver_world = (0.4, 0.4, 0.4)
	target_fs = 16000
	n_fft = 128

	if not config_path.exists():
		raise FileNotFoundError(f"Config not found: {config_path}")

	config = load_config(config_path)
	validate_scene_config(config)

	print("Running BEM...")
	freq_bins_hz, bem_spec = run_bem_spec(config_path, receiver_world)

	print("Running FDTD (k-Wave)...")
	fdtd_spec = run_fdtd_spec(config, out_dir, receiver_world, target_fs=target_fs, n_fft=n_fft)
	if len(fdtd_spec) != len(freq_bins_hz):
		raise ValueError(
			f"Frequency bin mismatch: BEM={len(freq_bins_hz)}, FDTD={len(fdtd_spec)}"
		)

	save_plot_and_metrics(out_dir, freq_bins_hz, bem_spec, fdtd_spec)


if __name__ == "__main__":
	main()
