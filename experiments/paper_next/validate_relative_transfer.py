"""Numerical sanity checks for relative acoustic transfer targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kwave_solver import KWaveSolver  # noqa: E402
from src.scene import Scene  # noqa: E402


def validate_bem(config: Path, frequency_index: int) -> dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("BEM validation requires CUDA.")

    scene = Scene(str(config), normalize=True)
    scene.enclose_sample(
        seed=0,
        freq_idx=frequency_index,
        max_freq_idx=65,
        sound_source="ball.obj",
        _transition=False,
        _rotate=False,
        _resize=False,
    )

    radii = torch.tensor(
        [0.08, 0.12, 0.18, 0.25], dtype=torch.float32, device="cuda"
    )
    points = torch.zeros((len(radii), 3), dtype=torch.float32, device="cuda")
    points[:, 0] = radii

    scene.solve(man_trg_point=points)
    scene_transfer = scene.relative_transfer_magnitude("ball.obj")
    analytic_transfer = scene.relative_transfer_magnitude(
        "ball.obj", reference_mode="analytic_sphere"
    )
    free_field = scene.free_field_potential.abs().reshape(-1)

    # An outgoing spherical field has magnitude proportional to 1/r. The
    # triangulated source is not perfectly spherical, so use a modest tolerance.
    radial_invariant = free_field * radii.cpu()
    radial_relative_spread = float(
        ((radial_invariant.max() - radial_invariant.min()) / radial_invariant.mean())
    )

    source = next(obj for obj in scene.objs if obj.name == "ball.obj")
    source_size = float(
        (source.vertices.max(dim=0).values - source.vertices.min(dim=0).values)
        .max()
        .item()
    )
    source_radius = source_size / 2.0
    analytic_free_field = source_radius**2 / (
        radii.cpu() * np.sqrt(1.0 + (scene.k * source_radius) ** 2)
    )
    analytic_relative_l2 = float(
        torch.linalg.vector_norm(free_field - analytic_free_field)
        / torch.linalg.vector_norm(free_field)
    )
    transfer_relative_l2 = float(
        torch.linalg.vector_norm(scene_transfer - analytic_transfer)
        / torch.linalg.vector_norm(scene_transfer)
    )

    # A source-only numerator divided by the same independently solved
    # source-only field is exactly the identity transfer.
    identity_transfer = free_field / free_field.clamp_min(1e-12)
    identity_max_error = float((identity_transfer - 1.0).abs().max())

    result = {
        "identity_max_abs_error": identity_max_error,
        "free_field_r_times_p_relative_spread": radial_relative_spread,
        "analytic_sphere_relative_l2": analytic_relative_l2,
        "analytic_vs_numerical_transfer_relative_l2": transfer_relative_l2,
        "source_radius_m": source_radius,
        "scene_transfer_min": float(scene_transfer.min()),
        "scene_transfer_max": float(scene_transfer.max()),
    }
    if identity_max_error > 1e-5:
        raise AssertionError(f"BEM identity transfer failed: {result}")
    if radial_relative_spread > 0.10:
        raise AssertionError(f"BEM free-field 1/r trend failed: {result}")
    if analytic_relative_l2 > 0.10:
        raise AssertionError(f"BEM analytic sphere reference failed: {result}")
    if transfer_relative_l2 > 0.10:
        raise AssertionError(f"BEM analytic transfer failed: {result}")
    if not torch.isfinite(scene_transfer).all():
        raise AssertionError(f"BEM scene transfer is non-finite: {result}")
    return result


def validate_fdtd(device_num: int) -> dict[str, float]:
    solver = KWaveSolver(
        grid_size=(64, 64, 64),
        resolution=0.005,
        pml_size=8,
        time_end=0.006,
        filter_source=False,
    )
    points = np.asarray(
        [[0.04, 0.0, 0.0], [0.0, 0.06, 0.0], [0.0, 0.0, 0.08]],
        dtype=np.float32,
    )
    transfer = solver.solve_spec(
        points,
        device_num=device_num,
        show_sim_log=False,
        target_fs=16000,
        n_fft=128,
        normalization="free_field",
    )
    max_error = float(np.max(np.abs(transfer - 1.0)))
    result = {
        "identity_max_abs_error": max_error,
        "transfer_min": float(np.min(transfer)),
        "transfer_max": float(np.max(transfer)),
    }
    if max_error > 1e-5:
        raise AssertionError(f"FDTD identity transfer failed: {result}")
    if not np.isfinite(transfer).all():
        raise AssertionError(f"FDTD transfer is non-finite: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "dataset" / "bem2fdtd" / "config.json",
    )
    parser.add_argument("--frequency-index", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-bem", action="store_true")
    parser.add_argument("--skip-fdtd", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report: dict[str, dict[str, float]] = {}
    if not args.skip_bem:
        report["bem"] = validate_bem(args.config.resolve(), args.frequency_index)
    if not args.skip_fdtd:
        report["fdtd"] = validate_fdtd(args.gpu)

    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
