"""Generate the versioned Pro relative-transfer test set without overwrites."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kwave_solver import KWaveSolver  # noqa: E402
from src.scene import Scene  # noqa: E402


LEGACY_STEM = re.compile(
    r"^out_(?P<worker>\d+)_(?P<object>\d+)_(?P<sample>\d+)$"
)


def find_template(mesh_dir: Path, object_id: int) -> Path:
    candidates = []
    for path in mesh_dir.glob("*.obj"):
        match = LEGACY_STEM.match(path.stem)
        if match and int(match.group("object")) == object_id:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No object {object_id} template found in {mesh_dir}"
        )
    return sorted(candidates)[0]


def prepare_bem_scene(
    work_dir: Path, template: Path, source_mesh: Path, query_count: int
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_mesh, work_dir / "ball.obj")
    shutil.copy2(template, work_dir / "obstacle.obj")
    config = {
        "obj": [
            {
                "mesh": "ball.obj",
                "size": 0.02,
                "position": [0, 0, 0],
                "vibration": "w",
            },
            {
                "mesh": "obstacle.obj",
                "size": 0.2,
                "position": [0, 0, 0],
            },
        ],
        "solver": {
            "src_sample_num": 1,
            "trg_sample_num": query_count,
            "freq_min": 100,
            "freq_max": 8000,
            "trg_pos_min": [-0.2, -0.2, -0.2],
            "trg_pos_max": [0.2, 0.2, 0.2],
        },
    }
    config_path = work_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def generate_bem(
    category: str,
    object_id: int,
    sample_id: int,
    template: Path,
    source_mesh: Path,
    output: Path,
    query_count: int,
    seed: int,
) -> tuple[Path, Path]:
    stem = f"bem_{category}_obj{object_id:03d}_sample{sample_id:03d}"
    data_path = output / "test_data" / f"{stem}.pt"
    mesh_path = output / "test_mesh" / f"{stem}.obj"
    work_dir = output / "generation_metadata" / stem
    config_path = prepare_bem_scene(
        work_dir, template.resolve(), source_mesh.resolve(), query_count
    )

    scene = Scene(str(config_path))
    x = torch.zeros((query_count, 3), dtype=torch.float32)
    y = torch.zeros((query_count, 65), dtype=torch.float32)
    scene_seed = seed + object_id * 1009 + sample_id

    for frequency_index in range(65):
        if category == "general":
            scene.my_sample(
                seed=scene_seed,
                freq_idx=frequency_index,
                max_freq_idx=65,
                sound_source="ball.obj",
            )
        elif category == "enclosed":
            scene.enclose_sample(
                seed=scene_seed,
                freq_idx=frequency_index,
                max_freq_idx=65,
                sound_source="ball.obj",
            )
        else:
            raise ValueError(f"Unsupported BEM category: {category}")

        scene.solve()
        x.copy_(scene.trg_points.cpu())
        y[:, frequency_index] = scene.relative_transfer_magnitude(
            "ball.obj", reference_mode="analytic_sphere"
        )

    torch.save(
        {
            "x": x,
            "y": y,
            "solver": "bem",
            "target": "relative_transfer",
            "reference": "analytic_sphere",
            "category": category,
            "object_id": object_id,
            "sample_id": sample_id,
            "seed": scene_seed,
            "source_template": str(template.resolve()),
        },
        data_path,
    )
    scene.get_obstacle_mesh().export(mesh_path)
    return mesh_path, data_path


def sample_listener_points(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.uniform(0.4, 0.8, size=(count, 1))
    return (directions * radii).astype(np.float32)


def configure_runtime_library_path() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    lib_dir = str(Path(conda_prefix) / "lib")
    current = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        f"{lib_dir}:{current}" if current else lib_dir
    )


def generate_fdtd(
    object_id: int,
    sample_id: int,
    template: Path,
    output: Path,
    query_count: int,
    seed: int,
    device_num: int,
) -> tuple[Path, Path]:
    configure_runtime_library_path()
    stem = f"fdtd_teapot_obj{object_id:03d}_sample{sample_id:03d}"
    data_path = output / "test_data" / f"{stem}.pt"
    mesh_path = output / "test_mesh" / f"{stem}.obj"
    scene_seed = seed + object_id * 1009 + sample_id

    mesh = trimesh.load_mesh(template, process=False)
    solver = KWaveSolver(resolution=0.005, pml_size=5, time_end=0.03)
    solver.add_object(mesh)
    x = sample_listener_points(query_count, scene_seed)
    y = solver.solve_spec(
        x,
        device_num=device_num,
        show_sim_log=False,
        target_fs=16000,
        n_fft=128,
        normalization="free_field",
    )
    torch.save(
        {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "solver": "fdtd",
            "target": "relative_transfer",
            "reference": "simulated_free_field",
            "category": "teapot",
            "object_id": object_id,
            "sample_id": sample_id,
            "seed": scene_seed,
            "source_template": str(template.resolve()),
        },
        data_path,
    )
    mesh.export(mesh_path)
    return mesh_path, data_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category", choices=("general", "enclosed", "teapot"), required=True
    )
    parser.add_argument("--object-id", type=int, required=True)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--template-mesh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-mesh",
        type=Path,
        default=ROOT / "dataset" / "fix" / "ball.obj",
    )
    parser.add_argument("--query-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--device-num", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "test_mesh").mkdir(exist_ok=True)
    (args.output / "test_data").mkdir(exist_ok=True)
    (args.output / "generation_metadata").mkdir(exist_ok=True)

    template = find_template(args.template_mesh_dir, args.object_id)
    prefix = "fdtd_teapot" if args.category == "teapot" else f"bem_{args.category}"
    stem = f"{prefix}_obj{args.object_id:03d}_sample{args.sample_id:03d}"
    destinations = [
        args.output / "test_mesh" / f"{stem}.obj",
        args.output / "test_data" / f"{stem}.pt",
    ]
    if not args.overwrite and any(path.exists() for path in destinations):
        raise FileExistsError(
            f"Refusing to overwrite existing sample {stem}; use --overwrite."
        )

    if args.category == "teapot":
        generated = generate_fdtd(
            args.object_id,
            args.sample_id,
            template,
            args.output,
            args.query_count,
            args.seed,
            args.device_num,
        )
    else:
        generated = generate_bem(
            args.category,
            args.object_id,
            args.sample_id,
            template,
            args.source_mesh,
            args.output,
            args.query_count,
            args.seed,
        )
    print("\n".join(str(path) for path in generated))


if __name__ == "__main__":
    main()
