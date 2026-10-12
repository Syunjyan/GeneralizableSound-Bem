"""Generate the manifest-defined full Pro relative-transfer dataset."""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "experiments" / "paper_next" / "pro_relative_full_manifest.json"
DEFAULT_OUTPUT = Path(
    "/data5/houxinyun/GeneralizableSound/data/pro_relative_full.staging"
)
SOURCE_MESH = ROOT / "dataset" / "fix" / "ball.obj"


def write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def output_paths(output: Path, scene: dict) -> tuple[Path, Path]:
    split = scene["split"]
    stem = scene["stem"]
    return (
        output / f"{split}_mesh" / f"{stem}.obj",
        output / f"{split}_data" / f"{stem}.pt",
    )


def sample_is_complete(output: Path, scene: dict) -> bool:
    mesh_path, data_path = output_paths(output, scene)
    if not mesh_path.is_file() or mesh_path.stat().st_size == 0 or not data_path.is_file():
        return False
    try:
        import torch

        sample = torch.load(data_path, map_location="cpu", weights_only=False)
        x = sample["x"]
        y = sample["y"]
        return bool(
            tuple(x.shape) == (scene["query_count"], 3)
            and tuple(y.shape) == (scene["query_count"], 65)
            and torch.isfinite(x).all()
            and torch.isfinite(y).all()
            and (y >= 0).all()
            and sample.get("solver") == scene["solver"]
            and sample.get("target") == "relative_transfer"
            and sample.get("source_sha256") == scene["source_sha256"]
            and sample.get("frequency_endpoints") == "nearest_resolved_bin"
        )
    except Exception:
        return False


def atomic_save(sample: dict, mesh, output: Path, scene: dict) -> None:
    import torch

    mesh_path, data_path = output_paths(output, scene)
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    temp_mesh = mesh_path.with_name(f"{mesh_path.stem}.tmp.obj")
    temp_data = data_path.with_name(f"{data_path.name}.tmp")
    for path in (temp_mesh, temp_data):
        path.unlink(missing_ok=True)

    x = sample["x"]
    y = sample["y"]
    if tuple(x.shape) != (scene["query_count"], 3):
        raise ValueError(f"Unexpected x shape for {scene['stem']}: {tuple(x.shape)}")
    if tuple(y.shape) != (scene["query_count"], 65):
        raise ValueError(f"Unexpected y shape for {scene['stem']}: {tuple(y.shape)}")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise FloatingPointError(f"Non-finite output for {scene['stem']}")
    if torch.any(y < 0):
        raise FloatingPointError(f"Negative transfer magnitude for {scene['stem']}")
    if not torch.equal(y[:, 0], y[:, 1]) or not torch.equal(y[:, -1], y[:, -2]):
        raise ValueError(f"Endpoint convention violated for {scene['stem']}")

    mesh.export(temp_mesh)
    torch.save(sample, temp_data)
    os.replace(temp_mesh, mesh_path)
    os.replace(temp_data, data_path)


def prepare_bem_config(work_dir: Path, source_template: Path, query_count: int) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_MESH, work_dir / "ball.obj")
    shutil.copy2(source_template, work_dir / "obstacle.obj")
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


def generate_bem(output: Path, scene_spec: dict):
    import torch

    from src.scene import Scene

    source_template = ROOT / scene_spec["source"]
    work_dir = output / "_work" / scene_spec["stem"]
    config_path = prepare_bem_config(
        work_dir, source_template, scene_spec["query_count"]
    )
    scene = Scene(str(config_path))
    x = torch.zeros((scene_spec["query_count"], 3), dtype=torch.float32)
    y = torch.zeros((scene_spec["query_count"], 65), dtype=torch.float32)

    for frequency_index in range(1, 64):
        if scene_spec["category"] == "general":
            scene.my_sample(
                seed=scene_spec["seed"],
                freq_idx=frequency_index,
                max_freq_idx=65,
                sound_source="ball.obj",
            )
        else:
            scene.enclose_sample(
                seed=scene_spec["seed"],
                freq_idx=frequency_index,
                max_freq_idx=65,
                sound_source="ball.obj",
            )
        scene.solve()
        x.copy_(scene.trg_points.cpu())
        y[:, frequency_index] = scene.relative_transfer_magnitude(
            "ball.obj", reference_mode="analytic_sphere"
        )

    y[:, 0] = y[:, 1]
    y[:, -1] = y[:, -2]
    obstacle = scene.get_obstacle_mesh()
    sample = {
        "x": x,
        "y": y,
        "solver": "bem",
        "target": "relative_transfer",
        "reference": "analytic_sphere",
        "category": scene_spec["category"],
        "object_id": scene_spec["object_id"],
        "sample_id": scene_spec["sample_id"],
        "seed": scene_spec["seed"],
        "source_template": scene_spec["source"],
        "source_sha256": scene_spec["source_sha256"],
        "frequency_endpoints": "nearest_resolved_bin",
    }
    atomic_save(sample, obstacle, output, scene_spec)
    shutil.rmtree(work_dir, ignore_errors=True)


def listener_points(count: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed + 17)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.uniform(0.4, 0.8, size=(count, 1))
    return (directions * radii).astype(np.float32)


def generate_fdtd(output: Path, scene_spec: dict):
    import numpy as np
    import torch
    import trimesh

    from kwave_solver import KWaveSolver
    from scripts.teapot_sample import sample as sample_teapot

    source_dir = ROOT / scene_spec["source"]
    body = trimesh.load_mesh(source_dir / "hushen.obj", process=False)
    lid = trimesh.load_mesh(source_dir / "gaizi.obj", process=False)
    rng = np.random.default_rng(scene_spec["seed"])
    target_size = 0.2 * (1.0 + float(rng.random()) * 0.5)
    body, lid = sample_teapot(
        body,
        lid,
        seed=scene_spec["seed"],
        body_target_size=target_size,
    )
    obstacle = trimesh.util.concatenate([body, lid])
    solver = KWaveSolver(resolution=0.005, pml_size=5, time_end=0.03)
    solver.add_object(obstacle)
    x = listener_points(scene_spec["query_count"], scene_spec["seed"])
    y = solver.solve_spec(
        x,
        device_num=0,
        show_sim_log=False,
        target_fs=16000,
        n_fft=128,
        normalization="free_field",
    )
    sample = {
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(y),
        "solver": "fdtd",
        "target": "relative_transfer",
        "reference": "simulated_free_field",
        "category": "teapot",
        "object_id": scene_spec["object_id"],
        "sample_id": scene_spec["sample_id"],
        "seed": scene_spec["seed"],
        "source_template": scene_spec["source"],
        "source_sha256": scene_spec["source_sha256"],
        "frequency_endpoints": "nearest_resolved_bin",
    }
    atomic_save(sample, obstacle, output, scene_spec)


def configure_worker(gpu_id: int, output: Path) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
    cache = output / "_kwave_cache" / f"gpu_{gpu_id}"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(cache)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        lib_dir = str(Path(conda_prefix) / "lib")
        current = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{current}" if current else lib_dir
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def worker(gpu_id: int, jobs: list[dict], output_text: str) -> None:
    output = Path(output_text)
    configure_worker(gpu_id, output)
    log_path = output / "_logs" / f"gpu_{gpu_id}.jsonl"
    for scene_spec in jobs:
        started = time.time()
        record = {
            "stem": scene_spec["stem"],
            "split": scene_spec["split"],
            "category": scene_spec["category"],
            "gpu": gpu_id,
        }
        try:
            if sample_is_complete(output, scene_spec):
                write_jsonl(
                    log_path, {**record, "status": "skipped", "seconds": 0.0}
                )
                continue
            if scene_spec["solver"] == "bem":
                generate_bem(output, scene_spec)
            else:
                generate_fdtd(output, scene_spec)
            write_jsonl(
                log_path,
                {
                    **record,
                    "status": "completed",
                    "seconds": round(time.time() - started, 3),
                },
            )
        except Exception as error:
            write_jsonl(
                log_path,
                {
                    **record,
                    "status": "failed",
                    "seconds": round(time.time() - started, 3),
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
            )
        finally:
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def detect_idle_gpus(limit: int) -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    candidates = []
    for line in result.stdout.splitlines():
        index, memory, power = [value.strip() for value in line.split(",")]
        candidates.append((float(power), float(memory), int(index)))
    candidates.sort()
    return [index for _, _, index in candidates[:limit]]


def select_jobs(manifest: dict, args) -> list[dict]:
    categories = set(args.categories)
    splits = set(args.splits)
    selected = [
        scene
        for scene in manifest["scenes"]
        if scene["category"] in categories and scene["split"] in splits
    ]
    if args.limit_per_category is None:
        return selected

    counts: dict[tuple[str, str], int] = {}
    limited = []
    for scene in selected:
        key = (scene["split"], scene["category"])
        if counts.get(key, 0) >= args.limit_per_category:
            continue
        limited.append(scene)
        counts[key] = counts.get(key, 0) + 1
    return limited


def summarize_output(output: Path, jobs: list[dict]) -> dict:
    complete = [scene for scene in jobs if sample_is_complete(output, scene)]
    return {
        "selected": len(jobs),
        "complete": len(complete),
        "incomplete": len(jobs) - len(complete),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpus", type=int, nargs="*")
    parser.add_argument("--max-gpus", type=int, default=4)
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=("general", "enclosed", "teapot"),
        default=("general", "enclosed", "teapot"),
    )
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val"), default=("train", "val")
    )
    parser.add_argument("--limit-per-category", type=int)
    args = parser.parse_args()

    if not 1 <= args.max_gpus <= 4:
        raise ValueError("--max-gpus must be between 1 and 4.")
    gpu_ids = args.gpus or detect_idle_gpus(args.max_gpus)
    if not gpu_ids or len(gpu_ids) > 4 or len(gpu_ids) > args.max_gpus:
        raise ValueError("Select between one and four GPUs within --max-gpus.")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("GPU IDs must be unique.")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = select_jobs(manifest, args)
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (args.output / f"{split}_mesh").mkdir(exist_ok=True)
        (args.output / f"{split}_data").mkdir(exist_ok=True)
    shutil.copy2(args.manifest, args.output / "manifest.json")

    pending = [scene for scene in jobs if not sample_is_complete(args.output, scene)]
    assignments = [[] for _ in gpu_ids]
    for index, scene in enumerate(pending):
        assignments[index % len(gpu_ids)].append(scene)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "gpus": gpu_ids,
                "selected": len(jobs),
                "already_complete": len(jobs) - len(pending),
                "pending": len(pending),
            },
            indent=2,
        ),
        flush=True,
    )
    if pending:
        context = mp.get_context("spawn")
        processes = [
            context.Process(target=worker, args=(gpu, assigned, str(args.output)))
            for gpu, assigned in zip(gpu_ids, assignments)
            if assigned
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        failed_processes = [process.exitcode for process in processes if process.exitcode]
        if failed_processes:
            raise RuntimeError(f"Worker process failures: {failed_processes}")

    summary = summarize_output(args.output, jobs)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["incomplete"]:
        raise RuntimeError(f"{summary['incomplete']} selected scenes are incomplete.")


if __name__ == "__main__":
    main()
