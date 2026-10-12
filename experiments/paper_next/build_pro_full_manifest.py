"""Build the immutable, mesh-disjoint manifest for the full Pro dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "experiments" / "paper_next" / "pro_relative_full_manifest.json"
DEFAULT_SEED = 20260924

CATEGORY_CONFIG = {
    "general": {
        "solver": "bem",
        "root": ROOT / "dataset" / "fix" / "my_obstacles",
        "scene_count": 768,
        "query_count": 1000,
    },
    "enclosed": {
        "solver": "bem",
        "root": ROOT / "dataset" / "fix" / "enclosed_obstacles",
        "scene_count": 1356,
        "query_count": 1000,
    },
    "teapot": {
        "solver": "fdtd",
        "root": ROOT / "dataset" / "teapot" / "teapot_3",
        "scene_count": 435,
        "query_count": 2000,
    },
}

EXCLUDED_SOURCES = {
    "dataset/fix/enclosed_obstacles/97e3b7b4688c2a93f155d75bbf62b80.obj.obj": (
        "All 22 assigned BEM scenes fail because the source mesh contains "
        "248 zero-area triangles and produces singular boundary matrices."
    )
}


def sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def inventory(category: str, root: Path) -> list[dict]:
    if category == "teapot":
        sources = []
        for path in sorted(item for item in root.iterdir() if item.is_dir()):
            files = [path / "hushen.obj", path / "gaizi.obj"]
            if not all(item.is_file() for item in files):
                raise FileNotFoundError(f"Incomplete teapot source: {path}")
            match = re.search(r"_(\d+)_", path.name)
            if not match:
                raise ValueError(f"Cannot parse teapot object ID: {path.name}")
            sources.append(
                {
                    "object_id": int(match.group(1)),
                    "source": str(path.relative_to(ROOT)),
                    "source_files": [str(item.relative_to(ROOT)) for item in files],
                    "source_sha256": sha256_files(files),
                }
            )
        return sources

    files = sorted(root.glob("*.obj"))
    return [
        {
            "object_id": index,
            "source": str(path.relative_to(ROOT)),
            "source_files": [str(path.relative_to(ROOT))],
            "source_sha256": sha256_files([path]),
        }
        for index, path in enumerate(files)
    ]


def balanced_counts(total: int, source_count: int, rng: random.Random) -> list[int]:
    base, remainder = divmod(total, source_count)
    counts = [base] * source_count
    indices = list(range(source_count))
    rng.shuffle(indices)
    for index in indices[:remainder]:
        counts[index] += 1
    return counts


def split_sources(
    sources: list[dict], val_fraction: float, rng: random.Random
) -> tuple[list[dict], list[dict]]:
    shuffled = sources.copy()
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_fraction))
    val_hashes = {item["source_sha256"] for item in shuffled[:val_count]}
    train = [item for item in sources if item["source_sha256"] not in val_hashes]
    val = [item for item in sources if item["source_sha256"] in val_hashes]
    if len(train) + len(val) != len(sources):
        raise RuntimeError("Source split lost entries.")
    return train, val


def make_scenes(
    category: str,
    solver: str,
    train_sources: list[dict],
    val_sources: list[dict],
    total_count: int,
    query_count: int,
    val_fraction: float,
    rng: random.Random,
    seed: int,
) -> list[dict]:
    val_total = round(total_count * val_fraction)
    split_specs = (
        ("train", train_sources, total_count - val_total),
        ("val", val_sources, val_total),
    )
    scenes = []
    for split, sources, split_total in split_specs:
        counts = balanced_counts(split_total, len(sources), rng)
        for source, count in zip(sources, counts):
            object_id = source["object_id"]
            prefix = "fdtd_teapot" if solver == "fdtd" else f"bem_{category}"
            for sample_id in range(count):
                stem = f"{prefix}_obj{object_id:03d}_sample{sample_id:03d}"
                scene_seed = (
                    seed
                    + (1 if split == "val" else 0) * 100_000_000
                    + object_id * 1009
                    + sample_id
                    + {"general": 0, "enclosed": 10_000_000, "teapot": 20_000_000}[
                        category
                    ]
                )
                scenes.append(
                    {
                        "stem": stem,
                        "split": split,
                        "category": category,
                        "solver": solver,
                        "object_id": object_id,
                        "sample_id": sample_id,
                        "seed": scene_seed,
                        "query_count": query_count,
                        **{
                            key: value
                            for key, value in source.items()
                            if key != "object_id"
                        },
                    }
                )
    return scenes


def build_manifest(seed: int, val_fraction: float) -> dict:
    rng = random.Random(seed)
    category_summaries = {}
    all_scenes = []
    source_split_by_hash: dict[str, str] = {}

    for category, config in CATEGORY_CONFIG.items():
        sources = inventory(category, config["root"])
        train_sources, val_sources = split_sources(sources, val_fraction, rng)
        for split, entries in (("train", train_sources), ("val", val_sources)):
            for entry in entries:
                digest = entry["source_sha256"]
                previous = source_split_by_hash.setdefault(digest, split)
                if previous != split:
                    raise ValueError(
                        f"Source hash crosses train/val: {digest} ({previous}, {split})"
                    )

        scenes = make_scenes(
            category=category,
            solver=config["solver"],
            train_sources=train_sources,
            val_sources=val_sources,
            total_count=config["scene_count"],
            query_count=config["query_count"],
            val_fraction=val_fraction,
            rng=rng,
            seed=seed,
        )
        all_scenes.extend(scenes)
    exclusions = []
    for source, reason in EXCLUDED_SOURCES.items():
        excluded_scenes = [item for item in all_scenes if item["source"] == source]
        if not excluded_scenes:
            raise ValueError(f"Configured exclusion did not match the manifest: {source}")
        exclusions.append(
            {
                "source": source,
                "source_sha256": excluded_scenes[0]["source_sha256"],
                "split": excluded_scenes[0]["split"],
                "scene_count": len(excluded_scenes),
                "reason": reason,
            }
        )
    all_scenes = [
        item for item in all_scenes if item["source"] not in EXCLUDED_SOURCES
    ]

    for category, config in CATEGORY_CONFIG.items():
        scenes = [item for item in all_scenes if item["category"] == category]
        category_summaries[category] = {
            "solver": config["solver"],
            "source_meshes": len({item["source_sha256"] for item in scenes}),
            "train_meshes": len(
                {
                    item["source_sha256"]
                    for item in scenes
                    if item["split"] == "train"
                }
            ),
            "val_meshes": len(
                {
                    item["source_sha256"]
                    for item in scenes
                    if item["split"] == "val"
                }
            ),
            "scenes": len(scenes),
            "train_scenes": sum(item["split"] == "train" for item in scenes),
            "val_scenes": sum(item["split"] == "val" for item in scenes),
            "query_count": config["query_count"],
        }

    stems_by_split = [(item["split"], item["stem"]) for item in all_scenes]
    if len(stems_by_split) != len(set(stems_by_split)):
        raise ValueError("Manifest contains duplicate split/stem pairs.")

    return {
        "name": "pro_relative_full",
        "version": 2,
        "seed": seed,
        "target": "relative_transfer",
        "reference": {
            "bem": "analytic_sphere",
            "fdtd": "simulated_free_field",
        },
        "frequency_endpoints": "nearest_resolved_bin",
        "val_fraction": val_fraction,
        "scene_count": len(all_scenes),
        "train_scene_count": sum(item["split"] == "train" for item in all_scenes),
        "val_scene_count": sum(item["split"] == "val" for item in all_scenes),
        "categories": category_summaries,
        "exclusions": exclusions,
        "scenes": sorted(all_scenes, key=lambda item: (item["split"], item["stem"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    manifest = build_manifest(args.seed, args.val_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scene_count": manifest["scene_count"],
                "train_scene_count": manifest["train_scene_count"],
                "val_scene_count": manifest["val_scene_count"],
                "categories": manifest["categories"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
