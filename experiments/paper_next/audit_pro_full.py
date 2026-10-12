"""Audit a generated full Pro dataset against its immutable manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "experiments" / "paper_next" / "pro_relative_full_manifest.json"


def audit(dataset: Path, manifest: dict) -> dict:
    errors = []
    counts = Counter()
    source_splits: dict[str, set[str]] = defaultdict(set)
    global_min = float("inf")
    global_max = float("-inf")

    expected_pairs = {
        (
            scene["split"],
            scene["stem"],
        )
        for scene in manifest["scenes"]
    }
    actual_pairs = set()
    for split in ("train", "val"):
        meshes = {
            path.stem for path in (dataset / f"{split}_mesh").glob("*.obj")
        }
        data = {path.stem for path in (dataset / f"{split}_data").glob("*.pt")}
        for stem in sorted(meshes ^ data):
            errors.append(f"Unpaired output: {split}/{stem}")
        actual_pairs.update((split, stem) for stem in meshes & data)

    for pair in sorted(expected_pairs - actual_pairs):
        errors.append(f"Missing expected output: {pair[0]}/{pair[1]}")
    for pair in sorted(actual_pairs - expected_pairs):
        errors.append(f"Unexpected output: {pair[0]}/{pair[1]}")

    for scene in manifest["scenes"]:
        source_splits[scene["source_sha256"]].add(scene["split"])
        split = scene["split"]
        stem = scene["stem"]
        counts[(split, scene["category"], scene["solver"])] += 1
        data_path = dataset / f"{split}_data" / f"{stem}.pt"
        if not data_path.is_file():
            continue
        try:
            sample = torch.load(data_path, map_location="cpu", weights_only=False)
            x = sample["x"]
            y = sample["y"]
            expected_x = (scene["query_count"], 3)
            expected_y = (scene["query_count"], 65)
            if tuple(x.shape) != expected_x:
                errors.append(f"{split}/{stem}: x shape {tuple(x.shape)} != {expected_x}")
            if tuple(y.shape) != expected_y:
                errors.append(f"{split}/{stem}: y shape {tuple(y.shape)} != {expected_y}")
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                errors.append(f"{split}/{stem}: non-finite values")
            if torch.any(y < 0):
                errors.append(f"{split}/{stem}: negative transfer magnitude")
            if not torch.equal(y[:, 0], y[:, 1]) or not torch.equal(
                y[:, -1], y[:, -2]
            ):
                errors.append(f"{split}/{stem}: endpoint convention mismatch")
            expected_metadata = {
                "solver": scene["solver"],
                "target": "relative_transfer",
                "source_sha256": scene["source_sha256"],
                "frequency_endpoints": "nearest_resolved_bin",
            }
            for key, expected in expected_metadata.items():
                if sample.get(key) != expected:
                    errors.append(
                        f"{split}/{stem}: {key}={sample.get(key)!r}, expected {expected!r}"
                    )
            global_min = min(global_min, float(y.min()))
            global_max = max(global_max, float(y.max()))
        except Exception as error:
            errors.append(f"{split}/{stem}: failed to read: {error!r}")

    cross_split_hashes = sorted(
        digest for digest, splits in source_splits.items() if len(splits) > 1
    )
    if cross_split_hashes:
        errors.append(
            f"{len(cross_split_hashes)} source mesh hashes cross train/val"
        )

    return {
        "dataset": str(dataset.resolve()),
        "manifest_name": manifest["name"],
        "expected_scenes": len(expected_pairs),
        "paired_scenes": len(actual_pairs & expected_pairs),
        "unexpected_pairs": len(actual_pairs - expected_pairs),
        "counts": [
            {
                "split": split,
                "category": category,
                "solver": solver,
                "count": count,
            }
            for (split, category, solver), count in sorted(counts.items())
        ],
        "source_meshes": {
            "train": len(
                {
                    scene["source_sha256"]
                    for scene in manifest["scenes"]
                    if scene["split"] == "train"
                }
            ),
            "val": len(
                {
                    scene["source_sha256"]
                    for scene in manifest["scenes"]
                    if scene["split"] == "val"
                }
            ),
        },
        "cross_split_source_hashes": cross_split_hashes,
        "global_y_min": None if global_min == float("inf") else global_min,
        "global_y_max": None if global_max == float("-inf") else global_max,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = audit(args.dataset, manifest)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
