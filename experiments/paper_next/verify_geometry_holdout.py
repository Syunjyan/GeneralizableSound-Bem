"""Verify Pro test geometries are absent from a legacy Pro training set."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


TEAPOT_TRAIN_RE = re.compile(
    r"^teapot_data_out_\d+_(?P<object>\d+)_\d+$"
)


def vertex_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dataset", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    train_mesh_dir = args.train_dataset / "train_mesh"
    test_mesh_dir = args.test_dataset / "test_mesh"
    test_data_dir = args.test_dataset / "test_data"

    train_counts = {"general": set(), "enclosed": set()}
    train_teapot_ids = set()
    for path in train_mesh_dir.glob("*.obj"):
        if path.stem.startswith("shapenet_out_"):
            train_counts["general"].add(vertex_count(path))
        elif path.stem.startswith("shapenet_bbw_out_"):
            train_counts["enclosed"].add(vertex_count(path))
        else:
            match = TEAPOT_TRAIN_RE.match(path.stem)
            if match:
                train_teapot_ids.add(int(match.group("object")))

    rows = []
    violations = []
    for mesh_path in sorted(test_mesh_dir.glob("*.obj")):
        data_path = test_data_dir / f"{mesh_path.stem}.pt"
        sample = torch.load(data_path, map_location="cpu", weights_only=False)
        category = str(sample["category"])
        object_id = int(sample["object_id"])
        count = vertex_count(mesh_path)

        if category in train_counts:
            held_out = count not in train_counts[category]
            reason = (
                "vertex_count_absent_from_training"
                if held_out
                else "vertex_count_present_in_training"
            )
        elif category == "teapot":
            held_out = object_id not in train_teapot_ids
            reason = (
                "object_id_absent_from_training"
                if held_out
                else "object_id_present_in_training"
            )
        else:
            raise ValueError(f"Unsupported category {category!r} in {data_path}")

        row = {
            "scene": mesh_path.stem,
            "category": category,
            "object_id": object_id,
            "vertices": count,
            "held_out": held_out,
            "reason": reason,
        }
        rows.append(row)
        if not held_out:
            violations.append(row)

    report = {
        "test_dataset": str(args.test_dataset.resolve()),
        "train_dataset": str(args.train_dataset.resolve()),
        "train_general_vertex_counts": len(train_counts["general"]),
        "train_enclosed_vertex_counts": len(train_counts["enclosed"]),
        "train_teapot_object_ids": sorted(train_teapot_ids),
        "scenes": rows,
        "violations": violations,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
