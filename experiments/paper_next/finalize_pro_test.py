"""Apply the frozen endpoint convention to an existing Pro test dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    data_dir = args.dataset / "test_data"
    paths = sorted(data_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No test .pt files found in {data_dir}")

    changed = 0
    for path in paths:
        sample = torch.load(path, map_location="cpu", weights_only=False)
        if sample.get("target") != "relative_transfer":
            raise ValueError(f"Unexpected target in {path}: {sample.get('target')}")
        y = sample["y"]
        if y.ndim != 2 or y.shape[1] != 65:
            raise ValueError(f"Expected y shape (N, 65), got {tuple(y.shape)} in {path}")

        endpoint_summary = {
            "original_bin0_min": float(y[:, 0].min()),
            "original_bin0_max": float(y[:, 0].max()),
            "original_bin64_min": float(y[:, -1].min()),
            "original_bin64_max": float(y[:, -1].max()),
        }
        y[:, 0] = y[:, 1]
        y[:, -1] = y[:, -2]
        sample["y"] = y
        sample["frequency_endpoints"] = "nearest_resolved_bin"
        sample["discarded_endpoint_summary"] = endpoint_summary

        if args.apply:
            temporary = path.with_suffix(".pt.tmp")
            torch.save(sample, temporary)
            os.replace(temporary, path)
        changed += 1
        print(f"{'updated' if args.apply else 'would update'} {path.name}")

    print(f"samples={changed} apply={args.apply}")


if __name__ == "__main__":
    main()
