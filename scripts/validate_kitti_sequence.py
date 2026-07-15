#!/usr/bin/env python3
"""Validate a KITTI odometry sequence and optionally create a manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from loop_closure.datasets.kitti_odometry import (
    discover_images,
    load_poses,
    load_timestamps,
    validate_sequence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one KITTI odometry sequence."
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("KITTI_ODOMETRY_ROOT"),
        help=(
            "KITTI dataset root. Defaults to the "
            "KITTI_ODOMETRY_ROOT environment variable."
        ),
    )
    parser.add_argument("--sequence", default="05")
    parser.add_argument("--camera", default="image_0")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional output CSV containing frame metadata.",
    )
    return parser.parse_args()


def write_manifest(
    dataset_root: Path,
    sequence: str,
    camera: str,
    output_path: Path,
) -> None:
    sequence = sequence.zfill(2)
    image_directory = dataset_root / "sequences" / sequence / camera

    images = discover_images(image_directory)
    poses = load_poses(dataset_root / "poses" / f"{sequence}.txt")
    timestamps = load_timestamps(
        dataset_root / "sequences" / sequence / "times.txt"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "dataset",
                "sequence",
                "camera",
                "frame_id",
                "image_path",
                "timestamp_seconds",
                "x",
                "y",
                "z",
            ],
        )
        writer.writeheader()

        for index, image_path in enumerate(images):
            translation = poses[index, :, 3]

            writer.writerow(
                {
                    "dataset": "KITTI_odometry",
                    "sequence": sequence,
                    "camera": camera,
                    "frame_id": image_path.stem,
                    "image_path": image_path.relative_to(dataset_root).as_posix(),
                    "timestamp_seconds": float(timestamps[index]),
                    "x": float(translation[0]),
                    "y": float(translation[1]),
                    "z": float(translation[2]),
                }
            )


def main() -> int:
    args = parse_args()

    if not args.root:
        raise SystemExit(
            "KITTI root not provided. Use --root or define "
            "KITTI_ODOMETRY_ROOT."
        )

    root = Path(args.root).expanduser().resolve()

    summary = validate_sequence(
        root=root,
        sequence=args.sequence,
        camera=args.camera,
    )

    print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))

    if args.manifest:
        write_manifest(
            dataset_root=root,
            sequence=summary.sequence,
            camera=summary.camera,
            output_path=args.manifest,
        )
        print(f"Manifest written to: {args.manifest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
