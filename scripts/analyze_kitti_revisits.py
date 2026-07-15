#!/usr/bin/env python3
"""Analyze potential KITTI revisits using ground-truth poses."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Callable, TypeVar

from loop_closure.datasets.kitti_odometry import load_poses
from loop_closure.datasets.kitti_revisits import analyze_revisits


T = TypeVar("T")


def parse_number_list(
    value: str,
    cast: Callable[[str], T],
) -> list[T]:
    values = [item.strip() for item in value.split(",") if item.strip()]

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one numeric value is required."
        )

    try:
        return [cast(item) for item in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid numeric list: {value}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze pose-based revisits in KITTI."
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("KITTI_ODOMETRY_ROOT"),
        help=(
            "KITTI root. Defaults to KITTI_ODOMETRY_ROOT."
        ),
    )
    parser.add_argument("--sequence", default="05")
    parser.add_argument(
        "--spatial-radii",
        default="2,5,10",
        help="Comma-separated spatial radii in meters.",
    )
    parser.add_argument(
        "--temporal-exclusions",
        default="30,50,100,200",
        help=(
            "Comma-separated minimum temporal separations "
            "in frames."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.root:
        raise SystemExit(
            "KITTI root not provided. Use --root or define "
            "KITTI_ODOMETRY_ROOT."
        )

    sequence = str(args.sequence).zfill(2)
    root = Path(args.root).expanduser().resolve()

    poses = load_poses(root / "poses" / f"{sequence}.txt")

    # KITTI vehicle trajectories are evaluated on the x-z ground plane.
    positions = poses[:, [0, 2], 3]

    radii = parse_number_list(args.spatial_radii, float)
    exclusions = parse_number_list(
        args.temporal_exclusions,
        int,
    )

    summaries = analyze_revisits(
        positions=positions,
        spatial_radii_m=radii,
        temporal_exclusions=exclusions,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "sequence",
        "coordinate_axes",
        "frame_count",
        "spatial_radius_m",
        "temporal_exclusion_frames",
        "candidate_pairs",
        "positive_pairs",
        "positive_prevalence",
        "query_frames_with_positive",
    ]

    with args.output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for summary in summaries:
            writer.writerow(
                {
                    "dataset": "KITTI_odometry",
                    "sequence": sequence,
                    "coordinate_axes": "x,z",
                    "frame_count": positions.shape[0],
                    **summary.to_dict(),
                }
            )

    print(f"Frames analyzed: {positions.shape[0]}")
    print("Coordinate axes: x,z")
    print(f"Results written to: {args.output}")

    for summary in summaries:
        print(
            f"radius={summary.spatial_radius_m:g} m, "
            f"exclusion={summary.temporal_exclusion_frames} frames, "
            f"candidates={summary.candidate_pairs}, "
            f"positives={summary.positive_pairs}, "
            f"prevalence={summary.positive_prevalence:.6f}, "
            f"queries={summary.query_frames_with_positive}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
