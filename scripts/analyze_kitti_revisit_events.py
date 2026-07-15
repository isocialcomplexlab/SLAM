#!/usr/bin/env python3
"""Analyze KITTI revisit events and traversal direction."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from loop_closure.datasets.kitti_odometry import load_poses
from loop_closure.datasets.kitti_revisit_events import (
    extract_ground_plane_pose,
    summarize_revisit_events,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate KITTI positive revisit pairs and event summaries."
        )
    )

    parser.add_argument(
        "--root",
        default=os.environ.get("KITTI_ODOMETRY_ROOT"),
    )
    parser.add_argument("--sequence", default="05")
    parser.add_argument(
        "--spatial-radius",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--temporal-exclusion",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--maximum-query-gap",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--events-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def write_csv(
    output_path: Path,
    rows: list[dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(
            f"No rows available for output: {output_path}"
        )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    if not args.root:
        raise SystemExit(
            "KITTI root not provided. Use --root or define "
            "KITTI_ODOMETRY_ROOT."
        )

    root = Path(args.root).expanduser().resolve()
    sequence = str(args.sequence).zfill(2)

    poses = load_poses(root / "poses" / f"{sequence}.txt")
    positions, headings = extract_ground_plane_pose(poses)

    events, positive_pairs = summarize_revisit_events(
        positions=positions,
        headings_deg=headings,
        spatial_radius_m=args.spatial_radius,
        temporal_exclusion_frames=args.temporal_exclusion,
        maximum_query_gap=args.maximum_query_gap,
    )

    event_rows = [
        {
            "dataset": "KITTI_odometry",
            "sequence": sequence,
            "spatial_radius_m": args.spatial_radius,
            "temporal_exclusion_frames": (
                args.temporal_exclusion
            ),
            **event.to_dict(),
        }
        for event in events
    ]

    pair_rows = [
        {
            "dataset": "KITTI_odometry",
            "sequence": sequence,
            "spatial_radius_m": args.spatial_radius,
            "temporal_exclusion_frames": (
                args.temporal_exclusion
            ),
            **record,
        }
        for record in positive_pairs
    ]

    write_csv(args.events_output, event_rows)
    write_csv(args.pairs_output, pair_rows)

    print(f"Frames: {positions.shape[0]}")
    print(f"Positive pairs: {len(pair_rows)}")
    print(f"Revisit events: {len(event_rows)}")
    print(f"Events written to: {args.events_output}")
    print(f"Pairs written to: {args.pairs_output}")

    for event in events:
        print(
            f"event={event.event_id}, "
            f"queries={event.query_start}-{event.query_end}, "
            f"references={event.reference_start}-"
            f"{event.reference_end}, "
            f"pairs={event.positive_pairs}, "
            f"same={event.same_direction_pairs}, "
            f"opposite={event.opposite_direction_pairs}, "
            f"oblique={event.oblique_pairs}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
