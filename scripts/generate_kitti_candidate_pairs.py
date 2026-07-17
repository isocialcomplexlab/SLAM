#!/usr/bin/env python3
"""Generate complete KITTI candidate-pair sets for each pilot split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from loop_closure.datasets.candidate_pairs import (
    iter_candidate_pairs,
    pair_keys,
    validate_positive_pair_preservation,
)
from loop_closure.datasets.kitti_odometry import load_poses
from loop_closure.datasets.kitti_revisit_events import (
    extract_ground_plane_pose,
)


OUTPUT_FIELDS = [
    "dataset",
    "sequence",
    "camera",
    "split",
    "pair_id",
    "query_frame",
    "reference_frame",
    "query_image_path",
    "reference_image_path",
    "temporal_separation_frames",
    "spatial_distance_m",
    "heading_difference_deg",
    "traversal_class",
    "pair_class",
    "binary_label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate full query-reference candidate sets for "
            "KITTI revisit-event splits."
        )
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("KITTI_ODOMETRY_ROOT"),
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--positive-splits-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--negative-radius",
        type=float,
        default=10.0,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as binary_file:
        for block in iter(
            lambda: binary_file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No rows found in {path}")

    return rows


def image_path(
    sequence: str,
    camera: str,
    frame: int,
) -> str:
    return (
        f"sequences/{sequence}/{camera}/"
        f"{frame:06d}.png"
    )


def main() -> int:
    args = parse_args()

    if not args.root:
        raise SystemExit(
            "KITTI root not provided. Use --root or define "
            "KITTI_ODOMETRY_ROOT."
        )

    config = load_json(args.split_config)

    sequence = str(config["sequence"]).zfill(2)
    camera = str(config["camera"])
    positive_radius = float(config["spatial_radius_m"])
    temporal_exclusion = int(
        config["temporal_exclusion_frames"]
    )

    assignments = config.get("event_assignments")

    if not isinstance(assignments, dict) or not assignments:
        raise ValueError(
            "Split configuration has no event_assignments."
        )

    root = Path(args.root).expanduser().resolve()
    poses = load_poses(root / "poses" / f"{sequence}.txt")
    positions, headings = extract_ground_plane_pose(poses)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_summaries: dict[str, dict[str, object]] = {}
    split_frames: dict[str, set[int]] = {}
    input_checksums: dict[str, str] = {}

    for split_name in assignments:
        source_path = (
            args.positive_splits_dir
            / f"{split_name}_positive_pairs.csv"
        )
        source_rows = read_csv(source_path)
        source_positive_keys = pair_keys(source_rows)

        query_frames = sorted(
            {
                int(row["query_frame"])
                for row in source_rows
            }
        )
        reference_frames = sorted(
            {
                int(row["reference_frame"])
                for row in source_rows
            }
        )

        output_path = (
            args.output_dir / f"{split_name}_pairs.csv"
        )

        class_counts: Counter[str] = Counter()
        traversal_counts: Counter[str] = Counter()
        generated_positive_keys: set[
            tuple[int, int]
        ] = set()
        candidate_count = 0

        with output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=OUTPUT_FIELDS,
            )
            writer.writeheader()

            for pair in iter_candidate_pairs(
                positions=positions,
                headings_deg=headings,
                query_frames=query_frames,
                reference_frames=reference_frames,
                positive_radius_m=positive_radius,
                negative_radius_m=args.negative_radius,
                temporal_exclusion_frames=temporal_exclusion,
            ):
                query_frame = int(pair["query_frame"])
                reference_frame = int(
                    pair["reference_frame"]
                )

                pair_class = str(pair["pair_class"])
                traversal_class = str(
             pair["traversal_class"]
                )

                if pair_class == "positive":
                    generated_positive_keys.add(
                        (query_frame, reference_frame)
                    )

                class_counts[pair_class] += 1
                traversal_counts[traversal_class] += 1
                candidate_count += 1

                writer.writerow(
                    {
                        "dataset": config["dataset"],
                        "sequence": sequence,
                        "camera": camera,
                        "split": split_name,
                        "pair_id": (
                            f"{sequence}_"
                            f"{query_frame:06d}_"
                            f"{reference_frame:06d}"
                        ),
                        "query_frame": query_frame,
                        "reference_frame": reference_frame,
                        "query_image_path": image_path(
                            sequence,
                            camera,
                            query_frame,
                        ),
                        "reference_image_path": image_path(
                            sequence,
                            camera,
                            reference_frame,
                        ),
                        "temporal_separation_frames": pair[
                            "temporal_separation_frames"
                        ],
                        "spatial_distance_m": (
                            f"{float(pair['spatial_distance_m']):.9f}"
                        ),
                        "heading_difference_deg": (
                            f"{float(pair['heading_difference_deg']):.9f}"
                        ),
                        "traversal_class": traversal_class,
                        "pair_class": pair_class,
                        "binary_label": pair["binary_label"],
                    }
                )

        preservation = validate_positive_pair_preservation(
            source_positive_pairs=source_positive_keys,
            generated_positive_pairs=generated_positive_keys,
        )

        evaluated_count = (
            class_counts["positive"]
            + class_counts["negative"]
        )

        positive_prevalence = (
            class_counts["positive"] / evaluated_count
            if evaluated_count
            else 0.0
        )

        split_frames[split_name] = (
            set(query_frames) | set(reference_frames)
        )

        input_checksums[source_path.name] = sha256_file(
            source_path
        )

        split_summaries[split_name] = {
            "source_file": source_path.as_posix(),
            "output_file": output_path.name,
            "query_frames": len(query_frames),
            "reference_frames": len(reference_frames),
            "candidate_pairs": candidate_count,
            "positive_pairs": class_counts["positive"],
            "ambiguous_pairs": class_counts["ambiguous"],
            "negative_pairs": class_counts["negative"],
            "binary_evaluation_pairs": evaluated_count,
            "positive_prevalence_excluding_ambiguous": (
                positive_prevalence
            ),
            "traversal_classes": dict(
                sorted(traversal_counts.items())
            ),
            "positive_preservation": preservation,
            "output_sha256": sha256_file(output_path),
        }

        print(
            f"{split_name}: candidates={candidate_count}, "
            f"positive={class_counts['positive']}, "
            f"ambiguous={class_counts['ambiguous']}, "
            f"negative={class_counts['negative']}, "
            f"prevalence={positive_prevalence:.6f}"
        )

    calibration_test_overlap = (
        split_frames["calibration"]
        & split_frames["test"]
    )

    if calibration_test_overlap:
        raise ValueError(
            "Generated calibration and test sets share frames: "
            f"{sorted(calibration_test_overlap)[:20]}"
        )

    summary = {
        "dataset": config["dataset"],
        "sequence": sequence,
        "camera": camera,
        "protocol": {
            "positive_radius_m": positive_radius,
            "ambiguous_interval_m": (
                f"({positive_radius}, {args.negative_radius})"
            ),
            "negative_radius_m": args.negative_radius,
            "temporal_exclusion_frames": temporal_exclusion,
            "candidate_strategy": (
                "full Cartesian product of the query and reference "
                "frames belonging to each complete revisit event"
            ),
            "ambiguous_policy": (
                "retained for audit and excluded from primary "
                "binary metrics"
            ),
        },
        "inputs": {
            "split_config": args.split_config.as_posix(),
            "split_config_sha256": sha256_file(
                args.split_config
            ),
            "positive_split_checksums": input_checksums,
        },
        "splits": split_summaries,
        "independence_checks": {
            "calibration_test_frame_overlap_count": len(
                calibration_test_overlap
            ),
            "calibration_test_frame_overlap": sorted(
                calibration_test_overlap
            ),
        },
    }

    summary_path = (
        args.output_dir / "candidate_pair_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    print("Calibration/test frame overlap: 0")
    print(f"Summary written to: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
