#!/usr/bin/env python3
"""Create leakage-aware KITTI revisit-event splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from loop_closure.datasets.revisit_split import (
    assign_pairs_to_events,
    frames_used,
    partition_pairs,
    summarize_partition,
    validate_disjoint_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split positive KITTI revisit pairs by complete events."
        )
    )
    parser.add_argument(
        "--events-input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--pairs-input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    with path.open(
        encoding="utf-8",
        newline="",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")

    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        raise ValueError(f"No rows available for output: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    original_fields = [
        field
        for field in rows[0]
        if field not in {"split", "event_id"}
    ]

    fieldnames = [
        "split",
        "event_id",
        *original_fields,
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as binary_file:
        for block in iter(
            lambda: binary_file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")

    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    assignments = config.get("event_assignments")

    if not isinstance(assignments, dict):
        raise ValueError(
            "Configuration must contain an event_assignments object."
        )

    required_splits = {"calibration", "test", "stress"}
    missing = required_splits - set(assignments)

    if missing:
        raise ValueError(
            f"Configuration is missing splits: {sorted(missing)}"
        )

    return config


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    events = read_csv(args.events_input)
    pairs = read_csv(args.pairs_input)

    assigned_pairs = assign_pairs_to_events(
        pair_rows=pairs,
        event_rows=events,
    )

    partitions = partition_pairs(
        assigned_pairs=assigned_pairs,
        event_assignments=config["event_assignments"],
    )

    calibration_test_overlap = validate_disjoint_frames(
        partitions,
        left_split="calibration",
        right_split="test",
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_files: dict[str, str] = {}

    for split_name, split_rows in partitions.items():
        output_path = (
            args.output_dir
            / f"{split_name}_positive_pairs.csv"
        )

        write_csv(output_path, split_rows)
        output_files[split_name] = output_path.name

    calibration_frames = frames_used(
        partitions["calibration"]
    )
    test_frames = frames_used(partitions["test"])
    stress_frames = frames_used(partitions["stress"])

    summary = {
        "dataset": config.get("dataset"),
        "sequence": config.get("sequence"),
        "camera": config.get("camera"),
        "protocol": {
            "spatial_radius_m": config.get(
                "spatial_radius_m"
            ),
            "temporal_exclusion_frames": config.get(
                "temporal_exclusion_frames"
            ),
            "event_assignments": config.get(
                "event_assignments"
            ),
            "rationale": config.get("rationale"),
            "final_protocol_note": config.get(
                "final_protocol_note"
            ),
        },
        "inputs": {
            "events_csv": args.events_input.as_posix(),
            "events_csv_sha256": sha256_file(
                args.events_input
            ),
            "positive_pairs_csv": (
                args.pairs_input.as_posix()
            ),
            "positive_pairs_csv_sha256": sha256_file(
                args.pairs_input
            ),
            "config_json": args.config.as_posix(),
            "config_json_sha256": sha256_file(
                args.config
            ),
        },
        "outputs": output_files,
        "splits": {
            split_name: summarize_partition(rows)
            for split_name, rows in partitions.items()
        },
        "independence_checks": {
            "calibration_test_frame_overlap_count": len(
                calibration_test_overlap
            ),
            "calibration_test_frame_overlap": sorted(
                calibration_test_overlap
            ),
            "calibration_stress_frame_overlap_count": len(
                calibration_frames & stress_frames
            ),
            "test_stress_frame_overlap_count": len(
                test_frames & stress_frames
            ),
        },
    }

    summary_path = args.output_dir / "split_summary.json"

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(
            summary,
            summary_file,
            indent=2,
            ensure_ascii=False,
        )
        summary_file.write("\n")

    print("Split generation completed.")

    for split_name, rows in partitions.items():
        details = summarize_partition(rows)

        print(
            f"{split_name}: "
            f"events={details['event_ids']}, "
            f"positive_pairs={details['positive_pairs']}, "
            f"unique_frames={details['unique_all_frames']}"
        )

    print(
        "Calibration/test frame overlap: "
        f"{len(calibration_test_overlap)}"
    )
    print(f"Summary written to: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
