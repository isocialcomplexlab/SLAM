#!/usr/bin/env python3
"""Generate L=1 pair scores for the controlled 2048-D full-image arm."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from loop_closure.evaluation.resnet101_full_image_pair_scores import (
    REQUIRED_PAIR_FIELDS,
    collect_output_fields,
    load_full_image_descriptor_index,
    score_full_image_pair_row,
)


EXPECTED_SPLITS = {
    "calibration": {
        "rows": 78624,
        "classes": {
            "positive": 3985,
            "ambiguous": 3572,
            "negative": 71067,
        },
    },
    "test": {
        "rows": 14170,
        "classes": {
            "positive": 1269,
            "ambiguous": 1186,
            "negative": 11715,
        },
    },
    "stress": {
        "rows": 9636,
        "classes": {
            "positive": 1296,
            "ambiguous": 1226,
            "negative": 7114,
        },
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def serializable(row: dict[str, object]) -> dict[str, object]:
    return {
        key: "" if value is None else value
        for key, value in row.items()
    }


def update_range(
    ranges: dict[str, dict[str, float]],
    field: str,
    value: object,
) -> None:
    number = float(value)
    if field not in ranges:
        ranges[field] = {
            "minimum": number,
            "maximum": number,
        }
    else:
        ranges[field]["minimum"] = min(
            ranges[field]["minimum"],
            number,
        )
        ranges[field]["maximum"] = max(
            ranges[field]["maximum"],
            number,
        )


def generate_split(
    split: str,
    *,
    pairs_path: Path,
    descriptor_index,
    output_dir: Path,
    chunk_size: int,
    descriptor_npz: Path,
) -> dict[str, object]:
    expected = EXPECTED_SPLITS[split]
    output_path = output_dir / f"{split}_scores.csv"
    report_path = output_dir / f"{split}_scores_report.json"
    temp_path = output_dir / f".{split}_scores.csv.tmp"

    for path in (output_path, report_path, temp_path):
        require(not path.exists(), f"refusing overwrite: {path}")

    counts = Counter()
    seen: set[tuple[int, int]] = set()
    ranges: dict[str, dict[str, float]] = {}
    processed = 0
    next_progress = chunk_size
    started = time.perf_counter()

    try:
        with pairs_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            input_fields = list(reader.fieldnames or [])
            missing = REQUIRED_PAIR_FIELDS - set(input_fields)
            require(
                not missing,
                f"{split}: missing pair columns {sorted(missing)}",
            )

            output_fields = collect_output_fields(input_fields)

            with temp_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as target:
                writer = csv.DictWriter(
                    target,
                    fieldnames=output_fields,
                    extrasaction="raise",
                )
                writer.writeheader()

                for line_number, row in enumerate(reader, start=2):
                    require(
                        row["split"] == split,
                        f"{split}: split contamination at line {line_number}",
                    )
                    if "dataset" in row:
                        require(
                            row["dataset"] == "KITTI_odometry",
                            f"{split}: unexpected dataset line {line_number}",
                        )
                    if "sequence" in row:
                        require(
                            row["sequence"] in {"05", "5"},
                            f"{split}: unexpected sequence line {line_number}",
                        )
                    if "camera" in row:
                        require(
                            row["camera"] == "image_0",
                            f"{split}: unexpected camera line {line_number}",
                        )

                    identity = (
                        int(row["query_frame"]),
                        int(row["reference_frame"]),
                    )
                    require(
                        identity not in seen,
                        f"{split}: duplicate pair {identity}",
                    )
                    seen.add(identity)

                    scored = score_full_image_pair_row(
                        row,
                        descriptor_index=descriptor_index,
                        expected_split=split,
                    )
                    writer.writerow(serializable(scored))

                    pair_class = str(scored["pair_class"])
                    counts[pair_class] += 1
                    for field in (
                        "legacy_exact_distance",
                        "legacy_exact_score",
                        "corrected_l2_distance",
                        "corrected_l2_score",
                    ):
                        update_range(ranges, field, scored[field])

                    processed += 1
                    if processed >= next_progress:
                        elapsed = time.perf_counter() - started
                        rate = processed / elapsed if elapsed > 0 else 0.0
                        print(
                            f"[{split}] {processed}/{expected['rows']} "
                            f"pairs — {rate:.1f} pairs/s",
                            flush=True,
                        )
                        while next_progress <= processed:
                            next_progress += chunk_size

        actual_classes = {
            name: int(counts.get(name, 0))
            for name in ("positive", "ambiguous", "negative")
        }
        require(
            processed == expected["rows"],
            f"{split}: row count {processed} != {expected['rows']}",
        )
        require(
            actual_classes == expected["classes"],
            f"{split}: class counts {actual_classes} != {expected['classes']}",
        )
        require(
            len(seen) == processed,
            f"{split}: unique pair count mismatch",
        )

        os.replace(temp_path, output_path)
        elapsed = time.perf_counter() - started

        report = {
            "status": "passed",
            "dataset": "KITTI_odometry",
            "sequence": "05",
            "camera_id": "image_0",
            "split": split,
            "preprocessing_mode": "full_image",
            "descriptor_model": "resnet101_imagenet1k_v1",
            "descriptor_dimension": 2048,
            "descriptor_file": str(descriptor_npz.resolve()),
            "descriptor_sha256": sha256(descriptor_npz),
            "pairs_file": str(pairs_path.resolve()),
            "pairs_sha256": sha256(pairs_path),
            "scores_file": str(output_path.resolve()),
            "scores_sha256": sha256(output_path),
            "row_count": processed,
            "unique_pair_count": len(seen),
            "class_counts": actual_classes,
            "binary_row_count": (
                actual_classes["positive"] + actual_classes["negative"]
            ),
            "ambiguous_row_count": actual_classes["ambiguous"],
            "score_ranges": ranges,
            "score_semantics": {
                "legacy_exact_distance": (
                    "L-infinity/Chebyshev; smaller is more similar"
                ),
                "legacy_exact_score": (
                    "negative L-infinity distance; larger is more similar"
                ),
                "corrected_l2_distance": (
                    "Euclidean L2; smaller is more similar"
                ),
                "corrected_l2_score": (
                    "negative Euclidean distance; larger is more similar"
                ),
            },
            "threshold_policy": (
                "Calibrate independently for full_image on the calibration "
                "split, then freeze on test/stress."
            ),
            "elapsed_seconds": elapsed,
            "pairs_per_second": processed / elapsed if elapsed > 0 else 0.0,
            "output_size_bytes": output_path.stat().st_size,
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        if temp_path.exists():
            temp_path.unlink()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--descriptor-npz", type=Path, required=True)
    p.add_argument("--pairs-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--chunk-size", type=int, default=4096)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    require(args.chunk_size > 0, "chunk-size must be positive")
    require(args.descriptor_npz.is_file(), "descriptor NPZ missing")
    require(args.pairs_dir.is_dir(), "pairs directory missing")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    descriptor_index = load_full_image_descriptor_index(args.descriptor_npz)
    require(
        len(descriptor_index) == 2761,
        f"descriptor index size={len(descriptor_index)}, expected 2761",
    )

    reports = []
    for split in ("calibration", "test", "stress"):
        pairs_path = args.pairs_dir / f"{split}_pairs.csv"
        require(
            pairs_path.is_file(),
            f"candidate pair file missing: {pairs_path}",
        )
        print(
            f"Generating full-image ResNet-101 scores: {split}",
            flush=True,
        )
        reports.append(
            generate_split(
                split,
                pairs_path=pairs_path,
                descriptor_index=descriptor_index,
                output_dir=args.output_dir,
                chunk_size=args.chunk_size,
                descriptor_npz=args.descriptor_npz,
            )
        )

    summary = {
        "status": "passed",
        "validation_status": (
            "R2_M2_FULL_IMAGE_PAIR_SCORE_GENERATION_PASSED_V1"
        ),
        "dataset": "KITTI_odometry",
        "sequence": "05",
        "preprocessing_mode": "full_image",
        "descriptor_dimension": 2048,
        "descriptor_sha256": sha256(args.descriptor_npz),
        "reports": reports,
        "total_row_count": sum(int(report["row_count"]) for report in reports),
        "total_binary_row_count": sum(
            int(report["binary_row_count"])
            for report in reports
        ),
        "total_ambiguous_row_count": sum(
            int(report["ambiguous_row_count"])
            for report in reports
        ),
        "threshold_policy": {
            "calibrate_on": "full_image calibration split",
            "freeze_on": ["test", "stress"],
            "reuse_split_left_right_threshold": False,
        },
        "next_executable_stage": (
            "R2_M2_PREPROCESSING_ABLATION_L1_EVALUATION_V1"
        ),
    }

    summary_path = args.output_dir / "pair_scores_summary.json"
    require(
        not summary_path.exists(),
        f"refusing overwrite: {summary_path}",
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("FULL-IMAGE PAIR SCORE GENERATION COMPLETE")
    print(f"total_rows={summary['total_row_count']}")
    print(f"total_binary={summary['total_binary_row_count']}")
    print(f"total_ambiguous={summary['total_ambiguous_row_count']}")


if __name__ == "__main__":
    main()
