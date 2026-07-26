from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from loop_closure.evaluation.resnet101_l1_evaluation import (
    load_binary_score_split,
)
from loop_closure.evaluation.resnet101_full_image_pair_scores import (
    FULL_IMAGE_DESCRIPTOR_LENGTH,
    full_image_descriptor_distances,
    load_full_image_descriptor_index,
    score_candidate_csv,
    score_full_image_pair_row,
)


def descriptor(values: list[float]) -> np.ndarray:
    result = np.zeros(
        FULL_IMAGE_DESCRIPTOR_LENGTH,
        dtype=np.float32,
    )
    result[: len(values)] = values
    return result


def make_index():
    return {
        ("image_0", 0): descriptor([0.0, 0.0, 0.0, 0.0]),
        ("image_0", 1): descriptor([0.3, 0.4, 0.0, 0.0]),
        ("image_0", 2): descriptor([3.0, 4.0, 0.0, 0.0]),
    }


def test_full_image_distance_semantics_are_linf_and_flat_l2() -> None:
    legacy, corrected = full_image_descriptor_distances(
        descriptor([0.0, 0.0]),
        descriptor([0.3, 0.4]),
    )

    assert legacy == pytest.approx(0.4)
    assert corrected == pytest.approx(0.5)


def test_full_image_pair_score_negates_distances() -> None:
    row = {
        "split": "test",
        "camera": "image_0",
        "query_frame": "1",
        "reference_frame": "0",
        "spatial_distance_m": "4.5",
        "pair_class": "positive",
        "binary_label": "1",
        "pair_id": "05_000001_000000",
    }

    scored = score_full_image_pair_row(
        row,
        descriptor_index=make_index(),
        expected_split="test",
    )

    assert scored["legacy_exact_distance"] == pytest.approx(0.4)
    assert scored["legacy_exact_score"] == pytest.approx(-0.4)
    assert scored["corrected_l2_distance"] == pytest.approx(0.5)
    assert scored["corrected_l2_score"] == pytest.approx(-0.5)
    assert scored["pair_id"] == row["pair_id"]
    assert scored["query_camera_id"] == "image_0"
    assert scored["reference_camera_id"] == "image_0"


def test_full_image_pair_score_preserves_ambiguous_blank_label() -> None:
    row = {
        "split": "calibration",
        "camera": "image_0",
        "query_frame": "2",
        "reference_frame": "0",
        "spatial_distance_m": "7.0",
        "pair_class": "ambiguous",
        "binary_label": "",
    }

    scored = score_full_image_pair_row(
        row,
        descriptor_index=make_index(),
        expected_split="calibration",
    )

    assert scored["pair_class"] == "ambiguous"
    assert scored["binary_label"] == ""


def test_full_image_pair_score_rejects_missing_descriptor() -> None:
    row = {
        "split": "test",
        "camera": "image_0",
        "query_frame": "99",
        "reference_frame": "0",
        "spatial_distance_m": "20.0",
        "pair_class": "negative",
        "binary_label": "0",
    }

    with pytest.raises(ValueError, match="missing query descriptor"):
        score_full_image_pair_row(
            row,
            descriptor_index=make_index(),
            expected_split="test",
        )


def test_full_image_npz_loader_accepts_only_2048_dimensions(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.npz"
    matrix = np.stack(
        [descriptor([0.0]), descriptor([1.0])],
        axis=0,
    )
    np.savez_compressed(
        valid,
        descriptors=matrix,
        camera_ids=np.asarray(["image_0", "image_0"]),
        frame_ids=np.asarray([0, 1], dtype=np.int64),
    )

    index = load_full_image_descriptor_index(valid)
    assert set(index) == {
        ("image_0", 0),
        ("image_0", 1),
    }
    assert index[("image_0", 0)].shape == (2048,)

    invalid = tmp_path / "invalid.npz"
    np.savez_compressed(
        invalid,
        descriptors=np.zeros((2, 4096), dtype=np.float32),
        camera_ids=np.asarray(["image_0", "image_0"]),
        frame_ids=np.asarray([0, 1], dtype=np.int64),
    )

    with pytest.raises(ValueError, match=r"\[N, 2048\]"):
        load_full_image_descriptor_index(invalid)


def test_scored_csv_schema_is_l1_compatible(tmp_path: Path) -> None:
    path = tmp_path / "test_pairs.csv"
    fields = [
        "dataset",
        "sequence",
        "camera",
        "split",
        "pair_id",
        "query_frame",
        "reference_frame",
        "spatial_distance_m",
        "pair_class",
        "binary_label",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "KITTI_odometry",
                "sequence": "05",
                "camera": "image_0",
                "split": "test",
                "pair_id": "p1",
                "query_frame": "1",
                "reference_frame": "0",
                "spatial_distance_m": "4.0",
                "pair_class": "positive",
                "binary_label": "1",
            }
        )
        writer.writerow(
            {
                "dataset": "KITTI_odometry",
                "sequence": "05",
                "camera": "image_0",
                "split": "test",
                "pair_id": "p2",
                "query_frame": "2",
                "reference_frame": "0",
                "spatial_distance_m": "12.0",
                "pair_class": "negative",
                "binary_label": "0",
            }
        )

    output_fields, rows = score_candidate_csv(
        path,
        descriptor_index=make_index(),
        expected_split="test",
    )

    for required in (
        "split",
        "pair_class",
        "binary_label",
        "legacy_exact_score",
        "corrected_l2_score",
    ):
        assert required in output_fields

    assert len(rows) == 2
    assert rows[0]["binary_label"] == "1"
    assert rows[1]["binary_label"] == "0"

    scored_path = tmp_path / "test_scores.csv"
    with scored_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)

    loaded = load_binary_score_split(
        scored_path,
        expected_split="test",
    )
    assert loaded.total_rows == 2
    assert loaded.binary_rows == 2
    assert loaded.ambiguous_rows == 0
    assert loaded.positive_rows == 1
    assert loaded.negative_rows == 1
