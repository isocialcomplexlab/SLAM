"""Pair-score helpers for the controlled 2048-D full-image ResNet-101 arm."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np


FULL_IMAGE_DESCRIPTOR_LENGTH: Final[int] = 2048

BASE_OUTPUT_FIELDS: Final[tuple[str, ...]] = (
    "split",
    "query_camera_id",
    "reference_camera_id",
    "query_frame",
    "reference_frame",
    "spatial_distance_m",
    "pair_class",
    "binary_label",
    "legacy_exact_distance",
    "legacy_exact_score",
    "corrected_l2_distance",
    "corrected_l2_score",
)

REQUIRED_PAIR_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "split",
        "query_frame",
        "reference_frame",
        "spatial_distance_m",
        "pair_class",
        "binary_label",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_full_image_descriptor_index(
    path: str | Path,
) -> dict[tuple[str, int], np.ndarray]:
    """Load the full-image NPZ and index descriptors by camera/frame identity."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    with np.load(source, allow_pickle=False) as archive:
        required = {"descriptors", "camera_ids", "frame_ids"}
        missing = required - set(archive.files)
        _require(
            not missing,
            f"descriptor NPZ missing arrays: {sorted(missing)}",
        )

        descriptors = np.asarray(archive["descriptors"])
        camera_ids = np.asarray(archive["camera_ids"])
        frame_ids = np.asarray(archive["frame_ids"])

    _require(
        descriptors.ndim == 2
        and descriptors.shape[1] == FULL_IMAGE_DESCRIPTOR_LENGTH,
        "descriptors must have shape [N, 2048]; "
        f"received {descriptors.shape}",
    )
    _require(
        descriptors.dtype == np.float32,
        f"descriptors must be float32; received {descriptors.dtype}",
    )
    _require(
        bool(np.isfinite(descriptors).all()),
        "descriptors contain NaN or infinite values",
    )
    _require(
        camera_ids.ndim == 1
        and frame_ids.ndim == 1
        and len(camera_ids) == len(frame_ids) == descriptors.shape[0],
        "descriptor identity arrays do not match descriptor row count",
    )

    result: dict[tuple[str, int], np.ndarray] = {}

    for row_index in range(descriptors.shape[0]):
        camera = str(camera_ids[row_index])
        frame = int(frame_ids[row_index])
        key = (camera, frame)
        _require(key not in result, f"duplicate descriptor identity: {key}")
        result[key] = np.ascontiguousarray(descriptors[row_index])

    return result


def full_image_descriptor_distances(
    query: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float]:
    """Return legacy L-infinity and corrected flattened L2 distances."""
    q = np.asarray(query)
    r = np.asarray(reference)

    _require(
        q.shape == (FULL_IMAGE_DESCRIPTOR_LENGTH,),
        f"query descriptor must have shape (2048,); received {q.shape}",
    )
    _require(
        r.shape == (FULL_IMAGE_DESCRIPTOR_LENGTH,),
        f"reference descriptor must have shape (2048,); received {r.shape}",
    )
    _require(
        q.dtype == np.float32 and r.dtype == np.float32,
        "query and reference descriptors must be float32",
    )
    _require(
        bool(np.isfinite(q).all()) and bool(np.isfinite(r).all()),
        "query/reference descriptors contain NaN or infinite values",
    )

    delta32 = q - r
    legacy = float(np.max(np.abs(delta32)))

    delta64 = q.astype(np.float64) - r.astype(np.float64)
    corrected = float(np.sqrt(np.dot(delta64, delta64)))

    return legacy, corrected


def _parse_int(row: Mapping[str, str], field: str) -> int:
    try:
        return int(str(row[field]).strip())
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid integer field {field}: {row.get(field)!r}"
        ) from exc


def _validate_class_and_label(pair_class: str, binary_label: str) -> None:
    _require(
        pair_class in {"positive", "ambiguous", "negative"},
        f"invalid pair_class: {pair_class!r}",
    )
    expected = {
        "positive": "1",
        "ambiguous": "",
        "negative": "0",
    }[pair_class]
    _require(
        binary_label == expected,
        f"pair_class={pair_class!r} requires binary_label={expected!r}; "
        f"received {binary_label!r}",
    )


def score_full_image_pair_row(
    row: Mapping[str, str],
    *,
    descriptor_index: Mapping[tuple[str, int], np.ndarray],
    expected_split: str,
    default_camera_id: str = "image_0",
) -> dict[str, object]:
    """Score one candidate pair while preserving its audit metadata."""
    missing = REQUIRED_PAIR_FIELDS - set(row)
    _require(
        not missing,
        f"candidate pair row missing fields: {sorted(missing)}",
    )

    split = str(row["split"]).strip()
    _require(
        split == expected_split,
        f"split contamination: {split!r} != {expected_split!r}",
    )

    pair_class = str(row["pair_class"]).strip()
    binary_label = str(row["binary_label"]).strip()
    _validate_class_and_label(pair_class, binary_label)

    query_frame = _parse_int(row, "query_frame")
    reference_frame = _parse_int(row, "reference_frame")

    row_camera = str(row.get("camera", default_camera_id)).strip()
    query_camera = str(
        row.get("query_camera_id", row_camera or default_camera_id)
    ).strip()
    reference_camera = str(
        row.get("reference_camera_id", row_camera or default_camera_id)
    ).strip()

    query_key = (query_camera, query_frame)
    reference_key = (reference_camera, reference_frame)

    _require(query_key in descriptor_index, f"missing query descriptor: {query_key}")
    _require(
        reference_key in descriptor_index,
        f"missing reference descriptor: {reference_key}",
    )

    legacy_distance, corrected_distance = full_image_descriptor_distances(
        descriptor_index[query_key],
        descriptor_index[reference_key],
    )

    try:
        spatial_distance = float(str(row["spatial_distance_m"]).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid spatial_distance_m: {row['spatial_distance_m']!r}"
        ) from exc

    _require(np.isfinite(spatial_distance), "spatial_distance_m must be finite")

    output: dict[str, object] = {
        "split": split,
        "query_camera_id": query_camera,
        "reference_camera_id": reference_camera,
        "query_frame": query_frame,
        "reference_frame": reference_frame,
        "spatial_distance_m": spatial_distance,
        "pair_class": pair_class,
        "binary_label": binary_label,
        "legacy_exact_distance": legacy_distance,
        "legacy_exact_score": -legacy_distance,
        "corrected_l2_distance": corrected_distance,
        "corrected_l2_score": -corrected_distance,
    }

    for field, value in row.items():
        if field not in output:
            output[field] = value

    return output


def collect_output_fields(input_fields: Sequence[str]) -> list[str]:
    """Create a stable output schema containing scores and all source metadata."""
    output = list(BASE_OUTPUT_FIELDS)
    seen = set(output)
    for field in input_fields:
        if field not in seen:
            output.append(field)
            seen.add(field)
    return output


def score_candidate_csv(
    pairs_path: str | Path,
    *,
    descriptor_index: Mapping[tuple[str, int], np.ndarray],
    expected_split: str,
) -> tuple[list[str], list[dict[str, object]]]:
    """Small in-memory helper used by unit tests and smoke validation."""
    source = Path(pairs_path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        input_fields = list(reader.fieldnames or [])
        missing = REQUIRED_PAIR_FIELDS - set(input_fields)
        _require(
            not missing,
            f"candidate pair CSV missing fields: {sorted(missing)}",
        )
        rows = [
            score_full_image_pair_row(
                row,
                descriptor_index=descriptor_index,
                expected_split=expected_split,
            )
            for row in reader
        ]

    return collect_output_fields(input_fields), rows
