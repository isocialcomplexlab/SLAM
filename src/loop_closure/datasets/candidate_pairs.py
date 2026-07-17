"""Deterministic candidate-pair generation for place recognition."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Iterator

import numpy as np

from loop_closure.datasets.kitti_revisit_events import (
    angular_difference_degrees,
)


PairKey = tuple[int, int]


def classify_distance(
    distance_m: float,
    positive_radius_m: float,
    negative_radius_m: float,
) -> tuple[str, int | None]:
    """Classify a geometric pair as positive, ambiguous, or negative."""

    if positive_radius_m <= 0:
        raise ValueError("Positive radius must be greater than zero.")

    if negative_radius_m <= positive_radius_m:
        raise ValueError(
            "Negative radius must be greater than the positive radius."
        )

    distance = float(distance_m)

    if not np.isfinite(distance) or distance < 0:
        raise ValueError(
            f"Distance must be finite and non-negative: {distance_m}"
        )

    if distance <= positive_radius_m:
        return "positive", 1

    if distance < negative_radius_m:
        return "ambiguous", None

    return "negative", 0


def classify_traversal(
    heading_difference_deg: float,
) -> str:
    """Classify relative traversal direction."""

    difference = float(heading_difference_deg)

    if difference <= 45.0:
        return "same_direction"

    if difference >= 135.0:
        return "opposite_direction"

    return "oblique"


def _normalize_frames(
    frames: Iterable[int],
    frame_count: int,
    field_name: str,
) -> np.ndarray:
    values = sorted({int(frame) for frame in frames})

    if not values:
        raise ValueError(f"{field_name} cannot be empty.")

    array = np.asarray(values, dtype=np.int64)

    if int(array.min()) < 0 or int(array.max()) >= frame_count:
        raise ValueError(
            f"{field_name} contains a frame outside [0, "
            f"{frame_count - 1}]: {values[:10]}"
        )

    return array


def iter_candidate_pairs(
    positions: np.ndarray,
    headings_deg: np.ndarray,
    query_frames: Iterable[int],
    reference_frames: Iterable[int],
    positive_radius_m: float,
    negative_radius_m: float,
    temporal_exclusion_frames: int,
) -> Iterator[dict[str, object]]:
    """Yield the full query-reference Cartesian product deterministically."""

    positions_array = np.asarray(positions, dtype=np.float64)
    headings_array = np.asarray(headings_deg, dtype=np.float64)

    if positions_array.ndim != 2 or positions_array.shape[1] != 2:
        raise ValueError(
            "Positions must have shape (N, 2). "
            f"Observed: {positions_array.shape}"
        )

    if (
        headings_array.ndim != 1
        or headings_array.shape[0] != positions_array.shape[0]
    ):
        raise ValueError(
            "Headings must have shape (N,) and match positions."
        )

    if temporal_exclusion_frames < 1:
        raise ValueError(
            "Temporal exclusion must be at least one frame."
        )

    query_array = _normalize_frames(
        query_frames,
        positions_array.shape[0],
        "query_frames",
    )
    reference_array = _normalize_frames(
        reference_frames,
        positions_array.shape[0],
        "reference_frames",
    )

    reference_positions = positions_array[reference_array]
    reference_headings = headings_array[reference_array]

    for query_frame in query_array:
        frame_separations = query_frame - reference_array

        invalid = reference_array[
            frame_separations < temporal_exclusion_frames
        ]

        if invalid.size:
            raise ValueError(
                f"Query frame {int(query_frame)} has references that "
                f"violate the temporal exclusion of "
                f"{temporal_exclusion_frames} frames. "
                f"Sample: {invalid[:10].tolist()}"
            )

        distances = np.linalg.norm(
            reference_positions - positions_array[query_frame],
            axis=1,
        )

        heading_differences = angular_difference_degrees(
            headings_array[query_frame],
            reference_headings,
        )

        for index, reference_frame in enumerate(reference_array):
            distance = float(distances[index])
            heading_difference = float(
                heading_differences[index]
            )

            pair_class, binary_label = classify_distance(
                distance_m=distance,
                positive_radius_m=positive_radius_m,
                negative_radius_m=negative_radius_m,
            )

            yield {
                "query_frame": int(query_frame),
                "reference_frame": int(reference_frame),
                "temporal_separation_frames": int(
                    query_frame - reference_frame
                ),
                "spatial_distance_m": distance,
                "heading_difference_deg": heading_difference,
                "traversal_class": classify_traversal(
                    heading_difference
                ),
                "pair_class": pair_class,
                "binary_label": (
                    "" if binary_label is None else binary_label
                ),
            }


def pair_keys(
    rows: Iterable[Mapping[str, object]],
) -> set[PairKey]:
    """Extract deterministic query-reference keys."""

    result: set[PairKey] = set()

    for row in rows:
        try:
            key = (
                int(row["query_frame"]),
                int(row["reference_frame"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid pair row: {row}"
            ) from exc

        if key in result:
            raise ValueError(f"Duplicate pair detected: {key}")

        result.add(key)

    return result


def validate_positive_pair_preservation(
    source_positive_pairs: set[PairKey],
    generated_positive_pairs: set[PairKey],
) -> dict[str, int]:
    """Require exact preservation of the source positive-pair set."""

    missing = source_positive_pairs - generated_positive_pairs
    extra = generated_positive_pairs - source_positive_pairs

    if missing or extra:
        raise ValueError(
            "Generated positive pairs do not match the source set. "
            f"Missing={len(missing)}, extra={len(extra)}, "
            f"missing_sample={sorted(missing)[:10]}, "
            f"extra_sample={sorted(extra)[:10]}"
        )

    return {
        "source_positive_pairs": len(source_positive_pairs),
        "generated_positive_pairs": len(
            generated_positive_pairs
        ),
        "missing_positive_pairs": 0,
        "extra_positive_pairs": 0,
    }
