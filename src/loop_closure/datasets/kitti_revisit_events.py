"""Revisit-event and heading analysis for KITTI odometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RevisitEvent:
    """Summary of a contiguous group of positive revisit queries."""

    event_id: int
    query_start: int
    query_end: int
    query_count: int
    reference_start: int
    reference_end: int
    positive_pairs: int
    minimum_distance_m: float
    median_distance_m: float
    maximum_distance_m: float
    median_heading_difference_deg: float
    same_direction_pairs: int
    oblique_pairs: int
    opposite_direction_pairs: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def angular_difference_degrees(
    first: np.ndarray | float,
    second: np.ndarray | float,
) -> np.ndarray:
    """Return absolute wrapped angular differences in [0, 180]."""

    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)

    difference = (
        first_array - second_array + 180.0
    ) % 360.0 - 180.0

    return np.abs(difference)


def extract_ground_plane_pose(
    poses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract x-z positions and camera-forward headings."""

    poses = np.asarray(poses, dtype=np.float64)

    if poses.ndim != 3 or poses.shape[1:] != (3, 4):
        raise ValueError(
            "Poses must have shape (N, 3, 4). "
            f"Observed shape: {poses.shape}"
        )

    positions = poses[:, [0, 2], 3]

    # The positive z-axis represents the camera-forward direction
    # in the KITTI camera coordinate system.
    forward_x = poses[:, 0, 2]
    forward_z = poses[:, 2, 2]

    headings = np.degrees(np.arctan2(forward_x, forward_z))
    headings = (headings + 360.0) % 360.0

    return positions, headings


def summarize_revisit_events(
    positions: np.ndarray,
    headings_deg: np.ndarray,
    spatial_radius_m: float,
    temporal_exclusion_frames: int,
    maximum_query_gap: int = 5,
) -> tuple[list[RevisitEvent], list[dict[str, object]]]:
    """Generate positive pairs and group contiguous revisit queries."""

    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings_deg, dtype=np.float64)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(
            "Positions must have shape (N, 2). "
            f"Observed shape: {positions.shape}"
        )

    if headings.ndim != 1 or headings.shape[0] != positions.shape[0]:
        raise ValueError(
            "Headings must have shape (N,) and match positions."
        )

    if spatial_radius_m <= 0:
        raise ValueError("Spatial radius must be positive.")

    if temporal_exclusion_frames < 1:
        raise ValueError("Temporal exclusion must be positive.")

    if maximum_query_gap < 0:
        raise ValueError("Maximum query gap cannot be negative.")

    pair_records: list[dict[str, object]] = []

    for query_index in range(
        temporal_exclusion_frames,
        positions.shape[0],
    ):
        reference_limit = (
            query_index - temporal_exclusion_frames + 1
        )

        reference_positions = positions[:reference_limit]

        distances = np.linalg.norm(
            reference_positions - positions[query_index],
            axis=1,
        )

        reference_indices = np.flatnonzero(
            distances <= spatial_radius_m
        )

        for reference_index in reference_indices:
            heading_difference = float(
                angular_difference_degrees(
                    headings[query_index],
                    headings[reference_index],
                )
            )

            if heading_difference <= 45.0:
                traversal_class = "same_direction"
            elif heading_difference >= 135.0:
                traversal_class = "opposite_direction"
            else:
                traversal_class = "oblique"

            pair_records.append(
                {
                    "query_frame": int(query_index),
                    "reference_frame": int(reference_index),
                    "spatial_distance_m": float(
                        distances[reference_index]
                    ),
                    "query_heading_deg": float(
                        headings[query_index]
                    ),
                    "reference_heading_deg": float(
                        headings[reference_index]
                    ),
                    "heading_difference_deg": heading_difference,
                    "traversal_class": traversal_class,
                }
            )

    if not pair_records:
        return [], []

    query_indices = sorted(
        {int(record["query_frame"]) for record in pair_records}
    )

    query_groups: list[list[int]] = []
    current_group = [query_indices[0]]

    for query_index in query_indices[1:]:
        if query_index - current_group[-1] <= maximum_query_gap:
            current_group.append(query_index)
        else:
            query_groups.append(current_group)
            current_group = [query_index]

    query_groups.append(current_group)

    events: list[RevisitEvent] = []

    for event_id, query_group in enumerate(query_groups, start=1):
        query_set = set(query_group)

        event_pairs = [
            record
            for record in pair_records
            if int(record["query_frame"]) in query_set
        ]

        references = np.asarray(
            [
                int(record["reference_frame"])
                for record in event_pairs
            ],
            dtype=int,
        )

        distances = np.asarray(
            [
                float(record["spatial_distance_m"])
                for record in event_pairs
            ],
            dtype=float,
        )

        heading_differences = np.asarray(
            [
                float(record["heading_difference_deg"])
                for record in event_pairs
            ],
            dtype=float,
        )

        classes = [
            str(record["traversal_class"])
            for record in event_pairs
        ]

        events.append(
            RevisitEvent(
                event_id=event_id,
                query_start=min(query_group),
                query_end=max(query_group),
                query_count=len(query_group),
                reference_start=int(references.min()),
                reference_end=int(references.max()),
                positive_pairs=len(event_pairs),
                minimum_distance_m=float(distances.min()),
                median_distance_m=float(np.median(distances)),
                maximum_distance_m=float(distances.max()),
                median_heading_difference_deg=float(
                    np.median(heading_differences)
                ),
                same_direction_pairs=classes.count(
                    "same_direction"
                ),
                oblique_pairs=classes.count("oblique"),
                opposite_direction_pairs=classes.count(
                    "opposite_direction"
                ),
            )
        )

    return events, pair_records
