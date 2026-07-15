"""Pose-based revisit analysis for KITTI odometry sequences."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RevisitSummary:
    """Summary for one spatial and temporal protocol combination."""

    spatial_radius_m: float
    temporal_exclusion_frames: int
    candidate_pairs: int
    positive_pairs: int
    positive_prevalence: float
    query_frames_with_positive: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_revisits(
    positions: np.ndarray,
    spatial_radii_m: Iterable[float],
    temporal_exclusions: Iterable[int],
) -> list[RevisitSummary]:
    """Count pose-based revisit pairs for several protocol parameters.

    A candidate pair (query, reference) is considered positive when its
    Euclidean position distance is no greater than the configured spatial
    radius. References closer than the temporal exclusion are omitted.

    Positions may contain two ground-plane coordinates or three spatial
    coordinates.
    """

    positions = np.asarray(positions, dtype=np.float64)

    if positions.ndim != 2 or positions.shape[1] not in {2, 3}:
        raise ValueError(
            "Positions must have shape (N, 2) or (N, 3). "
            f"Observed shape: {positions.shape}"
        )

    if positions.shape[0] < 2:
        raise ValueError("At least two positions are required.")

    if not np.isfinite(positions).all():
        raise ValueError("Positions contain non-finite values.")

    radii = sorted({float(value) for value in spatial_radii_m})
    exclusions = sorted({int(value) for value in temporal_exclusions})

    if not radii or any(value <= 0 for value in radii):
        raise ValueError("Spatial radii must contain positive values.")

    if not exclusions or any(value < 1 for value in exclusions):
        raise ValueError(
            "Temporal exclusions must contain positive frame counts."
        )

    frame_count = positions.shape[0]
    summaries: list[RevisitSummary] = []

    for exclusion in exclusions:
        if exclusion >= frame_count:
            raise ValueError(
                f"Temporal exclusion {exclusion} is not smaller than "
                f"the frame count {frame_count}."
            )

        candidate_pairs = 0
        positive_counts = {radius: 0 for radius in radii}
        positive_queries = {radius: set() for radius in radii}

        for query_index in range(exclusion, frame_count):
            # Includes reference_index == query_index - exclusion.
            reference_limit = query_index - exclusion + 1
            references = positions[:reference_limit]

            distances = np.linalg.norm(
                references - positions[query_index],
                axis=1,
            )

            candidate_pairs += int(distances.size)

            for radius in radii:
                match_count = int(
                    np.count_nonzero(distances <= radius)
                )

                positive_counts[radius] += match_count

                if match_count:
                    positive_queries[radius].add(query_index)

        for radius in radii:
            positive_pairs = positive_counts[radius]

            summaries.append(
                RevisitSummary(
                    spatial_radius_m=radius,
                    temporal_exclusion_frames=exclusion,
                    candidate_pairs=candidate_pairs,
                    positive_pairs=positive_pairs,
                    positive_prevalence=(
                        positive_pairs / candidate_pairs
                        if candidate_pairs
                        else 0.0
                    ),
                    query_frames_with_positive=len(
                        positive_queries[radius]
                    ),
                )
            )

    return summaries
