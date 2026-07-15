import numpy as np
import pytest

from loop_closure.datasets.kitti_revisit_events import (
    angular_difference_degrees,
    summarize_revisit_events,
)


def test_angular_difference_wraps_correctly() -> None:
    result = angular_difference_degrees(
        np.array([10.0, 350.0, 0.0]),
        np.array([350.0, 10.0, 180.0]),
    )

    assert result.tolist() == pytest.approx(
        [20.0, 20.0, 180.0]
    )


def test_revisit_events_group_queries_and_classify_heading() -> None:
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [10.0, 0.0],
            [0.1, 0.0],
        ],
        dtype=float,
    )

    headings = np.array(
        [0.0, 0.0, 0.0, 180.0, 180.0, 0.0, 180.0],
        dtype=float,
    )

    events, pairs = summarize_revisit_events(
        positions=positions,
        headings_deg=headings,
        spatial_radius_m=0.25,
        temporal_exclusion_frames=2,
        maximum_query_gap=1,
    )

    assert len(events) == 2
    assert len(pairs) >= 3
    assert events[0].query_start == 3
    assert events[0].query_end == 4
    assert events[0].opposite_direction_pairs >= 1
    assert events[1].query_start == 6


def test_revisit_events_reject_mismatched_headings() -> None:
    with pytest.raises(ValueError, match="Headings"):
        summarize_revisit_events(
            positions=np.array(
                [[0.0, 0.0], [1.0, 0.0]]
            ),
            headings_deg=np.array([0.0]),
            spatial_radius_m=1.0,
            temporal_exclusion_frames=1,
        )
