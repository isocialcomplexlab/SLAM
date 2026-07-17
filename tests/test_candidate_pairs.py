import numpy as np
import pytest

from loop_closure.datasets.candidate_pairs import (
    classify_distance,
    iter_candidate_pairs,
    validate_positive_pair_preservation,
)


def test_classify_distance_boundaries() -> None:
    assert classify_distance(5.0, 5.0, 10.0) == (
        "positive",
        1,
    )
    assert classify_distance(5.1, 5.0, 10.0) == (
        "ambiguous",
        None,
    )
    assert classify_distance(9.9, 5.0, 10.0) == (
        "ambiguous",
        None,
    )
    assert classify_distance(10.0, 5.0, 10.0) == (
        "negative",
        0,
    )


def test_iter_candidate_pairs_builds_cartesian_product() -> None:
    positions = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [6.0, 0.0],
        ]
    )
    headings = np.array([0.0, 0.0, 180.0])

    rows = list(
        iter_candidate_pairs(
            positions=positions,
            headings_deg=headings,
            query_frames=[2],
            reference_frames=[0, 1],
            positive_radius_m=3.0,
            negative_radius_m=5.0,
            temporal_exclusion_frames=1,
        )
    )

    assert len(rows) == 2
    assert rows[0]["pair_class"] == "negative"
    assert rows[1]["pair_class"] == "positive"
    assert rows[1]["traversal_class"] == (
        "opposite_direction"
    )


def test_positive_pair_preservation_accepts_exact_match() -> None:
    result = validate_positive_pair_preservation(
        source_positive_pairs={(100, 10), (101, 11)},
        generated_positive_pairs={(100, 10), (101, 11)},
    )

    assert result["missing_positive_pairs"] == 0
    assert result["extra_positive_pairs"] == 0


def test_positive_pair_preservation_rejects_difference() -> None:
    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        validate_positive_pair_preservation(
            source_positive_pairs={(100, 10)},
            generated_positive_pairs={(100, 11)},
        )
