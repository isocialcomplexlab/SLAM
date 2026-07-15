import numpy as np
import pytest

from loop_closure.datasets.kitti_revisits import analyze_revisits


def test_analyze_revisits_counts_pose_matches() -> None:
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.1, 0.0],
        ],
        dtype=float,
    )

    summaries = analyze_revisits(
        positions=positions,
        spatial_radii_m=[0.2],
        temporal_exclusions=[2],
    )

    assert len(summaries) == 1

    result = summaries[0]

    assert result.candidate_pairs == 3
    assert result.positive_pairs == 1
    assert result.query_frames_with_positive == 1
    assert result.positive_prevalence == pytest.approx(1 / 3)


def test_analyze_revisits_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        analyze_revisits(
            positions=np.array([0.0, 1.0, 2.0]),
            spatial_radii_m=[1.0],
            temporal_exclusions=[1],
        )


def test_analyze_revisits_rejects_invalid_radius() -> None:
    with pytest.raises(ValueError, match="positive"):
        analyze_revisits(
            positions=np.array(
                [[0.0, 0.0], [1.0, 0.0]]
            ),
            spatial_radii_m=[0.0],
            temporal_exclusions=[1],
        )
