import pytest

from loop_closure.datasets.revisit_split import (
    assign_pairs_to_events,
    partition_pairs,
    summarize_partition,
    validate_disjoint_frames,
)


def test_assign_and_partition_complete_events() -> None:
    events = [
        {
            "event_id": "1",
            "query_start": "100",
            "query_end": "110",
        },
        {
            "event_id": "2",
            "query_start": "200",
            "query_end": "210",
        },
    ]

    pairs = [
        {
            "query_frame": "101",
            "reference_frame": "10",
        },
        {
            "query_frame": "205",
            "reference_frame": "50",
        },
    ]

    assigned = assign_pairs_to_events(pairs, events)

    partitions = partition_pairs(
        assigned,
        {
            "calibration": [1],
            "test": [2],
        },
    )

    assert len(partitions["calibration"]) == 1
    assert len(partitions["test"]) == 1
    assert partitions["calibration"][0]["event_id"] == 1
    assert partitions["test"][0]["event_id"] == 2
    assert validate_disjoint_frames(partitions) == set()


def test_validate_disjoint_frames_rejects_leakage() -> None:
    partitions = {
        "calibration": [
            {
                "query_frame": 100,
                "reference_frame": 10,
            }
        ],
        "test": [
            {
                "query_frame": 200,
                "reference_frame": 100,
            }
        ],
    }

    with pytest.raises(ValueError, match="Frame leakage"):
        validate_disjoint_frames(partitions)


def test_assign_pairs_rejects_unmatched_query() -> None:
    events = [
        {
            "event_id": "1",
            "query_start": "100",
            "query_end": "110",
        }
    ]

    pairs = [
        {
            "query_frame": "500",
            "reference_frame": "10",
        }
    ]

    with pytest.raises(ValueError, match="matched 0 events"):
        assign_pairs_to_events(pairs, events)


def test_summarize_partition_counts_unique_frames() -> None:
    rows = [
        {
            "event_id": 1,
            "query_frame": 100,
            "reference_frame": 10,
        },
        {
            "event_id": 1,
            "query_frame": 101,
            "reference_frame": 10,
        },
    ]

    summary = summarize_partition(rows)

    assert summary["event_ids"] == [1]
    assert summary["positive_pairs"] == 2
    assert summary["unique_query_frames"] == 2
    assert summary["unique_reference_frames"] == 1
    assert summary["unique_all_frames"] == 3
