"""Leakage-aware splitting of pose-based revisit events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def _row_integer(
    row: Mapping[str, object],
    field: str,
) -> int:
    """Read one required integer field from a CSV-like row."""

    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid or missing integer field {field!r}: {row}"
        ) from exc


def assign_pairs_to_events(
    pair_rows: Sequence[Mapping[str, object]],
    event_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Attach exactly one revisit-event ID to every positive pair."""

    if not pair_rows:
        raise ValueError("At least one positive pair is required.")

    if not event_rows:
        raise ValueError("At least one revisit event is required.")

    event_ranges: list[tuple[int, int, int]] = []

    for row in event_rows:
        event_id = _row_integer(row, "event_id")
        query_start = _row_integer(row, "query_start")
        query_end = _row_integer(row, "query_end")

        if query_start > query_end:
            raise ValueError(
                f"Invalid query interval for event {event_id}: "
                f"{query_start}-{query_end}"
            )

        event_ranges.append(
            (event_id, query_start, query_end)
        )

    assigned: list[dict[str, object]] = []

    for row in pair_rows:
        query_frame = _row_integer(row, "query_frame")

        matching_events = [
            event_id
            for event_id, query_start, query_end in event_ranges
            if query_start <= query_frame <= query_end
        ]

        if len(matching_events) != 1:
            raise ValueError(
                f"Query frame {query_frame} matched "
                f"{len(matching_events)} events: {matching_events}"
            )

        record = dict(row)
        record["event_id"] = matching_events[0]
        assigned.append(record)

    return assigned


def partition_pairs(
    assigned_pairs: Sequence[Mapping[str, object]],
    event_assignments: Mapping[str, Sequence[int]],
) -> dict[str, list[dict[str, object]]]:
    """Partition complete revisit events into named splits."""

    if not assigned_pairs:
        raise ValueError("Assigned pairs cannot be empty.")

    if not event_assignments:
        raise ValueError("Event assignments cannot be empty.")

    event_to_split: dict[int, str] = {}

    for split_name, event_ids in event_assignments.items():
        if not split_name:
            raise ValueError("Split names cannot be empty.")

        for raw_event_id in event_ids:
            event_id = int(raw_event_id)

            if event_id in event_to_split:
                previous_split = event_to_split[event_id]

                raise ValueError(
                    f"Event {event_id} was assigned to both "
                    f"{previous_split!r} and {split_name!r}."
                )

            event_to_split[event_id] = split_name

    partitions: dict[str, list[dict[str, object]]] = {
        split_name: []
        for split_name in event_assignments
    }

    for row in assigned_pairs:
        event_id = _row_integer(row, "event_id")

        if event_id not in event_to_split:
            raise ValueError(
                f"Event {event_id} has no configured split."
            )

        split_name = event_to_split[event_id]

        record = dict(row)
        record["split"] = split_name
        partitions[split_name].append(record)

    empty_splits = [
        split_name
        for split_name, rows in partitions.items()
        if not rows
    ]

    if empty_splits:
        raise ValueError(
            f"Configured splits contain no pairs: {empty_splits}"
        )

    return partitions


def frames_used(
    pair_rows: Sequence[Mapping[str, object]],
) -> set[int]:
    """Return all query and reference frame IDs in a pair collection."""

    frames: set[int] = set()

    for row in pair_rows:
        frames.add(_row_integer(row, "query_frame"))
        frames.add(_row_integer(row, "reference_frame"))

    return frames


def validate_disjoint_frames(
    partitions: Mapping[
        str,
        Sequence[Mapping[str, object]],
    ],
    left_split: str = "calibration",
    right_split: str = "test",
) -> set[int]:
    """Reject frame leakage between two independent splits."""

    if left_split not in partitions:
        raise ValueError(f"Missing split: {left_split}")

    if right_split not in partitions:
        raise ValueError(f"Missing split: {right_split}")

    overlap = (
        frames_used(partitions[left_split])
        & frames_used(partitions[right_split])
    )

    if overlap:
        sample = sorted(overlap)[:20]

        raise ValueError(
            f"Frame leakage between {left_split!r} and "
            f"{right_split!r}: {len(overlap)} overlapping "
            f"frames. Sample: {sample}"
        )

    return overlap


def summarize_partition(
    pair_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize one positive-pair partition."""

    query_frames = {
        _row_integer(row, "query_frame")
        for row in pair_rows
    }

    reference_frames = {
        _row_integer(row, "reference_frame")
        for row in pair_rows
    }

    event_ids = sorted(
        {
            _row_integer(row, "event_id")
            for row in pair_rows
        }
    )

    all_frames = query_frames | reference_frames

    return {
        "event_ids": event_ids,
        "positive_pairs": len(pair_rows),
        "unique_query_frames": len(query_frames),
        "unique_reference_frames": len(reference_frames),
        "unique_all_frames": len(all_frames),
        "query_frame_min": (
            min(query_frames) if query_frames else None
        ),
        "query_frame_max": (
            max(query_frames) if query_frames else None
        ),
        "reference_frame_min": (
            min(reference_frames)
            if reference_frames
            else None
        ),
        "reference_frame_max": (
            max(reference_frames)
            if reference_frames
            else None
        ),
    }
