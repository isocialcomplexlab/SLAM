from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from loop_closure.evaluation.temporal_ablation import (
    PairScoreRow,
    PairScoreTable,
    build_temporal_window_split,
    evaluate_temporal_protocol,
    load_pair_score_table,
    write_temporal_artifacts,
)


FIELDNAMES = [
    "split",
    "query_frame",
    "reference_frame",
    "pair_class",
    "binary_label",
    "legacy_exact_score",
    "corrected_l2_score",
]


def row(
    query: int,
    reference: int,
    pair_class: str,
    legacy: float,
    corrected: float,
) -> PairScoreRow:
    labels = {
        "positive": 1,
        "negative": 0,
        "ambiguous": None,
    }

    return PairScoreRow(
        query_frame=query,
        reference_frame=reference,
        pair_class=pair_class,
        binary_label=labels[pair_class],
        scores={
            "legacy_exact": legacy,
            "corrected_l2": corrected,
        },
    )


def table(
    split: str,
    rows: list[PairScoreRow],
) -> PairScoreTable:
    return PairScoreTable(
        split=split,
        source_path=Path(
            f"{split}.csv"
        ),
        source_sha256="a" * 64,
        total_rows=len(rows),
        query_count=len(
            {
                item.query_frame
                for item in rows
            }
        ),
        rows=tuple(rows),
    )


def write_csv(
    path: Path,
    split: str,
    rows: list[
        tuple[
            int,
            int,
            str,
            str,
            float,
            float,
        ]
    ],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
        )
        writer.writeheader()

        for (
            query_frame,
            reference_frame,
            pair_class,
            binary_label,
            legacy_score,
            corrected_score,
        ) in rows:
            writer.writerow(
                {
                    "split": split,
                    "query_frame": query_frame,
                    "reference_frame": (
                        reference_frame
                    ),
                    "pair_class": pair_class,
                    "binary_label": binary_label,
                    "legacy_exact_score": (
                        legacy_score
                    ),
                    "corrected_l2_score": (
                        corrected_score
                    ),
                }
            )


def test_loader_sorts_rows_and_validates_classes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calibration.csv"

    write_csv(
        path,
        "calibration",
        [
            (10, 2, "negative", "0", 0.2, 2.0),
            (10, 0, "positive", "1", 0.9, 9.0),
            (10, 1, "ambiguous", "", 0.5, 5.0),
        ],
    )

    observed = load_pair_score_table(
        path,
        expected_split="calibration",
    )

    assert [
        item.reference_frame
        for item in observed.rows
    ] == [0, 1, 2]
    assert observed.total_rows == 3
    assert observed.query_count == 1


def test_loader_rejects_duplicate_pairs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calibration.csv"

    write_csv(
        path,
        "calibration",
        [
            (10, 0, "positive", "1", 0.9, 9.0),
            (10, 0, "positive", "1", 0.8, 8.0),
        ],
    )

    with pytest.raises(
        ValueError,
        match="duplicado",
    ):
        load_pair_score_table(
            path,
            expected_split="calibration",
        )


def test_l1_preserves_pair_labels_and_scores() -> None:
    source = table(
        "calibration",
        [
            row(10, 0, "positive", 0.9, 9.0),
            row(10, 1, "ambiguous", 0.5, 5.0),
            row(10, 2, "negative", 0.1, 1.0),
        ],
    )

    observed = build_temporal_window_split(
        source,
        sequence_length=1,
    )

    assert observed.contiguous_windows == 3
    assert observed.ambiguous_windows == 1
    assert observed.binary_windows == 2
    assert observed.labels.tolist() == [1, 0]
    assert observed.scores[
        "legacy_exact"
    ].tolist() == pytest.approx(
        [0.9, 0.1]
    )


def test_window_score_is_minimum_constituent_score() -> None:
    source = table(
        "calibration",
        [
            row(10, 0, "positive", 0.9, 9.0),
            row(10, 1, "positive", 0.7, 7.0),
            row(10, 2, "negative", 0.4, 4.0),
        ],
    )

    observed = build_temporal_window_split(
        source,
        sequence_length=2,
    )

    assert observed.labels.tolist() == [1, 0]
    assert observed.scores[
        "legacy_exact"
    ].tolist() == pytest.approx(
        [0.7, 0.4]
    )
    assert observed.scores[
        "corrected_l2"
    ].tolist() == pytest.approx(
        [7.0, 4.0]
    )


def test_window_with_ambiguous_constituent_is_excluded() -> None:
    source = table(
        "calibration",
        [
            row(10, 0, "positive", 0.9, 9.0),
            row(10, 1, "ambiguous", 0.8, 8.0),
            row(10, 2, "negative", 0.1, 1.0),
            row(10, 3, "negative", 0.2, 2.0),
        ],
    )

    observed = build_temporal_window_split(
        source,
        sequence_length=2,
    )

    assert observed.contiguous_windows == 3
    assert observed.ambiguous_windows == 2
    assert observed.binary_windows == 1
    assert observed.labels.tolist() == [0]


def test_nonconsecutive_references_do_not_form_window() -> None:
    source = table(
        "calibration",
        [
            row(10, 0, "positive", 0.9, 9.0),
            row(10, 1, "positive", 0.8, 8.0),
            row(10, 3, "negative", 0.2, 2.0),
            row(10, 4, "negative", 0.1, 1.0),
        ],
    )

    observed = build_temporal_window_split(
        source,
        sequence_length=2,
    )

    assert observed.candidate_window_positions == 3
    assert observed.contiguous_windows == 2
    assert (
        observed.nonconsecutive_windows_skipped
        == 1
    )
    assert observed.labels.tolist() == [1, 0]


def make_protocol_windows():
    definitions = {
        "calibration": [
            row(10, 0, "positive", 0.9, 9.0),
            row(10, 1, "positive", 0.8, 8.0),
            row(10, 2, "negative", 0.7, 7.0),
            row(10, 3, "negative", 0.1, 1.0),
        ],
        "test": [
            row(20, 0, "positive", 0.85, 8.5),
            row(20, 1, "positive", 0.75, 7.5),
            row(20, 2, "negative", 0.65, 6.5),
            row(20, 3, "negative", 0.2, 2.0),
        ],
        "stress": [
            row(30, 0, "positive", 0.79, 7.9),
            row(30, 1, "positive", 0.78, 7.8),
            row(30, 2, "negative", 0.3, 3.0),
            row(30, 3, "negative", 0.1, 1.0),
        ],
    }

    result = {}

    for length in (1, 2):
        result[length] = {}

        for split_name, rows in (
            definitions.items()
        ):
            result[length][split_name] = (
                build_temporal_window_split(
                    table(split_name, rows),
                    sequence_length=length,
                )
            )

    return result


def test_thresholds_are_calibrated_per_l_only_on_calibration() -> None:
    evaluation = evaluate_temporal_protocol(
        make_protocol_windows()
    )

    threshold_l1 = evaluation[
        "lengths"
    ]["1"]["thresholds"][
        "legacy_exact"
    ]["score_threshold"]
    threshold_l2 = evaluation[
        "lengths"
    ]["2"]["thresholds"][
        "legacy_exact"
    ]["score_threshold"]

    assert threshold_l1 == pytest.approx(0.8)
    assert threshold_l2 == pytest.approx(0.8)

    for length in ("1", "2"):
        for split_name in (
            "calibration",
            "test",
            "stress",
        ):
            for method in (
                "legacy_exact",
                "corrected_l2",
            ):
                result = evaluation[
                    "lengths"
                ][length]["splits"][
                    split_name
                ]["methods"][method]

                assert result[
                    "threshold_source_split"
                ] == "calibration"
                assert result[
                    "recalibrated_on_this_split"
                ] is False


def test_writer_creates_expected_artifacts(
    tmp_path: Path,
) -> None:
    evaluation = evaluate_temporal_protocol(
        make_protocol_windows()
    )
    paths = write_temporal_artifacts(
        evaluation,
        tmp_path,
    )

    assert len(paths) == 8
    assert (
        tmp_path / "metrics_l1_temporal.json"
    ).is_file()
    assert (
        tmp_path / "metrics_l2.json"
    ).is_file()
    assert (
        tmp_path / "temporal_ablation.csv"
    ).is_file()
    assert (
        tmp_path
        / "temporal_ablation_manifest.json"
    ).is_file()

    manifest = json.loads(
        (
            tmp_path
            / "temporal_ablation_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest[
        "sequence_lengths"
    ] == [1, 2]
    assert len(manifest["artifacts"]) == 7
