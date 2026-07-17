from __future__ import annotations

import csv
from pathlib import Path

import pytest

from loop_closure.evaluation.resnet101_l1_evaluation import (
    evaluate_l1_protocol,
    load_binary_score_split,
)


FIELDNAMES = [
    "split",
    "pair_class",
    "binary_label",
    "legacy_exact_score",
    "corrected_l2_score",
]


def write_split(
    path: Path,
    split: str,
    rows: list[
        tuple[str, str, float, float]
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
            pair_class,
            label,
            legacy_score,
            corrected_score,
        ) in rows:
            writer.writerow(
                {
                    "split": split,
                    "pair_class": pair_class,
                    "binary_label": label,
                    "legacy_exact_score": (
                        legacy_score
                    ),
                    "corrected_l2_score": (
                        corrected_score
                    ),
                }
            )


def make_splits(
    tmp_path: Path,
):
    definitions = {
        "calibration": [
            ("positive", "1", 0.9, 9.0),
            ("positive", "1", 0.8, 8.0),
            ("negative", "0", 0.7, 7.0),
            ("negative", "0", 0.1, 1.0),
            ("ambiguous", "", 0.5, 5.0),
        ],
        "test": [
            ("positive", "1", 0.85, 8.5),
            ("negative", "0", 0.75, 7.5),
            ("ambiguous", "", 0.6, 6.0),
        ],
        "stress": [
            ("positive", "1", 0.79, 7.9),
            ("negative", "0", 0.2, 2.0),
            ("ambiguous", "", 0.4, 4.0),
        ],
    }

    result = {}

    for split_name, rows in (
        definitions.items()
    ):
        path = (
            tmp_path
            / f"{split_name}.csv"
        )

        write_split(
            path,
            split_name,
            rows,
        )

        result[split_name] = (
            load_binary_score_split(
                path,
                expected_split=split_name,
            )
        )

    return result


def test_loader_excludes_ambiguous_pairs(
    tmp_path: Path,
) -> None:
    splits = make_splits(tmp_path)
    calibration = splits["calibration"]

    assert calibration.total_rows == 5
    assert calibration.binary_rows == 4
    assert calibration.ambiguous_rows == 1
    assert calibration.positive_rows == 2
    assert calibration.negative_rows == 2


def test_thresholds_are_calibrated_only_on_calibration(
    tmp_path: Path,
) -> None:
    evaluation = evaluate_l1_protocol(
        make_splits(tmp_path)
    )

    assert evaluation[
        "thresholds"
    ][
        "legacy_exact"
    ][
        "score_threshold"
    ] == pytest.approx(0.8)

    assert evaluation[
        "thresholds"
    ][
        "corrected_l2"
    ][
        "score_threshold"
    ] == pytest.approx(8.0)

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
                "splits"
            ][split_name]["methods"][method]

            assert result[
                "threshold_source_split"
            ] == "calibration"

            assert result[
                "recalibrated_on_this_split"
            ] is False


def test_stress_uses_calibration_threshold_without_refit(
    tmp_path: Path,
) -> None:
    evaluation = evaluate_l1_protocol(
        make_splits(tmp_path)
    )

    legacy_stress = evaluation[
        "splits"
    ][
        "stress"
    ][
        "methods"
    ][
        "legacy_exact"
    ][
        "operating_point"
    ]

    corrected_stress = evaluation[
        "splits"
    ][
        "stress"
    ][
        "methods"
    ][
        "corrected_l2"
    ][
        "operating_point"
    ]

    assert legacy_stress[
        "threshold"
    ] == pytest.approx(0.8)

    assert corrected_stress[
        "threshold"
    ] == pytest.approx(8.0)

    assert legacy_stress[
        "false_negative"
    ] == 1

    assert corrected_stress[
        "false_negative"
    ] == 1
