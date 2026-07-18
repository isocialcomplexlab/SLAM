from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.plot_kitti05_resnet101_temporal_ablation import (
    EXPECTED_LENGTHS,
    load_test_rows,
    save_metric_figure,
    validate_pdf,
)


FIELDNAMES = [
    "sequence_length",
    "split",
    "method",
    "binary_windows",
    "positive_windows",
    "negative_windows",
    "roc_auc",
    "pr_auc",
    "average_precision",
    "f1",
]


def write_metrics(path: Path) -> None:
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

        for method_index, method in enumerate(
            (
                "legacy_exact",
                "corrected_l2",
            )
        ):
            for length in EXPECTED_LENGTHS:
                writer.writerow(
                    {
                        "sequence_length": length,
                        "split": "test",
                        "method": method,
                        "binary_windows": 100,
                        "positive_windows": 20,
                        "negative_windows": 80,
                        "roc_auc": (
                            0.7
                            + 0.01 * method_index
                        ),
                        "pr_auc": (
                            0.4
                            + 0.01 * method_index
                        ),
                        "average_precision": (
                            0.41
                            + 0.01 * method_index
                            - 0.001 * length
                        ),
                        "f1": (
                            0.42
                            + 0.01 * method_index
                            - 0.001 * length
                        ),
                    }
                )


def test_load_test_rows_requires_all_lengths(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.csv"
    write_metrics(path)

    observed = load_test_rows(path)

    assert set(observed) == {
        "legacy_exact",
        "corrected_l2",
    }

    for rows in observed.values():
        assert [
            item["sequence_length"]
            for item in rows
        ] == EXPECTED_LENGTHS


def test_save_metric_figure_creates_valid_pdf(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "metrics.csv"
    output_path = tmp_path / "figure.pdf"
    write_metrics(csv_path)

    rows = load_test_rows(csv_path)

    save_metric_figure(
        rows=rows,
        metric="average_precision",
        ylabel="Average precision",
        title="Test figure",
        output_path=output_path,
    )

    validate_pdf(output_path)


def test_loader_rejects_incomplete_lengths(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.csv"
    write_metrics(path)

    rows = list(
        csv.DictReader(
            path.open(
                "r",
                encoding="utf-8",
                newline="",
            )
        )
    )
    rows = [
        row
        for row in rows
        if not (
            row["method"] == "legacy_exact"
            and row["sequence_length"] == "5"
        )
    ]

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
        writer.writerows(rows)

    with pytest.raises(
        ValueError,
        match="esperado",
    ):
        load_test_rows(path)
