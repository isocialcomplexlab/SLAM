#!/usr/bin/env python3
"""Gera figuras da ablação temporal ResNet-101 no KITTI 05."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


METHOD_LABELS = {
    "legacy_exact": "Legacy exact (L∞)",
    "corrected_l2": "Corrected L2",
}

EXPECTED_LENGTHS = [1, 2, 3, 5]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_test_rows(
    path: Path,
) -> dict[str, list[dict[str, float]]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    grouped: dict[str, list[dict[str, float]]] = {
        method: []
        for method in METHOD_LABELS
    }

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
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
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise ValueError(
                f"{path}: colunas ausentes: "
                f"{sorted(missing)}."
            )

        seen: set[tuple[str, int]] = set()

        for line_number, row in enumerate(
            reader,
            start=2,
        ):
            if row["split"] != "test":
                continue

            method = row["method"]

            if method not in METHOD_LABELS:
                continue

            try:
                length = int(
                    row["sequence_length"]
                )
                record = {
                    "sequence_length": length,
                    "binary_windows": int(
                        row["binary_windows"]
                    ),
                    "positive_windows": int(
                        row["positive_windows"]
                    ),
                    "negative_windows": int(
                        row["negative_windows"]
                    ),
                    "roc_auc": float(
                        row["roc_auc"]
                    ),
                    "pr_auc": float(
                        row["pr_auc"]
                    ),
                    "average_precision": float(
                        row["average_precision"]
                    ),
                    "f1": float(row["f1"]),
                }
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}, linha {line_number}: "
                    "valor inválido."
                ) from error

            key = (method, length)

            if key in seen:
                raise ValueError(
                    f"{path}: combinação duplicada "
                    f"{key}."
                )

            seen.add(key)

            for metric in (
                "roc_auc",
                "pr_auc",
                "average_precision",
                "f1",
            ):
                value = float(record[metric])

                if (
                    not math.isfinite(value)
                    or not 0.0 <= value <= 1.0
                ):
                    raise ValueError(
                        f"{path}, linha "
                        f"{line_number}: {metric} "
                        "fora de [0,1]."
                    )

            grouped[method].append(record)

    for method, rows in grouped.items():
        rows.sort(
            key=lambda item: int(
                item["sequence_length"]
            )
        )

        observed = [
            int(item["sequence_length"])
            for item in rows
        ]

        if observed != EXPECTED_LENGTHS:
            raise ValueError(
                f"{method}: L observado={observed}; "
                f"esperado={EXPECTED_LENGTHS}."
            )

    return grouped


def save_metric_figure(
    *,
    rows: dict[
        str,
        list[dict[str, float]],
    ],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(7.2, 5.0),
        constrained_layout=True,
    )

    for method, label in (
        METHOD_LABELS.items()
    ):
        lengths = [
            int(item["sequence_length"])
            for item in rows[method]
        ]
        values = [
            float(item[metric])
            for item in rows[method]
        ]

        axis.plot(
            lengths,
            values,
            marker="o",
            linewidth=1.8,
            label=label,
        )

        for length, value in zip(
            lengths,
            values,
        ):
            axis.annotate(
                f"{value:.3f}",
                (length, value),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )

    axis.set_xticks(EXPECTED_LENGTHS)
    axis.set_xlabel(
        "Temporal reference-window length L"
    )
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(
        True,
        linewidth=0.5,
        alpha=0.35,
    )
    axis.legend(loc="best")

    all_values = [
        float(item[metric])
        for method_rows in rows.values()
        for item in method_rows
    ]
    lower = max(
        0.0,
        min(all_values) - 0.04,
    )
    upper = min(
        1.0,
        max(all_values) + 0.04,
    )

    if upper - lower < 0.08:
        center = (
            max(all_values)
            + min(all_values)
        ) / 2.0
        lower = max(0.0, center - 0.04)
        upper = min(1.0, center + 0.04)

    axis.set_ylim(lower, upper)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def validate_pdf(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.stat().st_size < 1000:
        raise ValueError(
            f"{path}: PDF muito pequeno."
        )

    with path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise ValueError(
                f"{path}: assinatura PDF inválida."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--paper-figures-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ap_path = (
        args.output_dir
        / "temporal_ablation_average_precision.pdf"
    )
    f1_path = (
        args.output_dir
        / "temporal_ablation_f1.pdf"
    )
    manifest_path = (
        args.output_dir
        / "temporal_ablation_figure_manifest.json"
    )

    paper_ap = (
        args.paper_figures_dir
        / "Fig_13_KITTI05_ResNet101_Temporal_AP.pdf"
    )
    paper_f1 = (
        args.paper_figures_dir
        / "Fig_14_KITTI05_ResNet101_Temporal_F1.pdf"
    )

    outputs = [
        ap_path,
        f1_path,
        manifest_path,
        paper_ap,
        paper_f1,
    ]

    if not args.overwrite:
        existing = [
            path
            for path in outputs
            if path.exists()
        ]

        if existing:
            raise FileExistsError(
                "Artefatos já existem: "
                + ", ".join(
                    str(path)
                    for path in existing
                )
            )

    rows = load_test_rows(
        args.metrics_csv
    )

    save_metric_figure(
        rows=rows,
        metric="average_precision",
        ylabel="Average precision",
        title=(
            "KITTI 05 — Temporal ablation "
            "on the held-out test event"
        ),
        output_path=ap_path,
    )

    save_metric_figure(
        rows=rows,
        metric="f1",
        ylabel="F1 score",
        title=(
            "KITTI 05 — Calibrated operating-point "
            "temporal ablation"
        ),
        output_path=f1_path,
    )

    validate_pdf(ap_path)
    validate_pdf(f1_path)

    args.paper_figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copyfile(
        ap_path,
        paper_ap,
    )
    shutil.copyfile(
        f1_path,
        paper_f1,
    )

    validate_pdf(paper_ap)
    validate_pdf(paper_f1)

    if (
        sha256_file(ap_path)
        != sha256_file(paper_ap)
    ):
        raise RuntimeError(
            "A cópia da figura AP diverge."
        )

    if (
        sha256_file(f1_path)
        != sha256_file(paper_f1)
    ):
        raise RuntimeError(
            "A cópia da figura F1 diverge."
        )

    manifest: dict[str, Any] = {
        "dataset": "KITTI",
        "sequence": "05",
        "split": "test",
        "evaluation_level": (
            "temporal_reference_window"
        ),
        "sequence_lengths": EXPECTED_LENGTHS,
        "source": {
            "path": str(args.metrics_csv),
            "sha256": sha256_file(
                args.metrics_csv
            ),
        },
        "figures": [
            {
                "metric": "average_precision",
                "path": str(ap_path),
                "paper_copy": str(paper_ap),
                "sha256": sha256_file(
                    ap_path
                ),
                "size_bytes": (
                    ap_path.stat().st_size
                ),
            },
            {
                "metric": "f1",
                "path": str(f1_path),
                "paper_copy": str(paper_f1),
                "sha256": sha256_file(
                    f1_path
                ),
                "size_bytes": (
                    f1_path.stat().st_size
                ),
            },
        ],
        "methods": list(
            METHOD_LABELS
        ),
        "interpretation": {
            "corrected_l2_best_test_roc_auc_L": 2,
            "corrected_l2_best_test_average_precision_L": 1,
            "corrected_l2_best_test_f1_L": 1,
            "conclusion": (
                "Longer strict temporal windows did not "
                "improve AP or calibrated F1 on the "
                "held-out test event."
            ),
        },
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Figuras geradas:")
    print(f"  {ap_path}")
    print(f"  {f1_path}")
    print("Cópias para o artigo:")
    print(f"  {paper_ap}")
    print(f"  {paper_f1}")
    print(f"Manifesto: {manifest_path}")


if __name__ == "__main__":
    main()
