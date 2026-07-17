#!/usr/bin/env python3
"""Gera as figuras ROC e precisão-revocação do piloto KITTI 05."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


METHODS = {
    "legacy_exact": "Legacy exact (L∞)",
    "corrected_l2": "Corrected L2",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_curve(
    path: Path,
    *,
    x_field: str,
    y_field: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)

    x_values: list[float] = []
    y_values: list[float] = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        missing = {
            x_field,
            y_field,
        } - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"{path}: colunas ausentes: {sorted(missing)}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                x_value = float(row[x_field])
                y_value = float(row[y_field])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}, linha {line_number}: ponto inválido."
                ) from error

            if not (
                math.isfinite(x_value)
                and math.isfinite(y_value)
            ):
                raise ValueError(
                    f"{path}, linha {line_number}: ponto não finito."
                )

            x_values.append(x_value)
            y_values.append(y_value)

    if len(x_values) < 2:
        raise ValueError(
            f"{path}: a curva precisa de pelo menos dois pontos."
        )

    return (
        np.asarray(x_values, dtype=np.float64),
        np.asarray(y_values, dtype=np.float64),
    )


def load_test_metrics(
    path: Path,
) -> dict[str, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    metrics: dict[str, dict[str, float]] = {}

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "split",
            "method",
            "positive_prevalence",
            "roc_auc",
            "pr_auc",
            "average_precision",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"{path}: colunas ausentes: {sorted(missing)}"
            )

        for row in reader:
            if row["split"] != "test":
                continue

            method = row["method"]

            if method not in METHODS:
                continue

            metrics[method] = {
                "positive_prevalence": float(
                    row["positive_prevalence"]
                ),
                "roc_auc": float(row["roc_auc"]),
                "pr_auc": float(row["pr_auc"]),
                "average_precision": float(
                    row["average_precision"]
                ),
            }

    if set(metrics) != set(METHODS):
        raise ValueError(
            "As métricas do split test não contêm os dois métodos."
        )

    return metrics


def save_roc_figure(
    *,
    curves_dir: Path,
    metrics: dict[str, dict[str, float]],
    output_path: Path,
) -> list[Path]:
    source_paths: list[Path] = []

    figure, axis = plt.subplots(
        figsize=(7.2, 5.4),
        constrained_layout=True,
    )

    for method, label in METHODS.items():
        curve_path = (
            curves_dir
            / f"test_{method}_roc.csv"
        )
        source_paths.append(curve_path)

        false_positive_rate, true_positive_rate = load_curve(
            curve_path,
            x_field="false_positive_rate",
            y_field="true_positive_rate",
        )

        axis.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=1.8,
            label=(
                f"{label} "
                f"(AUC={metrics[method]['roc_auc']:.3f})"
            ),
        )

    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.0,
        label="Chance",
    )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(
        "KITTI 05 — Test split ROC (descriptor level, L=1)"
    )
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend(loc="lower right")
    figure.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(figure)

    return source_paths


def save_pr_figure(
    *,
    curves_dir: Path,
    metrics: dict[str, dict[str, float]],
    output_path: Path,
) -> list[Path]:
    source_paths: list[Path] = []

    figure, axis = plt.subplots(
        figsize=(7.2, 5.4),
        constrained_layout=True,
    )

    for method, label in METHODS.items():
        curve_path = (
            curves_dir
            / f"test_{method}_pr.csv"
        )
        source_paths.append(curve_path)

        recall, precision = load_curve(
            curve_path,
            x_field="recall",
            y_field="precision",
        )

        axis.plot(
            recall,
            precision,
            linewidth=1.8,
            label=(
                f"{label} "
                f"(AP={metrics[method]['average_precision']:.3f})"
            ),
        )

    prevalence_values = {
        round(
            values["positive_prevalence"],
            15,
        )
        for values in metrics.values()
    }

    if len(prevalence_values) != 1:
        raise ValueError(
            "A prevalência positiva diverge entre os métodos."
        )

    prevalence = prevalence_values.pop()

    axis.axhline(
        prevalence,
        linestyle="--",
        linewidth=1.0,
        label=(
            f"Prevalence baseline "
            f"({prevalence:.3f})"
        ),
    )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(
        "KITTI 05 — Test split precision–recall "
        "(descriptor level, L=1)"
    )
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend(loc="best")
    figure.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(figure)

    return source_paths


def validate_pdf(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.stat().st_size < 1000:
        raise ValueError(
            f"{path}: PDF inesperadamente pequeno."
        )

    with path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise ValueError(
                f"{path}: assinatura PDF inválida."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metrics-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
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

    curves_dir = args.metrics_dir / "curves"
    metrics_path = (
        args.metrics_dir
        / "metrics_summary.csv"
    )

    output_dir = args.output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    roc_path = output_dir / "roc_curve.pdf"
    pr_path = (
        output_dir
        / "precision_recall_curve.pdf"
    )
    manifest_path = (
        output_dir
        / "figure_manifest.json"
    )

    outputs = [
        roc_path,
        pr_path,
        manifest_path,
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

    metrics = load_test_metrics(metrics_path)

    roc_sources = save_roc_figure(
        curves_dir=curves_dir,
        metrics=metrics,
        output_path=roc_path,
    )

    pr_sources = save_pr_figure(
        curves_dir=curves_dir,
        metrics=metrics,
        output_path=pr_path,
    )

    validate_pdf(roc_path)
    validate_pdf(pr_path)

    source_paths = [
        metrics_path,
        *roc_sources,
        *pr_sources,
    ]

    manifest: dict[str, Any] = {
        "dataset": "KITTI",
        "sequence": "05",
        "split": "test",
        "evaluation_level": "descriptor_L1",
        "ambiguous_pairs": (
            "excluded_from_binary_metrics"
        ),
        "methods": list(METHODS),
        "metrics": metrics,
        "sources": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in source_paths
        ],
        "figures": [
            {
                "path": str(roc_path),
                "type": "roc",
                "sha256": sha256_file(roc_path),
                "size_bytes": roc_path.stat().st_size,
            },
            {
                "path": str(pr_path),
                "type": "precision_recall",
                "sha256": sha256_file(pr_path),
                "size_bytes": pr_path.stat().st_size,
            },
        ],
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
    print(f"  ROC: {roc_path}")
    print(f"  PR : {pr_path}")
    print(f"  Manifesto: {manifest_path}")

    for item in manifest["figures"]:
        print(
            f"  {item['type']}: "
            f"{item['size_bytes']} bytes, "
            f"SHA-256={item['sha256']}"
        )


if __name__ == "__main__":
    main()
