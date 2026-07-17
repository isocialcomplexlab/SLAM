"""Avaliação L=1 de scores ResNet-101 com calibração isolada."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from loop_closure.evaluation.binary_metrics import (
    calibrate_threshold_max_f1,
    compute_ranking_metrics,
    evaluate_at_threshold,
    precision_recall_curve_binary,
    roc_curve_binary,
)


SCORE_METHODS = {
    "legacy_exact": {
        "score_field": "legacy_exact_score",
        "distance_field": "legacy_exact_distance",
        "distance_name": "L-infinity/Chebyshev",
    },
    "corrected_l2": {
        "score_field": "corrected_l2_score",
        "distance_field": "corrected_l2_distance",
        "distance_name": "Euclidean L2",
    },
}


@dataclass(frozen=True)
class BinaryScoreSplit:
    """Dados binários de um único split experimental."""

    split: str
    source_path: Path
    source_sha256: str
    total_rows: int
    binary_rows: int
    ambiguous_rows: int
    positive_rows: int
    negative_rows: int
    labels: np.ndarray
    scores: dict[str, np.ndarray]


def sha256_file(path: Path) -> str:
    """Calcula o SHA-256 de um arquivo."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_binary_score_split(
    path: str | Path,
    *,
    expected_split: str,
) -> BinaryScoreSplit:
    """Carrega scores, excluindo ambíguos das métricas binárias."""
    source_path = Path(path)

    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    required_columns = {
        "split",
        "pair_class",
        "binary_label",
        *(
            config["score_field"]
            for config in SCORE_METHODS.values()
        ),
    }

    labels: list[int] = []
    score_values: dict[str, list[float]] = {
        method: []
        for method in SCORE_METHODS
    }

    total_rows = 0
    ambiguous_rows = 0
    positive_rows = 0
    negative_rows = 0

    with source_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        missing_columns = (
            required_columns
            - set(reader.fieldnames or [])
        )

        if missing_columns:
            raise ValueError(
                f"{source_path}: colunas ausentes: "
                f"{sorted(missing_columns)}"
            )

        for line_number, row in enumerate(
            reader,
            start=2,
        ):
            total_rows += 1

            row_split = row["split"].strip()

            if row_split != expected_split:
                raise ValueError(
                    f"{source_path}, linha {line_number}: "
                    f"split={row_split!r}, "
                    f"esperado={expected_split!r}."
                )

            pair_class = row["pair_class"].strip()
            raw_label = row["binary_label"].strip()

            if pair_class == "ambiguous":
                if raw_label:
                    raise ValueError(
                        f"{source_path}, linha {line_number}: "
                        "par ambíguo possui rótulo binário."
                    )

                ambiguous_rows += 1
                continue

            if raw_label not in {"0", "1"}:
                raise ValueError(
                    f"{source_path}, linha {line_number}: "
                    f"binary_label inválido: {raw_label!r}."
                )

            label = int(raw_label)

            if pair_class == "positive":
                if label != 1:
                    raise ValueError(
                        "Classe positiva deve possuir rótulo 1."
                    )
                positive_rows += 1

            elif pair_class == "negative":
                if label != 0:
                    raise ValueError(
                        "Classe negativa deve possuir rótulo 0."
                    )
                negative_rows += 1

            else:
                raise ValueError(
                    f"{source_path}, linha {line_number}: "
                    f"pair_class inválida: {pair_class!r}."
                )

            labels.append(label)

            for method, config in SCORE_METHODS.items():
                try:
                    score = float(
                        row[config["score_field"]]
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{source_path}, linha {line_number}: "
                        f"score inválido para {method}."
                    ) from error

                if not math.isfinite(score):
                    raise ValueError(
                        f"{source_path}, linha {line_number}: "
                        f"score não finito para {method}."
                    )

                score_values[method].append(score)

    label_array = np.asarray(
        labels,
        dtype=np.int64,
    )

    score_arrays = {
        method: np.asarray(
            values,
            dtype=np.float64,
        )
        for method, values in score_values.items()
    }

    binary_rows = len(label_array)

    if binary_rows != positive_rows + negative_rows:
        raise RuntimeError(
            "Contagem de pares binários inconsistente."
        )

    if total_rows != binary_rows + ambiguous_rows:
        raise RuntimeError(
            "Contagem total de pares inconsistente."
        )

    for method, values in score_arrays.items():
        if len(values) != binary_rows:
            raise RuntimeError(
                f"Quantidade de scores inconsistente para {method}."
            )

    return BinaryScoreSplit(
        split=expected_split,
        source_path=source_path,
        source_sha256=sha256_file(source_path),
        total_rows=total_rows,
        binary_rows=binary_rows,
        ambiguous_rows=ambiguous_rows,
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        labels=label_array,
        scores=score_arrays,
    )


def evaluate_l1_protocol(
    splits: Mapping[str, BinaryScoreSplit],
) -> dict[str, Any]:
    """Calibra em calibration e avalia test/stress sem reajuste."""
    required_splits = {
        "calibration",
        "test",
        "stress",
    }

    if set(splits) != required_splits:
        raise ValueError(
            "São necessários exatamente calibration, test e stress."
        )

    for split_name, data in splits.items():
        if data.split != split_name:
            raise ValueError(
                f"A chave {split_name!r} não corresponde "
                f"ao conteúdo {data.split!r}."
            )

    calibration = splits["calibration"]
    thresholds: dict[str, dict[str, Any]] = {}

    for method, config in SCORE_METHODS.items():
        calibrated = calibrate_threshold_max_f1(
            calibration.labels,
            calibration.scores[method],
        )

        score_threshold = float(
            calibrated["threshold"]
        )

        thresholds[method] = {
            **calibrated,
            "calibrated_on_split": "calibration",
            "score_field": config["score_field"],
            "distance_field": config["distance_field"],
            "distance_name": config["distance_name"],
            "score_semantics": "larger_is_more_similar",
            "distance_semantics": "smaller_is_more_similar",
            "score_threshold": score_threshold,
            "distance_threshold": -score_threshold,
            "applied_without_recalibration_to": [
                "test",
                "stress",
            ],
        }

    split_results: dict[str, Any] = {}

    for split_name in (
        "calibration",
        "test",
        "stress",
    ):
        data = splits[split_name]
        method_results: dict[str, Any] = {}

        for method in SCORE_METHODS:
            threshold = float(
                thresholds[method]["score_threshold"]
            )

            method_results[method] = {
                "ranking": compute_ranking_metrics(
                    data.labels,
                    data.scores[method],
                ),
                "operating_point": evaluate_at_threshold(
                    data.labels,
                    data.scores[method],
                    threshold,
                ),
                "threshold_source_split": "calibration",
                "recalibrated_on_this_split": False,
            }

        split_results[split_name] = {
            "source_file": str(data.source_path),
            "source_sha256": data.source_sha256,
            "total_rows": data.total_rows,
            "binary_rows": data.binary_rows,
            "ambiguous_rows_excluded_from_binary_metrics": (
                data.ambiguous_rows
            ),
            "positive_rows": data.positive_rows,
            "negative_rows": data.negative_rows,
            "methods": method_results,
        }

    return {
        "dataset": "KITTI",
        "sequence": "05",
        "evaluation_level": "descriptor_L1",
        "protocol": {
            "calibration_split": "calibration",
            "threshold_criterion": "maximum_f1",
            "test_threshold_recalibration": False,
            "stress_threshold_recalibration": False,
            "ambiguous_pairs": (
                "preserved_in_score_files_"
                "but_excluded_from_binary_metrics"
            ),
            "positive_definition": (
                "spatial_distance_m <= 5.0"
            ),
            "ambiguous_definition": (
                "5.0 < spatial_distance_m < 10.0"
            ),
            "negative_definition": (
                "spatial_distance_m >= 10.0"
            ),
            "prediction_rule": (
                "score >= calibrated_threshold"
            ),
        },
        "thresholds": thresholds,
        "splits": split_results,
    }


def _check_output_paths(
    paths: list[Path],
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return

    existing = [
        path
        for path in paths
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Artefatos já existentes: "
            + ", ".join(
                str(path)
                for path in existing
            )
        )


def write_l1_artifacts(
    evaluation: Mapping[str, Any],
    splits: Mapping[str, BinaryScoreSplit],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Grava métricas, limiares, matrizes e curvas."""
    root = Path(output_dir)
    curves_dir = root / "curves"

    root.mkdir(
        parents=True,
        exist_ok=True,
    )
    curves_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    primary_paths = [
        root / "threshold.json",
        root / "metrics_l1.json",
        root / "metrics_summary.csv",
        root / "confusion_matrices.csv",
        root / "curves_manifest.json",
    ]

    curve_paths: list[Path] = []

    for split_name in (
        "calibration",
        "test",
        "stress",
    ):
        for method in SCORE_METHODS:
            curve_paths.extend(
                [
                    curves_dir
                    / f"{split_name}_{method}_roc.csv",
                    curves_dir
                    / f"{split_name}_{method}_pr.csv",
                ]
            )

    output_paths = primary_paths + curve_paths

    _check_output_paths(
        output_paths,
        overwrite=overwrite,
    )

    threshold_payload = {
        "dataset": evaluation["dataset"],
        "sequence": evaluation["sequence"],
        "calibration_split": "calibration",
        "thresholds": evaluation["thresholds"],
    }

    (root / "threshold.json").write_text(
        json.dumps(
            threshold_payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    (root / "metrics_l1.json").write_text(
        json.dumps(
            evaluation,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary_fields = [
        "split",
        "method",
        "sample_count",
        "positive_count",
        "negative_count",
        "positive_prevalence",
        "roc_auc",
        "pr_auc",
        "average_precision",
        "threshold",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "precision",
        "recall",
        "specificity",
        "f1",
        "accuracy",
        "balanced_accuracy",
        "matthews_correlation_coefficient",
    ]

    with (root / "metrics_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=summary_fields,
        )
        writer.writeheader()

        for split_name in (
            "calibration",
            "test",
            "stress",
        ):
            for method in SCORE_METHODS:
                result = evaluation[
                    "splits"
                ][split_name]["methods"][method]

                combined = {
                    "split": split_name,
                    "method": method,
                    **result["ranking"],
                    **result["operating_point"],
                }

                writer.writerow(
                    {
                        field: combined.get(field, "")
                        for field in summary_fields
                    }
                )

    confusion_fields = [
        "split",
        "method",
        "threshold_source_split",
        "threshold",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    ]

    with (root / "confusion_matrices.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=confusion_fields,
        )
        writer.writeheader()

        for split_name in (
            "calibration",
            "test",
            "stress",
        ):
            for method in SCORE_METHODS:
                result = evaluation[
                    "splits"
                ][split_name]["methods"][method]

                operating = result["operating_point"]

                writer.writerow(
                    {
                        "split": split_name,
                        "method": method,
                        "threshold_source_split": (
                            "calibration"
                        ),
                        "threshold": operating[
                            "threshold"
                        ],
                        "true_positive": operating[
                            "true_positive"
                        ],
                        "false_positive": operating[
                            "false_positive"
                        ],
                        "true_negative": operating[
                            "true_negative"
                        ],
                        "false_negative": operating[
                            "false_negative"
                        ],
                    }
                )

    curve_manifest: list[dict[str, Any]] = []

    for split_name in (
        "calibration",
        "test",
        "stress",
    ):
        data = splits[split_name]

        for method in SCORE_METHODS:
            labels = data.labels
            scores = data.scores[method]

            fpr, tpr, roc_thresholds = (
                roc_curve_binary(
                    labels,
                    scores,
                )
            )

            roc_path = (
                curves_dir
                / f"{split_name}_{method}_roc.csv"
            )

            with roc_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "false_positive_rate",
                        "true_positive_rate",
                        "threshold",
                    ]
                )

                for values in zip(
                    fpr,
                    tpr,
                    roc_thresholds,
                ):
                    writer.writerow(values)

            precision, recall, pr_thresholds = (
                precision_recall_curve_binary(
                    labels,
                    scores,
                )
            )

            pr_path = (
                curves_dir
                / f"{split_name}_{method}_pr.csv"
            )

            with pr_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "precision",
                        "recall",
                        "threshold",
                    ]
                )

                for index in range(
                    len(precision)
                ):
                    threshold: float | str = ""

                    if index < len(pr_thresholds):
                        threshold = float(
                            pr_thresholds[index]
                        )

                    writer.writerow(
                        [
                            float(precision[index]),
                            float(recall[index]),
                            threshold,
                        ]
                    )

            curve_manifest.extend(
                [
                    {
                        "split": split_name,
                        "method": method,
                        "curve": "roc",
                        "path": str(roc_path),
                        "sha256": sha256_file(
                            roc_path
                        ),
                        "point_count": len(fpr),
                    },
                    {
                        "split": split_name,
                        "method": method,
                        "curve": "precision_recall",
                        "path": str(pr_path),
                        "sha256": sha256_file(
                            pr_path
                        ),
                        "point_count": len(
                            precision
                        ),
                    },
                ]
            )

    (root / "curves_manifest.json").write_text(
        json.dumps(
            {
                "dataset": evaluation["dataset"],
                "sequence": evaluation["sequence"],
                "curves": curve_manifest,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return output_paths
