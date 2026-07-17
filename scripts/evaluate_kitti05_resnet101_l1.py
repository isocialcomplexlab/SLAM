#!/usr/bin/env python3
"""Executa a avaliação L=1 dos scores ResNet-101 no KITTI 05."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loop_closure.evaluation.resnet101_l1_evaluation import (
    evaluate_l1_protocol,
    load_binary_score_split,
    write_l1_artifacts,
)


EXPECTED_COUNTS = {
    "calibration": {
        "total": 78624,
        "binary": 75052,
        "ambiguous": 3572,
        "positive": 3985,
        "negative": 71067,
    },
    "test": {
        "total": 14170,
        "binary": 12984,
        "ambiguous": 1186,
        "positive": 1269,
        "negative": 11715,
    },
    "stress": {
        "total": 9636,
        "binary": 8410,
        "ambiguous": 1226,
        "positive": 1296,
        "negative": 7114,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Avalia scores ResNet-101 L=1 usando "
            "limiares calibrados somente em calibration."
        )
    )

    parser.add_argument(
        "--scores-dir",
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
    splits = {}

    for split_name in (
        "calibration",
        "test",
        "stress",
    ):
        score_path = (
            args.scores_dir
            / f"{split_name}_scores.csv"
        )

        print(
            f"Carregando {split_name}: "
            f"{score_path}",
            flush=True,
        )

        data = load_binary_score_split(
            score_path,
            expected_split=split_name,
        )

        actual_counts = {
            "total": data.total_rows,
            "binary": data.binary_rows,
            "ambiguous": data.ambiguous_rows,
            "positive": data.positive_rows,
            "negative": data.negative_rows,
        }

        expected_counts = EXPECTED_COUNTS[
            split_name
        ]

        if actual_counts != expected_counts:
            raise ValueError(
                f"{split_name}: contagens inesperadas. "
                f"Esperado={expected_counts}; "
                f"obtido={actual_counts}."
            )

        splits[split_name] = data

        print(
            f"  contagens: {actual_counts}",
            flush=True,
        )

    evaluation = evaluate_l1_protocol(
        splits
    )

    write_l1_artifacts(
        evaluation,
        splits,
        args.output_dir,
        overwrite=args.overwrite,
    )

    print()
    print("=" * 79)
    print("AVALIAÇÃO L=1 CONCLUÍDA")
    print("=" * 79)

    for method, threshold_data in (
        evaluation["thresholds"].items()
    ):
        print()
        print(method)
        print(
            "  limiar de score    :",
            threshold_data[
                "score_threshold"
            ],
        )
        print(
            "  limiar de distância:",
            threshold_data[
                "distance_threshold"
            ],
        )
        print(
            "  F1 na calibração   :",
            threshold_data["f1"],
        )

        for split_name in (
            "calibration",
            "test",
            "stress",
        ):
            result = evaluation[
                "splits"
            ][split_name]["methods"][method]

            ranking = result["ranking"]
            operating = result[
                "operating_point"
            ]

            print(
                f"  {split_name:11s} "
                f"ROC-AUC="
                f"{ranking['roc_auc']:.6f} "
                f"PR-AUC="
                f"{ranking['pr_auc']:.6f} "
                f"AP="
                f"{ranking['average_precision']:.6f} "
                f"F1="
                f"{operating['f1']:.6f}"
            )

    print()
    print(
        "Artefatos:",
        json.dumps(
            {
                "threshold": str(
                    args.output_dir
                    / "threshold.json"
                ),
                "metrics_l1": str(
                    args.output_dir
                    / "metrics_l1.json"
                ),
                "metrics_summary": str(
                    args.output_dir
                    / "metrics_summary.csv"
                ),
                "confusion_matrices": str(
                    args.output_dir
                    / "confusion_matrices.csv"
                ),
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
