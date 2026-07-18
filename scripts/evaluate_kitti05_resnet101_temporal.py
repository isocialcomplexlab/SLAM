#!/usr/bin/env python3
"""Executa a ablação temporal ResNet-101 no KITTI 05."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loop_closure.evaluation.resnet101_l1_evaluation import (
    SCORE_METHODS,
)
from loop_closure.evaluation.temporal_ablation import (
    SPLIT_ORDER,
    build_temporal_window_split,
    evaluate_temporal_protocol,
    load_pair_score_table,
    validate_l1_equivalence,
    write_temporal_artifacts,
)


EXPECTED_SOURCE_COUNTS = {
    "calibration": 78624,
    "test": 14170,
    "stress": 9636,
}


def parse_lengths(value: str) -> list[int]:
    try:
        lengths = sorted(
            {
                int(item.strip())
                for item in value.split(",")
                if item.strip()
            }
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Lista de L inválida: {value!r}."
        ) from error

    if not lengths or any(
        length < 1
        for length in lengths
    ):
        raise argparse.ArgumentTypeError(
            "Os comprimentos L devem ser "
            "inteiros positivos."
        )

    return lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Avalia janelas de uma consulta contra "
            "L referências consecutivas."
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
        "--descriptor-l1-metrics",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--lengths",
        type=parse_lengths,
        default=parse_lengths("1,2,3,5"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = {}

    for split_name in SPLIT_ORDER:
        path = (
            args.scores_dir
            / f"{split_name}_scores.csv"
        )
        print(
            f"Carregando {split_name}: {path}",
            flush=True,
        )
        table = load_pair_score_table(
            path,
            expected_split=split_name,
        )

        expected = EXPECTED_SOURCE_COUNTS[
            split_name
        ]

        if table.total_rows != expected:
            raise ValueError(
                f"{split_name}: esperado "
                f"{expected} pares, obtidos "
                f"{table.total_rows}."
            )

        tables[split_name] = table
        print(
            f"  pares={table.total_rows}, "
            f"queries={table.query_count}",
            flush=True,
        )

    windows = {}

    for length in args.lengths:
        windows[length] = {}

        print()
        print(f"Construindo L={length}...")

        for split_name in SPLIT_ORDER:
            data = build_temporal_window_split(
                tables[split_name],
                sequence_length=length,
            )
            windows[length][split_name] = data

            print(
                f"  {split_name:11s} "
                f"contiguous="
                f"{data.contiguous_windows} "
                f"binary={data.binary_windows} "
                f"positive="
                f"{data.positive_windows} "
                f"negative="
                f"{data.negative_windows} "
                f"ambiguous="
                f"{data.ambiguous_windows} "
                f"skipped_gaps="
                f"{data.nonconsecutive_windows_skipped}"
            )

    evaluation = evaluate_temporal_protocol(
        windows
    )

    if 1 in args.lengths:
        equivalence = validate_l1_equivalence(
            evaluation,
            args.descriptor_l1_metrics,
        )
        evaluation["l1_equivalence"] = (
            equivalence
        )
        print()
        print(
            "Equivalência L=1:",
            json.dumps(
                equivalence,
                ensure_ascii=False,
            ),
        )

    paths = write_temporal_artifacts(
        evaluation,
        args.output_dir,
        overwrite=args.overwrite,
    )

    print()
    print("=" * 79)
    print("ABLAÇÃO TEMPORAL CONCLUÍDA")
    print("=" * 79)

    for length in args.lengths:
        length_data = evaluation[
            "lengths"
        ][str(length)]

        for method in SCORE_METHODS:
            threshold = length_data[
                "thresholds"
            ][method]["score_threshold"]

            test_result = length_data[
                "splits"
            ]["test"]["methods"][method]
            ranking = test_result["ranking"]
            operating = test_result[
                "operating_point"
            ]

            print(
                f"L={length} "
                f"{method:13s} "
                f"threshold={threshold:.9f} "
                f"ROC-AUC="
                f"{ranking['roc_auc']:.6f} "
                f"PR-AUC="
                f"{ranking['pr_auc']:.6f} "
                f"AP="
                f"{ranking['average_precision']:.6f} "
                f"F1={operating['f1']:.6f}"
            )

    print()
    print("Artefatos:")

    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
