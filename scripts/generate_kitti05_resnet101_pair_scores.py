#!/usr/bin/env python3
"""Gera scores contínuos ResNet-101 para os splits do KITTI 05."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from loop_closure.evaluation.resnet101_pair_scores import (
    BASE_OUTPUT_FIELDS,
    compute_pair_score_rows,
    load_descriptor_index_from_npz,
)


EXPECTED = {
    "calibration": {
        "rows": 78624,
        "classes": {
            "positive": 3985,
            "ambiguous": 3572,
            "negative": 71067,
        },
    },
    "test": {
        "rows": 14170,
        "classes": {
            "positive": 1269,
            "ambiguous": 1186,
            "negative": 11715,
        },
    },
    "stress": {
        "rows": 9636,
        "classes": {
            "positive": 1296,
            "ambiguous": 1226,
            "negative": 7114,
        },
    },
}

REQUIRED_PAIR_COLUMNS = {
    "split",
    "query_frame",
    "reference_frame",
    "spatial_distance_m",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def serializable_row(
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: "" if value is None else value
        for key, value in row.items()
    }


def update_score_range(
    ranges: dict[str, dict[str, float]],
    field: str,
    value: Any,
) -> None:
    number = float(value)

    if field not in ranges:
        ranges[field] = {
            "minimum": number,
            "maximum": number,
        }
        return

    ranges[field]["minimum"] = min(
        ranges[field]["minimum"],
        number,
    )
    ranges[field]["maximum"] = max(
        ranges[field]["maximum"],
        number,
    )


def generate_split(
    *,
    split: str,
    pairs_path: Path,
    descriptor_index: dict[
        tuple[str, int],
        np.ndarray,
    ],
    descriptor_npz: Path,
    output_path: Path,
    report_path: Path,
    camera_id: str,
    chunk_size: int,
    progress_every: int,
) -> dict[str, Any]:
    if split not in EXPECTED:
        raise ValueError(
            f"Split não suportado: {split}"
        )

    if not pairs_path.is_file():
        raise FileNotFoundError(
            f"Arquivo de pares não encontrado: {pairs_path}"
        )

    if output_path.exists():
        raise FileExistsError(
            f"O arquivo de saída já existe: {output_path}"
        )

    if report_path.exists():
        raise FileExistsError(
            f"O relatório já existe: {report_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    expected = EXPECTED[split]
    processed = 0
    next_progress = progress_every
    seen_pairs: set[tuple[int, int]] = set()
    class_counts: Counter[str] = Counter()
    score_ranges: dict[str, dict[str, float]] = {}
    started = time.perf_counter()

    try:
        with pairs_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as source:
            reader = csv.DictReader(source)
            input_fields = list(reader.fieldnames or [])

            missing = (
                REQUIRED_PAIR_COLUMNS
                - set(input_fields)
            )

            if missing:
                raise ValueError(
                    f"{pairs_path} não contém as colunas "
                    f"obrigatórias: {sorted(missing)}"
                )

            output_fields = list(
                BASE_OUTPUT_FIELDS
            )
            known_fields = set(output_fields)

            for field in input_fields:
                if field not in known_fields:
                    output_fields.append(field)
                    known_fields.add(field)

            with temporary_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as target:
                writer = csv.DictWriter(
                    target,
                    fieldnames=output_fields,
                    extrasaction="raise",
                )
                writer.writeheader()

                chunk: list[dict[str, str]] = []

                def process_chunk() -> None:
                    nonlocal processed
                    nonlocal next_progress

                    if not chunk:
                        return

                    scored_rows = compute_pair_score_rows(
                        pairs=chunk,
                        descriptor_index=descriptor_index,
                        camera_id=camera_id,
                        expected_split=split,
                    )

                    for scored in scored_rows:
                        pair_class = str(
                            scored["pair_class"]
                        )
                        class_counts[pair_class] += 1

                        for field in (
                            "legacy_exact_distance",
                            "legacy_exact_score",
                            "corrected_l2_distance",
                            "corrected_l2_score",
                        ):
                            update_score_range(
                                score_ranges,
                                field,
                                scored[field],
                            )

                        writer.writerow(
                            serializable_row(scored)
                        )

                    processed += len(scored_rows)
                    chunk.clear()

                    if processed >= next_progress:
                        elapsed = (
                            time.perf_counter()
                            - started
                        )
                        rate = (
                            processed / elapsed
                            if elapsed > 0
                            else 0.0
                        )

                        print(
                            f"[{split}] "
                            f"{processed}/{expected['rows']} "
                            f"pares — {rate:.1f} pares/s",
                            flush=True,
                        )

                        while (
                            next_progress
                            <= processed
                        ):
                            next_progress += (
                                progress_every
                            )

                for source_row_number, row in enumerate(
                    reader,
                    start=2,
                ):
                    row_split = str(
                        row.get("split", "")
                    ).strip()

                    if row_split != split:
                        raise ValueError(
                            "Contaminação entre splits em "
                            f"{pairs_path}, linha "
                            f"{source_row_number}: "
                            f"esperado={split!r}, "
                            f"recebido={row_split!r}."
                        )

                    try:
                        query_frame = int(
                            row["query_frame"]
                        )
                        reference_frame = int(
                            row["reference_frame"]
                        )
                    except (
                        TypeError,
                        ValueError,
                    ) as error:
                        raise ValueError(
                            "Frame inválido em "
                            f"{pairs_path}, linha "
                            f"{source_row_number}."
                        ) from error

                    identity = (
                        query_frame,
                        reference_frame,
                    )

                    if identity in seen_pairs:
                        raise ValueError(
                            "Par duplicado em "
                            f"{pairs_path}, linha "
                            f"{source_row_number}: "
                            f"{identity}."
                        )

                    seen_pairs.add(identity)
                    chunk.append(row)

                    if len(chunk) >= chunk_size:
                        process_chunk()

                process_chunk()

        elapsed = time.perf_counter() - started

        actual_classes = {
            name: int(class_counts.get(name, 0))
            for name in (
                "positive",
                "ambiguous",
                "negative",
            )
        }

        if processed != expected["rows"]:
            raise ValueError(
                f"{split}: total inesperado. "
                f"Esperado={expected['rows']}, "
                f"obtido={processed}."
            )

        if actual_classes != expected["classes"]:
            raise ValueError(
                f"{split}: classes inesperadas. "
                f"Esperado={expected['classes']}, "
                f"obtido={actual_classes}."
            )

        os.replace(
            temporary_path,
            output_path,
        )

        binary_rows = (
            actual_classes["positive"]
            + actual_classes["negative"]
        )

        report = {
            "dataset": "KITTI",
            "sequence": "05",
            "camera_id": camera_id,
            "split": split,
            "descriptor_model": (
                "resnet101_imagenet1k_v1"
            ),
            "descriptor_dimension": 4096,
            "descriptor_file": str(
                descriptor_npz
            ),
            "descriptor_sha256": sha256_file(
                descriptor_npz
            ),
            "pairs_file": str(pairs_path),
            "pairs_sha256": sha256_file(
                pairs_path
            ),
            "scores_file": str(output_path),
            "scores_sha256": sha256_file(
                output_path
            ),
            "row_count": processed,
            "unique_pair_count": len(
                seen_pairs
            ),
            "class_counts": actual_classes,
            "binary_row_count": binary_rows,
            "ambiguous_row_count": (
                actual_classes["ambiguous"]
            ),
            "score_semantics": {
                "legacy_exact_distance": (
                    "L-infinity/Chebyshev; "
                    "smaller is more similar"
                ),
                "legacy_exact_score": (
                    "negative L-infinity distance; "
                    "larger is more similar"
                ),
                "corrected_l2_distance": (
                    "Euclidean L2; "
                    "smaller is more similar"
                ),
                "corrected_l2_score": (
                    "negative Euclidean distance; "
                    "larger is more similar"
                ),
            },
            "ground_truth_definition": {
                "positive": (
                    "spatial_distance_m <= 5.0"
                ),
                "ambiguous": (
                    "5.0 < spatial_distance_m < 10.0"
                ),
                "negative": (
                    "spatial_distance_m >= 10.0"
                ),
            },
            "score_ranges": score_ranges,
            "elapsed_seconds": elapsed,
            "pairs_per_second": (
                processed / elapsed
                if elapsed > 0
                else None
            ),
            "output_size_bytes": (
                output_path.stat().st_size
            ),
        }

        report_path.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            f"[{split}] concluído: "
            f"{processed} pares, "
            f"{elapsed:.3f} s, "
            f"{report['pairs_per_second']:.1f} pares/s",
            flush=True,
        )

        return report

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera scores contínuos ResNet-101 "
            "para calibração, teste e stress."
        )
    )

    parser.add_argument(
        "--descriptor-npz",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--pairs-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--camera-id",
        default="image_0",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.chunk_size <= 0:
        raise ValueError(
            "--chunk-size deve ser positivo."
        )

    if args.progress_every <= 0:
        raise ValueError(
            "--progress-every deve ser positivo."
        )

    print(
        "Carregando descritores:",
        args.descriptor_npz,
        flush=True,
    )

    descriptor_index = (
        load_descriptor_index_from_npz(
            args.descriptor_npz
        )
    )

    print(
        f"Descritores carregados: "
        f"{len(descriptor_index)}",
        flush=True,
    )

    reports: list[dict[str, Any]] = []

    for split in (
        "calibration",
        "test",
        "stress",
    ):
        pairs_path = (
            args.pairs_root
            / f"{split}_pairs.csv"
        )
        output_path = (
            args.output_dir
            / f"{split}_scores.csv"
        )
        report_path = (
            args.output_dir
            / f"{split}_scores_report.json"
        )

        print()
        print(
            f"Iniciando split: {split}",
            flush=True,
        )

        reports.append(
            generate_split(
                split=split,
                pairs_path=pairs_path,
                descriptor_index=(
                    descriptor_index
                ),
                descriptor_npz=(
                    args.descriptor_npz
                ),
                output_path=output_path,
                report_path=report_path,
                camera_id=args.camera_id,
                chunk_size=args.chunk_size,
                progress_every=(
                    args.progress_every
                ),
            )
        )

    summary_path = (
        args.output_dir
        / "pair_scores_summary.json"
    )

    summary = {
        "dataset": "KITTI",
        "sequence": "05",
        "splits_are_separate": True,
        "total_row_count": sum(
            report["row_count"]
            for report in reports
        ),
        "total_binary_row_count": sum(
            report["binary_row_count"]
            for report in reports
        ),
        "total_ambiguous_row_count": sum(
            report["ambiguous_row_count"]
            for report in reports
        ),
        "reports": reports,
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 79)
    print("GERAÇÃO CONCLUÍDA")
    print("=" * 79)
    print(
        "Total de pares:",
        summary["total_row_count"],
    )
    print(
        "Pares binários:",
        summary[
            "total_binary_row_count"
        ],
    )
    print(
        "Pares ambíguos:",
        summary[
            "total_ambiguous_row_count"
        ],
    )
    print(
        "Resumo:",
        summary_path,
    )


if __name__ == "__main__":
    main()
