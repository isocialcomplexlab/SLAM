#!/usr/bin/env python3

"""Audita arquivos CSV de pares candidatos do piloto KITTI 05."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("results/pilot/kitti05_resnet")
OUTPUT = ROOT / "candidate_pair_files_audit.json"

QUERY_COLUMNS = (
    "query_frame",
    "query_frame_id",
    "query_idx",
    "query_id",
    "query",
    "frame_query",
)

REFERENCE_COLUMNS = (
    "reference_frame",
    "reference_frame_id",
    "reference_idx",
    "reference_id",
    "reference",
    "database_frame",
    "db_frame",
    "frame_reference",
)

LABEL_COLUMNS = (
    "label",
    "binary_label",
    "class",
    "pair_class",
    "is_positive",
    "target",
)

SPLIT_COLUMNS = (
    "split",
    "subset",
    "partition",
    "event",
    "event_id",
)

DISTANCE_COLUMNS = (
    "distance_m",
    "spatial_distance_m",
    "distance",
    "pose_distance_m",
)


def first_present(
    fieldnames: list[str],
    candidates: tuple[str, ...],
) -> str | None:
    lookup = {name.lower(): name for name in fieldnames}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def parse_number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def update_range(
    ranges: dict[str, dict[str, float]],
    column: str,
    value: str,
) -> None:
    number = parse_number(value)

    if number is None:
        return

    current = ranges.setdefault(
        column,
        {"minimum": number, "maximum": number},
    )

    current["minimum"] = min(current["minimum"], number)
    current["maximum"] = max(current["maximum"], number)


def inspect_csv(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        fieldnames = list(reader.fieldnames or [])

        query_column = first_present(
            fieldnames,
            QUERY_COLUMNS,
        )
        reference_column = first_present(
            fieldnames,
            REFERENCE_COLUMNS,
        )
        label_column = first_present(
            fieldnames,
            LABEL_COLUMNS,
        )
        split_column = first_present(
            fieldnames,
            SPLIT_COLUMNS,
        )
        distance_column = first_present(
            fieldnames,
            DISTANCE_COLUMNS,
        )

        row_count = 0
        duplicate_pair_count = 0
        seen_pairs: set[tuple[str, str]] = set()

        label_counts: Counter[str] = Counter()
        split_counts: Counter[str] = Counter()
        ranges: dict[str, dict[str, float]] = {}
        examples: list[dict[str, str]] = []

        for row in reader:
            row_count += 1

            if len(examples) < 3:
                examples.append(
                    {
                        key: value
                        for key, value in row.items()
                        if key is not None
                    }
                )

            if query_column and reference_column:
                identity = (
                    row.get(query_column, ""),
                    row.get(reference_column, ""),
                )

                if identity in seen_pairs:
                    duplicate_pair_count += 1
                else:
                    seen_pairs.add(identity)

                update_range(
                    ranges,
                    query_column,
                    row.get(query_column, ""),
                )
                update_range(
                    ranges,
                    reference_column,
                    row.get(reference_column, ""),
                )

            if label_column:
                label_counts[
                    row.get(label_column, "").strip()
                ] += 1

            if split_column:
                split_counts[
                    row.get(split_column, "").strip()
                ] += 1

            if distance_column:
                update_range(
                    ranges,
                    distance_column,
                    row.get(distance_column, ""),
                )

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "columns": fieldnames,
        "row_count": row_count,
        "detected_columns": {
            "query": query_column,
            "reference": reference_column,
            "label": label_column,
            "split_or_event": split_column,
            "distance": distance_column,
        },
        "label_counts": dict(
            sorted(label_counts.items())
        ),
        "split_counts": dict(
            sorted(split_counts.items())
        ),
        "numeric_ranges": ranges,
        "unique_pair_count": (
            len(seen_pairs)
            if query_column and reference_column
            else None
        ),
        "duplicate_pair_count": (
            duplicate_pair_count
            if query_column and reference_column
            else None
        ),
        "first_rows": examples,
    }


def is_pair_candidate(path: Path) -> bool:
    name = path.name.lower()

    return any(
        token in name
        for token in (
            "pair",
            "candidate",
            "calibration",
            "test",
            "stress",
        )
    )


def main() -> None:
    if not ROOT.is_dir():
        raise FileNotFoundError(
            f"Diretório não encontrado: {ROOT}"
        )

    all_csv_files = sorted(ROOT.rglob("*.csv"))
    candidate_files = [
        path
        for path in all_csv_files
        if is_pair_candidate(path)
    ]

    if not candidate_files:
        raise FileNotFoundError(
            "Nenhum CSV relacionado a pares candidatos foi encontrado."
        )

    report = {
        "root": str(ROOT),
        "all_csv_files": [
            str(path)
            for path in all_csv_files
        ],
        "candidate_file_count": len(candidate_files),
        "candidate_files": [
            inspect_csv(path)
            for path in candidate_files
        ],
    }

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 79)
    print("AUDITORIA DOS ARQUIVOS DE PARES — KITTI 05")
    print("=" * 79)

    for item in report["candidate_files"]:
        print()
        print(item["path"])
        print(f"  linhas            : {item['row_count']}")
        print(f"  colunas           : {item['columns']}")
        print(
            "  colunas detectadas:",
            item["detected_columns"],
        )
        print(
            "  contagem de labels:",
            item["label_counts"],
        )
        print(
            "  contagem de splits:",
            item["split_counts"],
        )
        print(
            "  pares únicos      :",
            item["unique_pair_count"],
        )
        print(
            "  pares duplicados  :",
            item["duplicate_pair_count"],
        )
        print(
            "  intervalos        :",
            item["numeric_ranges"],
        )

    print()
    print(f"Relatório salvo em: {OUTPUT}")
    print("=" * 79)


if __name__ == "__main__":
    main()
