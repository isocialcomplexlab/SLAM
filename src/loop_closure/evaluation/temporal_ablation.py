"""Ablação temporal de scores ResNet-101 no KITTI 05.

A semântica temporal preserva o comportamento auditado no notebook legado:
um único descritor de consulta é comparado com referências adjacentes no banco
histórico. Para uma janela de comprimento L, o score contínuo é o menor score
de similaridade entre os L pares, equivalente ao negativo da maior distância.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from loop_closure.evaluation.binary_metrics import (
    calibrate_threshold_max_f1,
    compute_ranking_metrics,
    evaluate_at_threshold,
)
from loop_closure.evaluation.resnet101_l1_evaluation import (
    SCORE_METHODS,
)


SPLIT_ORDER = (
    "calibration",
    "test",
    "stress",
)


@dataclass(frozen=True)
class PairScoreRow:
    """Um par consulta-referência com scores contínuos."""

    query_frame: int
    reference_frame: int
    pair_class: str
    binary_label: int | None
    scores: dict[str, float]


@dataclass(frozen=True)
class PairScoreTable:
    """Tabela validada de scores de um split."""

    split: str
    source_path: Path
    source_sha256: str
    total_rows: int
    query_count: int
    rows: tuple[PairScoreRow, ...]


@dataclass(frozen=True)
class TemporalWindowSplit:
    """Janelas temporais binárias de um split e comprimento L."""

    split: str
    sequence_length: int
    source_path: Path
    source_sha256: str
    source_pair_rows: int
    source_query_count: int
    candidate_window_positions: int
    contiguous_windows: int
    nonconsecutive_windows_skipped: int
    ambiguous_windows: int
    binary_windows: int
    positive_windows: int
    negative_windows: int
    query_count_with_contiguous_windows: int
    labels: np.ndarray
    scores: dict[str, np.ndarray]


def sha256_file(path: str | Path) -> str:
    """Calcula SHA-256 de um arquivo."""

    source = Path(path)
    digest = hashlib.sha256()

    with source.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _parse_frame(
    value: object,
    *,
    field: str,
    path: Path,
    line_number: int,
) -> int:
    try:
        frame = int(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}, linha {line_number}: "
            f"{field} inválido."
        ) from error

    if frame < 0:
        raise ValueError(
            f"{path}, linha {line_number}: "
            f"{field} negativo."
        )

    return frame


def _parse_pair_class(
    row: Mapping[str, str],
    *,
    path: Path,
    line_number: int,
) -> tuple[str, int | None]:
    pair_class = str(
        row.get("pair_class", "")
    ).strip()
    raw_label = str(
        row.get("binary_label", "")
    ).strip()

    expected_labels = {
        "positive": "1",
        "negative": "0",
        "ambiguous": "",
    }

    if pair_class not in expected_labels:
        raise ValueError(
            f"{path}, linha {line_number}: "
            f"pair_class inválida: {pair_class!r}."
        )

    if raw_label != expected_labels[pair_class]:
        raise ValueError(
            f"{path}, linha {line_number}: "
            f"binary_label={raw_label!r} incompatível "
            f"com pair_class={pair_class!r}."
        )

    if pair_class == "positive":
        return pair_class, 1

    if pair_class == "negative":
        return pair_class, 0

    return pair_class, None


def load_pair_score_table(
    path: str | Path,
    *,
    expected_split: str,
) -> PairScoreTable:
    """Carrega a grade completa de scores de um split."""

    source_path = Path(path)

    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    required_columns = {
        "split",
        "query_frame",
        "reference_frame",
        "pair_class",
        "binary_label",
        *(
            config["score_field"]
            for config in SCORE_METHODS.values()
        ),
    }

    rows: list[PairScoreRow] = []
    identities: set[tuple[int, int]] = set()
    queries: set[int] = set()

    with source_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = required_columns - fieldnames

        if missing:
            raise ValueError(
                f"{source_path}: colunas ausentes: "
                f"{sorted(missing)}."
            )

        for line_number, row in enumerate(
            reader,
            start=2,
        ):
            split = str(
                row.get("split", "")
            ).strip()

            if split != expected_split:
                raise ValueError(
                    f"{source_path}, linha {line_number}: "
                    f"split={split!r}; esperado "
                    f"{expected_split!r}."
                )

            query_frame = _parse_frame(
                row.get("query_frame"),
                field="query_frame",
                path=source_path,
                line_number=line_number,
            )
            reference_frame = _parse_frame(
                row.get("reference_frame"),
                field="reference_frame",
                path=source_path,
                line_number=line_number,
            )

            identity = (
                query_frame,
                reference_frame,
            )

            if identity in identities:
                raise ValueError(
                    f"{source_path}, linha {line_number}: "
                    f"par duplicado {identity}."
                )

            identities.add(identity)
            queries.add(query_frame)

            pair_class, binary_label = (
                _parse_pair_class(
                    row,
                    path=source_path,
                    line_number=line_number,
                )
            )

            scores: dict[str, float] = {}

            for method, config in (
                SCORE_METHODS.items()
            ):
                field = config["score_field"]

                try:
                    value = float(row[field])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{source_path}, linha "
                        f"{line_number}: score inválido "
                        f"para {method}."
                    ) from error

                if not math.isfinite(value):
                    raise ValueError(
                        f"{source_path}, linha "
                        f"{line_number}: score não finito "
                        f"para {method}."
                    )

                scores[method] = value

            rows.append(
                PairScoreRow(
                    query_frame=query_frame,
                    reference_frame=reference_frame,
                    pair_class=pair_class,
                    binary_label=binary_label,
                    scores=scores,
                )
            )

    if not rows:
        raise ValueError(
            f"{source_path}: nenhuma linha de dados."
        )

    rows.sort(
        key=lambda item: (
            item.query_frame,
            item.reference_frame,
        )
    )

    return PairScoreTable(
        split=expected_split,
        source_path=source_path,
        source_sha256=sha256_file(
            source_path
        ),
        total_rows=len(rows),
        query_count=len(queries),
        rows=tuple(rows),
    )


def build_temporal_window_split(
    table: PairScoreTable,
    *,
    sequence_length: int,
) -> TemporalWindowSplit:
    """Constrói janelas de uma consulta e L referências consecutivas.

    A janela é positiva quando todos os L pares geométricos são positivos.
    Ela é ambígua e excluída das métricas quando pelo menos um constituinte é
    ambíguo. Caso contrário, uma janela com pelo menos um par negativo é
    negativa. O score é o mínimo dos L scores de similaridade.
    """

    if isinstance(sequence_length, bool):
        raise TypeError(
            "sequence_length deve ser inteiro."
        )

    length = int(sequence_length)

    if length < 1:
        raise ValueError(
            "sequence_length deve ser pelo menos 1."
        )

    grouped: dict[int, list[PairScoreRow]] = (
        defaultdict(list)
    )

    for row in table.rows:
        grouped[row.query_frame].append(row)

    labels: list[int] = []
    score_values: dict[str, list[float]] = {
        method: []
        for method in SCORE_METHODS
    }

    candidate_positions = 0
    contiguous_windows = 0
    skipped_nonconsecutive = 0
    ambiguous_windows = 0
    positive_windows = 0
    negative_windows = 0
    queries_with_windows = 0

    for query_frame in sorted(grouped):
        query_rows = sorted(
            grouped[query_frame],
            key=lambda item: item.reference_frame,
        )

        available = len(query_rows) - length + 1

        if available <= 0:
            continue

        candidate_positions += available
        query_has_window = False

        for start in range(available):
            window = query_rows[
                start : start + length
            ]
            references = [
                item.reference_frame
                for item in window
            ]

            if any(
                right - left != 1
                for left, right in zip(
                    references,
                    references[1:],
                )
            ):
                skipped_nonconsecutive += 1
                continue

            query_has_window = True
            contiguous_windows += 1

            if any(
                item.binary_label is None
                for item in window
            ):
                ambiguous_windows += 1
                continue

            window_labels = [
                int(item.binary_label)
                for item in window
                if item.binary_label is not None
            ]

            label = (
                1
                if all(
                    value == 1
                    for value in window_labels
                )
                else 0
            )

            labels.append(label)

            if label == 1:
                positive_windows += 1
            else:
                negative_windows += 1

            for method in SCORE_METHODS:
                score_values[method].append(
                    min(
                        item.scores[method]
                        for item in window
                    )
                )

        if query_has_window:
            queries_with_windows += 1

    if (
        contiguous_windows
        + skipped_nonconsecutive
        != candidate_positions
    ):
        raise RuntimeError(
            "Contagem de posições de janela inconsistente."
        )

    binary_windows = len(labels)

    if (
        binary_windows
        != positive_windows + negative_windows
    ):
        raise RuntimeError(
            "Contagem binária de janelas inconsistente."
        )

    if (
        contiguous_windows
        != binary_windows + ambiguous_windows
    ):
        raise RuntimeError(
            "Contagem total de janelas contíguas "
            "inconsistente."
        )

    # A construção de janelas não exige as duas classes. Isso permite
    # auditar corretamente splits ou subconjuntos que contenham somente
    # positivos ou somente negativos. A exigência de ambas as classes fica
    # na etapa de avaliação, imediatamente antes do cálculo das métricas.
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

    for method, values in score_arrays.items():
        if len(values) != binary_windows:
            raise RuntimeError(
                f"{table.split}, L={length}: "
                f"scores inconsistentes para {method}."
            )

    return TemporalWindowSplit(
        split=table.split,
        sequence_length=length,
        source_path=table.source_path,
        source_sha256=table.source_sha256,
        source_pair_rows=table.total_rows,
        source_query_count=table.query_count,
        candidate_window_positions=(
            candidate_positions
        ),
        contiguous_windows=contiguous_windows,
        nonconsecutive_windows_skipped=(
            skipped_nonconsecutive
        ),
        ambiguous_windows=ambiguous_windows,
        binary_windows=binary_windows,
        positive_windows=positive_windows,
        negative_windows=negative_windows,
        query_count_with_contiguous_windows=(
            queries_with_windows
        ),
        labels=label_array,
        scores=score_arrays,
    )


def evaluate_temporal_protocol(
    windows: Mapping[
        int,
        Mapping[str, TemporalWindowSplit],
    ],
) -> dict[str, Any]:
    """Calibra cada combinação método-L somente em calibration."""

    if not windows:
        raise ValueError(
            "Nenhuma janela temporal foi fornecida."
        )

    lengths = sorted(
        int(length)
        for length in windows
    )

    required_splits = set(SPLIT_ORDER)
    thresholds: dict[str, dict[str, Any]] = {}
    results_by_length: dict[str, Any] = {}

    for length in lengths:
        split_map = windows[length]

        if set(split_map) != required_splits:
            raise ValueError(
                f"L={length}: são necessários exatamente "
                "calibration, test e stress."
            )

        for split_name, data in (
            split_map.items()
        ):
            if data.split != split_name:
                raise ValueError(
                    f"L={length}: chave {split_name!r} "
                    f"não corresponde a {data.split!r}."
                )

            if data.sequence_length != length:
                raise ValueError(
                    f"L={length}: objeto contém "
                    f"L={data.sequence_length}."
                )

        for split_name, data in split_map.items():
            if (
                data.positive_windows == 0
                or data.negative_windows == 0
            ):
                raise ValueError(
                    f"{split_name}, L={length}: a avaliação "
                    "exige janelas positivas e negativas."
                )

        calibration = split_map["calibration"]
        length_thresholds: dict[str, Any] = {}

        for method, config in (
            SCORE_METHODS.items()
        ):
            calibrated = (
                calibrate_threshold_max_f1(
                    calibration.labels,
                    calibration.scores[method],
                )
            )
            score_threshold = float(
                calibrated["threshold"]
            )

            length_thresholds[method] = {
                **calibrated,
                "sequence_length": length,
                "calibrated_on_split": (
                    "calibration"
                ),
                "score_field": config[
                    "score_field"
                ],
                "distance_field": config[
                    "distance_field"
                ],
                "distance_name": config[
                    "distance_name"
                ],
                "window_score_aggregation": (
                    "minimum_similarity_score_"
                    "equivalent_to_negative_"
                    "maximum_distance"
                ),
                "score_semantics": (
                    "larger_is_more_similar"
                ),
                "score_threshold": score_threshold,
                "distance_threshold": (
                    -score_threshold
                ),
                "applied_without_recalibration_to": [
                    "test",
                    "stress",
                ],
            }

        thresholds[str(length)] = (
            length_thresholds
        )

        split_results: dict[str, Any] = {}

        for split_name in SPLIT_ORDER:
            data = split_map[split_name]
            method_results: dict[str, Any] = {}

            for method in SCORE_METHODS:
                threshold = float(
                    length_thresholds[method][
                        "score_threshold"
                    ]
                )

                method_results[method] = {
                    "ranking": (
                        compute_ranking_metrics(
                            data.labels,
                            data.scores[method],
                        )
                    ),
                    "operating_point": (
                        evaluate_at_threshold(
                            data.labels,
                            data.scores[method],
                            threshold,
                        )
                    ),
                    "threshold_source_split": (
                        "calibration"
                    ),
                    "recalibrated_on_this_split": (
                        False
                    ),
                }

            split_results[split_name] = {
                "source_file": str(
                    data.source_path
                ),
                "source_sha256": (
                    data.source_sha256
                ),
                "source_pair_rows": (
                    data.source_pair_rows
                ),
                "source_query_count": (
                    data.source_query_count
                ),
                "candidate_window_positions": (
                    data.candidate_window_positions
                ),
                "contiguous_windows": (
                    data.contiguous_windows
                ),
                "nonconsecutive_windows_skipped": (
                    data.nonconsecutive_windows_skipped
                ),
                "ambiguous_windows_excluded": (
                    data.ambiguous_windows
                ),
                "binary_windows": (
                    data.binary_windows
                ),
                "positive_windows": (
                    data.positive_windows
                ),
                "negative_windows": (
                    data.negative_windows
                ),
                "query_count_with_contiguous_windows": (
                    data.query_count_with_contiguous_windows
                ),
                "methods": method_results,
            }

        results_by_length[str(length)] = {
            "sequence_length": length,
            "thresholds": length_thresholds,
            "splits": split_results,
        }

    return {
        "dataset": "KITTI",
        "sequence": "05",
        "evaluation_level": (
            "temporal_reference_window"
        ),
        "sequence_lengths": lengths,
        "protocol": {
            "legacy_semantics": (
                "one_query_descriptor_compared_"
                "against_L_consecutive_reference_"
                "frames"
            ),
            "contiguity_rule": (
                "reference_frame_increments_by_one"
            ),
            "window_score": (
                "minimum_constituent_similarity_"
                "score_equivalent_to_negative_"
                "maximum_constituent_distance"
            ),
            "positive_window": (
                "all_L_constituent_pairs_are_"
                "geometric_positives"
            ),
            "negative_window": (
                "no_ambiguous_constituents_and_"
                "at_least_one_geometric_negative"
            ),
            "ambiguous_window": (
                "at_least_one_constituent_pair_"
                "is_ambiguous"
            ),
            "ambiguous_policy": (
                "excluded_from_binary_metrics"
            ),
            "calibration_split": "calibration",
            "threshold_criterion": (
                "maximum_f1_separately_for_each_"
                "method_and_L"
            ),
            "test_threshold_recalibration": (
                False
            ),
            "stress_threshold_recalibration": (
                False
            ),
            "prediction_rule": (
                "window_score_greater_than_or_"
                "equal_to_calibrated_threshold"
            ),
        },
        "thresholds": thresholds,
        "lengths": results_by_length,
    }


def validate_l1_equivalence(
    temporal_evaluation: Mapping[str, Any],
    descriptor_l1_metrics_path: str | Path,
    *,
    absolute_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Confirma que L=1 reproduz exatamente a avaliação descritor-a-descritor."""

    source_path = Path(
        descriptor_l1_metrics_path
    )

    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    descriptor = json.loads(
        source_path.read_text(
            encoding="utf-8",
        )
    )

    temporal_l1 = temporal_evaluation[
        "lengths"
    ]["1"]

    compared_values = 0

    for method in SCORE_METHODS:
        temporal_threshold = float(
            temporal_l1["thresholds"][method][
                "score_threshold"
            ]
        )
        descriptor_threshold = float(
            descriptor["thresholds"][method][
                "score_threshold"
            ]
        )

        if not math.isclose(
            temporal_threshold,
            descriptor_threshold,
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        ):
            raise ValueError(
                f"L=1: threshold divergente para "
                f"{method}: temporal="
                f"{temporal_threshold}, descriptor="
                f"{descriptor_threshold}."
            )

        compared_values += 1

        for split_name in SPLIT_ORDER:
            temporal_result = temporal_l1[
                "splits"
            ][split_name]["methods"][method]
            descriptor_result = descriptor[
                "splits"
            ][split_name]["methods"][method]

            for section in (
                "ranking",
                "operating_point",
            ):
                temporal_values = temporal_result[
                    section
                ]
                descriptor_values = (
                    descriptor_result[section]
                )

                for key, descriptor_value in (
                    descriptor_values.items()
                ):
                    if key not in temporal_values:
                        continue

                    temporal_value = (
                        temporal_values[key]
                    )

                    if isinstance(
                        descriptor_value,
                        (int, float),
                    ) and not isinstance(
                        descriptor_value,
                        bool,
                    ):
                        if not math.isclose(
                            float(temporal_value),
                            float(descriptor_value),
                            rel_tol=0.0,
                            abs_tol=absolute_tolerance,
                        ):
                            raise ValueError(
                                "L=1 divergente: "
                                f"split={split_name}, "
                                f"method={method}, "
                                f"section={section}, "
                                f"key={key}, "
                                f"temporal={temporal_value}, "
                                f"descriptor="
                                f"{descriptor_value}."
                            )

                        compared_values += 1

    return {
        "status": "passed",
        "reference_file": str(source_path),
        "reference_sha256": sha256_file(
            source_path
        ),
        "absolute_tolerance": (
            absolute_tolerance
        ),
        "compared_numeric_values": (
            compared_values
        ),
    }


def _check_outputs(
    paths: Sequence[Path],
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
            "Artefatos temporais já existem: "
            + ", ".join(
                str(path)
                for path in existing
            )
        )


def write_temporal_artifacts(
    evaluation: Mapping[str, Any],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Grava os artefatos pequenos da ablação temporal."""

    root = Path(output_dir)
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    lengths = [
        int(value)
        for value in evaluation[
            "sequence_lengths"
        ]
    ]

    per_length_paths = [
        (
            root / "metrics_l1_temporal.json"
            if length == 1
            else root / f"metrics_l{length}.json"
        )
        for length in lengths
    ]

    paths = [
        root / "temporal_thresholds.json",
        root / "metrics_temporal.json",
        root / "temporal_ablation.csv",
        root / "temporal_confusion_matrices.csv",
        root / "temporal_window_counts.csv",
        *per_length_paths,
        root / "temporal_ablation_manifest.json",
    ]

    _check_outputs(
        paths,
        overwrite=overwrite,
    )

    (root / "temporal_thresholds.json").write_text(
        json.dumps(
            {
                "dataset": evaluation["dataset"],
                "sequence": evaluation["sequence"],
                "protocol": evaluation[
                    "protocol"
                ],
                "thresholds": evaluation[
                    "thresholds"
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    (root / "metrics_temporal.json").write_text(
        json.dumps(
            evaluation,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary_fields = [
        "sequence_length",
        "split",
        "method",
        "source_file",
        "source_sha256",
        "candidate_window_positions",
        "contiguous_windows",
        "nonconsecutive_windows_skipped",
        "ambiguous_windows_excluded",
        "binary_windows",
        "positive_windows",
        "negative_windows",
        "positive_prevalence",
        "score_threshold",
        "distance_threshold",
        "roc_auc",
        "pr_auc",
        "average_precision",
        "precision",
        "recall",
        "specificity",
        "f1",
        "accuracy",
        "balanced_accuracy",
        "matthews_correlation_coefficient",
    ]

    with (
        root / "temporal_ablation.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=summary_fields,
        )
        writer.writeheader()

        for length in lengths:
            length_data = evaluation[
                "lengths"
            ][str(length)]

            for split_name in SPLIT_ORDER:
                split_data = length_data[
                    "splits"
                ][split_name]

                for method in SCORE_METHODS:
                    method_data = split_data[
                        "methods"
                    ][method]
                    ranking = method_data[
                        "ranking"
                    ]
                    operating = method_data[
                        "operating_point"
                    ]
                    threshold = length_data[
                        "thresholds"
                    ][method]

                    writer.writerow(
                        {
                            "sequence_length": length,
                            "split": split_name,
                            "method": method,
                            "source_file": split_data[
                                "source_file"
                            ],
                            "source_sha256": split_data[
                                "source_sha256"
                            ],
                            "candidate_window_positions": split_data[
                                "candidate_window_positions"
                            ],
                            "contiguous_windows": split_data[
                                "contiguous_windows"
                            ],
                            "nonconsecutive_windows_skipped": split_data[
                                "nonconsecutive_windows_skipped"
                            ],
                            "ambiguous_windows_excluded": split_data[
                                "ambiguous_windows_excluded"
                            ],
                            "binary_windows": split_data[
                                "binary_windows"
                            ],
                            "positive_windows": split_data[
                                "positive_windows"
                            ],
                            "negative_windows": split_data[
                                "negative_windows"
                            ],
                            "positive_prevalence": ranking[
                                "positive_prevalence"
                            ],
                            "score_threshold": threshold[
                                "score_threshold"
                            ],
                            "distance_threshold": threshold[
                                "distance_threshold"
                            ],
                            "roc_auc": ranking[
                                "roc_auc"
                            ],
                            "pr_auc": ranking[
                                "pr_auc"
                            ],
                            "average_precision": ranking[
                                "average_precision"
                            ],
                            "precision": operating[
                                "precision"
                            ],
                            "recall": operating[
                                "recall"
                            ],
                            "specificity": operating[
                                "specificity"
                            ],
                            "f1": operating["f1"],
                            "accuracy": operating[
                                "accuracy"
                            ],
                            "balanced_accuracy": operating[
                                "balanced_accuracy"
                            ],
                            "matthews_correlation_coefficient": operating[
                                "matthews_correlation_coefficient"
                            ],
                        }
                    )

    confusion_fields = [
        "sequence_length",
        "split",
        "method",
        "threshold",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    ]

    with (
        root / "temporal_confusion_matrices.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=confusion_fields,
        )
        writer.writeheader()

        for length in lengths:
            length_data = evaluation[
                "lengths"
            ][str(length)]

            for split_name in SPLIT_ORDER:
                for method in SCORE_METHODS:
                    operating = length_data[
                        "splits"
                    ][split_name]["methods"][
                        method
                    ]["operating_point"]

                    writer.writerow(
                        {
                            "sequence_length": length,
                            "split": split_name,
                            "method": method,
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

    count_fields = [
        "sequence_length",
        "split",
        "source_pair_rows",
        "source_query_count",
        "candidate_window_positions",
        "contiguous_windows",
        "nonconsecutive_windows_skipped",
        "ambiguous_windows_excluded",
        "binary_windows",
        "positive_windows",
        "negative_windows",
        "query_count_with_contiguous_windows",
    ]

    with (
        root / "temporal_window_counts.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=count_fields,
        )
        writer.writeheader()

        for length in lengths:
            for split_name in SPLIT_ORDER:
                split_data = evaluation[
                    "lengths"
                ][str(length)]["splits"][
                    split_name
                ]

                writer.writerow(
                    {
                        field: (
                            length
                            if field
                            == "sequence_length"
                            else split_name
                            if field == "split"
                            else split_data[field]
                        )
                        for field in count_fields
                    }
                )

    for length, path in zip(
        lengths,
        per_length_paths,
    ):
        path.write_text(
            json.dumps(
                {
                    "dataset": evaluation[
                        "dataset"
                    ],
                    "sequence": evaluation[
                        "sequence"
                    ],
                    "evaluation_level": evaluation[
                        "evaluation_level"
                    ],
                    "protocol": evaluation[
                        "protocol"
                    ],
                    "l1_equivalence": (
                        evaluation.get(
                            "l1_equivalence"
                        )
                        if length == 1
                        else None
                    ),
                    **evaluation["lengths"][
                        str(length)
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    manifest_path = (
        root / "temporal_ablation_manifest.json"
    )
    artifact_paths = [
        path
        for path in paths
        if path != manifest_path
    ]

    manifest_path.write_text(
        json.dumps(
            {
                "dataset": evaluation[
                    "dataset"
                ],
                "sequence": evaluation[
                    "sequence"
                ],
                "evaluation_level": evaluation[
                    "evaluation_level"
                ],
                "sequence_lengths": lengths,
                "sources": sorted(
                    {
                        split_data[
                            "source_file"
                        ]: split_data[
                            "source_sha256"
                        ]
                        for length_data in evaluation[
                            "lengths"
                        ].values()
                        for split_data in length_data[
                            "splits"
                        ].values()
                    }.items()
                ),
                "artifacts": [
                    {
                        "path": str(path),
                        "sha256": sha256_file(
                            path
                        ),
                        "size_bytes": (
                            path.stat().st_size
                        ),
                    }
                    for path in artifact_paths
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return paths
