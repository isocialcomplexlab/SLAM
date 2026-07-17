"""Cálculo de scores contínuos para pares de descritores ResNet-101.

São mantidas duas variantes metodologicamente separadas:

- ``legacy_exact``: distância L-infinito/Chebyshev;
- ``corrected_l2``: distância Euclidiana do vetor achatado.

Para ROC, PR e Average Precision, o score é o negativo da distância, de modo
que valores maiores representem pares mais semelhantes.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


DESCRIPTOR_LENGTH = 4096

BASE_OUTPUT_FIELDS = [
    "split",
    "query_camera_id",
    "reference_camera_id",
    "query_frame",
    "reference_frame",
    "spatial_distance_m",
    "pair_class",
    "binary_label",
    "legacy_exact_distance",
    "legacy_exact_score",
    "corrected_l2_distance",
    "corrected_l2_score",
]


def classify_spatial_distance(
    distance_m: float,
) -> tuple[str, int | None]:
    """Classifica um par pela distância espacial de ground truth."""
    try:
        normalized = float(distance_m)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "distance_m deve ser um número real."
        ) from error

    if not math.isfinite(normalized):
        raise ValueError(
            "distance_m deve ser finita."
        )

    if normalized < 0:
        raise ValueError(
            "distance_m não pode ser negativa."
        )

    if normalized <= 5.0:
        return "positive", 1

    if normalized < 10.0:
        return "ambiguous", None

    return "negative", 0


def build_descriptor_index(
    *,
    descriptors: np.ndarray,
    camera_ids: np.ndarray,
    frame_ids: np.ndarray,
) -> dict[tuple[str, int], np.ndarray]:
    """Constrói um índice explícito por ``(camera_id, frame_id)``."""
    descriptor_array = np.asarray(descriptors)
    camera_array = np.asarray(camera_ids)
    frame_array = np.asarray(frame_ids)

    if (
        descriptor_array.ndim != 2
        or descriptor_array.shape[1] != DESCRIPTOR_LENGTH
    ):
        raise ValueError(
            "descriptors deve possuir forma [N, 4096]; "
            f"recebido {descriptor_array.shape}."
        )

    sample_count = descriptor_array.shape[0]

    if camera_array.ndim != 1 or len(camera_array) != sample_count:
        raise ValueError(
            "camera_ids deve possuir um elemento por descritor."
        )

    if frame_array.ndim != 1 or len(frame_array) != sample_count:
        raise ValueError(
            "frame_ids deve possuir um elemento por descritor."
        )

    if not np.issubdtype(
        descriptor_array.dtype,
        np.floating,
    ):
        raise TypeError(
            "descriptors deve possuir dtype de ponto flutuante."
        )

    if not np.isfinite(descriptor_array).all():
        raise ValueError(
            "descriptors contém valor não finito, NaN ou infinito."
        )

    index: dict[tuple[str, int], np.ndarray] = {}

    for row_index in range(sample_count):
        camera_id = str(camera_array[row_index])

        try:
            frame_id = int(frame_array[row_index])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"frame_id inválido na posição {row_index}."
            ) from error

        identity = (camera_id, frame_id)

        if identity in index:
            raise ValueError(
                "Identidade de descritor duplicada: "
                f"{identity}."
            )

        index[identity] = np.asarray(
            descriptor_array[row_index],
            dtype=np.float32,
        ).copy()

    return index


def load_descriptor_index_from_npz(
    path: str | Path,
) -> dict[tuple[str, int], np.ndarray]:
    """Carrega descritores de um NPZ sem permitir pickle."""
    archive_path = Path(path).expanduser()

    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Arquivo de descritores não encontrado: {archive_path}"
        )

    with np.load(
        archive_path,
        allow_pickle=False,
    ) as archive:
        required = {
            "descriptors",
            "camera_ids",
            "frame_ids",
        }

        missing = required - set(archive.files)

        if missing:
            raise ValueError(
                "O arquivo NPZ não contém os campos obrigatórios: "
                f"{sorted(missing)}."
            )

        descriptors = archive["descriptors"]
        camera_ids = archive["camera_ids"]
        frame_ids = archive["frame_ids"]

        return build_descriptor_index(
            descriptors=descriptors,
            camera_ids=camera_ids,
            frame_ids=frame_ids,
        )


def _parse_frame_id(
    value: Any,
    *,
    column_name: str,
) -> int:
    try:
        frame_id = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{column_name} inválido: {value!r}."
        ) from error

    if frame_id < 0:
        raise ValueError(
            f"{column_name} não pode ser negativo."
        )

    return frame_id


def _parse_spatial_distance(value: Any) -> float:
    try:
        distance = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"spatial_distance_m inválida: {value!r}."
        ) from error

    if not math.isfinite(distance):
        raise ValueError(
            "spatial_distance_m deve ser finita."
        )

    return distance


def _legacy_exact_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    return float(
        np.max(
            np.abs(
                first.astype(np.float64, copy=False)
                - second.astype(np.float64, copy=False)
            )
        )
    )


def _corrected_l2_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    difference = (
        first.astype(np.float64, copy=False)
        - second.astype(np.float64, copy=False)
    )

    return float(
        np.linalg.norm(
            difference,
            ord=2,
        )
    )


def compute_pair_score_rows(
    *,
    pairs: Iterable[Mapping[str, Any]],
    descriptor_index: Mapping[
        tuple[str, int],
        np.ndarray,
    ],
    camera_id: str,
    expected_split: str,
) -> list[dict[str, Any]]:
    """Calcula os dois scores contínuos para uma coleção de pares."""
    if not isinstance(camera_id, str) or not camera_id:
        raise ValueError(
            "camera_id deve ser uma string não vazia."
        )

    if not isinstance(expected_split, str) or not expected_split:
        raise ValueError(
            "expected_split deve ser uma string não vazia."
        )

    materialized_pairs = list(pairs)
    output_rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, int, int]] = set()

    for row_number, original in enumerate(
        materialized_pairs,
        start=1,
    ):
        row = dict(original)

        required_columns = {
            "split",
            "query_frame",
            "reference_frame",
            "spatial_distance_m",
        }

        missing = required_columns - set(row)

        if missing:
            raise ValueError(
                f"Par na linha {row_number} não contém "
                f"as colunas {sorted(missing)}."
            )

        split = str(row["split"]).strip()

        if split != expected_split:
            raise ValueError(
                "Contaminação entre splits detectada: "
                f"esperado={expected_split!r}, recebido={split!r}."
            )

        query_frame = _parse_frame_id(
            row["query_frame"],
            column_name="query_frame",
        )
        reference_frame = _parse_frame_id(
            row["reference_frame"],
            column_name="reference_frame",
        )

        pair_identity = (
            split,
            query_frame,
            reference_frame,
        )

        if pair_identity in seen_pairs:
            raise ValueError(
                "Par duplicado encontrado: "
                f"{pair_identity}."
            )

        seen_pairs.add(pair_identity)

        query_identity = (
            camera_id,
            query_frame,
        )
        reference_identity = (
            camera_id,
            reference_frame,
        )

        if query_identity not in descriptor_index:
            raise KeyError(
                "Descritor de consulta não encontrado para "
                f"{query_identity}."
            )

        if reference_identity not in descriptor_index:
            raise KeyError(
                "Descritor de referência não encontrado para "
                f"{reference_identity}."
            )

        query_descriptor = np.asarray(
            descriptor_index[query_identity],
            dtype=np.float32,
        )
        reference_descriptor = np.asarray(
            descriptor_index[reference_identity],
            dtype=np.float32,
        )

        for name, descriptor in (
            ("query", query_descriptor),
            ("reference", reference_descriptor),
        ):
            if descriptor.shape != (DESCRIPTOR_LENGTH,):
                raise ValueError(
                    f"O descritor {name} deve possuir 4096 valores; "
                    f"recebido {descriptor.shape}."
                )

            if not np.isfinite(descriptor).all():
                raise ValueError(
                    f"O descritor {name} contém valor não finito."
                )

        spatial_distance = _parse_spatial_distance(
            row["spatial_distance_m"]
        )

        pair_class, binary_label = (
            classify_spatial_distance(
                spatial_distance
            )
        )

        legacy_distance = _legacy_exact_distance(
            query_descriptor,
            reference_descriptor,
        )
        corrected_distance = _corrected_l2_distance(
            query_descriptor,
            reference_descriptor,
        )

        output = dict(row)

        output.update(
            {
                "split": split,
                "query_camera_id": camera_id,
                "reference_camera_id": camera_id,
                "query_frame": query_frame,
                "reference_frame": reference_frame,
                "spatial_distance_m": spatial_distance,
                "pair_class": pair_class,
                "binary_label": binary_label,
                "legacy_exact_distance": legacy_distance,
                "legacy_exact_score": -legacy_distance,
                "corrected_l2_distance": corrected_distance,
                "corrected_l2_score": -corrected_distance,
            }
        )

        output_rows.append(output)

    return output_rows


def _collect_fieldnames(
    rows: list[Mapping[str, Any]],
) -> list[str]:
    """Reúne campos obrigatórios e metadados adicionais."""
    fieldnames = list(BASE_OUTPUT_FIELDS)
    seen = set(fieldnames)

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    return fieldnames


def save_pair_score_rows(
    rows: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Salva os scores em CSV, deixando ambíguos sem rótulo binário."""
    materialized_rows = [
        dict(row)
        for row in rows
    ]

    if not materialized_rows:
        raise ValueError(
            "Não há linhas de scores para salvar."
        )

    path = Path(output_path).expanduser()

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"O arquivo já existe: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = _collect_fieldnames(
        materialized_rows
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
        )

        writer.writeheader()

        for row in materialized_rows:
            serializable = {
                key: (
                    ""
                    if value is None
                    else value
                )
                for key, value in row.items()
            }

            writer.writerow(serializable)

    return path
