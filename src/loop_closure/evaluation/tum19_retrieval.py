from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


PRIMARY_ERROR_FIELDS = (
    "top1_spatial_error_m",
    "best_spatial_error_at_5_m",
    "best_spatial_error_at_10_m",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_descriptor_arrays(
    descriptors: np.ndarray,
    frame_ids: np.ndarray,
    *,
    expected_dim: int,
    expected_frame_ids: Sequence[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    descriptors = np.asarray(descriptors)
    frame_ids = np.asarray(frame_ids)
    _require(descriptors.ndim == 2, "descriptors deve possuir forma [N, D]")
    _require(descriptors.shape[1] == expected_dim, f"descriptor dimension inesperada: {descriptors.shape[1]} != {expected_dim}")
    _require(descriptors.dtype == np.float32, f"descriptors deve usar float32, recebido {descriptors.dtype}")
    _require(np.isfinite(descriptors).all(), "descriptors contém NaN/Inf")
    _require(frame_ids.ndim == 1, "frame_ids deve possuir forma [N]")
    _require(frame_ids.shape[0] == descriptors.shape[0], "frame_ids e descriptors devem possuir o mesmo N")
    _require(np.issubdtype(frame_ids.dtype, np.integer), "frame_ids deve usar dtype inteiro")
    frame_ids = frame_ids.astype(np.int64, copy=False)
    _require(len(np.unique(frame_ids)) == len(frame_ids), "frame_ids contém duplicatas")
    if expected_frame_ids is not None:
        expected = np.asarray(expected_frame_ids, dtype=np.int64)
        _require(np.array_equal(frame_ids, expected), "frame_ids não correspondem exatamente à ordem congelada do manifest")
    return np.ascontiguousarray(descriptors), np.ascontiguousarray(frame_ids)


def corrected_l2_distance_matrix(
    queries: np.ndarray,
    gallery: np.ndarray,
    *,
    query_batch_size: int = 8,
) -> np.ndarray:
    queries = np.asarray(queries)
    gallery = np.asarray(gallery)
    _require(queries.ndim == 2, "queries deve possuir forma [Q, D]")
    _require(gallery.ndim == 2, "gallery deve possuir forma [G, D]")
    _require(queries.shape[1] == gallery.shape[1], "queries e gallery devem ter a mesma dimensão")
    _require(np.issubdtype(queries.dtype, np.floating) and np.issubdtype(gallery.dtype, np.floating), "queries/gallery devem usar ponto flutuante")
    _require(np.isfinite(queries).all() and np.isfinite(gallery).all(), "queries/gallery contém NaN/Inf")
    _require(
        isinstance(query_batch_size, int) and query_batch_size > 0,
        "query_batch_size deve ser inteiro > 0",
    )

    q = queries.astype(np.float32, copy=False)
    g = gallery.astype(np.float32, copy=False)
    output = np.empty((q.shape[0], g.shape[0]), dtype=np.float32)

    # Preserve the original per-pair corrected flattened-L2 arithmetic while
    # bounding peak memory. The former all-at-once broadcast would materialize
    # Q x G x D arrays; for TUM19 split_left_right this exceeds 6 GB per
    # temporary. Batching only the query axis changes memory use, not the
    # distance definition.
    for start in range(0, q.shape[0], query_batch_size):
        stop = min(start + query_batch_size, q.shape[0])
        differences = q[start:stop, None, :] - g[None, :, :]
        squared = np.square(differences, dtype=np.float32)
        sums = np.sum(squared, axis=2, dtype=np.float32)
        output[start:stop] = np.sqrt(sums, dtype=np.float32)

    return output


def rank_gallery(descriptor_distances: np.ndarray, gallery_frame_ids: np.ndarray) -> np.ndarray:
    distances = np.asarray(descriptor_distances)
    gallery_frame_ids = np.asarray(gallery_frame_ids, dtype=np.int64)
    _require(distances.ndim == 1, "descriptor_distances deve ser 1-D")
    _require(distances.shape[0] == gallery_frame_ids.shape[0], "distâncias e IDs da galeria têm tamanhos diferentes")
    _require(np.isfinite(distances).all(), "distâncias contém NaN/Inf")
    return np.lexsort((gallery_frame_ids, distances))


def evaluate_mode(
    *,
    mode: str,
    descriptors: np.ndarray,
    frame_ids: np.ndarray,
    gallery_frame_ids: Sequence[int] | np.ndarray,
    query_frame_ids: Sequence[int] | np.ndarray,
    query_splits: Mapping[int, str],
    geometry_distances_m: np.ndarray,
    exact_nearest_reference_frames: Mapping[int, int],
    expected_dim: int,
) -> list[dict[str, Any]]:
    descriptors, frame_ids = validate_descriptor_arrays(descriptors, frame_ids, expected_dim=expected_dim)
    gallery_ids = np.asarray(gallery_frame_ids, dtype=np.int64)
    query_ids = np.asarray(query_frame_ids, dtype=np.int64)
    geometry = np.asarray(geometry_distances_m, dtype=np.float64)
    _require(geometry.shape == (len(query_ids), len(gallery_ids)), "geometry_distances_m deve possuir forma [Q, G]")
    _require(np.isfinite(geometry).all(), "geometry_distances_m contém NaN/Inf")
    descriptor_by_frame = {int(frame): descriptors[index] for index, frame in enumerate(frame_ids)}
    missing = [int(frame) for frame in np.concatenate([gallery_ids, query_ids]) if int(frame) not in descriptor_by_frame]
    _require(not missing, f"descritores ausentes para frames: {missing[:10]}")
    gallery_desc = np.stack([descriptor_by_frame[int(frame)] for frame in gallery_ids], axis=0)
    query_desc = np.stack([descriptor_by_frame[int(frame)] for frame in query_ids], axis=0)
    descriptor_matrix = corrected_l2_distance_matrix(query_desc, gallery_desc)
    rows: list[dict[str, Any]] = []
    for query_index, query_frame in enumerate(query_ids):
        q = int(query_frame)
        order = rank_gallery(descriptor_matrix[query_index], gallery_ids)
        ranked_gallery = gallery_ids[order]
        ranked_geometry = geometry[query_index, order]
        exact_frame = int(exact_nearest_reference_frames[q])
        exact_positions = np.flatnonzero(ranked_gallery == exact_frame)
        _require(len(exact_positions) == 1, f"referência geométrica exata ausente/duplicada para query {q}")
        exact_rank = int(exact_positions[0]) + 1
        rows.append({
            "query_frame": q,
            "split": str(query_splits[q]),
            "mode": mode,
            "descriptor_top1_reference_frame": int(ranked_gallery[0]),
            "top1_spatial_error_m": float(ranked_geometry[0]),
            "best_spatial_error_at_5_m": float(np.min(ranked_geometry[: min(5, len(ranked_geometry))])),
            "best_spatial_error_at_10_m": float(np.min(ranked_geometry[: min(10, len(ranked_geometry))])),
            "exact_nearest_reference_frame": exact_frame,
            "exact_nearest_rank": exact_rank,
            "reciprocal_rank": 1.0 / exact_rank,
            "recall_at_1": int(exact_rank <= 1),
            "recall_at_5": int(exact_rank <= 5),
            "recall_at_10": int(exact_rank <= 10),
        })
    return rows


def _descriptive(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    _require(array.size > 0, "lista de métricas vazia")
    _require(np.isfinite(array).all(), "métrica contém NaN/Inf")
    return {"mean": float(np.mean(array)), "median": float(np.median(array)), "p95": float(np.quantile(array, 0.95))}


def summarize_mode_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(rows), "rows não pode ser vazio")
    groups: dict[str, list[Mapping[str, Any]]] = {"all": list(rows)}
    for row in rows:
        groups.setdefault(str(row["split"]), []).append(row)
    output: dict[str, Any] = {}
    for group_name, group_rows in groups.items():
        output[group_name] = {
            "count": len(group_rows),
            **{field: _descriptive(float(row[field]) for row in group_rows) for field in PRIMARY_ERROR_FIELDS},
            "MRR_exact_nearest": float(np.mean([float(row["reciprocal_rank"]) for row in group_rows])),
            "Recall@1_exact_nearest": float(np.mean([int(row["recall_at_1"]) for row in group_rows])),
            "Recall@5_exact_nearest": float(np.mean([int(row["recall_at_5"]) for row in group_rows])),
            "Recall@10_exact_nearest": float(np.mean([int(row["recall_at_10"]) for row in group_rows])),
        }
    return output


def pair_mode_rows(split_rows: Sequence[Mapping[str, Any]], full_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    split_index = {int(row["query_frame"]): row for row in split_rows}
    full_index = {int(row["query_frame"]): row for row in full_rows}
    _require(set(split_index) == set(full_index), "os dois modos devem possuir exatamente as mesmas queries")
    output = []
    for query_frame in sorted(split_index):
        split = split_index[query_frame]
        full = full_index[query_frame]
        _require(str(split["split"]) == str(full["split"]), f"split divergente para query {query_frame}")
        output.append({
            "query_frame": query_frame,
            "split": str(split["split"]),
            "delta_top1_spatial_error_m_full_minus_split": float(full["top1_spatial_error_m"]) - float(split["top1_spatial_error_m"]),
            "delta_best_spatial_error_at_5_m_full_minus_split": float(full["best_spatial_error_at_5_m"]) - float(split["best_spatial_error_at_5_m"]),
            "delta_best_spatial_error_at_10_m_full_minus_split": float(full["best_spatial_error_at_10_m"]) - float(split["best_spatial_error_at_10_m"]),
            "delta_exact_nearest_rank_full_minus_split": int(full["exact_nearest_rank"]) - int(split["exact_nearest_rank"]),
        })
    return output


def summarize_paired_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(rows), "paired rows não pode ser vazio")
    groups: dict[str, list[Mapping[str, Any]]] = {"all": list(rows)}
    for row in rows:
        groups.setdefault(str(row["split"]), []).append(row)
    fields = (
        "delta_top1_spatial_error_m_full_minus_split",
        "delta_best_spatial_error_at_5_m_full_minus_split",
        "delta_best_spatial_error_at_10_m_full_minus_split",
        "delta_exact_nearest_rank_full_minus_split",
    )
    output: dict[str, Any] = {}
    for group_name, group_rows in groups.items():
        metrics = {}
        for field in fields:
            values = np.asarray([float(row[field]) for row in group_rows], dtype=np.float64)
            metrics[field] = {
                **_descriptive(values),
                "wins_full_image": int(np.count_nonzero(values < 0)),
                "ties": int(np.count_nonzero(values == 0)),
                "losses_full_image": int(np.count_nonzero(values > 0)),
            }
        output[group_name] = {"count": len(group_rows), "metrics": metrics}
    return output
