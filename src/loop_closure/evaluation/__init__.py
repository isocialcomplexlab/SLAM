"""Avaliação de descritores e detecção de fechamento de ciclo."""

from loop_closure.evaluation.resnet101_pair_scores import (
    build_descriptor_index,
    classify_spatial_distance,
    compute_pair_score_rows,
    load_descriptor_index_from_npz,
    save_pair_score_rows,
)

__all__ = [
    "build_descriptor_index",
    "classify_spatial_distance",
    "compute_pair_score_rows",
    "load_descriptor_index_from_npz",
    "save_pair_score_rows",
]
