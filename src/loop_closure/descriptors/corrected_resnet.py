"""Distância corrigida para descritores ResNet.

Esta implementação é metodologicamente separada da reprodução
``legacy_exact``. Os tensores são achatados antes do cálculo da distância
Euclidiana L2.
"""

from __future__ import annotations

import math

import torch


def _validate_descriptor(
    descriptor: torch.Tensor,
    *,
    argument_name: str,
) -> None:
    """Valida um tensor de descritor antes do cálculo da distância."""
    if not isinstance(descriptor, torch.Tensor):
        raise TypeError(
            f"{argument_name} deve ser torch.Tensor; "
            f"recebido {type(descriptor).__name__}."
        )

    if not descriptor.is_floating_point():
        raise TypeError(
            f"{argument_name} deve possuir dtype de ponto flutuante; "
            f"recebido dtype={descriptor.dtype}."
        )

    if descriptor.numel() == 0:
        raise ValueError(
            f"{argument_name} não pode ser um descritor vazio."
        )

    if not bool(torch.isfinite(descriptor).all()):
        raise ValueError(
            f"{argument_name} contém valores NaN ou infinitos."
        )


def _validate_threshold(threshold: float) -> float:
    """Valida e normaliza o limiar de distância."""
    if isinstance(threshold, bool) or not isinstance(
        threshold,
        (int, float),
    ):
        raise TypeError(
            "threshold deve ser um número real."
        )

    normalized = float(threshold)

    if not math.isfinite(normalized):
        raise ValueError(
            "threshold deve ser finito."
        )

    if normalized < 0:
        raise ValueError(
            "threshold não pode ser negativo."
        )

    return normalized


def corrected_l2_distance(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:
    """Calcula a distância Euclidiana entre descritores achatados.

    São aceitas formas diferentes, como ``[1, D, 1, 1]`` e ``[D]``, desde
    que os tensores tenham o mesmo número total de elementos.
    """
    _validate_descriptor(first, argument_name="first")
    _validate_descriptor(second, argument_name="second")

    if first.numel() != second.numel():
        raise ValueError(
            "Os descritores devem possuir o mesmo número de elementos; "
            f"recebidos {first.numel()} e {second.numel()}."
        )

    first_flat = first.detach().reshape(-1)
    second_flat = second.detach().reshape(-1)

    # O cálculo é feito no dispositivo e dtype dos tensores. Caso estejam em
    # dispositivos ou dtypes distintos, o segundo descritor é convertido de
    # forma explícita para permitir uma comparação determinística.
    second_flat = second_flat.to(
        device=first_flat.device,
        dtype=first_flat.dtype,
    )

    distance = torch.linalg.vector_norm(
        first_flat - second_flat,
        ord=2,
    )

    return float(distance.item())


def corrected_l2_match(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    threshold: float,
) -> bool:
    """Aplica a comparação inclusiva ``L2 distance <= threshold``."""
    normalized_threshold = _validate_threshold(threshold)

    return (
        corrected_l2_distance(first, second)
        <= normalized_threshold
    )
