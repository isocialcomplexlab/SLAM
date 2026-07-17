"""Reprodução explícita da semântica do notebook ResNet-101 legado.

Este módulo não implementa a extração de descritores e não carrega pesos.
Ele preserva somente:

1. a distância produzida pelo uso legado de ``torch.cdist``;
2. a comparação inclusiva com o limiar;
3. a regra de adjacência dentro do banco histórico.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


def _validate_legacy_descriptor(
    descriptor: torch.Tensor,
    *,
    argument_name: str,
) -> None:
    """Valida a forma de descritor observada no notebook legado."""
    if not isinstance(descriptor, torch.Tensor):
        raise TypeError(
            f"{argument_name} deve ser torch.Tensor; "
            f"recebido {type(descriptor).__name__}."
        )

    if descriptor.ndim != 4:
        raise ValueError(
            f"{argument_name} deve ter quatro dimensões [1, D, 1, 1]; "
            f"recebido shape={tuple(descriptor.shape)}."
        )

    if descriptor.shape[0] != 1:
        raise ValueError(
            f"{argument_name} deve ter batch_size=1; "
            f"recebido shape={tuple(descriptor.shape)}."
        )

    if tuple(descriptor.shape[-2:]) != (1, 1):
        raise ValueError(
            f"{argument_name} deve terminar em [1, 1]; "
            f"recebido shape={tuple(descriptor.shape)}."
        )

    if not descriptor.is_floating_point():
        raise TypeError(
            f"{argument_name} deve possuir dtype de ponto flutuante; "
            f"recebido dtype={descriptor.dtype}."
        )


def legacy_exact_distance(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:
    """Reproduz a distância efetivamente calculada no notebook.

    Para entradas com forma ``[1, D, 1, 1]``, ``torch.cdist`` interpreta:

    - ``[1, D]`` como dimensões de batch;
    - ``P = 1``;
    - ``M = 1``.

    O ``torch.max`` subsequente resulta em:

        max(abs(first - second))

    Portanto, esta função representa a distância L-infinito/Chebyshev entre
    os componentes, e não a distância Euclidiana do vetor achatado.
    """
    _validate_legacy_descriptor(first, argument_name="first")
    _validate_legacy_descriptor(second, argument_name="second")

    if first.shape != second.shape:
        raise ValueError(
            "Os descritores devem possuir a mesma forma; "
            f"recebidos {tuple(first.shape)} e {tuple(second.shape)}."
        )

    distances = torch.cdist(first, second, p=2)
    return float(torch.max(distances).item())


def legacy_exact_match(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    threshold: float = 0.25,
) -> bool:
    """Aplica a comparação inclusiva ``distance <= threshold``."""
    distance = legacy_exact_distance(first, second)
    return distance <= threshold


def legacy_belief_generation(
    database: Iterable[torch.Tensor],
    descriptor: torch.Tensor,
    *,
    threshold: float = 0.25,
) -> int:
    """Reproduz a regra ``belief_generation`` do notebook legado.

    Um único descritor atual é comparado, em ordem, com o banco histórico.

    Uma não correspondência zera a sequência corrente. Para uma sequência de
    ``r`` correspondências adjacentes, com ``r >= 3``, a contribuição ao
    contador é ``r - 2``.
    """
    _validate_legacy_descriptor(
        descriptor,
        argument_name="descriptor",
    )

    consecutive_matches = 0
    loop_count = 0

    for database_descriptor in database:
        if legacy_exact_match(
            descriptor,
            database_descriptor,
            threshold=threshold,
        ):
            consecutive_matches += 1
        else:
            consecutive_matches = 0

        if consecutive_matches >= 3:
            loop_count += 1

    return loop_count
