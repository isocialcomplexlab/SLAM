from __future__ import annotations

import math

import pytest
import torch

from loop_closure.descriptors.legacy_resnet import (
    legacy_belief_generation,
    legacy_exact_distance,
    legacy_exact_match,
)


def descriptor(values: list[float]) -> torch.Tensor:
    """
    Reproduz a forma salva pelo notebook legado:

        [batch=1, channels=D, height=1, width=1]
    """
    return torch.tensor(
        values,
        dtype=torch.float32,
    ).reshape(1, len(values), 1, 1)


def test_legacy_exact_distance_is_chebyshev_not_flattened_l2() -> None:
    first = descriptor([0.0, 0.0, 0.0, 0.0])
    second = descriptor([0.3, 0.4, 0.0, 0.0])

    observed = legacy_exact_distance(first, second)

    expected_chebyshev = 0.4
    flattened_l2 = math.sqrt((0.3**2) + (0.4**2))

    assert observed == pytest.approx(expected_chebyshev)
    assert observed != pytest.approx(flattened_l2)


def test_legacy_exact_distance_matches_maximum_absolute_component_difference() -> None:
    first = descriptor([-1.0, 2.0, 4.0, 8.0])
    second = descriptor([-0.5, 1.0, 7.0, 7.75])

    observed = legacy_exact_distance(first, second)

    assert observed == pytest.approx(3.0)


def test_legacy_exact_match_uses_inclusive_threshold() -> None:
    first = descriptor([0.0, 0.0, 0.0])
    exactly_at_threshold = descriptor([0.25, 0.0, 0.0])
    above_threshold = descriptor([0.2501, 0.0, 0.0])

    assert legacy_exact_match(
        first,
        exactly_at_threshold,
        threshold=0.25,
    )

    assert not legacy_exact_match(
        first,
        above_threshold,
        threshold=0.25,
    )


def test_legacy_belief_requires_three_adjacent_database_matches() -> None:
    query = descriptor([0.0, 0.0])

    database = [
        descriptor([0.10, 0.00]),
        descriptor([0.20, 0.00]),
        descriptor([0.24, 0.00]),
    ]

    observed = legacy_belief_generation(
        database,
        query,
        threshold=0.25,
    )

    assert observed == 1


def test_legacy_belief_does_not_combine_non_adjacent_matches() -> None:
    query = descriptor([0.0, 0.0])

    database = [
        descriptor([0.10, 0.00]),
        descriptor([0.20, 0.00]),
        descriptor([1.00, 0.00]),
        descriptor([0.10, 0.00]),
        descriptor([0.20, 0.00]),
    ]

    observed = legacy_belief_generation(
        database,
        query,
        threshold=0.25,
    )

    assert observed == 0


def test_legacy_belief_resets_run_after_database_non_match() -> None:
    query = descriptor([0.0, 0.0])

    database = [
        descriptor([0.10, 0.00]),
        descriptor([0.20, 0.00]),
        descriptor([1.00, 0.00]),
        descriptor([0.10, 0.00]),
        descriptor([0.20, 0.00]),
        descriptor([0.24, 0.00]),
    ]

    observed = legacy_belief_generation(
        database,
        query,
        threshold=0.25,
    )

    assert observed == 1


def test_legacy_belief_counts_run_length_minus_two() -> None:
    query = descriptor([0.0, 0.0])

    database = [
        descriptor([0.10, 0.00]),
        descriptor([0.15, 0.00]),
        descriptor([0.20, 0.00]),
        descriptor([0.24, 0.00]),
        descriptor([0.05, 0.00]),
    ]

    observed = legacy_belief_generation(
        database,
        query,
        threshold=0.25,
    )

    # Para uma sequência de r=5 correspondências:
    # lp = r - 2 = 3.
    assert observed == 3


def test_legacy_belief_sums_multiple_adjacent_runs() -> None:
    query = descriptor([0.0, 0.0])

    database = [
        # Primeira sequência: r=3, contribuição 1.
        descriptor([0.10, 0.00]),
        descriptor([0.15, 0.00]),
        descriptor([0.20, 0.00]),
        # Interrompe a sequência.
        descriptor([1.00, 0.00]),
        # Segunda sequência: r=4, contribuição 2.
        descriptor([0.05, 0.00]),
        descriptor([0.10, 0.00]),
        descriptor([0.15, 0.00]),
        descriptor([0.20, 0.00]),
    ]

    observed = legacy_belief_generation(
        database,
        query,
        threshold=0.25,
    )

    assert observed == 3


def test_legacy_belief_uses_one_query_against_ordered_database() -> None:
    """
    Este teste documenta que L=3 no legado não representa três consultas
    consecutivas.

    Um único descritor de consulta é comparado com três elementos adjacentes
    do banco histórico.
    """
    current_query = descriptor([10.0, 10.0])

    ordered_database = [
        descriptor([10.10, 10.00]),
        descriptor([10.20, 10.00]),
        descriptor([10.24, 10.00]),
    ]

    observed = legacy_belief_generation(
        ordered_database,
        current_query,
        threshold=0.25,
    )

    assert observed > 0
