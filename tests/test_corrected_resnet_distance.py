from __future__ import annotations

import pytest
import torch

from loop_closure.descriptors.corrected_resnet import (
    corrected_l2_distance,
    corrected_l2_match,
)


def legacy_shape(values: list[float]) -> torch.Tensor:
    """Descritor na forma produzida pelo notebook: [1, D, 1, 1]."""
    return torch.tensor(
        values,
        dtype=torch.float32,
    ).reshape(1, len(values), 1, 1)


def flat_descriptor(values: list[float]) -> torch.Tensor:
    """Descritor revisado armazenado como vetor unidimensional."""
    return torch.tensor(values, dtype=torch.float32)


def test_corrected_l2_flattens_legacy_descriptor_before_distance() -> None:
    first = legacy_shape([0.0, 0.0, 0.0, 0.0])
    second = legacy_shape([0.3, 0.4, 0.0, 0.0])

    observed = corrected_l2_distance(first, second)

    assert observed == pytest.approx(0.5)


def test_corrected_l2_differs_from_legacy_chebyshev_semantics() -> None:
    first = legacy_shape([0.0, 0.0, 0.0, 0.0])
    second = legacy_shape([0.3, 0.4, 0.0, 0.0])

    corrected_distance = corrected_l2_distance(first, second)
    legacy_chebyshev_distance = 0.4

    assert corrected_distance == pytest.approx(0.5)
    assert corrected_distance != pytest.approx(legacy_chebyshev_distance)


def test_corrected_l2_accepts_flat_descriptors() -> None:
    first = flat_descriptor([1.0, 2.0, 3.0])
    second = flat_descriptor([4.0, 6.0, 3.0])

    observed = corrected_l2_distance(first, second)

    assert observed == pytest.approx(5.0)


def test_corrected_l2_accepts_equivalent_shapes_with_same_element_count() -> None:
    first = legacy_shape([0.0, 0.0, 0.0, 0.0])
    second = flat_descriptor([0.0, 3.0, 4.0, 0.0])

    observed = corrected_l2_distance(first, second)

    assert observed == pytest.approx(5.0)


def test_corrected_l2_returns_zero_for_identical_descriptors() -> None:
    first = flat_descriptor([0.1, -0.2, 0.3])
    second = flat_descriptor([0.1, -0.2, 0.3])

    assert corrected_l2_distance(first, second) == pytest.approx(0.0)


def test_corrected_l2_match_uses_inclusive_threshold() -> None:
    first = flat_descriptor([0.0, 0.0])
    exactly_at_threshold = flat_descriptor([0.3, 0.4])
    above_threshold = flat_descriptor([0.3001, 0.4])

    assert corrected_l2_match(
        first,
        exactly_at_threshold,
        threshold=0.5,
    )

    assert not corrected_l2_match(
        first,
        above_threshold,
        threshold=0.5,
    )


def test_corrected_l2_rejects_different_descriptor_lengths() -> None:
    first = flat_descriptor([0.0, 1.0, 2.0])
    second = flat_descriptor([0.0, 1.0])

    with pytest.raises(ValueError, match="mesmo número de elementos"):
        corrected_l2_distance(first, second)


@pytest.mark.parametrize(
    "invalid",
    [
        torch.tensor([1, 2, 3], dtype=torch.int64),
        torch.tensor([True, False]),
    ],
)
def test_corrected_l2_rejects_non_floating_tensors(
    invalid: torch.Tensor,
) -> None:
    valid = flat_descriptor([1.0, 2.0, 3.0])

    with pytest.raises(TypeError, match="ponto flutuante"):
        corrected_l2_distance(valid, invalid)


def test_corrected_l2_rejects_empty_descriptors() -> None:
    empty = torch.tensor([], dtype=torch.float32)

    with pytest.raises(ValueError, match="vazio"):
        corrected_l2_distance(empty, empty)
