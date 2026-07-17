from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from loop_closure.descriptors.resnet101 import (
    concatenate_right_left_avgpool,
    legacy_resnet101_crops,
)


MEAN = 0.36
STD = 0.28


def synthetic_rgb_image() -> Image.Image:
    """
    Cria uma imagem RGB 256x256 em que cada pixel codifica sua posição:

    R = x
    G = y
    B = (x + y) mod 256
    """
    x = np.arange(256, dtype=np.uint8)
    y = np.arange(256, dtype=np.uint8)

    xx = np.tile(x, (256, 1))
    yy = np.tile(y[:, None], (1, 256))
    blue = ((xx.astype(np.uint16) + yy.astype(np.uint16)) % 256).astype(
        np.uint8
    )

    array = np.stack([xx, yy, blue], axis=-1)
    return Image.fromarray(array, mode="RGB")


def normalized(value: int) -> float:
    return ((value / 255.0) - MEAN) / STD


def test_legacy_resnet101_crops_return_right_then_left() -> None:
    image = synthetic_rgb_image()

    right, left = legacy_resnet101_crops(image)

    assert right.shape == (3, 124, 124)
    assert left.shape == (3, 124, 124)

    # Primeiro pixel do crop direito:
    # original em x=124, y=4.
    assert right[0, 0, 0].item() == pytest.approx(
        normalized(124),
        abs=1e-6,
    )
    assert right[1, 0, 0].item() == pytest.approx(
        normalized(4),
        abs=1e-6,
    )
    assert right[2, 0, 0].item() == pytest.approx(
        normalized(128),
        abs=1e-6,
    )

    # Primeiro pixel do crop esquerdo:
    # original em x=4, y=4.
    assert left[0, 0, 0].item() == pytest.approx(
        normalized(4),
        abs=1e-6,
    )
    assert left[1, 0, 0].item() == pytest.approx(
        normalized(4),
        abs=1e-6,
    )
    assert left[2, 0, 0].item() == pytest.approx(
        normalized(8),
        abs=1e-6,
    )


def test_legacy_resnet101_crop_boundaries_are_exact() -> None:
    image = synthetic_rgb_image()

    right, left = legacy_resnet101_crops(image)

    # Último pixel do crop direito:
    # x = 124 + 123 = 247
    # y = 4 + 123 = 127
    assert right[0, -1, -1].item() == pytest.approx(
        normalized(247),
        abs=1e-6,
    )
    assert right[1, -1, -1].item() == pytest.approx(
        normalized(127),
        abs=1e-6,
    )
    assert right[2, -1, -1].item() == pytest.approx(
        normalized((247 + 127) % 256),
        abs=1e-6,
    )

    # Último pixel do crop esquerdo:
    # x = 4 + 123 = 127
    # y = 4 + 123 = 127
    assert left[0, -1, -1].item() == pytest.approx(
        normalized(127),
        abs=1e-6,
    )
    assert left[1, -1, -1].item() == pytest.approx(
        normalized(127),
        abs=1e-6,
    )
    assert left[2, -1, -1].item() == pytest.approx(
        normalized(254),
        abs=1e-6,
    )


def test_legacy_resnet101_crops_resize_before_cropping() -> None:
    image = Image.new(
        mode="RGB",
        size=(640, 480),
        color=(100, 120, 140),
    )

    right, left = legacy_resnet101_crops(image)

    assert right.shape == (3, 124, 124)
    assert left.shape == (3, 124, 124)


def test_legacy_resnet101_crops_convert_grayscale_to_rgb() -> None:
    image = Image.new(
        mode="L",
        size=(256, 256),
        color=128,
    )

    right, left = legacy_resnet101_crops(image)

    assert right.shape == (3, 124, 124)
    assert left.shape == (3, 124, 124)

    assert torch.allclose(right[0], right[1])
    assert torch.allclose(right[1], right[2])
    assert torch.allclose(left[0], left[1])
    assert torch.allclose(left[1], left[2])


def test_concatenation_produces_flat_4096_descriptor() -> None:
    right = torch.zeros(
        (1, 2048, 1, 1),
        dtype=torch.float32,
    )
    left = torch.ones(
        (1, 2048, 1, 1),
        dtype=torch.float32,
    )

    descriptor = concatenate_right_left_avgpool(right, left)

    assert descriptor.shape == (4096,)
    assert descriptor.dtype == torch.float32

    assert torch.equal(
        descriptor[:2048],
        torch.zeros(2048, dtype=torch.float32),
    )
    assert torch.equal(
        descriptor[2048:],
        torch.ones(2048, dtype=torch.float32),
    )


def test_concatenation_preserves_right_then_left_order() -> None:
    right = torch.arange(
        0,
        2048,
        dtype=torch.float32,
    ).reshape(1, 2048, 1, 1)

    left = torch.arange(
        10000,
        12048,
        dtype=torch.float32,
    ).reshape(1, 2048, 1, 1)

    descriptor = concatenate_right_left_avgpool(right, left)

    assert descriptor[0].item() == pytest.approx(0.0)
    assert descriptor[2047].item() == pytest.approx(2047.0)
    assert descriptor[2048].item() == pytest.approx(10000.0)
    assert descriptor[-1].item() == pytest.approx(12047.0)


def test_concatenation_rejects_wrong_channel_count() -> None:
    right = torch.zeros(
        (1, 1024, 1, 1),
        dtype=torch.float32,
    )
    left = torch.zeros(
        (1, 2048, 1, 1),
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="2048"):
        concatenate_right_left_avgpool(right, left)


def test_concatenation_rejects_batch_greater_than_one() -> None:
    right = torch.zeros(
        (2, 2048, 1, 1),
        dtype=torch.float32,
    )
    left = torch.zeros(
        (2, 2048, 1, 1),
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="batch"):
        concatenate_right_left_avgpool(right, left)


def test_concatenation_rejects_non_avgpool_spatial_shape() -> None:
    right = torch.zeros(
        (1, 2048, 2, 2),
        dtype=torch.float32,
    )
    left = torch.zeros(
        (1, 2048, 1, 1),
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match=r"1.*2048.*1.*1"):
        concatenate_right_left_avgpool(right, left)


def test_concatenation_rejects_non_floating_tensors() -> None:
    right = torch.zeros(
        (1, 2048, 1, 1),
        dtype=torch.int64,
    )
    left = torch.zeros(
        (1, 2048, 1, 1),
        dtype=torch.int64,
    )

    with pytest.raises(TypeError, match="ponto flutuante"):
        concatenate_right_left_avgpool(right, left)
