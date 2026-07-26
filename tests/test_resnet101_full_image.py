from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

import loop_closure.descriptors.resnet101_full_image as full_image_module
from loop_closure.descriptors.resnet101_full_image import (
    FULL_IMAGE_DESCRIPTOR_LENGTH,
    extract_resnet101_full_image_descriptor,
    full_image_resnet101_tensor,
)


def normalized(value: int) -> float:
    return ((value / 255.0) - 0.36) / 0.28


def synthetic_rgb_image() -> Image.Image:
    x = np.arange(256, dtype=np.uint8)
    y = np.arange(256, dtype=np.uint8)
    xx = np.tile(x, (256, 1))
    yy = np.tile(y[:, None], (1, 256))
    blue = (
        (xx.astype(np.uint16) + yy.astype(np.uint16)) % 256
    ).astype(np.uint8)
    array = np.stack([xx, yy, blue], axis=-1)
    return Image.fromarray(array, mode="RGB")


def test_full_image_preprocessing_preserves_whole_256_image() -> None:
    tensor = full_image_resnet101_tensor(synthetic_rgb_image())

    assert tensor.shape == (3, 256, 256)
    assert tensor.dtype == torch.float32
    assert tensor.is_contiguous()
    assert torch.isfinite(tensor).all()

    assert tensor[0, 0, 0].item() == pytest.approx(
        normalized(0), abs=1e-6
    )
    assert tensor[0, 0, 255].item() == pytest.approx(
        normalized(255), abs=1e-6
    )
    assert tensor[1, 255, 0].item() == pytest.approx(
        normalized(255), abs=1e-6
    )


def test_full_image_preprocessing_converts_grayscale_to_rgb() -> None:
    image = Image.new("L", (640, 480), color=128)
    tensor = full_image_resnet101_tensor(image)

    assert tensor.shape == (3, 256, 256)
    assert torch.allclose(tensor[0], tensor[1])
    assert torch.allclose(tensor[1], tensor[2])


def test_full_image_preprocessing_is_not_a_124_crop() -> None:
    image = Image.new("RGB", (256, 256), color=(1, 2, 3))
    tensor = full_image_resnet101_tensor(image)

    assert tensor.shape[-2:] == (256, 256)
    assert tensor.shape[-2:] != (124, 124)


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eval_calls = 0
        self.train()

    def eval(self):
        self.eval_calls += 1
        return super().eval()


def test_full_image_extractor_uses_one_forward_and_returns_2048(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, ...]] = []

    def fake_forward(
        model: nn.Module,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        del model
        calls.append(tuple(batch.shape))
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        return torch.arange(
            FULL_IMAGE_DESCRIPTOR_LENGTH,
            dtype=torch.float32,
            device=batch.device,
        ).reshape(1, FULL_IMAGE_DESCRIPTOR_LENGTH, 1, 1)

    monkeypatch.setattr(
        full_image_module,
        "forward_resnet101_avgpool",
        fake_forward,
    )

    model = DummyModel()
    descriptor = extract_resnet101_full_image_descriptor(
        model,
        Image.new("RGB", (1226, 370), color=(20, 40, 60)),
        device="cpu",
    )

    assert model.eval_calls == 1
    assert not model.training
    assert calls == [(1, 3, 256, 256)]

    assert descriptor.shape == (2048,)
    assert descriptor.dtype == torch.float32
    assert descriptor.device.type == "cpu"
    assert descriptor.requires_grad is False
    assert descriptor.is_contiguous()
    assert torch.isfinite(descriptor).all()
    assert descriptor[0].item() == pytest.approx(0.0)
    assert descriptor[-1].item() == pytest.approx(2047.0)
