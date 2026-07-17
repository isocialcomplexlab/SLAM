from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from PIL import Image
from torch import nn

import loop_closure.descriptors.resnet101 as resnet101_module
from loop_closure.descriptors.resnet101 import (
    extract_resnet101_descriptor,
    forward_resnet101_avgpool,
)


class RecordingLayer(nn.Module):
    """Camada falsa que registra ordem e estado de execução."""

    def __init__(
        self,
        name: str,
        execution_log: list[str],
        inference_flags: list[bool],
        grad_flags: list[bool],
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.execution_log = execution_log
        self.inference_flags = inference_flags
        self.grad_flags = grad_flags
        self.transform = transform

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.execution_log.append(self.name)
        self.inference_flags.append(torch.is_inference_mode_enabled())
        self.grad_flags.append(torch.is_grad_enabled())

        if self.transform is not None:
            return self.transform(inputs)

        return inputs


class ExplodingFullyConnectedLayer(nn.Module):
    """Falha imediatamente caso a camada fc seja executada."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise AssertionError("A camada fc não deve participar do descritor.")


class FakeResNet101(nn.Module):
    """Modelo mínimo com a mesma interface estrutural da ResNet-101."""

    def __init__(self) -> None:
        super().__init__()

        self.execution_log: list[str] = []
        self.inference_flags: list[bool] = []
        self.grad_flags: list[bool] = []
        self.received_shapes: list[tuple[int, ...]] = []
        self.eval_calls = 0

        def avgpool_transform(inputs: torch.Tensor) -> torch.Tensor:
            self.received_shapes.append(tuple(inputs.shape))

            value = inputs.mean().reshape(1, 1, 1, 1)

            return value.expand(
                inputs.shape[0],
                2048,
                1,
                1,
            ).clone()

        self.conv1 = RecordingLayer(
            "conv1",
            self.execution_log,
            self.inference_flags,
            self.grad_flags,
        )
        self.bn1 = RecordingLayer(
            "bn1",
            self.execution_log,
            self.inference_flags,
            self.grad_flags,
        )
        self.relu = RecordingLayer(
            "relu",
            self.execution_log,
            self.inference_flags,
            self.grad_flags,
        )
        self.maxpool = RecordingLayer(
            "maxpool",
            self.execution_log,
            self.inference_flags,
            self.grad_flags,
        )
        self.layer1 = RecordingLayer(
            "layer1",
            self.execution_log,
            self.inference_flags,
            self.grad_flags,
        )
        self.layer2 = RecordingLayer(
            "layer2",
            self.execution_log,
            self.inference_flags,
            self.grad_flags,
        )
        self.layer3 = RecordingLayer(
            "layer3",
            self.execution_log,
            self.inference_flags,
            self.grad_flags,
        )
        self.layer4 = RecordingLayer(
            "layer4",
            self.execution_log,
            self.inference_flags,
            self.grad_flags,
        )
        self.avgpool = RecordingLayer(
            "avgpool",
            self.execution_log,
            self.inference_flags,
            self.grad_flags,
            transform=avgpool_transform,
        )
        self.fc = ExplodingFullyConnectedLayer()

        # Garante que o modelo começa em modo de treinamento.
        self.train()

    def eval(self) -> FakeResNet101:
        self.eval_calls += 1
        super().eval()
        return self


EXPECTED_FORWARD_ORDER = [
    "conv1",
    "bn1",
    "relu",
    "maxpool",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
    "avgpool",
]


def test_forward_resnet101_avgpool_uses_exact_legacy_layer_order() -> None:
    model = FakeResNet101()
    batch = torch.ones((1, 3, 124, 124), dtype=torch.float32)

    output = forward_resnet101_avgpool(model, batch)

    assert model.execution_log == EXPECTED_FORWARD_ORDER
    assert output.shape == (1, 2048, 1, 1)


def test_forward_resnet101_avgpool_does_not_execute_fc() -> None:
    model = FakeResNet101()
    batch = torch.ones((1, 3, 124, 124), dtype=torch.float32)

    # ExplodingFullyConnectedLayer geraria AssertionError caso fc fosse usada.
    output = forward_resnet101_avgpool(model, batch)

    assert output.shape == (1, 2048, 1, 1)
    assert "fc" not in model.execution_log


def test_extractor_calls_eval_and_uses_inference_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeResNet101()

    right = torch.ones((3, 124, 124), dtype=torch.float32)
    left = torch.full((3, 124, 124), 2.0, dtype=torch.float32)

    monkeypatch.setattr(
        resnet101_module,
        "legacy_resnet101_crops",
        lambda image: (right, left),
    )

    image = Image.new("RGB", (256, 256))

    descriptor = extract_resnet101_descriptor(
        model,
        image,
        device="cpu",
    )

    assert model.eval_calls == 1
    assert not model.training

    assert model.inference_flags
    assert all(model.inference_flags)

    assert model.grad_flags
    assert not any(model.grad_flags)

    assert descriptor.requires_grad is False


def test_extractor_processes_right_before_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeResNet101()

    right = torch.ones((3, 124, 124), dtype=torch.float32)
    left = torch.full((3, 124, 124), 2.0, dtype=torch.float32)

    monkeypatch.setattr(
        resnet101_module,
        "legacy_resnet101_crops",
        lambda image: (right, left),
    )

    image = Image.new("RGB", (256, 256))

    descriptor = extract_resnet101_descriptor(
        model,
        image,
        device="cpu",
    )

    assert descriptor.shape == (4096,)

    assert torch.equal(
        descriptor[:2048],
        torch.ones(2048, dtype=torch.float32),
    )
    assert torch.equal(
        descriptor[2048:],
        torch.full((2048,), 2.0, dtype=torch.float32),
    )

    assert model.execution_log == (
        EXPECTED_FORWARD_ORDER + EXPECTED_FORWARD_ORDER
    )


def test_extractor_adds_batch_dimension_to_each_crop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeResNet101()

    right = torch.ones((3, 124, 124), dtype=torch.float32)
    left = torch.ones((3, 124, 124), dtype=torch.float32)

    monkeypatch.setattr(
        resnet101_module,
        "legacy_resnet101_crops",
        lambda image: (right, left),
    )

    image = Image.new("RGB", (256, 256))

    extract_resnet101_descriptor(
        model,
        image,
        device="cpu",
    )

    assert model.received_shapes == [
        (1, 3, 124, 124),
        (1, 3, 124, 124),
    ]


def test_extractor_returns_detached_cpu_float32_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeResNet101()

    right = torch.ones(
        (3, 124, 124),
        dtype=torch.float32,
        requires_grad=True,
    )
    left = torch.ones(
        (3, 124, 124),
        dtype=torch.float32,
        requires_grad=True,
    )

    monkeypatch.setattr(
        resnet101_module,
        "legacy_resnet101_crops",
        lambda image: (right, left),
    )

    image = Image.new("RGB", (256, 256))

    descriptor = extract_resnet101_descriptor(
        model,
        image,
        device="cpu",
    )

    assert descriptor.shape == (4096,)
    assert descriptor.device.type == "cpu"
    assert descriptor.dtype == torch.float32
    assert descriptor.requires_grad is False


def test_forward_rejects_input_without_batch_dimension() -> None:
    model = FakeResNet101()
    invalid = torch.ones((3, 124, 124), dtype=torch.float32)

    with pytest.raises(ValueError, match=r"\[1, 3, H, W\]"):
        forward_resnet101_avgpool(model, invalid)


def test_forward_rejects_batch_greater_than_one() -> None:
    model = FakeResNet101()
    invalid = torch.ones((2, 3, 124, 124), dtype=torch.float32)

    with pytest.raises(ValueError, match="batch"):
        forward_resnet101_avgpool(model, invalid)


def test_forward_rejects_non_rgb_input() -> None:
    model = FakeResNet101()
    invalid = torch.ones((1, 1, 124, 124), dtype=torch.float32)

    with pytest.raises(ValueError, match="3 canais"):
        forward_resnet101_avgpool(model, invalid)


def test_forward_rejects_non_floating_input() -> None:
    model = FakeResNet101()
    invalid = torch.ones((1, 3, 124, 124), dtype=torch.int64)

    with pytest.raises(TypeError, match="ponto flutuante"):
        forward_resnet101_avgpool(model, invalid)
