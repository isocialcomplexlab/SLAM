from __future__ import annotations

import pytest
import torch
from torch import nn
from torchvision.models import ResNet101_Weights

import loop_closure.descriptors.resnet101 as resnet101_module
from loop_closure.descriptors.resnet101 import (
    DEFAULT_RESNET101_WEIGHTS,
    create_resnet101_model,
)


class TrackingModel(nn.Module):
    """Modelo falso para registrar configuração sem baixar pesos."""

    def __init__(self) -> None:
        super().__init__()
        self.eval_calls = 0
        self.to_calls: list[torch.device] = []

    def eval(self) -> TrackingModel:
        self.eval_calls += 1
        super().eval()
        return self

    def to(
        self,
        device: torch.device | str,
        *args: object,
        **kwargs: object,
    ) -> TrackingModel:
        self.to_calls.append(torch.device(device))
        return self


def test_default_weights_are_explicit_imagenet1k_v1() -> None:
    assert (
        DEFAULT_RESNET101_WEIGHTS
        is ResNet101_Weights.IMAGENET1K_V1
    )


def test_factory_passes_imagenet1k_v1_to_torchvision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_weights: list[ResNet101_Weights | None] = []
    fake_model = TrackingModel()

    def fake_resnet101(
        *,
        weights: ResNet101_Weights | None,
    ) -> TrackingModel:
        received_weights.append(weights)
        return fake_model

    monkeypatch.setattr(
        resnet101_module.models,
        "resnet101",
        fake_resnet101,
    )

    observed = create_resnet101_model(device="cpu")

    assert observed is fake_model
    assert received_weights == [
        ResNet101_Weights.IMAGENET1K_V1
    ]


def test_factory_places_model_on_requested_device_and_calls_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = TrackingModel()

    monkeypatch.setattr(
        resnet101_module.models,
        "resnet101",
        lambda *, weights: fake_model,
    )

    observed = create_resnet101_model(device="cpu")

    assert observed is fake_model
    assert fake_model.to_calls == [torch.device("cpu")]
    assert fake_model.eval_calls == 1
    assert not fake_model.training


def test_factory_allows_explicit_weights_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_weights: list[ResNet101_Weights | None] = []
    fake_model = TrackingModel()

    def fake_resnet101(
        *,
        weights: ResNet101_Weights | None,
    ) -> TrackingModel:
        received_weights.append(weights)
        return fake_model

    monkeypatch.setattr(
        resnet101_module.models,
        "resnet101",
        fake_resnet101,
    )

    create_resnet101_model(
        device="cpu",
        weights=None,
    )

    assert received_weights == [None]


def test_factory_does_not_replace_imagenet1k_v1_with_default_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_weights: list[ResNet101_Weights | None] = []
    fake_model = TrackingModel()

    def fake_resnet101(
        *,
        weights: ResNet101_Weights | None,
    ) -> TrackingModel:
        received_weights.append(weights)
        return fake_model

    monkeypatch.setattr(
        resnet101_module.models,
        "resnet101",
        fake_resnet101,
    )

    create_resnet101_model(device="cpu")

    assert received_weights[0] is ResNet101_Weights.IMAGENET1K_V1
    assert DEFAULT_RESNET101_WEIGHTS is not ResNet101_Weights.DEFAULT


def test_factory_rejects_invalid_weights_value() -> None:
    with pytest.raises(TypeError, match="ResNet101_Weights"):
        create_resnet101_model(
            device="cpu",
            weights="IMAGENET1K_V1",  # type: ignore[arg-type]
        )
