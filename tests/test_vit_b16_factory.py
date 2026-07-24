from __future__ import annotations

import pytest
import torch
from torchvision.models import ViT_B_16_Weights

from loop_closure.descriptors.vit_b16 import (
    DEFAULT_VIT_B16_WEIGHTS,
    create_vit_b16_model,
)


class FakeModel:
    def __init__(self) -> None:
        self.to_calls = []
        self.eval_calls = 0

    def to(self, device):
        self.to_calls.append(torch.device(device))
        return self

    def eval(self):
        self.eval_calls += 1
        return self


def test_default_weights_are_explicit_imagenet1k_v1():
    assert DEFAULT_VIT_B16_WEIGHTS is ViT_B_16_Weights.IMAGENET1K_V1


def test_factory_passes_explicit_weights_and_calls_to_eval():
    calls = []
    fake = FakeModel()

    def factory(*, weights):
        calls.append(weights)
        return fake

    result = create_vit_b16_model(device="cpu", factory=factory)
    assert result is fake
    assert calls == [ViT_B_16_Weights.IMAGENET1K_V1]
    assert fake.to_calls == [torch.device("cpu")]
    assert fake.eval_calls == 1


def test_factory_allows_explicit_member_override():
    calls = []
    fake = FakeModel()

    def factory(*, weights):
        calls.append(weights)
        return fake

    create_vit_b16_model(
        weights=ViT_B_16_Weights.DEFAULT,
        factory=factory,
    )
    assert calls == [ViT_B_16_Weights.DEFAULT]


def test_factory_rejects_enum_class_instead_of_member():
    with pytest.raises(TypeError, match="explicit ViT_B_16_Weights member"):
        create_vit_b16_model(weights=ViT_B_16_Weights)  # type: ignore[arg-type]


def test_factory_rejects_arbitrary_value():
    with pytest.raises(TypeError, match="explicit ViT_B_16_Weights member"):
        create_vit_b16_model(weights="DEFAULT")  # type: ignore[arg-type]
