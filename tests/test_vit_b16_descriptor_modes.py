from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from loop_closure.descriptors.vit_b16 import (
    DEFAULT_FINAL_BLOCK_INDEX,
    DEFAULT_INTERMEDIATE_BLOCK_INDEX,
    DEFAULT_REGION_ORDER,
    DESCRIPTOR_DIMENSIONS,
    extract_vit_b16_descriptors,
    forward_vit_b16_region,
    legacy_vit_b16_regions,
)


class AddLayer(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, x):
        return x + self.value


class FakeViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.class_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.encoder = SimpleNamespace(
            pos_embedding=torch.zeros(1, 197, 768),
            dropout=nn.Identity(),
            layers=nn.Sequential(*[AddLayer(float(i + 1)) for i in range(12)]),
            ln=nn.Identity(),
        )
        self.heads = nn.Linear(768, 1000, bias=False)
        with torch.no_grad():
            self.heads.weight.zero_()
            self.heads.weight[:, 0] = 1.0

    def _process_input(self, batch):
        base = batch.mean(dim=(1, 2, 3), keepdim=False)
        tokens = base[:, None, None].expand(-1, 196, 768).clone()
        return tokens


class EvalTrackingFakeViT(FakeViT):
    def __init__(self) -> None:
        super().__init__()
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return super().eval()


def synthetic_image() -> Image.Image:
    data = np.zeros((300, 400, 3), dtype=np.uint8)
    data[:, :200, 0] = 32
    data[:, 200:, 0] = 224
    data[:, :, 1] = np.arange(400, dtype=np.uint16)[None, :] % 256
    return Image.fromarray(data, mode="RGB")


def test_approved_constants_are_preserved():
    assert DEFAULT_REGION_ORDER == ('right', 'left')
    assert DEFAULT_INTERMEDIATE_BLOCK_INDEX == 5
    assert DEFAULT_FINAL_BLOCK_INDEX == 11
    assert DESCRIPTOR_DIMENSIONS == {
        "legacy_logits": 2000,
        "intermediate_cls": 1536,
        "intermediate_mean": 1536,
        "final_cls": 1536,
        "final_mean": 1536,
    }


def test_historical_preprocessing_returns_two_finite_224_rgb_regions():
    regions = legacy_vit_b16_regions(synthetic_image())
    assert len(regions) == 2
    for region in regions:
        assert region.shape == (3, 224, 224)
        assert region.dtype == torch.float32
        assert torch.isfinite(region).all()


def test_preprocessing_preserves_approved_region_order():
    by_default = legacy_vit_b16_regions(synthetic_image())
    explicit = legacy_vit_b16_regions(
        synthetic_image(), region_order=DEFAULT_REGION_ORDER
    )
    assert torch.equal(by_default[0], explicit[0])
    assert torch.equal(by_default[1], explicit[1])
    assert not torch.equal(by_default[0], by_default[1])


def test_forward_modes_use_requested_blocks_and_exclude_cls_from_mean():
    model = FakeViT().eval()
    batch = torch.zeros(1, 3, 224, 224)
    output = forward_vit_b16_region(model, batch)

    # Sum 1..6 after block index 5; sum 1..12 after final block index 11.
    assert torch.allclose(output["intermediate_cls"], torch.full((1, 768), 21.0))
    assert torch.allclose(output["intermediate_mean"], torch.full((1, 768), 21.0))
    assert torch.allclose(output["final_cls"], torch.full((1, 768), 78.0))
    assert torch.allclose(output["final_mean"], torch.full((1, 768), 78.0))
    assert torch.allclose(output["legacy_logits"], torch.full((1, 1000), 78.0))


def test_extractor_returns_all_two_region_dimensions_and_calls_eval():
    model = EvalTrackingFakeViT()
    output = extract_vit_b16_descriptors(synthetic_image(), model)
    assert model.eval_calls == 1
    assert set(output) == set(DESCRIPTOR_DIMENSIONS)
    for mode, expected in DESCRIPTOR_DIMENSIONS.items():
        assert output[mode].shape == (1, expected)
        assert output[mode].dtype == torch.float32
        assert output[mode].device.type == "cpu"
        assert torch.isfinite(output[mode]).all()


def test_extractor_supports_subset_of_modes():
    model = FakeViT()
    output = extract_vit_b16_descriptors(
        synthetic_image(),
        model,
        modes=("legacy_logits", "final_cls"),
    )
    assert set(output) == {"legacy_logits", "final_cls"}


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown descriptor modes"):
        extract_vit_b16_descriptors(
            synthetic_image(),
            FakeViT(),
            modes=("unknown",),
        )


def test_region_batch_shape_is_strictly_validated():
    with pytest.raises(ValueError, match=r"\[1, 3, 224, 224\]"):
        forward_vit_b16_region(FakeViT(), torch.zeros(3, 224, 224))
