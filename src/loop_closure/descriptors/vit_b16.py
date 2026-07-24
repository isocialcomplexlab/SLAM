from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Final, Literal, TypeAlias

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.transforms import functional as TF

DescriptorMode: TypeAlias = Literal[
    "legacy_logits",
    "intermediate_cls",
    "intermediate_mean",
    "final_cls",
    "final_mean",
]

DEFAULT_VIT_B16_WEIGHTS: Final = ViT_B_16_Weights.IMAGENET1K_V1
DEFAULT_REGION_ORDER: Final[tuple[str, str]] = ('right', 'left')
DEFAULT_INTERMEDIATE_BLOCK_INDEX: Final[int] = 5
DEFAULT_FINAL_BLOCK_INDEX: Final[int] = 11

MODE_TO_ARCHIVE_KEY: Final[dict[str, str]] = {
    "legacy_logits": "historical_logits_2region",
    "intermediate_cls": "intermediate_cls_2region",
    "intermediate_mean": "intermediate_patch_mean_2region",
    "final_cls": "final_cls_2region",
    "final_mean": "final_patch_mean_2region",
}

DESCRIPTOR_DIMENSIONS: Final[dict[str, int]] = {
    "legacy_logits": 2000,
    "intermediate_cls": 1536,
    "intermediate_mean": 1536,
    "final_cls": 1536,
    "final_mean": 1536,
}

_ALL_MODES: Final[tuple[DescriptorMode, ...]] = (
    "legacy_logits",
    "intermediate_cls",
    "intermediate_mean",
    "final_cls",
    "final_mean",
)


def _validate_weights(weights: ViT_B_16_Weights) -> None:
    if isinstance(weights, type) or not isinstance(weights, ViT_B_16_Weights):
        raise TypeError(
            "weights must be an explicit ViT_B_16_Weights member, "
            "for example ViT_B_16_Weights.IMAGENET1K_V1"
        )


def create_vit_b16_model(
    *,
    device: str | torch.device = "cpu",
    weights: ViT_B_16_Weights = DEFAULT_VIT_B16_WEIGHTS,
    factory=vit_b_16,
) -> nn.Module:
    """Create the explicit ImageNet-1K V1 ViT-B/16 used by the pipeline."""
    _validate_weights(weights)
    model = factory(weights=weights)
    model = model.to(device)
    model.eval()
    return model


def _as_rgb_pil(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    return image.convert("RGB")


def legacy_vit_b16_regions(
    image: Image.Image,
    *,
    region_order: tuple[str, str] = DEFAULT_REGION_ORDER,
) -> tuple[Tensor, Tensor]:
    """Apply the historical 256 resize, 224 crops, and normalization."""
    if len(region_order) != 2 or set(region_order) != {"left", "right"}:
        raise ValueError("region_order must contain left and right exactly once")

    resized = TF.resize(_as_rgb_pil(image), [256, 256])
    crops = {
        "left": TF.crop(resized, top=4, left=4, height=224, width=224),
        "right": TF.crop(resized, top=4, left=124, height=224, width=224),
    }

    normalized: dict[str, Tensor] = {}
    for name, crop in crops.items():
        tensor = TF.pil_to_tensor(crop).to(dtype=torch.float32).div_(255.0)
        normalized[name] = TF.normalize(
            tensor,
            mean=[0.36, 0.36, 0.36],
            std=[0.28, 0.28, 0.28],
        )

    return normalized[region_order[0]], normalized[region_order[1]]


def _validate_region_batch(batch: Tensor) -> None:
    if not isinstance(batch, Tensor):
        raise TypeError("batch must be a torch.Tensor")
    if batch.ndim != 4 or tuple(batch.shape) != (1, 3, 224, 224):
        raise ValueError("batch must have shape [1, 3, 224, 224]")
    if not torch.is_floating_point(batch):
        raise TypeError("batch must use a floating dtype")
    if not torch.isfinite(batch).all():
        raise ValueError("batch contains non-finite values")


def _encoder_layers(model: nn.Module) -> list[nn.Module]:
    layers = getattr(getattr(model, "encoder", None), "layers", None)
    if layers is None:
        raise TypeError("model does not expose encoder.layers")
    return list(layers)


def forward_vit_b16_region(
    model: nn.Module,
    batch: Tensor,
    *,
    intermediate_block_index: int = DEFAULT_INTERMEDIATE_BLOCK_INDEX,
    final_block_index: int = DEFAULT_FINAL_BLOCK_INDEX,
) -> dict[DescriptorMode, Tensor]:
    """Return historical logits and controlled internal-token descriptors."""
    _validate_region_batch(batch)
    layers = _encoder_layers(model)

    if not 0 <= intermediate_block_index < len(layers):
        raise ValueError("intermediate_block_index is outside encoder.layers")
    if not 0 <= final_block_index < len(layers):
        raise ValueError("final_block_index is outside encoder.layers")
    if final_block_index != len(layers) - 1:
        raise ValueError("final_block_index must select the last encoder block")
    if intermediate_block_index >= final_block_index:
        raise ValueError("intermediate block must precede final block")

    process_input = getattr(model, "_process_input", None)
    class_token = getattr(model, "class_token", None)
    encoder = getattr(model, "encoder", None)
    heads = getattr(model, "heads", None)

    if not callable(process_input) or class_token is None or encoder is None:
        raise TypeError("model does not expose torchvision ViT internals")
    if not callable(heads):
        raise TypeError("model.heads is not callable")

    tokens = process_input(batch)
    if tokens.ndim != 3:
        raise ValueError("_process_input must return [batch, patches, hidden]")

    batch_class_token = class_token.expand(tokens.shape[0], -1, -1)
    tokens = torch.cat([batch_class_token, tokens], dim=1)
    tokens = encoder.dropout(tokens + encoder.pos_embedding)

    intermediate_tokens: Tensor | None = None
    final_tokens: Tensor | None = None

    for index, layer in enumerate(layers):
        tokens = layer(tokens)
        if index == intermediate_block_index:
            intermediate_tokens = tokens
        if index == final_block_index:
            final_tokens = encoder.ln(tokens)

    if intermediate_tokens is None or final_tokens is None:
        raise RuntimeError("failed to capture requested encoder states")

    logits = heads(final_tokens[:, 0])
    result: dict[DescriptorMode, Tensor] = {
        "legacy_logits": logits,
        "intermediate_cls": intermediate_tokens[:, 0],
        "intermediate_mean": intermediate_tokens[:, 1:].mean(dim=1),
        "final_cls": final_tokens[:, 0],
        "final_mean": final_tokens[:, 1:].mean(dim=1),
    }

    for name, value in result.items():
        if value.ndim != 2 or value.shape[0] != 1:
            raise ValueError(f"{name} must have shape [1, dimension]")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")

    return result


def _normalize_modes(modes: Iterable[str] | None) -> tuple[DescriptorMode, ...]:
    if modes is None:
        return _ALL_MODES
    selected = tuple(modes)
    if not selected:
        raise ValueError("at least one descriptor mode is required")
    unknown = [item for item in selected if item not in _ALL_MODES]
    if unknown:
        raise ValueError(f"unknown descriptor modes: {unknown}")
    if len(set(selected)) != len(selected):
        raise ValueError("descriptor modes must be unique")
    return selected  # type: ignore[return-value]


def extract_vit_b16_descriptors(
    image: Image.Image,
    model: nn.Module,
    *,
    device: str | torch.device = "cpu",
    modes: Iterable[str] | None = None,
    region_order: tuple[str, str] = DEFAULT_REGION_ORDER,
    intermediate_block_index: int = DEFAULT_INTERMEDIATE_BLOCK_INDEX,
    final_block_index: int = DEFAULT_FINAL_BLOCK_INDEX,
) -> dict[DescriptorMode, Tensor]:
    """Extract all requested two-region ViT descriptors for one image."""
    selected = _normalize_modes(modes)
    regions = legacy_vit_b16_regions(image, region_order=region_order)
    per_region: list[Mapping[DescriptorMode, Tensor]] = []

    model.eval()
    with torch.inference_mode():
        for region in regions:
            per_region.append(
                forward_vit_b16_region(
                    model,
                    region.unsqueeze(0).to(device),
                    intermediate_block_index=intermediate_block_index,
                    final_block_index=final_block_index,
                )
            )

    output: dict[DescriptorMode, Tensor] = {}
    for mode in selected:
        descriptor = torch.cat(
            [per_region[0][mode], per_region[1][mode]],
            dim=1,
        ).detach().to(device="cpu", dtype=torch.float32)
        expected = DESCRIPTOR_DIMENSIONS[mode]
        if tuple(descriptor.shape) != (1, expected):
            raise ValueError(
                f"{mode} descriptor has shape {tuple(descriptor.shape)}, "
                f"expected [1, {expected}]"
            )
        if not torch.isfinite(descriptor).all():
            raise ValueError(f"{mode} descriptor contains non-finite values")
        output[mode] = descriptor

    return output
