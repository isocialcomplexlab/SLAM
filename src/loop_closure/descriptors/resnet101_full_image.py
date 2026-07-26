"""Controlled full-image ResNet-101 descriptor arm for preprocessing ablation.

This module is deliberately additive. The historical split-left/right
pipeline in :mod:`loop_closure.descriptors.resnet101` is not modified.

Controlled arm:
- convert PIL input to RGB;
- resize the whole image to 256 x 256;
- normalize all RGB channels with mean=0.36 and std=0.28;
- execute the existing ResNet-101 avgpool path once;
- flatten the single 2048-channel avgpool output.
"""

from __future__ import annotations

from typing import Final

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms import functional as TF

from loop_closure.descriptors.resnet101 import forward_resnet101_avgpool


FULL_IMAGE_SIZE: Final[tuple[int, int]] = (256, 256)
NORMALIZATION_MEAN: Final[tuple[float, float, float]] = (0.36, 0.36, 0.36)
NORMALIZATION_STD: Final[tuple[float, float, float]] = (0.28, 0.28, 0.28)
FULL_IMAGE_DESCRIPTOR_LENGTH: Final[int] = 2048
PREPROCESSING_MODE: Final[str] = "full_image"


def full_image_resnet101_tensor(image: Image.Image) -> Tensor:
    """Return the normalized whole-image tensor with shape ``[3, 256, 256]``."""
    if not isinstance(image, Image.Image):
        raise TypeError(
            "image must be a PIL.Image.Image; "
            f"received {type(image).__name__}"
        )

    rgb = image.convert("RGB")
    resized = TF.resize(rgb, list(FULL_IMAGE_SIZE))
    tensor = TF.pil_to_tensor(resized).to(dtype=torch.float32).div_(255.0)
    normalized = TF.normalize(
        tensor,
        mean=list(NORMALIZATION_MEAN),
        std=list(NORMALIZATION_STD),
    )

    if tuple(normalized.shape) != (3, 256, 256):
        raise RuntimeError(
            "full-image preprocessing produced an unexpected shape: "
            f"{tuple(normalized.shape)}"
        )
    if normalized.dtype != torch.float32:
        raise RuntimeError("full-image preprocessing must produce torch.float32")
    if not bool(torch.isfinite(normalized).all()):
        raise RuntimeError(
            "full-image preprocessing produced NaN or infinite values"
        )

    return normalized.contiguous()


@torch.inference_mode()
def extract_resnet101_full_image_descriptor(
    model: nn.Module,
    image: Image.Image,
    *,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Extract one 2048-D avgpool descriptor from the complete resized image."""
    if not isinstance(model, nn.Module):
        raise TypeError(
            "model must be a torch.nn.Module; "
            f"received {type(model).__name__}"
        )

    model.eval()
    tensor = full_image_resnet101_tensor(image)
    batch = tensor.unsqueeze(0).to(device=device)

    avgpool = forward_resnet101_avgpool(model, batch)
    descriptor = (
        avgpool.detach()
        .reshape(-1)
        .cpu()
        .to(dtype=torch.float32)
        .contiguous()
    )

    if descriptor.numel() != FULL_IMAGE_DESCRIPTOR_LENGTH:
        raise RuntimeError(
            "full-image avgpool descriptor must contain exactly "
            f"{FULL_IMAGE_DESCRIPTOR_LENGTH} values; "
            f"received {descriptor.numel()}"
        )
    if descriptor.ndim != 1:
        raise RuntimeError(
            f"descriptor must be one-dimensional; shape={tuple(descriptor.shape)}"
        )
    if not bool(torch.isfinite(descriptor).all()):
        raise RuntimeError("full-image descriptor contains NaN or infinite values")
    if descriptor.requires_grad:
        raise RuntimeError("full-image descriptor must be detached")

    return descriptor
