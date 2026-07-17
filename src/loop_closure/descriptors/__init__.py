"""Descritores e funções de comparação para detecção de loop closure."""

from loop_closure.descriptors.corrected_resnet import (
    corrected_l2_distance,
    corrected_l2_match,
)
from loop_closure.descriptors.legacy_resnet import (
    legacy_belief_generation,
    legacy_exact_distance,
    legacy_exact_match,
)
from loop_closure.descriptors.resnet101_batch import (
    KittiImageSample,
    ResNet101DescriptorBatch,
    discover_kitti_image_samples,
    extract_resnet101_descriptor_batch,
    save_resnet101_descriptor_batch,
)
from loop_closure.descriptors.resnet101 import (
    DEFAULT_RESNET101_WEIGHTS,
    concatenate_right_left_avgpool,
    create_resnet101_model,
    extract_resnet101_descriptor,
    forward_resnet101_avgpool,
    legacy_resnet101_crops,
)

__all__ = [
    "save_resnet101_descriptor_batch",
    "extract_resnet101_descriptor_batch",
    "discover_kitti_image_samples",
    "ResNet101DescriptorBatch",
    "KittiImageSample",
    "DEFAULT_RESNET101_WEIGHTS",
    "concatenate_right_left_avgpool",
    "corrected_l2_distance",
    "corrected_l2_match",
    "create_resnet101_model",
    "extract_resnet101_descriptor",
    "forward_resnet101_avgpool",
    "legacy_belief_generation",
    "legacy_exact_distance",
    "legacy_exact_match",
    "legacy_resnet101_crops",
]
