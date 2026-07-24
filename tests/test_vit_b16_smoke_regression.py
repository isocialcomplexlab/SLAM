from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from loop_closure.descriptors.vit_b16 import (
    DEFAULT_REGION_ORDER,
    MODE_TO_ARCHIVE_KEY,
    create_vit_b16_model,
    extract_vit_b16_descriptors,
)

DEFAULT_DATASET = Path(
    "/home/wagner/datasets/KITTI_odometry/extracted/dataset/sequences/05/image_0"
)
DEFAULT_REFERENCE = Path(
    "/mnt/c/Users/Wagner/Downloads/VIT_THREE_FRAME_SEMANTIC_SMOKE_V1/"
    "vit_three_frame_descriptors_v1.npz"
)

DATASET = Path(os.environ.get("KITTI05_IMAGE0_DIR", DEFAULT_DATASET))
REFERENCE = Path(os.environ.get("VIT_SMOKE_NPZ", DEFAULT_REFERENCE))

pytestmark = pytest.mark.skipif(
    not DATASET.is_dir() or not REFERENCE.is_file(),
    reason="KITTI 05 frames or approved ViT smoke NPZ are unavailable",
)


def test_three_frame_descriptors_reproduce_approved_smoke_npz():
    with np.load(REFERENCE, allow_pickle=False) as archive:
        frames = archive["frames"].astype(np.int64)
        expected_order = tuple(str(item) for item in archive["region_order"].tolist())
        expected = {
            mode: archive[key].astype(np.float32)
            for mode, key in MODE_TO_ARCHIVE_KEY.items()
        }

    assert expected_order == DEFAULT_REGION_ORDER
    model = create_vit_b16_model(device="cpu")
    actual = {mode: [] for mode in MODE_TO_ARCHIVE_KEY}

    for frame in frames:
        path = DATASET / f"{int(frame):06d}.png"
        assert path.is_file()
        with Image.open(path) as image:
            descriptors = extract_vit_b16_descriptors(
                image,
                model,
                device="cpu",
            )
        for mode, tensor in descriptors.items():
            actual[mode].append(tensor.squeeze(0).numpy())

    for mode in MODE_TO_ARCHIVE_KEY:
        observed = np.stack(actual[mode]).astype(np.float32, copy=False)
        assert observed.shape == expected[mode].shape
        np.testing.assert_allclose(
            observed,
            expected[mode],
            rtol=1e-5,
            atol=2e-5,
            err_msg=f"smoke regression mismatch for {mode}",
        )
