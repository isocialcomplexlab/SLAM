from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from loop_closure.descriptors.resnet101_batch import (
    KittiImageSample,
    ResNet101DescriptorBatch,
    discover_kitti_image_samples,
    extract_resnet101_descriptor_batch,
    save_resnet101_descriptor_batch,
)


DESCRIPTOR_LENGTH = 4096


def save_test_image(
    path: Path,
    *,
    value: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    Image.new(
        mode="RGB",
        size=(16, 8),
        color=(value, value, value),
    ).save(path)


def create_camera(
    sequence_root: Path,
    camera_id: str,
    frame_ids: list[int],
) -> None:
    for frame_id in frame_ids:
        save_test_image(
            sequence_root
            / camera_id
            / f"{frame_id:06d}.png",
            value=frame_id,
        )


def test_discovery_returns_explicit_camera_and_frame_identity(
    tmp_path: Path,
) -> None:
    sequence_root = tmp_path / "05"

    create_camera(
        sequence_root,
        "image_0",
        [0, 1, 2],
    )

    samples = discover_kitti_image_samples(
        sequence_root,
        camera_ids=("image_0",),
    )

    assert samples == [
        KittiImageSample(
            camera_id="image_0",
            frame_id=0,
            image_path=(
                sequence_root
                / "image_0"
                / "000000.png"
            ).resolve(),
        ),
        KittiImageSample(
            camera_id="image_0",
            frame_id=1,
            image_path=(
                sequence_root
                / "image_0"
                / "000001.png"
            ).resolve(),
        ),
        KittiImageSample(
            camera_id="image_0",
            frame_id=2,
            image_path=(
                sequence_root
                / "image_0"
                / "000002.png"
            ).resolve(),
        ),
    ]


def test_discovery_preserves_requested_camera_order(
    tmp_path: Path,
) -> None:
    sequence_root = tmp_path / "05"

    create_camera(sequence_root, "image_0", [0, 1])
    create_camera(sequence_root, "image_1", [0, 1])

    samples = discover_kitti_image_samples(
        sequence_root,
        camera_ids=("image_1", "image_0"),
    )

    observed = [
        (sample.camera_id, sample.frame_id)
        for sample in samples
    ]

    assert observed == [
        ("image_1", 0),
        ("image_1", 1),
        ("image_0", 0),
        ("image_0", 1),
    ]


def test_discovery_sorts_frames_numerically(
    tmp_path: Path,
) -> None:
    sequence_root = tmp_path / "05"

    create_camera(
        sequence_root,
        "image_0",
        [10, 2, 1],
    )

    samples = discover_kitti_image_samples(
        sequence_root,
        camera_ids=("image_0",),
        require_contiguous=False,
    )

    assert [
        sample.frame_id
        for sample in samples
    ] == [1, 2, 10]


def test_discovery_rejects_missing_camera_directory(
    tmp_path: Path,
) -> None:
    sequence_root = tmp_path / "05"
    sequence_root.mkdir()

    with pytest.raises(
        FileNotFoundError,
        match="image_0",
    ):
        discover_kitti_image_samples(
            sequence_root,
            camera_ids=("image_0",),
        )


def test_discovery_rejects_non_numeric_image_filename(
    tmp_path: Path,
) -> None:
    sequence_root = tmp_path / "05"

    save_test_image(
        sequence_root / "image_0" / "frame.png",
        value=0,
    )

    with pytest.raises(
        ValueError,
        match="frame",
    ):
        discover_kitti_image_samples(
            sequence_root,
            camera_ids=("image_0",),
        )


def test_discovery_rejects_duplicate_frame_ids(
    tmp_path: Path,
) -> None:
    sequence_root = tmp_path / "05"

    save_test_image(
        sequence_root / "image_0" / "000000.png",
        value=0,
    )
    save_test_image(
        sequence_root / "image_0" / "000000.jpg",
        value=0,
    )

    with pytest.raises(
        ValueError,
        match="duplicado",
    ):
        discover_kitti_image_samples(
            sequence_root,
            camera_ids=("image_0",),
        )


def test_discovery_rejects_missing_frame_in_contiguous_camera(
    tmp_path: Path,
) -> None:
    sequence_root = tmp_path / "05"

    create_camera(
        sequence_root,
        "image_0",
        [0, 2],
    )

    with pytest.raises(
        ValueError,
        match="contígua",
    ):
        discover_kitti_image_samples(
            sequence_root,
            camera_ids=("image_0",),
        )


def test_batch_processes_each_sample_exactly_once(
    tmp_path: Path,
) -> None:
    sequence_root = tmp_path / "05"

    create_camera(
        sequence_root,
        "image_0",
        [0, 1, 2],
    )

    samples = discover_kitti_image_samples(
        sequence_root,
        camera_ids=("image_0",),
    )

    calls: list[int] = []

    def fake_extractor(
        model: object,
        image: Image.Image,
        *,
        device: str,
    ) -> torch.Tensor:
        del model
        del device

        pixel_value = int(image.getpixel((0, 0))[0])
        calls.append(pixel_value)

        return torch.full(
            (DESCRIPTOR_LENGTH,),
            float(pixel_value),
            dtype=torch.float32,
        )

    result = extract_resnet101_descriptor_batch(
        model=object(),
        samples=samples,
        extractor=fake_extractor,
        device="cpu",
    )

    assert calls == [0, 1, 2]
    assert len(calls) == len(set(calls))

    assert result.descriptors.shape == (
        3,
        DESCRIPTOR_LENGTH,
    )
    assert result.descriptors.dtype == np.float32

    assert result.camera_ids.tolist() == [
        "image_0",
        "image_0",
        "image_0",
    ]
    assert result.frame_ids.tolist() == [0, 1, 2]

    assert np.all(result.descriptors[0] == 0.0)
    assert np.all(result.descriptors[1] == 1.0)
    assert np.all(result.descriptors[2] == 2.0)


def test_batch_rejects_duplicate_camera_frame_identity(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "000000.png"
    save_test_image(image_path, value=0)

    duplicate_samples = [
        KittiImageSample(
            camera_id="image_0",
            frame_id=0,
            image_path=image_path.resolve(),
        ),
        KittiImageSample(
            camera_id="image_0",
            frame_id=0,
            image_path=image_path.resolve(),
        ),
    ]

    with pytest.raises(
        ValueError,
        match="duplicada",
    ):
        extract_resnet101_descriptor_batch(
            model=object(),
            samples=duplicate_samples,
            extractor=lambda model, image, device: torch.zeros(
                DESCRIPTOR_LENGTH,
                dtype=torch.float32,
            ),
            device="cpu",
        )


def test_batch_rejects_invalid_descriptor_shape(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "000000.png"
    save_test_image(image_path, value=0)

    samples = [
        KittiImageSample(
            camera_id="image_0",
            frame_id=0,
            image_path=image_path.resolve(),
        )
    ]

    with pytest.raises(
        ValueError,
        match="4096",
    ):
        extract_resnet101_descriptor_batch(
            model=object(),
            samples=samples,
            extractor=lambda model, image, device: torch.zeros(
                2048,
                dtype=torch.float32,
            ),
            device="cpu",
        )


def test_batch_rejects_non_finite_descriptor(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "000000.png"
    save_test_image(image_path, value=0)

    samples = [
        KittiImageSample(
            camera_id="image_0",
            frame_id=0,
            image_path=image_path.resolve(),
        )
    ]

    invalid = torch.zeros(
        DESCRIPTOR_LENGTH,
        dtype=torch.float32,
    )
    invalid[10] = float("nan")

    with pytest.raises(
        ValueError,
        match="NaN|infinito|finit",
    ):
        extract_resnet101_descriptor_batch(
            model=object(),
            samples=samples,
            extractor=lambda model, image, device: invalid,
            device="cpu",
        )


def test_npz_round_trip_uses_no_pickle_or_object_arrays(
    tmp_path: Path,
) -> None:
    result = ResNet101DescriptorBatch(
        descriptors=np.arange(
            2 * DESCRIPTOR_LENGTH,
            dtype=np.float32,
        ).reshape(2, DESCRIPTOR_LENGTH),
        camera_ids=np.asarray(
            ["image_0", "image_0"],
            dtype=np.str_,
        ),
        frame_ids=np.asarray(
            [10, 11],
            dtype=np.int64,
        ),
        image_paths=np.asarray(
            [
                "/dataset/05/image_0/000010.png",
                "/dataset/05/image_0/000011.png",
            ],
            dtype=np.str_,
        ),
    )

    output_path = tmp_path / "descriptors.npz"

    save_resnet101_descriptor_batch(
        result,
        output_path,
    )

    assert output_path.is_file()

    with np.load(
        output_path,
        allow_pickle=False,
    ) as archive:
        assert set(archive.files) == {
            "descriptors",
            "camera_ids",
            "frame_ids",
            "image_paths",
        }

        assert archive["descriptors"].shape == (
            2,
            DESCRIPTOR_LENGTH,
        )
        assert archive["descriptors"].dtype == np.float32
        assert archive["frame_ids"].dtype == np.int64

        for name in archive.files:
            assert archive[name].dtype.kind != "O"

        assert archive["camera_ids"].tolist() == [
            "image_0",
            "image_0",
        ]
        assert archive["frame_ids"].tolist() == [10, 11]


def test_save_refuses_accidental_overwrite(
    tmp_path: Path,
) -> None:
    result = ResNet101DescriptorBatch(
        descriptors=np.zeros(
            (1, DESCRIPTOR_LENGTH),
            dtype=np.float32,
        ),
        camera_ids=np.asarray(
            ["image_0"],
            dtype=np.str_,
        ),
        frame_ids=np.asarray(
            [0],
            dtype=np.int64,
        ),
        image_paths=np.asarray(
            ["/dataset/05/image_0/000000.png"],
            dtype=np.str_,
        ),
    )

    output_path = tmp_path / "descriptors.npz"

    save_resnet101_descriptor_batch(
        result,
        output_path,
    )

    with pytest.raises(FileExistsError):
        save_resnet101_descriptor_batch(
            result,
            output_path,
        )
