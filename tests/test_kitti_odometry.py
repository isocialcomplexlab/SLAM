from pathlib import Path

import pytest

from loop_closure.datasets.kitti_odometry import validate_sequence


def create_fake_sequence(
    root: Path,
    *,
    image_count: int = 3,
    pose_count: int = 3,
    timestamp_count: int = 3,
) -> None:
    image_directory = root / "sequences" / "05" / "image_0"
    image_directory.mkdir(parents=True)

    poses_directory = root / "poses"
    poses_directory.mkdir(parents=True)

    for frame_id in range(image_count):
        (image_directory / f"{frame_id:06d}.png").write_bytes(b"test")

    pose_line = "1 0 0 0 0 1 0 0 0 0 1 0\n"
    (poses_directory / "05.txt").write_text(
        pose_line * pose_count,
        encoding="utf-8",
    )

    sequence_directory = root / "sequences" / "05"
    timestamps = "".join(
        f"{frame_id * 0.1:.6f}\n"
        for frame_id in range(timestamp_count)
    )
    (sequence_directory / "times.txt").write_text(
        timestamps,
        encoding="utf-8",
    )
    (sequence_directory / "calib.txt").write_text(
        "P0: 1 0 0 0 0 1 0 0 0 0 1 0\n",
        encoding="utf-8",
    )


def test_validate_sequence_accepts_aligned_files(tmp_path: Path) -> None:
    create_fake_sequence(tmp_path)

    summary = validate_sequence(tmp_path, "05", "image_0")

    assert summary.image_count == 3
    assert summary.pose_count == 3
    assert summary.timestamp_count == 3
    assert summary.first_frame == "000000.png"
    assert summary.last_frame == "000002.png"


def test_validate_sequence_rejects_count_mismatch(
    tmp_path: Path,
) -> None:
    create_fake_sequence(tmp_path, image_count=3, pose_count=2)

    with pytest.raises(ValueError, match="counts do not match"):
        validate_sequence(tmp_path, "05", "image_0")
