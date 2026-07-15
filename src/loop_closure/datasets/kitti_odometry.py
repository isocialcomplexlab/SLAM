"""Utilities for validating and reading KITTI odometry sequences."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class KittiSequenceSummary:
    """Validated KITTI sequence metadata."""

    root: str
    sequence: str
    camera: str
    image_directory: str
    poses_file: str
    times_file: str
    calibration_file: str
    image_count: int
    pose_count: int
    timestamp_count: int
    first_frame: str
    last_frame: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def discover_images(image_directory: Path) -> list[Path]:
    """Return sequence images in deterministic frame order."""

    if not image_directory.is_dir():
        raise FileNotFoundError(
            f"KITTI image directory not found: {image_directory}"
        )

    images = sorted(
        path
        for path in image_directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )

    if not images:
        raise ValueError(f"No images found in: {image_directory}")

    frame_ids: list[int] = []

    for path in images:
        try:
            frame_ids.append(int(path.stem))
        except ValueError as exc:
            raise ValueError(
                f"Image filename is not a numeric KITTI frame: {path.name}"
            ) from exc

    expected = list(range(len(images)))

    if frame_ids != expected:
        raise ValueError(
            "KITTI frame numbering is not contiguous from zero. "
            f"First IDs: {frame_ids[:5]}; last IDs: {frame_ids[-5:]}"
        )

    return images


def load_poses(poses_file: Path) -> np.ndarray:
    """Load KITTI poses as an array with shape (N, 3, 4)."""

    if not poses_file.is_file():
        raise FileNotFoundError(f"KITTI poses file not found: {poses_file}")

    values = np.loadtxt(poses_file, dtype=np.float64)
    values = np.atleast_2d(values)

    if values.ndim != 2 or values.shape[1] != 12:
        raise ValueError(
            "Each KITTI pose must contain exactly 12 numeric values. "
            f"Observed shape: {values.shape}"
        )

    if not np.isfinite(values).all():
        raise ValueError("KITTI poses contain non-finite values.")

    return values.reshape(-1, 3, 4)


def load_timestamps(times_file: Path) -> np.ndarray:
    """Load KITTI frame timestamps."""

    if not times_file.is_file():
        raise FileNotFoundError(
            f"KITTI timestamps file not found: {times_file}"
        )

    timestamps = np.loadtxt(times_file, dtype=np.float64)
    timestamps = np.atleast_1d(timestamps)

    if not np.isfinite(timestamps).all():
        raise ValueError("KITTI timestamps contain non-finite values.")

    if timestamps.size > 1 and np.any(np.diff(timestamps) < 0):
        raise ValueError("KITTI timestamps are not monotonically increasing.")

    return timestamps


def validate_sequence(
    root: str | Path,
    sequence: str = "05",
    camera: str = "image_0",
) -> KittiSequenceSummary:
    """Validate the files and alignment of one KITTI odometry sequence."""

    dataset_root = Path(root).expanduser().resolve()
    sequence = str(sequence).zfill(2)

    sequence_directory = dataset_root / "sequences" / sequence
    image_directory = sequence_directory / camera
    poses_file = dataset_root / "poses" / f"{sequence}.txt"
    times_file = sequence_directory / "times.txt"
    calibration_file = sequence_directory / "calib.txt"

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"KITTI odometry root not found: {dataset_root}"
        )

    if not calibration_file.is_file():
        raise FileNotFoundError(
            f"KITTI calibration file not found: {calibration_file}"
        )

    images = discover_images(image_directory)
    poses = load_poses(poses_file)
    timestamps = load_timestamps(times_file)

    counts = {
        "images": len(images),
        "poses": int(poses.shape[0]),
        "timestamps": int(timestamps.shape[0]),
    }

    if len(set(counts.values())) != 1:
        raise ValueError(
            "KITTI image, pose, and timestamp counts do not match: "
            f"{counts}"
        )

    return KittiSequenceSummary(
        root=str(dataset_root),
        sequence=sequence,
        camera=camera,
        image_directory=str(image_directory),
        poses_file=str(poses_file),
        times_file=str(times_file),
        calibration_file=str(calibration_file),
        image_count=len(images),
        pose_count=int(poses.shape[0]),
        timestamp_count=int(timestamps.shape[0]),
        first_frame=images[0].name,
        last_frame=images[-1].name,
    )
