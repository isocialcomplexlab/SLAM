#!/usr/bin/env python3
"""Extract controlled full-image ResNet-101 descriptors for KITTI odometry 05."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from PIL import Image

from loop_closure.descriptors.resnet101 import (
    DEFAULT_RESNET101_WEIGHTS,
    create_resnet101_model,
)
from loop_closure.descriptors.resnet101_full_image import (
    FULL_IMAGE_DESCRIPTOR_LENGTH,
    PREPROCESSING_MODE,
    extract_resnet101_full_image_descriptor,
)


SEQUENCE_ID = "05"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence-root", type=Path, required=True)
    p.add_argument("--camera-id", default="image_0")
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--end-frame", type=int, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.start_frame < 0:
        raise ValueError("start-frame must be non-negative")
    if args.end_frame < args.start_frame:
        raise ValueError("end-frame must be >= start-frame")
    if not args.sequence_root.is_dir():
        raise FileNotFoundError(args.sequence_root)
    camera_dir = args.sequence_root / args.camera_id
    if not camera_dir.is_dir():
        raise FileNotFoundError(camera_dir)
    if not args.overwrite:
        for path in (args.output, args.report):
            if path.exists():
                raise FileExistsError(
                    f"refusing overwrite without --overwrite: {path}"
                )


def main() -> None:
    args = parse_args()
    validate_args(args)

    camera_dir = args.sequence_root / args.camera_id
    frame_ids = np.arange(
        args.start_frame,
        args.end_frame + 1,
        dtype=np.int64,
    )
    image_paths = [
        camera_dir / f"{int(frame):06d}.png"
        for frame in frame_ids
    ]

    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing KITTI frame: {missing[0]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    total_started = time.perf_counter()
    model_started = time.perf_counter()
    model = create_resnet101_model(
        device=args.device,
        weights=DEFAULT_RESNET101_WEIGHTS,
    )
    model_creation_seconds = time.perf_counter() - model_started

    descriptors: list[np.ndarray] = []
    extraction_started = time.perf_counter()

    for index, path in enumerate(image_paths, start=1):
        with Image.open(path) as image:
            descriptor = extract_resnet101_full_image_descriptor(
                model,
                image,
                device=args.device,
            )

        array = descriptor.numpy().astype(np.float32, copy=False)
        if array.shape != (FULL_IMAGE_DESCRIPTOR_LENGTH,):
            raise RuntimeError(
                f"frame {path.name}: descriptor shape={array.shape}"
            )
        if not np.isfinite(array).all():
            raise RuntimeError(f"frame {path.name}: non-finite descriptor")

        descriptors.append(array.copy())

        if index == 1 or index == len(image_paths) or index % 100 == 0:
            print(
                f"[full_image] {index}/{len(image_paths)} frame={path.stem}",
                flush=True,
            )

    extraction_seconds = time.perf_counter() - extraction_started
    matrix = np.stack(descriptors, axis=0).astype(np.float32, copy=False)

    expected_shape = (len(frame_ids), FULL_IMAGE_DESCRIPTOR_LENGTH)
    if matrix.shape != expected_shape:
        raise RuntimeError(
            f"descriptor matrix shape={matrix.shape}, expected={expected_shape}"
        )
    if not np.isfinite(matrix).all():
        raise RuntimeError("descriptor matrix contains NaN/Inf")

    camera_ids = np.asarray(
        [args.camera_id] * len(frame_ids),
        dtype=np.str_,
    )
    image_path_array = np.asarray(
        [str(path.resolve()) for path in image_paths],
        dtype=np.str_,
    )

    serialization_started = time.perf_counter()
    np.savez_compressed(
        args.output,
        descriptors=matrix,
        camera_ids=camera_ids,
        frame_ids=frame_ids,
        image_paths=image_path_array,
        preprocessing_mode=np.asarray(PREPROCESSING_MODE, dtype=np.str_),
        descriptor_dimension=np.asarray(
            FULL_IMAGE_DESCRIPTOR_LENGTH,
            dtype=np.int64,
        ),
    )
    serialization_seconds = time.perf_counter() - serialization_started
    total_seconds = time.perf_counter() - total_started

    with np.load(args.output, allow_pickle=False) as archive:
        saved = np.asarray(archive["descriptors"])
        saved_frames = np.asarray(archive["frame_ids"])
        saved_cameras = np.asarray(archive["camera_ids"])
        saved_mode = str(np.asarray(archive["preprocessing_mode"]).item())
        saved_dimension = int(
            np.asarray(archive["descriptor_dimension"]).item()
        )

    if not np.array_equal(saved, matrix):
        raise RuntimeError("descriptor NPZ round-trip mismatch")
    if not np.array_equal(saved_frames, frame_ids):
        raise RuntimeError("frame identity round-trip mismatch")
    if not np.array_equal(saved_cameras, camera_ids):
        raise RuntimeError("camera identity round-trip mismatch")
    if saved_mode != PREPROCESSING_MODE:
        raise RuntimeError("preprocessing mode round-trip mismatch")
    if saved_dimension != FULL_IMAGE_DESCRIPTOR_LENGTH:
        raise RuntimeError("descriptor dimension round-trip mismatch")

    report = {
        "status": "passed",
        "validation_status": (
            "R2_M2_FULL_IMAGE_DESCRIPTOR_EXTRACTION_PASSED_V1"
        ),
        "dataset": "KITTI odometry",
        "sequence": SEQUENCE_ID,
        "camera_id": args.camera_id,
        "preprocessing_mode": PREPROCESSING_MODE,
        "descriptor_dimension": FULL_IMAGE_DESCRIPTOR_LENGTH,
        "descriptor_semantics": (
            "single ResNet-101 avgpool output from the complete "
            "256x256 resized normalized RGB image"
        ),
        "forward_passes_per_frame": 1,
        "start_frame": int(frame_ids[0]),
        "end_frame": int(frame_ids[-1]),
        "sample_count": int(len(frame_ids)),
        "sequence_root": str(args.sequence_root.resolve()),
        "model": {
            "architecture": "ResNet-101",
            "weights": str(DEFAULT_RESNET101_WEIGHTS),
            "training": bool(model.training),
            "parameter_count": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
        },
        "output_file": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "validation": {
            "descriptor_shape": list(matrix.shape),
            "descriptor_dtype": str(matrix.dtype),
            "all_finite": bool(np.isfinite(matrix).all()),
            "unique_identity_count": int(
                len(set(zip(camera_ids.tolist(), frame_ids.tolist())))
            ),
            "first_frame_id": int(frame_ids[0]),
            "last_frame_id": int(frame_ids[-1]),
            "minimum": float(matrix.min()),
            "maximum": float(matrix.max()),
            "mean": float(matrix.mean()),
            "standard_deviation": float(matrix.std()),
        },
        "timing_seconds": {
            "model_creation": model_creation_seconds,
            "descriptor_extraction": extraction_seconds,
            "serialization": serialization_seconds,
            "total": total_seconds,
            "seconds_per_frame": extraction_seconds / len(frame_ids),
            "frames_per_second": (
                len(frame_ids) / extraction_seconds
                if extraction_seconds > 0
                else 0.0
            ),
        },
        "peak_process_rss_kib": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": PIL.__version__,
            "numpy": np.__version__,
        },
    }

    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("FULL-IMAGE DESCRIPTOR EXTRACTION COMPLETE")
    print(f"shape={matrix.shape}")
    print(f"dtype={matrix.dtype}")
    print(f"output_sha256={report['output_sha256']}")


if __name__ == "__main__":
    main()
