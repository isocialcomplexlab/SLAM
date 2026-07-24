#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from loop_closure.descriptors.vit_b16 import (
    DEFAULT_REGION_ORDER,
    DEFAULT_VIT_B16_WEIGHTS,
    MODE_TO_ARCHIVE_KEY,
    create_vit_b16_model,
    extract_vit_b16_descriptors,
)

ALL_MODES = tuple(MODE_TO_ARCHIVE_KEY)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_modes(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return ALL_MODES
    modes = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = [item for item in modes if item not in ALL_MODES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown descriptor modes: {unknown}")
    if not modes:
        raise argparse.ArgumentTypeError("at least one mode is required")
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--camera-id", default="image_0")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=2760)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--modes", type=parse_modes, default=ALL_MODES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> list[tuple[int, Path]]:
    camera_dir = args.sequence_root / args.camera_id
    if not camera_dir.is_dir():
        raise FileNotFoundError(f"camera directory absent: {camera_dir}")
    if args.start_frame < 0 or args.end_frame < args.start_frame:
        raise ValueError("invalid frame range")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")
    if args.report.exists() and not args.overwrite:
        raise FileExistsError(f"report already exists: {args.report}")

    samples = []
    for frame in range(args.start_frame, args.end_frame + 1):
        path = camera_dir / f"{frame:06d}.png"
        if not path.is_file():
            raise FileNotFoundError(f"frame absent: {path}")
        samples.append((frame, path))
    return samples


def main() -> int:
    args = parse_args()
    samples = validate_args(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    model = create_vit_b16_model(device=args.device)
    storage: dict[str, list[np.ndarray]] = {mode: [] for mode in args.modes}
    started = time.perf_counter()

    for index, (frame, path) in enumerate(samples, start=1):
        with Image.open(path) as image:
            descriptors = extract_vit_b16_descriptors(
                image,
                model,
                device=args.device,
                modes=args.modes,
            )
        for mode, tensor in descriptors.items():
            storage[mode].append(tensor.squeeze(0).numpy())
        if index == 1 or index % 100 == 0 or index == len(samples):
            print(f"processed {index}/{len(samples)} frame={frame:06d}")

    elapsed = time.perf_counter() - started
    arrays: dict[str, np.ndarray] = {
        "frames": np.asarray([frame for frame, _ in samples], dtype=np.int64),
        "camera_ids": np.asarray([args.camera_id] * len(samples)),
        "region_order": np.asarray(DEFAULT_REGION_ORDER),
    }
    for mode in args.modes:
        arrays[MODE_TO_ARCHIVE_KEY[mode]] = np.stack(storage[mode]).astype(
            np.float32, copy=False
        )

    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temp, args.output)

    report = {
        "status": "passed",
        "model": "vit_b_16",
        "weights": str(DEFAULT_VIT_B16_WEIGHTS),
        "device": args.device,
        "sequence_root": str(args.sequence_root.resolve()),
        "camera_id": args.camera_id,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "frame_count": len(samples),
        "modes": list(args.modes),
        "region_order": list(DEFAULT_REGION_ORDER),
        "elapsed_seconds": elapsed,
        "seconds_per_frame": elapsed / len(samples),
        "frames_per_second": len(samples) / elapsed,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
    }
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
