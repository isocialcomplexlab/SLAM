#!/usr/bin/env python3

"""Extração reproduzível de descritores ResNet-101 para o KITTI 05."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import torch
import torchvision

from loop_closure.descriptors.resnet101 import (
    DEFAULT_RESNET101_WEIGHTS,
    create_resnet101_model,
)
from loop_closure.descriptors.resnet101_batch import (
    discover_kitti_image_samples,
    extract_resnet101_descriptor_batch,
    save_resnet101_descriptor_batch,
)


SEQUENCE_ID = "05"
EXPECTED_DESCRIPTOR_LENGTH = 4096


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extrai descritores ResNet-101 de um intervalo explícito "
            "de frames do KITTI 05."
        )
    )

    parser.add_argument(
        "--sequence-root",
        type=Path,
        default=None,
        help=(
            "Diretório da sequência 05. Quando omitido, usa "
            "KITTI_ODOMETRY_ROOT."
        ),
    )
    parser.add_argument(
        "--camera-id",
        default="image_0",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        required=True,
        help="Frame final inclusivo.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def resolve_sequence_root(
    explicit_root: Path | None,
) -> Path:
    if explicit_root is not None:
        resolved = explicit_root.expanduser().resolve()

        if not resolved.is_dir():
            raise FileNotFoundError(
                f"Diretório da sequência não encontrado: {resolved}"
            )

        return resolved

    root_value = os.environ.get("KITTI_ODOMETRY_ROOT")

    if not root_value:
        raise EnvironmentError(
            "KITTI_ODOMETRY_ROOT não está definido."
        )

    dataset_root = Path(root_value).expanduser().resolve()

    candidates = [
        dataset_root / "sequences" / SEQUENCE_ID,
        dataset_root / "dataset" / "sequences" / SEQUENCE_ID,
        dataset_root / SEQUENCE_ID,
    ]

    for candidate in candidates:
        candidate = candidate.resolve()

        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Não foi possível localizar a sequência KITTI 05."
    )


def validate_arguments(args: argparse.Namespace) -> None:
    if args.start_frame < 0:
        raise ValueError(
            "start-frame não pode ser negativo."
        )

    if args.end_frame < args.start_frame:
        raise ValueError(
            "end-frame deve ser maior ou igual a start-frame."
        )


def validate_saved_archive(
    output_path: Path,
    *,
    expected_count: int,
    camera_id: str,
    start_frame: int,
    end_frame: int,
) -> dict[str, Any]:
    with np.load(
        output_path,
        allow_pickle=False,
    ) as archive:
        expected_names = {
            "descriptors",
            "camera_ids",
            "frame_ids",
            "image_paths",
        }

        if set(archive.files) != expected_names:
            raise AssertionError(
                f"Campos inesperados no NPZ: {archive.files}"
            )

        descriptors = archive["descriptors"]
        camera_ids = archive["camera_ids"]
        frame_ids = archive["frame_ids"]
        image_paths = archive["image_paths"]

        if descriptors.shape != (
            expected_count,
            EXPECTED_DESCRIPTOR_LENGTH,
        ):
            raise AssertionError(
                f"Forma inesperada: {descriptors.shape}"
            )

        if descriptors.dtype != np.float32:
            raise AssertionError(
                f"Dtype inesperado: {descriptors.dtype}"
            )

        if not np.isfinite(descriptors).all():
            raise AssertionError(
                "O arquivo contém descritores não finitos."
            )

        expected_frames = np.arange(
            start_frame,
            end_frame + 1,
            dtype=np.int64,
        )

        if not np.array_equal(
            frame_ids,
            expected_frames,
        ):
            raise AssertionError(
                "Os frame_ids salvos não correspondem "
                "ao intervalo solicitado."
            )

        if not np.all(camera_ids == camera_id):
            raise AssertionError(
                "O arquivo contém camera_id inesperado."
            )

        identities = {
            (str(camera), int(frame))
            for camera, frame in zip(
                camera_ids,
                frame_ids,
                strict=True,
            )
        }

        if len(identities) != expected_count:
            raise AssertionError(
                "Foram encontradas identidades duplicadas."
            )

        for name in archive.files:
            if archive[name].dtype.kind == "O":
                raise AssertionError(
                    f"{name} possui dtype object."
                )

        return {
            "descriptor_shape": list(descriptors.shape),
            "descriptor_dtype": str(descriptors.dtype),
            "all_finite": bool(np.isfinite(descriptors).all()),
            "minimum": float(descriptors.min()),
            "maximum": float(descriptors.max()),
            "mean": float(descriptors.mean()),
            "standard_deviation": float(descriptors.std()),
            "unique_identity_count": len(identities),
            "first_frame_id": int(frame_ids[0]),
            "last_frame_id": int(frame_ids[-1]),
            "first_image_path": str(image_paths[0]),
            "last_image_path": str(image_paths[-1]),
        }


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)

    sequence_root = resolve_sequence_root(
        args.sequence_root
    )

    all_samples = discover_kitti_image_samples(
        sequence_root,
        camera_ids=(args.camera_id,),
    )

    selected_samples = [
        sample
        for sample in all_samples
        if (
            args.start_frame
            <= sample.frame_id
            <= args.end_frame
        )
    ]

    expected_count = (
        args.end_frame
        - args.start_frame
        + 1
    )

    if len(selected_samples) != expected_count:
        raise AssertionError(
            "A quantidade de imagens selecionadas não corresponde "
            f"ao intervalo solicitado: esperado={expected_count}, "
            f"observado={len(selected_samples)}."
        )

    expected_frame_ids = list(
        range(
            args.start_frame,
            args.end_frame + 1,
        )
    )
    observed_frame_ids = [
        sample.frame_id
        for sample in selected_samples
    ]

    if observed_frame_ids != expected_frame_ids:
        raise AssertionError(
            "Os frames selecionados não formam o intervalo "
            "contíguo solicitado."
        )

    print("=" * 79)
    print("EXTRAÇÃO RESNET-101 — KITTI 05")
    print("=" * 79)
    print(f"Sequência       : {sequence_root}")
    print(f"Câmera          : {args.camera_id}")
    print(
        f"Frames          : "
        f"{args.start_frame}..{args.end_frame}"
    )
    print(f"Amostras        : {len(selected_samples)}")
    print(f"Dispositivo     : {args.device}")
    print(f"Pesos           : {DEFAULT_RESNET101_WEIGHTS}")
    print(f"Arquivo de saída: {args.output}")
    print()

    total_start = time.perf_counter()

    model_start = time.perf_counter()
    model = create_resnet101_model(
        device=args.device,
        weights=DEFAULT_RESNET101_WEIGHTS,
    )
    model_seconds = time.perf_counter() - model_start

    extraction_start = time.perf_counter()
    batch = extract_resnet101_descriptor_batch(
        model=model,
        samples=selected_samples,
        device=args.device,
    )
    extraction_seconds = (
        time.perf_counter()
        - extraction_start
    )

    save_start = time.perf_counter()
    output_path = save_resnet101_descriptor_batch(
        batch,
        args.output,
        overwrite=args.overwrite,
    )
    save_seconds = time.perf_counter() - save_start

    validation = validate_saved_archive(
        output_path,
        expected_count=expected_count,
        camera_id=args.camera_id,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )

    total_seconds = time.perf_counter() - total_start

    file_sha256 = hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()

    peak_rss_kib = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    seconds_per_frame = (
        extraction_seconds / expected_count
    )
    frames_per_second = (
        expected_count / extraction_seconds
        if extraction_seconds > 0
        else None
    )

    report = {
        "status": "passed",
        "dataset": "KITTI odometry",
        "sequence": SEQUENCE_ID,
        "sequence_root": str(sequence_root),
        "camera_id": args.camera_id,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "sample_count": expected_count,
        "weights": str(DEFAULT_RESNET101_WEIGHTS),
        "device": args.device,
        "output_file": str(output_path),
        "output_sha256": file_sha256,
        "validation": validation,
        "timing_seconds": {
            "model_creation": model_seconds,
            "descriptor_extraction": extraction_seconds,
            "serialization": save_seconds,
            "total": total_seconds,
            "seconds_per_frame": seconds_per_frame,
            "frames_per_second": frames_per_second,
        },
        "peak_process_rss_kib": peak_rss_kib,
        "model": {
            "training": model.training,
            "parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": PIL.__version__,
            "numpy": np.__version__,
        },
    }

    args.report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.report.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("RESULTADO")
    print(f"- status                 : {report['status']}")
    print(
        f"- shape                  : "
        f"{validation['descriptor_shape']}"
    )
    print(
        f"- valores finitos        : "
        f"{validation['all_finite']}"
    )
    print(
        f"- identidades únicas     : "
        f"{validation['unique_identity_count']}"
    )
    print(
        f"- primeiro frame         : "
        f"{validation['first_frame_id']}"
    )
    print(
        f"- último frame           : "
        f"{validation['last_frame_id']}"
    )
    print(
        f"- tempo de extração      : "
        f"{extraction_seconds:.6f} s"
    )
    print(
        f"- segundos por frame     : "
        f"{seconds_per_frame:.6f}"
    )
    print(f"- pico RSS               : {peak_rss_kib} KiB")
    print(f"- SHA-256 do NPZ         : {file_sha256}")
    print()
    print(f"Descritores: {output_path}")
    print(f"Relatório  : {args.report}")
    print("=" * 79)


if __name__ == "__main__":
    main()
