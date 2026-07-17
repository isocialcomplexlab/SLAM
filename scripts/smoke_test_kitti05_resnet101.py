#!/usr/bin/env python3

"""Smoke test real da ResNet-101 em um único frame do KITTI 05."""

from __future__ import annotations

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
from PIL import Image

from loop_closure.descriptors.resnet101 import (
    DEFAULT_RESNET101_WEIGHTS,
    create_resnet101_model,
    extract_resnet101_descriptor,
)


SEQUENCE = "05"
CAMERA_ID = "image_0"
FRAME_ID = 0
EXPECTED_DESCRIPTOR_LENGTH = 4096


def resolve_image_path(dataset_root: Path) -> Path:
    filename = f"{FRAME_ID:06d}.png"

    candidates = [
        dataset_root / "sequences" / SEQUENCE / CAMERA_ID / filename,
        dataset_root / "dataset" / "sequences" / SEQUENCE / CAMERA_ID / filename,
        dataset_root / SEQUENCE / CAMERA_ID / filename,
    ]

    checked: list[str] = []

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        checked.append(str(candidate))

        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Não foi possível localizar o frame do KITTI 05:\n"
        + "\n".join(f"  - {path}" for path in checked)
    )


def tensor_statistics(descriptor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(descriptor.shape),
        "numel": descriptor.numel(),
        "dtype": str(descriptor.dtype),
        "device": str(descriptor.device),
        "requires_grad": descriptor.requires_grad,
        "is_contiguous": descriptor.is_contiguous(),
        "all_finite": bool(torch.isfinite(descriptor).all()),
        "minimum": float(descriptor.min().item()),
        "maximum": float(descriptor.max().item()),
        "mean": float(descriptor.mean().item()),
        "standard_deviation": float(
            descriptor.std(unbiased=False).item()
        ),
        "l2_norm": float(
            torch.linalg.vector_norm(descriptor, ord=2).item()
        ),
    }


def main() -> None:
    root_value = os.environ.get("KITTI_ODOMETRY_ROOT")

    if not root_value:
        raise EnvironmentError(
            "KITTI_ODOMETRY_ROOT não está definido. Verifique .env.local."
        )

    dataset_root = Path(root_value).expanduser().resolve()
    image_path = resolve_image_path(dataset_root)

    output_directory = (
        Path("results")
        / "pilot"
        / "kitti05_resnet"
        / "smoke"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    descriptor_path = (
        output_directory
        / "kitti05_image0_frame000000_resnet101_imagenet1k_v1.npy"
    )
    report_path = (
        output_directory
        / "kitti05_image0_frame000000_resnet101_smoke.json"
    )

    print("=" * 79)
    print("SMOKE TEST REAL — KITTI 05 / RESNET-101")
    print("=" * 79)
    print(f"Imagem      : {image_path}")
    print(f"Dispositivo : cpu")
    print(f"Pesos       : {DEFAULT_RESNET101_WEIGHTS}")
    print()

    process_start = time.perf_counter()

    model_start = time.perf_counter()

    try:
        model = create_resnet101_model(
            device="cpu",
            weights=DEFAULT_RESNET101_WEIGHTS,
        )
    except Exception as error:
        print()
        print("Falha ao criar a ResNet-101 com os pesos pré-treinados.")
        print(
            "Caso os pesos ainda não estejam em cache, verifique a conexão "
            "necessária para o primeiro download."
        )
        print(f"{type(error).__name__}: {error}")
        raise

    model_seconds = time.perf_counter() - model_start

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    with Image.open(image_path) as opened_image:
        image = opened_image.convert("RGB")
        original_size = list(opened_image.size)
        converted_size = list(image.size)

        extraction_start = time.perf_counter()

        descriptor = extract_resnet101_descriptor(
            model,
            image,
            device="cpu",
        )

        extraction_seconds = time.perf_counter() - extraction_start

    total_seconds = time.perf_counter() - process_start

    statistics = tensor_statistics(descriptor)

    if descriptor.shape != (EXPECTED_DESCRIPTOR_LENGTH,):
        raise AssertionError(
            "Forma inesperada do descritor: "
            f"{tuple(descriptor.shape)}."
        )

    if descriptor.dtype != torch.float32:
        raise AssertionError(
            f"Dtype inesperado: {descriptor.dtype}."
        )

    if descriptor.device.type != "cpu":
        raise AssertionError(
            f"Dispositivo inesperado: {descriptor.device}."
        )

    if descriptor.requires_grad:
        raise AssertionError(
            "O descritor não pode requerer gradientes."
        )

    if not statistics["all_finite"]:
        raise AssertionError(
            "O descritor contém NaN ou infinito."
        )

    descriptor_array = descriptor.numpy()

    np.save(
        descriptor_path,
        descriptor_array,
        allow_pickle=False,
    )

    reloaded = np.load(
        descriptor_path,
        allow_pickle=False,
    )

    if reloaded.shape != (EXPECTED_DESCRIPTOR_LENGTH,):
        raise AssertionError(
            "O arquivo .npy salvo possui forma inesperada: "
            f"{reloaded.shape}."
        )

    if not np.array_equal(reloaded, descriptor_array):
        raise AssertionError(
            "O descritor recarregado difere do descritor original."
        )

    descriptor_sha256 = hashlib.sha256(
        descriptor_array.tobytes()
    ).hexdigest()

    peak_rss_kib = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    report = {
        "status": "passed",
        "dataset": "KITTI odometry",
        "sequence": SEQUENCE,
        "camera_id": CAMERA_ID,
        "frame_id": FRAME_ID,
        "image_path": str(image_path),
        "original_image_size": original_size,
        "converted_image_size": converted_size,
        "device": "cpu",
        "weights": str(DEFAULT_RESNET101_WEIGHTS),
        "model_training_mode_after_factory": model.training,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "descriptor": statistics,
        "descriptor_file": str(descriptor_path),
        "descriptor_sha256": descriptor_sha256,
        "serialization": {
            "format": "npy",
            "allow_pickle": False,
            "reload_verified": True,
        },
        "timing_seconds": {
            "model_creation_and_weight_loading": model_seconds,
            "descriptor_extraction": extraction_seconds,
            "total_process": total_seconds,
        },
        "peak_process_rss_kib": peak_rss_kib,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": PIL.__version__,
            "numpy": np.__version__,
        },
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("RESULTADO")
    print(f"- status                 : {report['status']}")
    print(f"- shape                  : {statistics['shape']}")
    print(f"- dtype                  : {statistics['dtype']}")
    print(f"- valores finitos        : {statistics['all_finite']}")
    print(f"- requires_grad          : {statistics['requires_grad']}")
    print(f"- modelo em training     : {model.training}")
    print(f"- norma L2               : {statistics['l2_norm']:.8f}")
    print(f"- tempo de criação       : {model_seconds:.6f} s")
    print(f"- tempo de extração      : {extraction_seconds:.6f} s")
    print(f"- pico RSS               : {peak_rss_kib} KiB")
    print(f"- SHA-256                : {descriptor_sha256}")
    print()
    print(f"Descritor salvo em: {descriptor_path}")
    print(f"Relatório salvo em : {report_path}")
    print("=" * 79)


if __name__ == "__main__":
    main()
