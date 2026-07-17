"""Extração determinística em lote de descritores ResNet-101.

Cada imagem é identificada explicitamente pelo par:

    (camera_id, frame_id)

O módulo não usa a ordem global do ImageFolder, não divide índices por dois e
não reproduz a sobreposição dos blocos do notebook legado.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from loop_closure.descriptors.resnet101 import (
    DESCRIPTOR_LENGTH,
    extract_resnet101_descriptor,
)


SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".ppm",
    ".pgm",
    ".pbm",
    ".pnm",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True, slots=True)
class KittiImageSample:
    """Identidade explícita de uma imagem KITTI."""

    camera_id: str
    frame_id: int
    image_path: Path


@dataclass(frozen=True, slots=True)
class ResNet101DescriptorBatch:
    """Descritores e metadados alinhados pelo primeiro eixo."""

    descriptors: np.ndarray
    camera_ids: np.ndarray
    frame_ids: np.ndarray
    image_paths: np.ndarray


DescriptorExtractor = Callable[..., torch.Tensor]


def _validate_camera_ids(
    camera_ids: Sequence[str],
) -> tuple[str, ...]:
    normalized = tuple(camera_ids)

    if not normalized:
        raise ValueError(
            "camera_ids deve conter ao menos uma câmera."
        )

    for camera_id in normalized:
        if not isinstance(camera_id, str) or not camera_id:
            raise ValueError(
                "Cada camera_id deve ser uma string não vazia."
            )

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "camera_ids contém uma câmera duplicada."
        )

    return normalized


def _discover_camera_samples(
    sequence_root: Path,
    camera_id: str,
    *,
    require_contiguous: bool,
) -> list[KittiImageSample]:
    camera_root = sequence_root / camera_id

    if not camera_root.is_dir():
        raise FileNotFoundError(
            f"Diretório da câmera {camera_id} não encontrado: "
            f"{camera_root}"
        )

    discovered: dict[int, Path] = {}

    image_paths = sorted(
        path
        for path in camera_root.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
    )

    if not image_paths:
        raise FileNotFoundError(
            f"Nenhuma imagem encontrada para {camera_id} em "
            f"{camera_root}."
        )

    for image_path in image_paths:
        if not image_path.stem.isdigit():
            raise ValueError(
                "O nome do arquivo deve representar numericamente o frame; "
                f"recebido {image_path.name}."
            )

        frame_id = int(image_path.stem)

        if frame_id in discovered:
            raise ValueError(
                f"Frame duplicado {frame_id} em {camera_id}: "
                f"{discovered[frame_id].name} e {image_path.name}."
            )

        discovered[frame_id] = image_path.resolve()

    sorted_frame_ids = sorted(discovered)

    if require_contiguous:
        expected = list(
            range(
                sorted_frame_ids[0],
                sorted_frame_ids[-1] + 1,
            )
        )

        if sorted_frame_ids != expected:
            missing = sorted(
                set(expected) - set(sorted_frame_ids)
            )

            raise ValueError(
                f"A sequência de frames de {camera_id} não é contígua; "
                f"frames ausentes: {missing[:20]}."
            )

    return [
        KittiImageSample(
            camera_id=camera_id,
            frame_id=frame_id,
            image_path=discovered[frame_id],
        )
        for frame_id in sorted_frame_ids
    ]


def discover_kitti_image_samples(
    sequence_root: str | Path,
    *,
    camera_ids: Sequence[str] = ("image_0",),
    require_contiguous: bool = True,
) -> list[KittiImageSample]:
    """Descobre imagens por câmera, preservando a ordem solicitada.

    Dentro de cada câmera, os frames são ordenados numericamente.
    """
    root = Path(sequence_root).expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(
            f"Raiz da sequência não encontrada: {root}"
        )

    normalized_camera_ids = _validate_camera_ids(camera_ids)

    samples: list[KittiImageSample] = []

    for camera_id in normalized_camera_ids:
        samples.extend(
            _discover_camera_samples(
                root,
                camera_id,
                require_contiguous=require_contiguous,
            )
        )

    return samples


def _validate_sample_identity(
    samples: Sequence[KittiImageSample],
) -> None:
    identities: set[tuple[str, int]] = set()

    for sample in samples:
        if not isinstance(sample, KittiImageSample):
            raise TypeError(
                "Todos os elementos de samples devem ser "
                "KittiImageSample."
            )

        identity = (
            sample.camera_id,
            sample.frame_id,
        )

        if identity in identities:
            raise ValueError(
                "Identidade duplicada encontrada no lote: "
                f"{identity}."
            )

        identities.add(identity)

        if not sample.image_path.is_file():
            raise FileNotFoundError(
                f"Imagem não encontrada: {sample.image_path}"
            )


def _descriptor_to_numpy(
    descriptor: torch.Tensor,
    *,
    sample: KittiImageSample,
) -> np.ndarray:
    if not isinstance(descriptor, torch.Tensor):
        raise TypeError(
            "O extrator deve retornar torch.Tensor; "
            f"recebido {type(descriptor).__name__} para "
            f"{sample.camera_id}/{sample.frame_id}."
        )

    if descriptor.shape != (DESCRIPTOR_LENGTH,):
        raise ValueError(
            "Cada descritor deve possuir exatamente 4096 valores; "
            f"recebido shape={tuple(descriptor.shape)} para "
            f"{sample.camera_id}/{sample.frame_id}."
        )

    if not descriptor.is_floating_point():
        raise TypeError(
            "O descritor deve possuir dtype de ponto flutuante."
        )

    if not bool(torch.isfinite(descriptor).all()):
        raise ValueError(
            "O descritor contém NaN ou valor infinito para "
            f"{sample.camera_id}/{sample.frame_id}."
        )

    return (
        descriptor
        .detach()
        .to(
            device="cpu",
            dtype=torch.float32,
        )
        .contiguous()
        .numpy()
        .copy()
    )


def extract_resnet101_descriptor_batch(
    *,
    model: Any,
    samples: Iterable[KittiImageSample],
    extractor: DescriptorExtractor = extract_resnet101_descriptor,
    device: str | torch.device = "cpu",
) -> ResNet101DescriptorBatch:
    """Extrai cada amostra exatamente uma vez, na ordem recebida."""
    materialized_samples = list(samples)

    _validate_sample_identity(materialized_samples)

    descriptor_rows: list[np.ndarray] = []
    camera_ids: list[str] = []
    frame_ids: list[int] = []
    image_paths: list[str] = []

    for sample in materialized_samples:
        with Image.open(sample.image_path) as opened_image:
            image = opened_image.convert("RGB")

            descriptor = extractor(
                model,
                image,
                device=device,
            )

        descriptor_rows.append(
            _descriptor_to_numpy(
                descriptor,
                sample=sample,
            )
        )
        camera_ids.append(sample.camera_id)
        frame_ids.append(sample.frame_id)
        image_paths.append(str(sample.image_path))

    if descriptor_rows:
        descriptors = np.stack(
            descriptor_rows,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )
    else:
        descriptors = np.empty(
            (0, DESCRIPTOR_LENGTH),
            dtype=np.float32,
        )

    return ResNet101DescriptorBatch(
        descriptors=descriptors,
        camera_ids=np.asarray(
            camera_ids,
            dtype=np.str_,
        ),
        frame_ids=np.asarray(
            frame_ids,
            dtype=np.int64,
        ),
        image_paths=np.asarray(
            image_paths,
            dtype=np.str_,
        ),
    )


def _validate_batch_arrays(
    batch: ResNet101DescriptorBatch,
) -> None:
    if not isinstance(batch, ResNet101DescriptorBatch):
        raise TypeError(
            "batch deve ser ResNet101DescriptorBatch."
        )

    descriptors = np.asarray(batch.descriptors)
    camera_ids = np.asarray(batch.camera_ids)
    frame_ids = np.asarray(batch.frame_ids)
    image_paths = np.asarray(batch.image_paths)

    if (
        descriptors.ndim != 2
        or descriptors.shape[1] != DESCRIPTOR_LENGTH
    ):
        raise ValueError(
            "descriptors deve possuir forma [N, 4096]; "
            f"recebido {descriptors.shape}."
        )

    sample_count = descriptors.shape[0]

    for name, array in (
        ("camera_ids", camera_ids),
        ("frame_ids", frame_ids),
        ("image_paths", image_paths),
    ):
        if array.ndim != 1 or len(array) != sample_count:
            raise ValueError(
                f"{name} deve possuir N={sample_count} elementos."
            )

    if descriptors.dtype != np.float32:
        raise TypeError(
            "descriptors deve possuir dtype float32."
        )

    if frame_ids.dtype != np.int64:
        raise TypeError(
            "frame_ids deve possuir dtype int64."
        )

    if not np.isfinite(descriptors).all():
        raise ValueError(
            "descriptors contém NaN ou infinito."
        )

    for name, array in (
        ("descriptors", descriptors),
        ("camera_ids", camera_ids),
        ("frame_ids", frame_ids),
        ("image_paths", image_paths),
    ):
        if array.dtype.kind == "O":
            raise TypeError(
                f"{name} não pode possuir dtype object."
            )


def save_resnet101_descriptor_batch(
    batch: ResNet101DescriptorBatch,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Salva o lote em NPZ, sem arrays object e sem pickle."""
    _validate_batch_arrays(batch)

    path = Path(output_path).expanduser()

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"O arquivo já existe: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        descriptors=np.asarray(
            batch.descriptors,
            dtype=np.float32,
        ),
        camera_ids=np.asarray(
            batch.camera_ids,
            dtype=np.str_,
        ),
        frame_ids=np.asarray(
            batch.frame_ids,
            dtype=np.int64,
        ),
        image_paths=np.asarray(
            batch.image_paths,
            dtype=np.str_,
        ),
    )

    return path
