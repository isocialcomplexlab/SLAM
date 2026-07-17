#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, has_file_allowed_extension


def resolve_sequence_root(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "sequences" / "05",
        dataset_root / "dataset" / "sequences" / "05",
        dataset_root / "05",
    ]

    if dataset_root.name == "05":
        candidates.insert(0, dataset_root)

    checked: list[str] = []

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        checked.append(str(candidate))

        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Não foi possível localizar a sequência KITTI 05.\n"
        + "\n".join(f"  - {path}" for path in checked)
    )


def reproduce_imagefolder_order(
    root: Path,
) -> tuple[list[str], dict[str, int], list[tuple[str, int]]]:
    """
    Reproduz a ordenação relevante do torchvision ImageFolder:

    1. classes = subdiretórios imediatos, em ordem lexicográfica;
    2. classes percorridas em ordem lexicográfica;
    3. diretórios e nomes de arquivo percorridos ordenadamente.
    """
    classes = sorted(entry.name for entry in root.iterdir() if entry.is_dir())

    if not classes:
        raise FileNotFoundError(
            f"Nenhum subdiretório de classe encontrado em {root}"
        )

    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
    samples: list[tuple[str, int]] = []

    for class_name in sorted(class_to_idx):
        class_index = class_to_idx[class_name]
        class_root = root / class_name

        for current_root, _, filenames in sorted(os.walk(class_root, followlinks=True)):
            for filename in sorted(filenames):
                path = Path(current_root) / filename

                if has_file_allowed_extension(str(path), IMG_EXTENSIONS):
                    samples.append((str(path.resolve()), class_index))

    return classes, class_to_idx, samples


def parse_frame_id(path: Path) -> int | None:
    if path.stem.isdigit():
        return int(path.stem)

    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else None


def sample_record(
    index: int,
    path_string: str,
    class_index: int,
    classes: list[str],
    sequence_root: Path,
) -> dict[str, Any]:
    path = Path(path_string)
    class_name = classes[class_index]

    return {
        "sample_index": index,
        "relative_path": str(path.relative_to(sequence_root)),
        "class_name": class_name,
        "class_index": class_index,
        "camera_id": class_name if re.fullmatch(r"image_\d+", class_name) else None,
        "frame_id": parse_frame_id(path),
    }


def main() -> None:
    root_value = os.environ.get("KITTI_ODOMETRY_ROOT")

    if not root_value:
        raise EnvironmentError(
            "KITTI_ODOMETRY_ROOT não está definido. Verifique .env.local."
        )

    dataset_root = Path(root_value).expanduser().resolve()
    sequence_root = resolve_sequence_root(dataset_root)

    classes, class_to_idx, samples = reproduce_imagefolder_order(sequence_root)

    records = [
        sample_record(index, path, class_index, classes, sequence_root)
        for index, (path, class_index) in enumerate(samples)
    ]

    class_counts = Counter(record["class_name"] for record in records)

    class_boundaries: dict[str, dict[str, Any]] = {}

    for class_name in classes:
        matching = [
            record for record in records if record["class_name"] == class_name
        ]

        class_boundaries[class_name] = {
            "sample_count": len(matching),
            "first_sample_index": (
                matching[0]["sample_index"] if matching else None
            ),
            "last_sample_index": (
                matching[-1]["sample_index"] if matching else None
            ),
            "first_relative_path": (
                matching[0]["relative_path"] if matching else None
            ),
            "last_relative_path": (
                matching[-1]["relative_path"] if matching else None
            ),
            "first_frame_id": (
                matching[0]["frame_id"] if matching else None
            ),
            "last_frame_id": (
                matching[-1]["frame_id"] if matching else None
            ),
        }

    class_transitions: list[dict[str, Any]] = []

    for previous, current in zip(records, records[1:]):
        if previous["class_name"] != current["class_name"]:
            class_transitions.append(
                {
                    "previous_sample_index": previous["sample_index"],
                    "previous_class": previous["class_name"],
                    "previous_path": previous["relative_path"],
                    "current_sample_index": current["sample_index"],
                    "current_class": current["class_name"],
                    "current_path": current["relative_path"],
                }
            )

    adjacent_same_frame_different_camera = []

    for first, second in zip(records, records[1:]):
        if (
            first["frame_id"] is not None
            and first["frame_id"] == second["frame_id"]
            and first["camera_id"] is not None
            and second["camera_id"] is not None
            and first["camera_id"] != second["camera_id"]
        ):
            adjacent_same_frame_different_camera.append(
                {
                    "first_index": first["sample_index"],
                    "first_path": first["relative_path"],
                    "second_index": second["sample_index"],
                    "second_path": second["relative_path"],
                    "frame_id": first["frame_id"],
                }
            )

    camera_records = [
        record for record in records if record["camera_id"] is not None
    ]

    camera_frame_sets: dict[str, set[int]] = {}

    for camera_name in sorted(
        {
            str(record["camera_id"])
            for record in camera_records
            if record["camera_id"] is not None
        }
    ):
        camera_frame_sets[camera_name] = {
            int(record["frame_id"])
            for record in camera_records
            if record["camera_id"] == camera_name
            and record["frame_id"] is not None
        }

    camera_alignment: dict[str, Any] = {}

    if "image_0" in camera_frame_sets and "image_1" in camera_frame_sets:
        image_0_frames = camera_frame_sets["image_0"]
        image_1_frames = camera_frame_sets["image_1"]

        camera_alignment = {
            "image_0_frame_count": len(image_0_frames),
            "image_1_frame_count": len(image_1_frames),
            "shared_frame_count": len(image_0_frames & image_1_frames),
            "only_image_0_count": len(image_0_frames - image_1_frames),
            "only_image_1_count": len(image_1_frames - image_0_frames),
            "image_0_before_image_1": (
                class_boundaries["image_0"]["last_sample_index"] is not None
                and class_boundaries["image_1"]["first_sample_index"] is not None
                and class_boundaries["image_0"]["last_sample_index"]
                < class_boundaries["image_1"]["first_sample_index"]
            ),
        }

    actual_imagefolder: dict[str, Any]

    try:
        dataset = ImageFolder(str(sequence_root))

        actual_samples = [
            (str(Path(path).resolve()), class_index)
            for path, class_index in dataset.samples
        ]

        actual_imagefolder = {
            "constructed_successfully": True,
            "classes": dataset.classes,
            "class_to_idx": dataset.class_to_idx,
            "sample_count": len(dataset.samples),
            "matches_reproduced_order": actual_samples == samples,
        }

    except Exception as error:
        actual_imagefolder = {
            "constructed_successfully": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "explanation": (
                "O ImageFolder padrão pode falhar quando um subdiretório "
                "imediato não contém extensões de imagem reconhecidas."
            ),
        }

    first_samples = records[:12]
    last_samples = records[-12:] if records else []

    boundary_samples: list[dict[str, Any]] = []

    for transition in class_transitions:
        previous_index = transition["previous_sample_index"]
        start = max(0, previous_index - 2)
        end = min(len(records), previous_index + 4)
        boundary_samples.extend(records[start:end])

    unique_boundary_samples = {
        record["sample_index"]: record for record in boundary_samples
    }

    report = {
        "dataset_root": str(dataset_root),
        "sequence_root": str(sequence_root),
        "image_extensions": list(IMG_EXTENSIONS),
        "immediate_subdirectories_considered_classes": classes,
        "class_to_idx": class_to_idx,
        "total_image_samples": len(records),
        "class_counts": dict(sorted(class_counts.items())),
        "class_boundaries": class_boundaries,
        "class_transition_count": len(class_transitions),
        "class_transitions": class_transitions,
        "adjacent_same_frame_different_camera_count": len(
            adjacent_same_frame_different_camera
        ),
        "adjacent_same_frame_different_camera_examples":
            adjacent_same_frame_different_camera[:20],
        "camera_alignment": camera_alignment,
        "actual_imagefolder": actual_imagefolder,
        "first_samples": first_samples,
        "samples_around_class_boundaries": [
            unique_boundary_samples[index]
            for index in sorted(unique_boundary_samples)
        ],
        "last_samples": last_samples,
    }

    output_directory = (
        Path("results") / "pilot" / "kitti05_resnet"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    json_path = output_directory / "imagefolder_order_audit.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 79)
    print("AUDITORIA DA ORDEM DO IMAGEFOLDER — KITTI 05")
    print("=" * 79)
    print(f"KITTI_ODOMETRY_ROOT : {dataset_root}")
    print(f"Raiz da sequência   : {sequence_root}")
    print(f"Classes imediatas   : {classes}")
    print(f"Total de amostras   : {len(records)}")
    print()

    print("CONTAGEM POR CLASSE")
    for class_name in classes:
        boundary = class_boundaries[class_name]
        print(
            f"- {class_name}: "
            f"{boundary['sample_count']} imagens; "
            f"índices {boundary['first_sample_index']}.."
            f"{boundary['last_sample_index']}; "
            f"frames {boundary['first_frame_id']}.."
            f"{boundary['last_frame_id']}"
        )

    print()
    print("IMAGEFOLDER REAL")
    print(
        json.dumps(
            actual_imagefolder,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("TRANSIÇÕES ENTRE CLASSES")
    if class_transitions:
        for transition in class_transitions:
            print(
                f"- índice {transition['previous_sample_index']} "
                f"({transition['previous_path']}) -> "
                f"índice {transition['current_sample_index']} "
                f"({transition['current_path']})"
            )
    else:
        print("- Nenhuma transição encontrada.")

    print()
    print("ALINHAMENTO IMAGE_0 / IMAGE_1")
    if camera_alignment:
        print(json.dumps(camera_alignment, indent=2, ensure_ascii=False))
    else:
        print("- image_0 e image_1 não foram ambas encontradas como classes.")

    print()
    print(
        "PARES ADJACENTES COM MESMO FRAME E CÂMERAS DIFERENTES: "
        f"{len(adjacent_same_frame_different_camera)}"
    )

    print()
    print("PRIMEIRAS 12 AMOSTRAS")
    for record in first_samples:
        print(
            f"{record['sample_index']:6d} | "
            f"{record['class_name']:12s} | "
            f"frame={str(record['frame_id']):>6s} | "
            f"{record['relative_path']}"
        )

    print()
    print("AMOSTRAS NAS FRONTEIRAS ENTRE CLASSES")
    for record in report["samples_around_class_boundaries"]:
        print(
            f"{record['sample_index']:6d} | "
            f"{record['class_name']:12s} | "
            f"frame={str(record['frame_id']):>6s} | "
            f"{record['relative_path']}"
        )

    print()
    print(f"Relatório JSON salvo em: {json_path}")
    print("=" * 79)


if __name__ == "__main__":
    main()
