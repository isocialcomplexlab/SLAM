from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from loop_closure.evaluation.resnet101_pair_scores import (
    build_descriptor_index,
    classify_spatial_distance,
    compute_pair_score_rows,
    load_descriptor_index_from_npz,
    save_pair_score_rows,
)


DESCRIPTOR_LENGTH = 4096


def descriptor(
    values: list[float],
) -> np.ndarray:
    result = np.zeros(
        DESCRIPTOR_LENGTH,
        dtype=np.float32,
    )
    result[: len(values)] = values
    return result


def make_index() -> dict[tuple[str, int], np.ndarray]:
    return build_descriptor_index(
        descriptors=np.stack(
            [
                descriptor([0.0, 0.0, 0.0, 0.0]),
                descriptor([0.3, 0.4, 0.0, 0.0]),
                descriptor([3.0, 4.0, 0.0, 0.0]),
            ],
            axis=0,
        ),
        camera_ids=np.asarray(
            ["image_0", "image_0", "image_0"],
            dtype=np.str_,
        ),
        frame_ids=np.asarray(
            [0, 1, 2],
            dtype=np.int64,
        ),
    )


@pytest.mark.parametrize(
    ("distance_m", "expected_class", "expected_label"),
    [
        (0.0, "positive", 1),
        (5.0, "positive", 1),
        (5.000001, "ambiguous", None),
        (9.999999, "ambiguous", None),
        (10.0, "negative", 0),
        (100.0, "negative", 0),
    ],
)
def test_spatial_classification_boundaries(
    distance_m: float,
    expected_class: str,
    expected_label: int | None,
) -> None:
    pair_class, binary_label = classify_spatial_distance(
        distance_m
    )

    assert pair_class == expected_class
    assert binary_label == expected_label


def test_descriptor_index_uses_explicit_camera_frame_identity() -> None:
    index = make_index()

    assert set(index) == {
        ("image_0", 0),
        ("image_0", 1),
        ("image_0", 2),
    }

    assert index[("image_0", 0)].shape == (
        DESCRIPTOR_LENGTH,
    )
    assert index[("image_0", 0)].dtype == np.float32


def test_descriptor_index_rejects_duplicate_identity() -> None:
    descriptors = np.stack(
        [
            descriptor([0.0]),
            descriptor([1.0]),
        ],
        axis=0,
    )

    with pytest.raises(
        ValueError,
        match="duplicada|duplicado",
    ):
        build_descriptor_index(
            descriptors=descriptors,
            camera_ids=np.asarray(
                ["image_0", "image_0"],
                dtype=np.str_,
            ),
            frame_ids=np.asarray(
                [0, 0],
                dtype=np.int64,
            ),
        )


def test_descriptor_index_rejects_non_finite_values() -> None:
    invalid = descriptor([0.0])
    invalid[100] = np.nan

    with pytest.raises(
        ValueError,
        match="finit|NaN|infinito",
    ):
        build_descriptor_index(
            descriptors=invalid.reshape(
                1,
                DESCRIPTOR_LENGTH,
            ),
            camera_ids=np.asarray(
                ["image_0"],
                dtype=np.str_,
            ),
            frame_ids=np.asarray(
                [0],
                dtype=np.int64,
            ),
        )


def test_npz_loader_reconstructs_descriptor_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "descriptors.npz"

    np.savez_compressed(
        path,
        descriptors=np.stack(
            [
                descriptor([1.0]),
                descriptor([2.0]),
            ],
            axis=0,
        ),
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
                "/dataset/image_0/000010.png",
                "/dataset/image_0/000011.png",
            ],
            dtype=np.str_,
        ),
    )

    index = load_descriptor_index_from_npz(path)

    assert set(index) == {
        ("image_0", 10),
        ("image_0", 11),
    }
    assert index[("image_0", 10)][0] == pytest.approx(
        1.0
    )
    assert index[("image_0", 11)][0] == pytest.approx(
        2.0
    )


def test_scores_preserve_legacy_and_corrected_semantics() -> None:
    rows = compute_pair_score_rows(
        pairs=[
            {
                "split": "calibration",
                "query_frame": "0",
                "reference_frame": "1",
                "spatial_distance_m": "3.0",
            }
        ],
        descriptor_index=make_index(),
        camera_id="image_0",
        expected_split="calibration",
    )

    assert len(rows) == 1
    row = rows[0]

    # Vetores:
    # a = [0.0, 0.0, ...]
    # b = [0.3, 0.4, ...]
    #
    # legacy_exact = max(abs(a-b)) = 0.4
    # corrected_l2 = sqrt(0.3² + 0.4²) = 0.5
    assert row["legacy_exact_distance"] == pytest.approx(
        0.4,
        abs=1e-6,
    )
    assert row["corrected_l2_distance"] == pytest.approx(
        0.5,
        abs=1e-6,
    )

    # Para ROC e PR, scores maiores devem indicar maior
    # probabilidade de correspondência.
    assert row["legacy_exact_score"] == pytest.approx(
        -0.4,
        abs=1e-6,
    )
    assert row["corrected_l2_score"] == pytest.approx(
        -0.5,
        abs=1e-6,
    )

    assert row["pair_class"] == "positive"
    assert row["binary_label"] == 1


def test_more_similar_pair_has_larger_ranking_score() -> None:
    rows = compute_pair_score_rows(
        pairs=[
            {
                "split": "test",
                "query_frame": "0",
                "reference_frame": "1",
                "spatial_distance_m": "3.0",
            },
            {
                "split": "test",
                "query_frame": "0",
                "reference_frame": "2",
                "spatial_distance_m": "15.0",
            },
        ],
        descriptor_index=make_index(),
        camera_id="image_0",
        expected_split="test",
    )

    close_pair = rows[0]
    far_pair = rows[1]

    assert (
        close_pair["legacy_exact_score"]
        > far_pair["legacy_exact_score"]
    )
    assert (
        close_pair["corrected_l2_score"]
        > far_pair["corrected_l2_score"]
    )


def test_ambiguous_pair_is_preserved_without_binary_label() -> None:
    rows = compute_pair_score_rows(
        pairs=[
            {
                "split": "stress",
                "query_frame": "0",
                "reference_frame": "1",
                "spatial_distance_m": "7.5",
            }
        ],
        descriptor_index=make_index(),
        camera_id="image_0",
        expected_split="stress",
    )

    assert rows[0]["pair_class"] == "ambiguous"
    assert rows[0]["binary_label"] is None

    # Os scores continuam presentes para auditoria, embora o par
    # não participe das métricas binárias.
    assert np.isfinite(
        rows[0]["legacy_exact_score"]
    )
    assert np.isfinite(
        rows[0]["corrected_l2_score"]
    )


def test_split_contamination_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="split|calibration|test",
    ):
        compute_pair_score_rows(
            pairs=[
                {
                    "split": "test",
                    "query_frame": "0",
                    "reference_frame": "1",
                    "spatial_distance_m": "3.0",
                }
     ],
            descriptor_index=make_index(),
            camera_id="image_0",
            expected_split="calibration",
        )


def test_missing_descriptor_is_rejected() -> None:
    with pytest.raises(
        KeyError,
        match="99|descritor",
    ):
        compute_pair_score_rows(
            pairs=[
                {
                    "split": "test",
                    "query_frame": "0",
                    "reference_frame": "99",
                    "spatial_distance_m": "3.0",
                }
            ],
            descriptor_index=make_index(),
            camera_id="image_0",
            expected_split="test",
        )


def test_duplicate_pair_identity_is_rejected() -> None:
    pair = {
        "split": "test",
        "query_frame": "0",
        "reference_frame": "1",
        "spatial_distance_m": "3.0",
    }

    with pytest.raises(
        ValueError,
        match="duplicado|duplicada",
    ):
        compute_pair_score_rows(
            pairs=[pair, dict(pair)],
            descriptor_index=make_index(),
            camera_id="image_0",
            expected_split="test",
        )


def test_pair_metadata_is_preserved() -> None:
    rows = compute_pair_score_rows(
        pairs=[
            {
                "split": "test",
                "query_frame": "0",
                "reference_frame": "1",
                "spatial_distance_m": "3.0",
                "query_heading_deg": "10.0",
                "reference_heading_deg": "12.5",
                "heading_difference_deg": "2.5",
                "traversal_class": "same_direction",
            }
        ],
        descriptor_index=make_index(),
        camera_id="image_0",
        expected_split="test",
    )

    row = rows[0]

    assert row["query_heading_deg"] == "10.0"
    assert row["reference_heading_deg"] == "12.5"
    assert row["heading_difference_deg"] == "2.5"
    assert row["traversal_class"] == "same_direction"


def test_score_csv_round_trip_and_ambiguous_blank_label(
    tmp_path: Path,
) -> None:
    rows = compute_pair_score_rows(
        pairs=[
            {
                "split": "stress",
                "query_frame": "0",
                "reference_frame": "1",
                "spatial_distance_m": "7.5",
                "traversal_class": "oblique",
            }
        ],
        descriptor_index=make_index(),
        camera_id="image_0",
        expected_split="stress",
    )

    output = tmp_path / "stress_scores.csv"

    save_pair_score_rows(
        rows,
        output,
    )

    assert output.is_file()

    with output.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        loaded = list(csv.DictReader(handle))

    assert len(loaded) == 1

    row = loaded[0]

    assert row["split"] == "stress"
    assert row["query_frame"] == "0"
    assert row["reference_frame"] == "1"
    assert row["pair_class"] == "ambiguous"
    assert row["binary_label"] == ""
    assert row["traversal_class"] == "oblique"

    assert float(
        row["legacy_exact_distance"]
    ) == pytest.approx(0.4, abs=1e-6)

    assert float(
        row["corrected_l2_distance"]
    ) == pytest.approx(0.5, abs=1e-6)


def test_score_csv_refuses_accidental_overwrite(
    tmp_path: Path,
) -> None:
    rows = compute_pair_score_rows(
        pairs=[
            {
                "split": "test",
                "query_frame": "0",
                "reference_frame": "1",
                "spatial_distance_m": "3.0",
            }
        ],
        descriptor_index=make_index(),
        camera_id="image_0",
        expected_split="test",
    )

    output = tmp_path / "scores.csv"

    save_pair_score_rows(rows, output)

    with pytest.raises(FileExistsError):
        save_pair_score_rows(rows, output)
