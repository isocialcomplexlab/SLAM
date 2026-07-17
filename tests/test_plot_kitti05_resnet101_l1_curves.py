from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(
    "scripts/plot_kitti05_resnet101_l1_curves.py"
)


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def prepare_fixture(
    tmp_path: Path,
) -> tuple[Path, Path]:
    metrics_dir = tmp_path / "metrics"
    curves_dir = metrics_dir / "curves"
    output_dir = tmp_path / "figures"

    write_csv(
        metrics_dir / "metrics_summary.csv",
        [
            "split",
            "method",
            "positive_prevalence",
            "roc_auc",
            "pr_auc",
            "average_precision",
        ],
        [
            {
                "split": "test",
                "method": "legacy_exact",
                "positive_prevalence": 0.1,
                "roc_auc": 0.7,
                "pr_auc": 0.3,
                "average_precision": 0.31,
            },
            {
                "split": "test",
                "method": "corrected_l2",
                "positive_prevalence": 0.1,
                "roc_auc": 0.8,
                "pr_auc": 0.4,
                "average_precision": 0.41,
            },
        ],
    )

    for method in (
        "legacy_exact",
        "corrected_l2",
    ):
        write_csv(
            curves_dir / f"test_{method}_roc.csv",
            [
                "false_positive_rate",
                "true_positive_rate",
                "threshold",
            ],
            [
                {
                    "false_positive_rate": 0.0,
                    "true_positive_rate": 0.0,
                    "threshold": 10.0,
                },
                {
                    "false_positive_rate": 0.2,
                    "true_positive_rate": 0.7,
                    "threshold": 0.5,
                },
                {
                    "false_positive_rate": 1.0,
                    "true_positive_rate": 1.0,
                    "threshold": 0.0,
                },
            ],
        )

        write_csv(
            curves_dir / f"test_{method}_pr.csv",
            [
                "precision",
                "recall",
                "threshold",
            ],
            [
                {
                    "precision": 0.1,
                    "recall": 1.0,
                    "threshold": 0.0,
                },
                {
                    "precision": 0.5,
                    "recall": 0.5,
                    "threshold": 0.5,
                },
                {
                    "precision": 1.0,
                    "recall": 0.0,
                    "threshold": "",
                },
            ],
        )

    return metrics_dir, output_dir


def test_script_generates_valid_pdf_files(
    tmp_path: Path,
) -> None:
    metrics_dir, output_dir = (
        prepare_fixture(tmp_path)
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--metrics-dir",
            str(metrics_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    for filename in (
        "roc_curve.pdf",
        "precision_recall_curve.pdf",
    ):
        path = output_dir / filename

        assert path.is_file()
        assert path.stat().st_size > 1000
        assert path.read_bytes().startswith(
            b"%PDF-"
        )


def test_script_creates_reproducibility_manifest(
    tmp_path: Path,
) -> None:
    metrics_dir, output_dir = (
        prepare_fixture(tmp_path)
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--metrics-dir",
            str(metrics_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    manifest = (
        output_dir
        / "figure_manifest.json"
    ).read_text(encoding="utf-8")

    assert '"split": "test"' in manifest
    assert '"legacy_exact"' in manifest
    assert '"corrected_l2"' in manifest
    assert '"sha256"' in manifest


def test_script_refuses_accidental_overwrite(
    tmp_path: Path,
) -> None:
    metrics_dir, output_dir = (
        prepare_fixture(tmp_path)
    )

    command = [
        sys.executable,
        str(SCRIPT),
        "--metrics-dir",
        str(metrics_dir),
        "--output-dir",
        str(output_dir),
    ]

    subprocess.run(
        command,
        check=True,
    )

    repeated = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    assert repeated.returncode != 0
    assert "já existem" in repeated.stderr
