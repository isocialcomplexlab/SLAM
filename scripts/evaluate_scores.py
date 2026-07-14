#!/usr/bin/env python3
"""Evaluate continuous loop-closure scores from a CSV file.

Expected columns:
  - actual: 0/1 ground-truth label
  - score: continuous score where larger means a stronger match

For Euclidean distance outputs, create score = -distance before evaluation.
Threshold selection must use calibration data. The selected threshold is then
frozen and supplied when evaluating held-out test data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loop_closure.metrics import evaluate_binary_scores, select_threshold_by_f1  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="CSV containing actual and score columns")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Frozen operational threshold. Omit only for calibration data.",
    )
    parser.add_argument("--actual-column", default="actual")
    parser.add_argument("--score-column", default="score")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.csv)
    required = {args.actual_column, args.score_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    y_true = frame[args.actual_column].astype(int).to_numpy()
    scores = frame[args.score_column].astype(float).to_numpy()

    threshold = args.threshold
    mode = "test"
    if threshold is None:
        threshold, _ = select_threshold_by_f1(y_true, scores)
        mode = "calibration"

    result = evaluate_binary_scores(y_true, scores, threshold=threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = {"mode": mode, **result.to_dict()}
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    RocCurveDisplay.from_predictions(y_true, scores)
    plt.tight_layout()
    plt.savefig(args.output_dir / "roc_curve.pdf")
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, scores)
    plt.tight_layout()
    plt.savefig(args.output_dir / "precision_recall_curve.pdf")
    plt.close()

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
