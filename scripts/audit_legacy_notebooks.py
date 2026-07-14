#!/usr/bin/env python3
"""Audit legacy notebooks for reproducibility and evaluation risks."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

PATTERNS = {
    "binary_roc": re.compile(r"RocCurveDisplay\.from_predictions\([^\n]*\[['\"]pred['\"]\]"),
    "binary_auc": re.compile(r"roc_auc_score\([^\n]*\[['\"]pred['\"]\]"),
    "absolute_colab_path": re.compile(r"/content/drive/"),
    "hard_coded_threshold": re.compile(r"(?:dist_limit|threshold|limiar)\s*=\s*[0-9]*\.?[0-9]+", re.I),
    "three_sequence_logic": re.compile(r"(?:i\s*<\s*3|len\([^\)]*\)\s*==\s*3|range\([^\)]*3[^\)]*\))"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("LoopClosure"))
    parser.add_argument("--output", type=Path, default=Path("results/legacy_notebook_audit.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, str | int]] = []
    for notebook in sorted(args.root.rglob("*.ipynb")):
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        row: dict[str, str | int] = {"notebook": str(notebook)}
        for name, pattern in PATTERNS.items():
            row[name] = len(pattern.findall(source))
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["notebook", *PATTERNS])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Audited {len(rows)} notebooks -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
