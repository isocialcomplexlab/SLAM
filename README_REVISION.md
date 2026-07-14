# IEEE Access Revision Workspace

This workspace extends the legacy Loop Closure Detection notebooks to address
the IEEE Access review of manuscript `Access-2026-26114`.

## Initial checks

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-revision.txt
PYTHONPATH=src pytest -q
python scripts/audit_legacy_notebooks.py LoopClosure
```

## Evaluate calibration scores

```bash
python scripts/evaluate_scores.py calibration_scores.csv \
  --output-dir results/pilot/calibration
```

Copy the reported threshold and evaluate held-out test data:

```bash
python scripts/evaluate_scores.py test_scores.csv \
  --output-dir results/pilot/test \
  --threshold <FROZEN_THRESHOLD>
```

Continuous scores are mandatory. For Euclidean distances, store
`score = -distance`.
