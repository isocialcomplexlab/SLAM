# Sprint 1 — Foundation and Evaluation Audit

## Goal

Build a reproducible pilot for KITTI 05 + ResNet that computes ROC and PR from
continuous descriptor scores and separates `L=1` from the legacy `L=3` rule.

## Tasks

- [x] Import the legacy code snapshot.
- [x] Add the LaTeX source, figures, bibliography, and response template.
- [x] Add the reviewer matrix.
- [x] Add Experimental Protocol V2 draft.
- [x] Add a legacy-notebook audit script.
- [x] Add continuous-score metric utilities and tests.
- [ ] Recover or regenerate KITTI 05 + ResNet pair-level distances.
- [ ] Define calibration/test manifests for the pilot.
- [ ] Freeze the calibration threshold.
- [ ] Generate descriptor-only `L=1` metrics.
- [ ] Generate legacy-comparable `L=3` metrics.
- [ ] Produce pilot PR/ROC curves and temporal-ablation table.

## Acceptance criteria

The sprint is complete when the following files are generated reproducibly:

```text
results/pilot/kitti05_resnet_scores.csv
results/pilot/kitti05_resnet_metrics.json
results/pilot/kitti05_resnet_precision_recall_curve.pdf
results/pilot/kitti05_resnet_roc_curve.pdf
results/pilot/kitti05_resnet_temporal_ablation.csv
```
