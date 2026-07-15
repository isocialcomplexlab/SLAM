# Legacy-to-Revision Traceability

This document records how the revised IEEE Access experimental pipeline
relates to the original Loop Closure Detection implementation available
under `LoopClosure/`.

The legacy notebooks are preserved without modification. Revised modules
reuse valid architectural and preprocessing decisions while correcting
methodological limitations identified during peer review.

## KITTI 05 + ResNet pilot

| Legacy component | Revised component | Preserved behavior | Revision or correction |
|---|---|---|---|
| `LoopClosure/KITTI/Group5/KITTI_05_resnet.ipynb` | `src/loop_closure/datasets/kitti_odometry.py` | KITTI sequence 05 and frame ordering | Explicit validation of images, poses, timestamps, and calibration |
| Google Drive absolute paths | `.env.local` and relative manifests | Same source dataset | Portable local dataset configuration |
| ResNet feature extraction | Planned `descriptors/resnet.py` | Architecture and selected representation | Reproducible batch-size-one extraction and explicit model metadata |
| Left/right image splitting | Planned preprocessing module | Original splitting strategy | Full-image versus split-image ablation |
| `torch.cdist(..., p=2)` | Planned matching module | Euclidean descriptor distance | Continuous score retained before thresholding |
| `torch.max(distance)` | Planned matching module | Legacy scalar reduction | Exact dimensional interpretation and alternative aggregation documented |
| Binary `df["pred"]` used for ROC | `src/loop_closure/metrics/` | None | ROC and PR computed from continuous scores |
| Fixed three-distance sequence | Planned temporal module | Temporal consistency concept | Ablation with sequence lengths L={1,2,3,5} |
| Empirical threshold without explicit split | Planned calibration module | Architecture-specific threshold | Held-out calibration set separated from test data |
| Accuracy, F1, ROC-AUC | Revised metrics module | Legacy metrics retained for comparison | Average Precision, PR curves, maximum F1, Precision, and Recall added |
| Manual dataset inspection | `scripts/validate_kitti_sequence.py` | Original sequence | Automated and tested dataset validation |
| Informal loop annotations | Planned pose-based pair generator | Revisit semantics | Explicit spatial, temporal, and heading metadata |

## Governing rules

1. Legacy notebooks remain available as historical evidence.
2. The revised pipeline first reproduces the original configuration.
3. Methodological corrections are evaluated separately.
4. Every modification is linked to a reviewer concern.
5. Results from the legacy and revised protocols are stored separately.
6. Dataset files and pretrained weights are not committed to Git.
7. Manuscript claims will be updated only after revised experiments are frozen.
