# KITTI 05 + ResNet Pilot

## Objective

Reproduce the KITTI sequence 05 evaluation using continuous descriptor
distances and separate descriptor-level evaluation from temporal verification.

## Methodological corrections

- ROC and PR must use continuous scores.
- Average Precision must be reported.
- Descriptor evaluation must use L=1.
- Temporal lengths L={1,2,3,5} must be evaluated separately.
- Calibration and test pairs must not overlap.
- Accuracy is treated as a secondary metric.

## Required outputs

- experiment_config.json
- pairs.csv
- scores.csv
- calibration_pairs.csv
- test_pairs.csv
- threshold.json
- metrics_l1.json
- metrics_l3.json
- temporal_ablation.csv
- precision_recall_curve.pdf
- roc_curve.pdf
- confusion_matrices.csv

## Legacy implementation findings

- The original notebook loads KITTI sequence 05 from Google Drive.
- The original ResNet descriptor tensors are not stored in the repository.
- The original metrics CSV and loop-closure output file are external.
- ROC and ROC-AUC were computed from the binary `pred` column.
- Continuous descriptor distances must be regenerated.
- The `.pt` files currently present in the repository belong to Gazebo experiments.

## Data discovery

A search was performed in the WSL home directory and in the Windows user
profile to locate the original KITTI 05 images, ResNet tensors, metrics CSV,
and loop-closure output files.

The next implementation decision depends on the discovered artifacts:

1. Images available:
   regenerate descriptors and continuous distances from the original frames.
2. Only descriptor tensors available:
   regenerate pairwise scores and all evaluation metrics.
3. Only binary prediction CSV available:
   use it exclusively for legacy auditing, not for the revised evaluation.
4. No artifacts available:
   obtain KITTI odometry sequence 05 again and rebuild the experiment.

## Data-discovery result

No KITTI 05 images, ResNet descriptors, continuous-distance files, or legacy
metrics were found in the WSL home directory or in the Windows user profile.

Decision: regenerate the KITTI 05 experiment from the official KITTI odometry
images and ground-truth poses.

## KITTI sequence validation

The official KITTI odometry sequence 05 was validated using an automated
dataset checker. Images, ground-truth poses, and timestamps are aligned by
frame index. A relative-path manifest was generated for reproducible
descriptor extraction and ground-truth construction.
