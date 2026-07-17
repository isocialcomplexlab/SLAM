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

## Pose-based revisit sensitivity analysis

A preliminary sensitivity analysis was introduced to evaluate how the number
of ground-truth revisit pairs varies with the spatial radius and temporal
exclusion.

The analysis uses the KITTI x-z ground plane and evaluates spatial radii of
2, 5, and 10 meters, together with temporal exclusions of 30, 50, 100, and
200 frames.

These values are not yet the final evaluation protocol. They will be used to
select and justify the ground-truth definition before descriptor extraction.
Heading differences will subsequently be stored as metadata so that
same-direction and opposite-traversal conditions can be analyzed separately.

## Revisit-event analysis

Using a 5 m positive radius and a 100-frame temporal exclusion, KITTI 05
contains three detected revisit events and 6,550 positive frame pairs.

Traversal-direction distribution:

- same direction: 5,693 pairs;
- oblique: 857 pairs;
- opposite direction: 0 pairs.

The third event cannot be considered independent from the first event because
its reference-frame interval overlaps the first event's reference and query
intervals. Its query interval is also temporally close to the second event.

For the pilot only, event 1 will be used for threshold calibration, event 2
for primary testing, and event 3 as an overlap/stress diagnostic. The final
paper protocol must prefer cross-sequence calibration to avoid same-sequence
leakage.

## Leakage-aware pilot split

The three KITTI 05 revisit events were partitioned by complete events:

- calibration: event 1, 3,985 positive pairs;
- test: event 2, 1,269 positive pairs;
- stress diagnostic: event 3, 1,296 positive pairs.

Calibration and test share no query or reference frames. The stress diagnostic
shares 49 frames with calibration and is therefore explicitly marked as
non-independent. It will not be included in the primary test metrics.

## Candidate-pair generation

Complete event-local query-reference Cartesian products were generated using:

- positive: spatial distance less than or equal to 5 m;
- ambiguous: spatial distance greater than 5 m and less than 10 m;
- negative: spatial distance greater than or equal to 10 m.

Results:

- calibration: 78,624 candidates, 3,985 positives, 3,572 ambiguous,
  and 71,067 negatives;
- test: 14,170 candidates, 1,269 positives, 1,186 ambiguous,
  and 11,715 negatives;
- stress: 9,636 candidates, 1,296 positives, 1,226 ambiguous,
  and 7,114 negatives.

Ambiguous pairs are retained for auditing but excluded from the primary binary
metrics. All original positive pairs were preserved exactly, and calibration
and test share no frames.
