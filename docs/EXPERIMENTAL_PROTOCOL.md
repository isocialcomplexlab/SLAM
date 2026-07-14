# Experimental Protocol V2 — Draft

## 1. Objective

Evaluate generic, task-adapted, and place-specific visual descriptors for Loop
Closure Detection while separating descriptor discrimination from temporal
verification and preventing calibration/test leakage.

## 2. Research questions

- **RQ1:** How competitive are generic CNN/Transformer descriptors against specialized VPR methods?
- **RQ2:** How does performance vary under stable viewpoints, opposite traversal, and camera rotation?
- **RQ3:** What is the effect of temporal window length `L ∈ {1,2,3,5}`?
- **RQ4:** Does left/right splitting improve performance over full-image extraction?
- **RQ5:** What are the accuracy/efficiency/storage trade-offs?
- **RQ6:** How do ViT layer and token-aggregation choices affect place discrimination?

## 3. Evaluation units

Each evaluated image pair must be represented by one row with at least:

```text
dataset, sequence, query_frame, reference_frame, actual,
distance, score, method, preprocessing, temporal_length, split
```

`score` must be continuous and ordered so that larger values indicate a stronger
loop-closure match. For Euclidean distances, use `score = -distance`.

## 4. Data partitions

- **Calibration/validation:** used to select operational thresholds and any non-learned configuration.
- **Test:** used only once after configurations and thresholds are frozen.
- **Training:** used only by methods requiring learned adaptation.

Required checks:

```text
calibration pairs ∩ test pairs = ∅
calibration frames ∩ test frames = ∅ when sequence-level separation is required
```

The final paper must report sequence IDs, frame ranges, positive/negative pair
counts, and class prevalence for every split.

## 5. Descriptor-only and temporal evaluation

- Primary descriptor comparison: `L = 1`.
- Temporal-consistency ablation: `L = 1, 2, 3, 5`.
- Report false-positive reduction and false-negative increase separately.

## 6. Metrics

Primary:

- Precision-Recall curve;
- Average Precision;
- maximum F1-score;
- Precision and Recall at the frozen operational threshold.

Secondary:

- ROC-AUC from continuous scores;
- F1 at the operational threshold;
- confusion matrix;
- Accuracy, explicitly identified as secondary;
- positive-class prevalence.

## 7. Methods

Existing:

- AlexNet;
- ConvNeXt;
- ResNet;
- Original VGG-16;
- Adapted and Fine-Tuned VGG-16;
- Vision Transformer.

Required baselines:

- NetVLAD;
- DBoW2.

Optional, subject to reproducibility and resources:

- Patch-NetVLAD.

## 8. Dataset expansion

Minimum target:

- KITTI 00, 02, 05, 06, 08;
- at least four TUM sequences covering diverse motion/viewpoint conditions.

Strong target:

- NewCollege or CityCentre as a standard place-recognition dataset.

Selection criteria must be defined before final evaluation and include revisit
availability, trajectory/pose support, operational diversity, and split safety.

## 9. Required ablations

1. `L = 1, 2, 3, 5`;
2. full image vs. left/right split and concatenation;
3. ViT CLS token vs. mean pooling;
4. ViT intermediate vs. final block.

## 10. Efficiency protocol

Measure with batch size 1 after warm-up:

- preprocessing time per image;
- inference time per image;
- matching time per query for 1k, 5k, and 10k descriptors;
- final descriptor dimension and bytes;
- parameter/model size;
- peak CPU/GPU memory.

Report hardware, software versions, precision, warm-up count, measured repeat
count, mean, and standard deviation. Disk-loading time must be separated.

## 11. Reproducibility outputs

Every final experiment should produce:

```text
scores.csv or scores.parquet
metrics.json
precision_recall_curve.pdf
roc_curve.pdf
confusion_matrix.csv
configuration.yaml/json
runtime_environment.txt
```
