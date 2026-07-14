# IEEE Access Revision Matrix

Manuscript ID: **Access-2026-26114**

This matrix links every reviewer concern to the corresponding system change,
experiment, manuscript section, and response-to-reviewers entry.

| ID | Reviewer concern | Required system/action | Evidence to generate | Manuscript impact | Status |
|---|---|---|---|---|---|
| R1-G1 | Positive overall assessment; no specific correction requested | Preserve strengths and report the scope of the revision | Revision summary | Response letter only | Planned |
| R2-M1 | No specialized VPR/LCD baseline | Add NetVLAD and DBoW2; optionally Patch-NetVLAD | AP, max-F1, PR/ROC, latency, storage | Related Work, Methodology, Results | Planned |
| R2-M2 | Only KITTI 05/08 and one TUM sequence | Add KITTI 00/02/06 and diverse TUM sequences; consider NewCollege/CityCentre | Dataset table and per-sequence results | Methodology, Results, Limitations | Planned |
| R2-M3 | Missing Precision-Recall and Average Precision | Compute PR/AP from continuous descriptor scores | PR curves, AP and max-F1 tables | Metrics and all Results sections | In progress |
| R2-M4 | Fixed three-distance heuristic confounds descriptor quality | Evaluate L = 1, 2, 3, 5; use L=1 as descriptor-only primary result | Temporal-ablation table and plot | Methodology and Results | Planned |
| R2-M5 | Architecture analysis is superficial | Add descriptor distributions, UMAP/t-SNE, error cases, Grad-CAM/attention rollout | Representation and attention figures | Results and Discussion | Planned |
| R2-M6 | Missing efficiency analysis | Measure preprocessing, inference, matching, memory, descriptor size | Efficiency table | Methodology and Results | Planned |
| R2-m1 | ViT extraction strategy underdeveloped | Compare final CLS, mean pooling, intermediate/final blocks | ViT ablation | Methodology, Results, Discussion | Planned |
| R2-m2 | Left/right splitting not ablated | Compare full image against split/concatenated image | Preprocessing ablation | Methodology and Results | Planned |
| R2-m3 | Threshold calibration is vague and may leak data | Create explicit calibration/test manifests and freeze thresholds | Split manifest and threshold table | Methodology and Reproducibility | Planned |
| R2-m4 | Large figures convey limited information | Compress sequence illustrations and add PR/analysis figures | Revised figure set | Results layout | Planned |
| R2-m5 | Conclusion merely repeats results | Derive condition-specific design guidance | Practitioner guidance table/text | Conclusion | Planned |
| R2-R1 | Important VPR references are missing | Add and critically discuss relevant NetVLAD, Patch-NetVLAD, DBoW2, DenseVLAD/view-synthesis literature | Updated bibliography | Related Work | Planned |
| R2-R2 | General AlexNet survey is tangential | Remove the survey if the primary AlexNet reference is sufficient | Reference audit | Related Work / bibliography | Planned |

## Completion rule

A concern is marked **Completed** only after all four items exist:

1. committed system/code change;
2. reproducible experiment output;
3. highlighted manuscript modification;
4. final `Author response` and `Author action` text.
