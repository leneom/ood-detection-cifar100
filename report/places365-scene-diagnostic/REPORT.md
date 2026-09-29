# Places365 scene acceptance heterogeneity

This post-result diagnostic partitions all 36,500 official Places365 validation images into 365 scenes of 100 images. The three existing CIFAR-100 models, six methods, both temperature variants and all four ID-validation thresholds are unchanged. Scene labels are used only to describe the saved decisions. No detector, model or threshold was selected with these scene results.

Each value below is a quantile of the 365 **scene-level three-seed mean OOD acceptance rates** at the fixed 95% ID-validation target, in percent. A lower acceptance rate means more images rejected as OOD. Full per-scene/per-seed counts for all four targets are in [scene_counts.csv](scene_counts.csv); the complete grid and exact seed aggregates are in [summary.json](summary.json).

## T1

| Method | Min | Q25 | Median | Q75 | Max | Overall OOD accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MSP | 60.33 | 77.00 | 82.67 | 87.67 | 96.67 | 82.42 |
| Energy | 62.67 | 78.67 | 84.00 | 89.00 | 98.00 | 83.43 |
| MaxLogit | 60.67 | 77.33 | 83.67 | 88.67 | 97.67 | 82.97 |
| Entropy | 58.67 | 76.67 | 83.00 | 88.00 | 97.33 | 82.18 |
| KNN-50 | 53.33 | 75.00 | 83.00 | 89.00 | 99.00 | 81.62 |
| Pooled Mahalanobis | 81.00 | 92.67 | 95.33 | 97.00 | 100.00 | 94.69 |

## calibrated

| Method | Min | Q25 | Median | Q75 | Max | Overall OOD accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MSP | 58.33 | 76.00 | 82.00 | 87.33 | 97.00 | 81.65 |
| Energy | 64.33 | 79.33 | 85.00 | 89.33 | 98.00 | 84.08 |
| MaxLogit | 60.67 | 77.33 | 83.67 | 88.67 | 97.67 | 82.97 |
| Entropy | 60.00 | 77.00 | 83.00 | 88.33 | 97.67 | 82.54 |
| KNN-50 | 53.33 | 75.00 | 83.00 | 89.00 | 99.00 | 81.62 |
| Pooled Mahalanobis | 81.00 | 92.67 | 95.33 | 97.00 | 100.00 | 94.69 |

## Interpretation

All 365 scenes and all 100 images per scene are retained. Because scenes have equal size, the average of the 365 scene acceptance rates equals the saved overall OOD acceptance rate for each seed; this identity is independently checked. Scene ranges describe this image collection after a 256→32 resize. They are neither confidence intervals nor evidence that scene semantics caused the differences. Comparisons remain post-result and do not update the completed [aggregate Places365 report](../places365-extension/REPORT.md).

The [fixed95 distribution figure](fixed95-scene-distribution.png) shows every method and both temperature variants; full numerical values and independent audit remain local.
