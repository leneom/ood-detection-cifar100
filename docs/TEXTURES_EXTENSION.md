# Predeclared Textures data extension

Declared 2026-09-27, before any Textures classifier inference or score inspection.

This adds a natural texture OOD test population to the independently trained CIFAR-100 study. It does not reproduce the full OpenOOD benchmark. Keep the original three-seed matrix unchanged; integrate this prepared cache only after its final evaluation and independent audit are complete. The already running SVHN queue must finish before a Textures GPU evaluation starts.

## Population and isolation

Use the Describable Textures Dataset (DTD) release r1.0.1 from the [Oxford dataset authors](https://www.robots.ox.ac.uk/~vgg/data/dtd/). Select exactly `dtd/labels/test1.txt`: official test split, partition 1, 1,880 images, 47 categories with 40 images each. The authors define disjoint train/validation/test splits; [torchvision's DTD interface](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.DTD.html) exposes the split and partition independently.

DTD train/validation images are not converted into evaluation samples. Verify that all three partition-1 lists are pairwise disjoint and together contain 5,640 unique paths. Preserve test-list order as the sample-index definition. Category labels and filenames are retained for provenance and annotations only; they do not fit the classifier, feature detectors, thresholds or temperature. No alternative DTD partition or transform is selected from OOD results.

## Source and transform

- Official archive: `https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz`.
- Archive length: 625,239,812 bytes, verified by the server's HTTP Content-Range before downloading.
- Published torchvision MD5, also verified against the installed torchvision 0.28.0 source: `fff73e5086ae6bdbea199a49dfb8a4c1`.
- Decode each test JPEG with Pillow and convert to RGB. No EXIF orientation transform is applied.
- Resize the shorter edge to 32 using Pillow bilinear resampling. The longer edge is `floor(32 * longer / shorter)`; preserve aspect ratio. Center-crop 32 by 32 with offsets `round((resized_dimension - 32) / 2)`, matching torchvision `Resize(32, BILINEAR)` followed by `CenterCrop(32)` on PIL input. There is no random augmentation or pixel normalization in the cache.
- Store contiguous NHWC uint8 images. A later evaluator applies the same CIFAR-100 normalization already specified in `PROTOCOL.md`.

Retain the source archive locally and read requested JPEG members directly from it; do not extract arbitrary archive paths. Save cache SHA-256, array-byte SHA-256, original split-list SHA-256, every selected JPEG SHA-256, original dimensions, class mapping, transformation and tool/package versions. Interrupted download ranges may be resumed, but a completed archive is accepted only after the full MD5 matches.

## Evaluation commitment

Later evaluate the same frozen ID-validation-selected checkpoints for all predeclared seeds 0, 1 and 2 and all six original detectors. Keep T=1, k=50, Mahalanobis shrinkage=0.01 and score directions unchanged. Detector fitting remains all 45,000 training ID features; operational thresholds remain ID-validation-only. Save raw logits, features and scores and report the 10,000 ID / 1,880 OOD counts and prevalence with AUPR. Report Textures separately; any expanded natural-OOD macro must explicitly list its dataset population and preserve the original matrix summary.

Data preparation is not an evaluation result. Do not claim Textures detection performance until the actual frozen-checkpoint runs and independent raw-score audits finish.
