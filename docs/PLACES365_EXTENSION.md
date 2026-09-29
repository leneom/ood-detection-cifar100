# Frozen Places365 validation population extension — 2026-09-29

This protocol is written after viewing the four original natural OOD populations, Tiny ImageNet, Fashion-MNIST and the class-holdout/RMD results, but **before downloading or decoding a Places365 image or scoring this population**. Places365 is a sequentially chosen additional scene-image OOD population, not an untouched method-selection test, a prospectively fixed multi-dataset benchmark, or an OpenOOD leaderboard reproduction. Preserve all earlier results unchanged.

## Source and fixed population

Use the official Places365-Standard **validation** images only, all 36,500 JPEGs (365 scene labels, 100 images each). Never use Places365 training or test images. The [Places365 development kit](https://github.com/zhoubolei/places_devkit#overview-of-places365-standard-data) documents the validation census, disjoint source splits and filename/label list. The [torchvision Places365 source](https://docs.pytorch.org/vision/main/_modules/torchvision/datasets/places365.html) gives the official archive URL and checksums:

- Small validation JPEG archive `http://data.csail.mit.edu/places/places365/val_256.tar`, required MD5 `e27b17d8d44f4af9a78502beb927f808`.
- Standard development kit `http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar`, required MD5 `35a0585fee1fa656440f3ab298f8479c`.
- Its `places365_val.txt` annotation file, required MD5 `e9f2fd57bfd9d07630173f4e8708e4b1`.
- `categories_places365.txt`, required MD5 `06c963b85866bd0649f97cb43dd16673`.

The same official-host HTTPS URLs may be used if HTTP redirects or is unavailable, provided archive MD5s and annotation/category MD5s match exactly. Save SHA-256, byte length, resolved URL and download logs. Do not choose an alternative mirror, split or subset after seeing outcomes. If official assets cannot be obtained intact, record the failure and leave this population unscored.

Require exactly the 36,500 unique official validation filenames and 365 labels with 100 rows each. Order rows lexicographically by official filename; save that order and labels. Reject missing/extra/duplicate annotations, unsafe tar paths, duplicate archive basenames, corrupt JPEGs and archive checksum mismatch. Preserve all validation rows with no class/image filtering. Scene labels are for integrity and possible later clearly labeled descriptive analysis only, never for fitting a detector or selecting a method. Compare preprocessed OOD pixels to the existing CIFAR-100 train/validation/test population for exact duplicates; reject any exact overlap and record it. Record within-OOD duplicate counts without silently dropping images.

Decode each JPEG with the installed Pillow RGB conversion and resize directly to 32×32 using `Image.Resampling.BILINEAR`, with no crop, flip or augmentation. Save contiguous uint8 NHWC `[36500,32,32,3]`, original dimensions, per-JPEG SHA-256, per-resized-image SHA-256, Pillow version, row order and full cache hash. The unchanged CIFAR-100 normalization is applied only by the frozen extractor. This 256-source-to-32 preprocessing can affect performance; it does not isolate semantic novelty from resolution/source change. Independently reconstruct the pixel cache from the verified archive and labels before model inference.

## Frozen models, scores and inference

Use only the completed three-way ID-validation-isolated CIFAR-100 ResNet-18 seeds 0/1/2 and their immutable ID-checkpoint-selected epochs 92/96/98. Reuse each seed's frozen 45,000-row training-ID feature bank, 10,000-row ID-test logits/scores, ID-only fitted temperature (1,500 separate validation rows) and four ID-only thresholds (1,500 other validation rows). Rehash the existing completion and per-seed checkpoint/extraction/scoring/audit artifacts before Places365 inference. Never retrain, change a checkpoint, fit a parameter on Places365, or use any OOD score to select a method.

Score the original six methods only: MSP, Energy, MaxLogit, Entropy, normalized KNN-50 and pooled Mahalanobis with fixed 1% shrinkage. Retain T=1 and existing ID-fitted-temperature variants; MaxLogit/KNN/Mahalanobis remain invariant. Keep the 80/90/95/99% ID-validation target thresholds and inclusive-tie rule unchanged. One guarded GPU inference process at a time. Save every OOD logit, 512-D feature, model prediction, source filename/label, raw score and acceptance mask; also retain exact ID score, checkpoint, model/source, training-feature and cache hashes.

## Fixed analysis and limits

Compare the same 10,000 CIFAR-100 ID-test examples with all 36,500 Places365 validation images; OOD prevalence is 36,500/46,500, approximately 78.49%. For each of three seeds × six methods × two variants, compute AUROC with half-credit ties, non-interpolated AUPR-out/in, benchmark FPR95-ID from the ID **test** order statistic, and separate OOD acceptance plus actual ID-test acceptance for all four frozen ID-validation thresholds. Report all seed points and mean/sample SD, counts/prevalence and ties. Do not choose a winner or merge this sequential population into the earlier four-population macro without a new prospective aggregation protocol.

Independently audit archive/annotation/pixel order, selected checkpoints and training-ID fit provenance, all raw score formulas and invariant controls, per-row thresholds/masks, metrics and displayed report values. Inspect figures and run regression checks before declaring complete. This one additional population does not prove universal OOD generalization, new method validity, or scene-semantic causality; scene/object overlap and the 256→32 image-source transformation remain material limits. Keep archive, cache, checkpoints, raw outputs and reports local. No upload, publication or external message.
