# Frozen Fashion-MNIST population extension — 2026-09-28

This declaration is written after the Tiny ImageNet aggregate and class-diagnostic results were seen, but before downloading, decoding, extracting, or scoring any Fashion-MNIST image. This is a **sequentially chosen additional real-image OOD population**, not a prospectively fixed multi-dataset benchmark, a new method validation, or an OpenOOD leaderboard reproduction. No earlier OOD test may be used to change the six methods, classifiers, ID fits or operating targets.

## Fixed source and population

Use only the official [Zalando Research Fashion-MNIST repository](https://github.com/zalandoresearch/fashion-mnist) test files from `data/fashion`, in the original IDX order:

- `https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz`, expected compressed MD5 `bef4ecab320f06d8554ea6380940ec79`.
- `https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz`, expected compressed MD5 `bb300cfdad3c16e7a12a480ee83cd310`.

The project's [official README](https://github.com/zalandoresearch/fashion-mnist#Get-the-Data) describes 10,000 test images, 28×28 grayscale, ten clothing/accessory categories and the above checksums. Download only those two test files; do not access its training images. Require exact gzip/IDX magic numbers, 10,000 rows, 28×28 uint8 images, labels 0–9 and 1,000 rows per label; reject errors rather than filtering or replacing rows. Save archive MD5 and SHA-256, decompressed IDX hashes, label counts and test-row indices. Check that no preprocessed 32×32 RGB row is an exact pixel duplicate of any CIFAR-100 train/test row; this is an integrity check, not a semantic-overlap proof.

## Frozen preprocessing and model inputs

Exactly reuse the existing MNIST evaluation transform in `src/ood/datasets.py`: float32 bilinear interpolation from 28×28 to 32×32 with `align_corners=False`, then round to nearest uint8 and clamp to [0,255], repeat the one gray channel three times, store contiguous uint8 NHWC [10000,32,32,3]. Pass it to unchanged `src/ood/data.py::preprocess` for CIFAR-100 normalization. Keep original official row order. No OOD-specific crop, augmentation, contrast inversion, colorization, normalization or threshold. Independently reconstruct every preprocessed pixel before model inference.

Use only the completed isolated CIFAR-100 seeds 0/1/2, selected epochs 92/96/98, their already frozen 45,000-row training-ID feature banks, 10,000 ID-test scores, 1,500-row ID temperature fit and disjoint 1,500-row ID threshold fit. Confirm all checkpoints/ID fits and prior 70-file Tiny ImageNet completion still hash-verify. Compute the original six methods MSP, Energy, MaxLogit, Entropy, normalized KNN-50 and pooled Mahalanobis (1% shrinkage), T=1 and already ID-fitted temperature; the latter three methods are invariant. Save raw logits, features, predictions, labels, row indices, every score and all 80/90/95/99 acceptance masks. One GPU inference process at a time; no new training, checkpoint selection or OOD-dependent detector fit.

## Fixed analysis and limits

For each of the three seeds × six methods × two variants, compare all 10,000 Fashion-MNIST test images to the same 10,000 CIFAR-100 ID-test images. Report AUROC with half-credit ties, non-interpolated AUPR-out/in at 50% OOD prevalence, benchmark FPR95-ID and achieved ID-test acceptance, and OOD acceptance plus actual ID-test acceptance under every frozen 80/90/95/99 ID-validation threshold. Preserve three individual seed points, mean and sample SD. Retain Fashion-MNIST label counts for integrity only; no class-wise method choice or category filtering. Independently audit archive/pixels, training-ID feature probes, logit scores, all metrics/threshold masks and displayed values; run full regression and inspect plots before reporting completion.

Fashion-MNIST has a strong grayscale, low-resolution and product-photo source difference from CIFAR-100. Good or poor detection here cannot isolate semantic novelty, demonstrate performance on natural color photographs, or be combined with the previously fixed four-population macro without a separate prospective aggregate protocol. Do not choose a method, parameter or future dataset based on this test. Keep all data, raw outputs, plots and logs local; no publication or external communication.
