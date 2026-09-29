# Tiny ImageNet confirmation amendment 1 — layout correction before decoding

The frozen protocol's phrase "convert to uint8 CHW" is a layout error. Static inspection of the unchanged `src/ood/data.py:preprocess` shows that every existing CIFAR/OOD loader supplies uint8 **NHWC**, and `preprocess` performs the NHWC→NCHW permutation before normalization. No Tiny ImageNet image has been decoded, extracted or scored when this amendment is frozen; the archive download may be in progress.

For this check, after Pillow RGB and fixed 64→32 bilinear resizing, store a contiguous uint8 array of shape **[10000, 32, 32, 3] (NHWC)**. Pass it directly to the unchanged `src.ood.features.extract`/CIFAR-100 `preprocess`. Do not transpose twice or alter numeric pixel values. All population, source checksum, ordering, fitting, method, threshold, metric and reporting choices in the original declaration remain unchanged.

Original declaration SHA-256: `9aa9724741dd3ea00fc2dd3b43f0945637b2757799fb667db22f5f9e6cdb86a5`. The amendment must be hash-bound before the first image decode.

