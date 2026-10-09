# Fashion-MNIST local runbook after GPU contention

The frozen [population protocol](FASHIONMNIST_EXTENSION.md) and its declaration in `runs/fashionmnist-extension/declaration.json` predate dataset access. Both official test gzip files match the publisher's MD5. `cache-manifest.json` and `cache-audit.json` verify the complete 10,000-row 28→32 RGB cache and zero exact CIFAR-100 pixel duplicates. No Fashion-MNIST classifier score or complete extraction exists yet.

The first controller `tools/run_fashionmnist_extraction.py` was launched while an unrelated GPU workload measured 99% utilization and about 11.5 GiB used. Its seed-0 child was stopped, and `extraction-status.json` records `extraction_failed_preserve_partial`. The `extract-seed0` directory contains only an `extracting` status and frozen source snapshots; do not treat it as a model result. The first launch, logs and partial folder must be preserved. Do not rerun the original controller, overwrite a result, delete the partial folder, or start seed 1/2 directly.

Use only `tools/run_fashionmnist_extraction_recovery.py` with the verified CUDA Python after checking live Python/GPU state. It first checks three GPU samples two seconds apart, each requiring <=4,096 MiB used and <=15% utilization; if any sample exceeds either limit it writes `recovery-status.json` as `deferred_gpu_busy_no_mutation` and exits without modifying the partial folder or launching inference. When idle, it verifies source hashes, moves the exact incomplete seed-0 folder to `interrupted-attempt-1` inside the project, copies first-attempt logs/status, then runs seeds 0/1/2 sequentially. It checks GPU again before each missing seed. Completed seed folders are verified on subsequent invocations; a new partial or source mismatch stops for review.

Example command (only after a fresh state check):

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
python tools/run_fashionmnist_extraction_recovery.py
```

The initial recovery preflight returned `deferred_gpu_busy_no_mutation` on three samples of 11.5 GiB / 98–99% utilization; no retry extraction began. Once all three seeds finish, independently audit the 45k training-ID KNN/Mahalanobis probes, then score only with saved ID temperatures/thresholds. Keep all raw logits/features/scores/masks, audit and report evidence local. Fashion-MNIST is a sequential extension, not part of the previously frozen four-population macro.

CPU-only continuation prepared while the GPU remained busy: `tools/audit_fashionmnist_features.py`, `tools/score_fashionmnist.py`, `tools/audit_fashionmnist_scores.py`, `tools/build_fashionmnist_report.py`, and `tools/audit_fashionmnist_report.py` compile but have not run on Fashion outputs. After `recovery-status.json` confirms all three extractions and each `extract-seedN/status.json` and manifest/hash validates, run the feature audit for each seed using the base CPU Python, then score and independently audit each seed in sequence. Only after all three score audits pass may the report builder and report auditor run. Keep `CUDA_VISIBLE_DEVICES=-1` for CPU stages. The Fashion report directory must not be created from a partial seed or interim metric.

Final gate after three audited score seeds: build the report, run the independent report audit, visually inspect both actual PNGs and save figure-inspection.json with each inspected PNG SHA-256, and run the full 91-test regression to report/fashionmnist-extension/regression.log. Only then run tools/verify_fashionmnist_completion.py --write; subsequent invocations without --write verify all 76 hash-bound inputs read-only. An incomplete or changed artifact must fail; do not create completion.json from an interim Fashion result.

Operational guard amendment (2026-09-28 15:08 local, before any successful Fashion extraction): the original 4,096 MiB / 15% GPU-utilization guard correctly deferred while WDDM desktop/browser graphics stayed at 27–48% despite the game exiting. With no experiment/game process, 2.3 GiB allocated, 2–3% memory utilization and ~29–30 W draw, the 15% cutoff was an overrestrictive desktop-idle proxy. The guarded recovery controller now requires three consecutive samples, two seconds apart, each satisfying used VRAM <=4,096 MiB, total GPU utilization <=55%, memory utilization <=10%, and power draw <=45 W; it repeats all four conditions before each missing seed. The first modified read-only triple was 2,320 MiB, 30/31/35% GPU, 2/3/2% memory, and 29–30 W, passing. This changes only scheduling safety, not the frozen classifier, dataset, method, temperature, threshold, or metric. Any failure still preserves partial evidence; never launch the original controller.
