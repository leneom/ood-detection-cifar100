# Additional class partitions: local runbook

Frozen scientific plan: [CLASS_HOLDOUT_REPLICATION.md](CLASS_HOLDOUT_REPLICATION.md). This page records commands and audit order, not new method choices. All commands run from the `ood_detection` root in Windows PowerShell with `C:\Users\zxzx\anaconda3\envs\ece488_clip\python.exe` unless stated otherwise.

## Current schedule

The single locked controller `tools/run_class_holdout_replication_training.py` was launched hidden once on 2026-09-28. It trains split 2032 seeds 0, 1, 2, then split 2033 seeds 0, 1, 2, each for the complete 100-epoch schedule. Before **each** model it samples the GPU three times two seconds apart and proceeds only when all samples meet the original low-load guard. Its root state is `runs/cifar100-class-holdout-replication/status.json`; stdout/stderr are `controller.stdout.log` and `controller.stderr.log`, and each model has separate logs and full resume artifacts inside its split. Do not start another controller while its Python PID or a trainer PID is live. If it defers for a busy GPU, inspect process exit/status and relaunch the same controller only after confirming no project job remains and the frozen sources still match. If it fails, preserve partial artifacts and investigate before any resume. Never restart with a changed trainer source or partition.

## Train-only audits, one completed model at a time

After the selected `final-best.pt`, `final-selection.json`, all 100 history rows and `status.json: complete` exist for a model, run the independent training audit with CUDA hidden. For example, for split 2032 seed 0:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
& 'C:\Users\zxzx\anaconda3\envs\ece488_clip\python.exe' -X utf8 tools/audit_class_holdout_replication_training.py --class-seed 2032 --seed 0
& 'C:\Users\zxzx\anaconda3\envs\ece488_clip\python.exe' -X utf8 tools/audit_class_holdout_replication_checkpoint_cpu.py --class-seed 2032 --seed 0
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

The first audit checks the exact 100-epoch trace, learning-rate milestones, first maximum on the 1,600-row checkpoint-ID role, all original row indices/class labels, source snapshots, full resume state and checkpoint hash. The second independently recomputes the frozen checkpoint's full 1,600-row ID-role logits on CPU and saves them. A CPU/CUDA borderline argmax difference occurred in the original split, so a CPU discrepancy is preserved as `cpu_cuda_discrepancy_requires_cuda_audit`, never silently treated as a failed or different ID-only checkpoint selection.

After **all six** schedules finish and the controller exits, a discrepant model may be checked on the original CUDA channels-last inference path under the same three-sample idle guard:

```powershell
& 'C:\Users\zxzx\anaconda3\envs\ece488_clip\python.exe' -X utf8 tools/audit_class_holdout_replication_checkpoint_cuda.py --class-seed 2032 --seed 0
```

The CUDA audit must reproduce the training-selected ID-role correct count exactly; it saves CUDA logits and all CPU/CUDA differing original training rows. It is only applicable when the CPU audit explicitly flagged a discrepancy. All these audits use CIFAR-100 **training-file ID role only**, never official ID test or OOD test. They are write-once; do not overwrite a failed or passing record.

## Test boundary

Wait for the full three-model schedule and audits for a split before its official test inference. The original 2031 split artifacts and detector settings are frozen. The next stage must save all raw logits, 512-D features, original test rows/labels, train-ID feature banks, and six fixed scores for every split/seed. Fit temperature and thresholds on the separate 1,200/1,200 ID roles only. Evaluate all 8,000 ID and 2,000 OOD official test examples. No OOD result may alter class partitions, checkpoint selection, fitting, thresholds or method list. A partial training seed is not a benchmark result.

## Controller-only resume amendment, 2026-09-28

After split2032 seed0 completed 100 epochs, one of the three idle-guard GPU samples had 29% memory utilization, so the original controller correctly deferred seed1 and exited. Its first resume then failed *before launching any new trainer*: the original controller wrote integer class-seed dictionary keys to JSON, but on resume compared the loaded string keys directly to the in-memory integer keys. The failed status, original controller source and immutable training launch were preserved in `controller-resume1-failure.json`, `controller-original-source.py` and `training-launch.json`. `controller-amendment.json` binds their hashes, the revised controller hash, both audited partitions and unchanged trainer-source hashes. The only controller code change canonicalizes class-seed keys when checking the existing launch and requires that amendment. No model, optimizer, partition, protocol, checkpoint or trainer source was edited. The amended controller resumed split2032 seed1 under the original three-sample GPU guard.

The independent CPU inference of split2032 seed0 gives 1,227/1,600 correct, versus 1,229/1,600 recorded on the training CUDA path. Both ID-only results and all CPU logits are preserved. This is **not** a passed cross-device checkpoint audit yet. Wait until all six schedules finish and the controller exits, then run the guarded CUDA-path audit to reproduce the selected 1,229/1,600 exactly or investigate further. Do not use any official test data for this resolution.

## Evaluation stages after the six-model controller exits

Use the verified Windows CUDA Python for the source-bound GPU extractor, one model at a time and only after all six schedules, all training audits, all consolidated ID checkpoint audits, and the independent 8,000/2,000 test-population audits pass. The extractor itself repeats the three-sample low-load GPU guard and reuses the unchanged original extraction/ID-only feature-fit implementation:

```powershell
& 'C:\Users\zxzx\anaconda3\envs\ece488_clip\python.exe' -X utf8 tools/extract_class_holdout_replication.py --class-seed 2032 --seed 0
```

The independent full raw-pixel/ID feature-bank audit and subsequent temperature/threshold scoring need SciPy/sklearn from base Anaconda, so use `C:\Users\zxzx\anaconda3\python.exe` for these CPU-only commands:

```powershell
& 'C:\Users\zxzx\anaconda3\python.exe' -X utf8 tools/audit_class_holdout_replication_features.py --class-seed 2032 --seed 0
& 'C:\Users\zxzx\anaconda3\python.exe' -X utf8 tools/score_class_holdout_replication.py --class-seed 2032 --seed 0
& 'C:\Users\zxzx\anaconda3\python.exe' -X utf8 tools/audit_class_holdout_replication_scores.py --class-seed 2032 --seed 0
```

These adapters invoke the unchanged original six-method extractor/scorer and independent auditors with source-bound class context. The feature auditor uses a declared per-split compatibility view containing only the official training-file hash it expects; this does not replace the frozen root replication declaration. Check every output manifest and write-once audit before proceeding to a three-partition summary. The scripts must not be run on an incomplete schedule or while a GPU trainer is active.

## Frozen nine-model report after all independent audits

The report specification was frozen in `docs/CLASS_HOLDOUT_REPLICATION_REPORT.md` and the write-once `runs/cifar100-class-holdout-replication/report-declaration.json` before either new split had an OOD score. For each new split, first prepare and independently audit all 10,000 official CIFAR-100 test rows **after** its full three-model train/ID gate:

```powershell
& 'C:\Users\zxzx\anaconda3\python.exe' -X utf8 tools/prepare_class_holdout_replication_test.py --class-seed 2032
& 'C:\Users\zxzx\anaconda3\python.exe' -X utf8 tools/audit_class_holdout_replication_test.py --class-seed 2032
```

Repeat for split 2033, then use the one-model-at-a-time GPU extraction and base-Anaconda CPU score audit commands above for every new model seed 0/1/2. Once all six new score audits pass, run:

```powershell
& 'C:\Users\zxzx\anaconda3\python.exe' -X utf8 tools/build_class_holdout_replication_report.py
& 'C:\Users\zxzx\anaconda3\python.exe' -X utf8 tools/audit_class_holdout_replication_report.py
& 'C:\Users\zxzx\anaconda3\python.exe' -X utf8 tools/verify_class_holdout_replication_completion.py --write
```

The builder independently gates the old 160-file baseline and all six new source-bound model/partition/test/score audits. It saves all nine points, each split's three-seed mean/sample SD, and the grand mean/sample SD of the three split means for all 48 ranking and 144 operating rows. The second script numerically rebuilds every reported point, CSV field and displayed all-method row without using the builder's statistics implementation. The report is not complete until this audit passes. No OOD result can select a method, checkpoint, temperature or threshold.

The final write-once completion command can be rerun without `--write` to rehash the original 160-file phase, all six new checkpoints and audit chains, every saved extraction/scoring output and source snapshot, and the independent report outputs. A premature completion call must refuse without writing a report.

## Optional all-method scientific figure

The full-grid report remains the numerical source of truth. A separate display specification was frozen before the two new partitions had OOD test scores in `docs/CLASS_HOLDOUT_REPLICATION_FIGURES.md`. Only **after** the nine-model report's independent audit and write-once completion gate pass, base Anaconda can render one 3×2 all-method figure (AUROC, actual ID acceptance and OOD acceptance at the same fixed95 ID-validation target; both temperature variants):

```powershell
& 'C:\Users\zxzx\anaconda3\python.exe' -X utf8 tools/plot_class_holdout_replication.py
& 'C:\Users\zxzx\anaconda3\python.exe' -X utf8 tools/audit_class_holdout_replication_figures.py
```

It refuses an incomplete report, saves separate PNG/PDF/figure-data and source hashes in `report/cifar100-class-holdout-replication-figures/`, and its independent numeric/file auditor checks all 108 figure-data rows, 324 seed values, hashes and PNG/PDF dimensions. Rendered-image inspection is still required before calling it a completed presentation artifact. No plot or OOD figure result exists yet.


## Completion record, 2026-09-29

The six fixed new 100-epoch schedules, all ID-only checkpoint audits, both official 8,000-ID/2,000-OOD test-population audits, and all six independent feature/score chains passed. The conditional split2032 seed0 CUDA ID-only audit reproduced 1,229/1,600 correct and retained both CUDA and CPU logits plus the two differing prediction rows. The frozen nine-model report passed its independent audit and the final 351-file completion verifier. The earlier "pending" statements above document the state at the time they were written and are superseded by this completion record.

Numerical results: `report/cifar100-class-holdout-replication/SUMMARY.md`, `metrics.csv`, `operating.csv`, and `summary.json`. Source/raw completion: `report/cifar100-class-holdout-replication/completion.json`. The visually reviewed optional presentation is the versioned `report/cifar100-class-holdout-replication-figures-v2/all-methods.png` and PDF, with separate numeric audit and manual visual-review records. The first figure rendering is retained as the audited pre-layout-correction artifact. Recheck the 351-file completion chain read-only with:

```powershell
& 'C:\Users\zxzx\anaconda3\python.exe' -X utf8 tools/verify_class_holdout_replication_completion.py
```

This is descriptive fixed-partition sensitivity; partition 2031 was viewed earlier, and the three fixed partitions do not form a confidence interval or a new untouched confirmation.
