# OOD Detection Study

**最新进展与证据边界：** [2026-09-29 项目状态](docs/CURRENT_STATE.md)。旧项目评估报告的部分待办已经完成，后续以此日期化索引和 [PROGRESS](docs/PROGRESS.md) 为准。

独立的 Out-of-Distribution Detection 研究项目。第一阶段的问题是：**固定同一个 ID classifier，仅替换 post-hoc score，MSP、Energy、MaxLogit 对自然图像 OOD 的表现如何？**

当前进度：原四个自然 OOD 测试人口上的完整基准和后续诊断、三路 ID validation 隔离复验均已完成；这些已看过的 OOD 测试仍属探索性。新增的 [Tiny ImageNet-200 冻结检验](report/tinyimagenet-confirmation/REPORT.md) 使用事先声明的完整 10,000 张官方 validation、三个已冻结模型和只由 ID 拟合的温度/阈值。全部六个主要 detector × T=1/校准版本、四个接受目标均保存三 seed 结果和原始逐图输出，并通过独立复算与 91 项回归（89 通过、2 项已知跳过）。这是单个新 OOD 人口，不是完整 OpenOOD 复现或普适泛化结论。见 [机器可读结果](report/tinyimagenet-confirmation/summary.json)、[完成证据](report/tinyimagenet-confirmation/completion.json)、[实时进展](docs/PROGRESS.md) 和 [中文学习指南](docs/LEARNING_GUIDE.zh-CN.md)。

[Tiny ImageNet 类别异质性诊断](report/tinyimagenet-class-heterogeneity/REPORT.md)也已完成：全部 200 个官方类别、三 seed、六方法和两温度版本的逐类别 AUROC 与四目标接受计数均已保存并独立审计。这是同一已评分 OOD test 的事后描述，不增加独立确认，也不用于挑选方法。

额外的 [Fashion-MNIST 冻结人口扩展](report/fashionmnist-extension/REPORT.md)现已完成官方 test 全 10,000 张、三 seed、六方法及 T=1/ID 温度版本的评分；原始 logits/features/scores/masks、独立审计、图表和 [76 文件完成证据](report/fashionmnist-extension/completion.json)均保存在本地。该数据集是在查看 Tiny ImageNet 结果后顺序选定，不并入原四人口宏平均；它的灰度/来源差异也限制语义解释。首次 GPU 拥堵中断与安全恢复见 [运行说明](docs/FASHIONMNIST_RUNBOOK.md)。

[CIFAR-100 同源 held-out-class 三 seed 研究](report/cifar100-class-holdout/SUMMARY.md)现已完成。[协议](docs/CIFAR100_CLASS_HOLDOUT.md)在划分类别前冻结：80 个 ID / 20 个训练排除的 OOD fine class、36k 训练 ID，以及 1,600/1,200/1,200 互斥的 checkpoint/温度/阈值 ID 验证角色。三颗新 80 类 ResNet-18 均跑满 100 epochs；官方 test 的 8,000 张 ID 和 2,000 张 held-out-class OOD 全数评分。六方法、T=1/ID 校准、四目标的原始 logits/features/scores/masks 与独立审计均保存在本地；[机器可读三 seed 结果](report/cifar100-class-holdout/summary.json)及[160 文件完成校验](report/cifar100-class-holdout/completion.json)可复核。ID-test accuracy 为 76.96±0.38%；T=1 Energy 的 OOD AUROC 为 73.57±0.73%（均值±训练 seed 样本 SD）。这与原 100 类研究更换了分类器和 ID 标签空间，不能把跨研究差异单独归因于同源图像；OOD test 未用于方法选择。Seed 2 的 CPU/CUDA checkpoint 验证出现单张边界样本差异，原 CUDA 选择结果完全复现，双路径 raw logits 和[运行记录](docs/CLASS_HOLDOUT_RUNBOOK.md)均已保存。

[同源留出类别异质性诊断](report/cifar100-class-holdout-classes/REPORT.md)也已完成：全部 20 个 held-out fine class 各 100 张官方 test 图，三 seed、六方法、两温度和四个固定 ID 接受目标的 720 个逐类 AUROC、2,880 个接受计数均有[原始数组与完整 CSV](report/cifar100-class-holdout-classes/class_values.npz)，经独立复算与[36 文件完成校验](report/cifar100-class-holdout-classes/completion.json)。它描述同一已看过的 OOD test，不增加独立确认，也不用于选择方法。

[同源留出类的配对图像 bootstrap](report/cifar100-class-holdout-bootstrap/REPORT.md)已完成：按已观察的 80 个 ID/20 个 OOD 类各自重复抽样，1,000 次都保持 8,000:2,000 test 比例，同一抽样权重配对用于全部三 seed、两温度和六方法。原始权重/指标、72 行[机器可读统计](report/cifar100-class-holdout-bootstrap/statistics.csv)、全量独立数值审计与[67 文件完成校验](report/cifar100-class-holdout-bootstrap/completion.json)保存在本地。T=1 Energy AUROC 原三训练 seed 为 73.57±0.73%，固定三模型均值的图像抽样描述性 2.5–97.5% 分位数为 72.58–74.53%；这仅针对已观察类别和模型，不是新类别或训练 seed 的置信区间，也不用于选方法。

[冻结 80 类模型的 CIFAR-10 扩展](report/class-holdout-cifar10-extension/REPORT.md)已完成：原三颗 100-epoch 选中 checkpoint、36k 训练 ID 特征库、ID 温度和四组 ID 阈值原样复用；全部 10,000 张官方 CIFAR-10 test 对照同一 8,000 张 ID test。六方法、T=1/校准、四阈值的原始 logits/features/scores/masks、独立审计和[125 文件完成证据](report/class-holdout-cifar10-extension/completion.json)均在本地。T=1 Energy 的 CIFAR-10 AUROC 为 76.49±0.58%，固定 95% ID-validation 目标的 OOD 接受率为 85.38±0.59%（三训练 seed 均值±样本 SD）。CIFAR-10 在项目早期已经使用过，且这项扩展是在查看同源留出类结果后选定，只属探索性；其 OOD 占比 55.56% 与留出类的 20% 不同，AUPR 不可直接横向解释。

[固定 20% OOD 比例的 AUPR 对照](report/class-holdout-prior-standardization/REPORT.md)已完成：不丢弃任何测试图或重算分数，只按每类人口赋予固定权重，对全部三 seed、六方法、两温度的 AUPR-out/in 使用相同的假设性 20% OOD 比例。144 个值都通过独立加权 scikit-learn 复算；原留出类的 20% 人口严格复现原 AUPR，[42 文件完成证据](report/class-holdout-prior-standardization/completion.json)只读通过。T=1 Energy 的 CIFAR-10 AUPR-out 从原 55.56% OOD 比例下的 76.26±0.43% 变为 20% 比例下的 39.94±0.59%；对应留出类为 37.38±1.00%。20% 只是展示约定，不是部署先验，也不消除数据集/语义差异或早期测试暴露。

## 实验设计

- ID：CIFAR-100。固定 `split_seed=2026`，每类 450 张训练、50 张 validation；官方 10,000 张 test 只用于评估。
- 原四个自然 OOD：CIFAR-10 官方 test 10,000 张；MNIST test 10,000 张（双线性 28→32、重复 RGB）；SVHN test 26,032 张；DTD 官方 test partition 1 共 1,880 张。它们此前已用于方法开发，仍属探索性。
- 新增冻结检验人口：Tiny ImageNet-200 官方 validation 全 10,000 张，64→32 双线性 RGB；只复用已冻结的三 seed 模型、ID 温度及阈值，没有 OOD 拟合或选择。
- 模型：适配 32×32 输入的 ResNet-18，从零开始、仅 ID cross-entropy 训练。旧 continual-learning 权重不作为本项目标准分类器。
- 优化：SGD，lr 0.1，momentum 0.9，weight decay 5e-4，batch 128；100 epoch 固定日程，50/75/90 epoch 后 lr ×0.1。
- checkpoint：只按 ID validation accuracy 选择；温度默认 T=1。所有方法共享同一个 checkpoint 和输入归一化。
- score：所有分数越大越像 OOD。MSP=`1-max softmax(z/T)`；Energy=`-T logsumexp(z/T)`；MaxLogit=`-max z`；Entropy 是额外诊断基线。
- 特征基线：normalized KNN（k=50）与 final-layer Mahalanobis（固定 1% shrinkage），只拟合 45,000 张训练 ID 的特征；不包含原论文全部扩展。
- 指标：AUROC、AUPR-out、AUPR-in、FPR95-ID。ID validation 的 95% 接受阈值及其 test 接受率另外报告。
- 精确断点续训：保存 optimizer、scheduler、GradScaler 和 Python/NumPy/CPU/CUDA RNG；校验配置、划分和训练源码 hash。

本协议是独立的 CIFAR 实验，**并非 OpenOOD v1.5 的完整复现**：缺少完整数据列表、预训练权重与其它标准化设置。CIFAR-10 的 near-OOD 归类参考 [OpenOOD 官方基准](https://github.com/Jingkang50/OpenOOD/wiki/OpenOOD-v1.5-methods-%26-benchmarks-overview)。

## 运行

在本目录运行。当前已验证 Windows `ece488_clip` 的 PyTorch CUDA 环境；不需要修改旧环境或旧项目文件。

```powershell
$py = 'C:\Users\zxzx\anaconda3\envs\ece488_clip\python.exe'
& $py -m unittest discover -s tests -v
& $py train.py --dataset synthetic --model tiny --epochs 3 --amp --output-dir runs/smoke-gpu
& $py evaluate.py --checkpoint runs/smoke-gpu/best.pt --ood noise constant --output-dir runs/smoke-gpu/evaluation

# --stop-after 可暂停，100 epoch 日程保持不变
& $py -u train.py --download --amp --epochs 100 --stop-after 10 --seed 0 --output-dir runs/cifar100-seed0
& $py -u train.py --amp --epochs 100 --seed 0 --output-dir runs/cifar100-seed0 --resume
& $py -u evaluate.py --checkpoint runs/cifar100-seed0/best.pt --ood cifar10 noise --download --output-dir runs/cifar100-seed0/eval-final
```

`--download` 首先复用 `../data/cifar-100-python.tar.gz`，由 torchvision 校验 archive/extracted file 的 MD5，再解压到本项目的 `data/`。CIFAR-10 从 torchvision 官方数据源下载。全部数据、checkpoint、实验原始输出默认仅保存在本地。

## 文件

`src/ood/`：数据、模型、训练、score、metrics、评估；`tests/`：指标和实验协议测试；`docs/PROTOCOL.md`：指标定义、证据边界；`docs/PROGRESS.md`：进展和下一步；`run_matrix.py`：顺序续训/评估/报告控制器；`tools/build_report.py`：科学曲线与 bootstrap；`tools/aggregate_results.py`：完整训练多 seed 统计；`runs/`：config、environment、split、history、checkpoint、原始 logits/scores、results。

## 阅读

- [Hendrycks & Gimpel, MSP baseline (ICLR 2017)](https://arxiv.org/abs/1610.02136)
- [Liu et al., Energy-based OOD Detection (NeurIPS 2020)](https://arxiv.org/abs/2010.03759) / [作者代码](https://github.com/weitliu/energy_ood)
- [OpenOOD 官方代码](https://github.com/Jingkang50/OpenOOD)

先建立可信基线和多 seed 证据，再讨论方法改进；初期 pilot 不支持论文级结论。

## 后台完整实验

原始控制器已经完成 seeds 0/1/2 各 100 epochs，以及 CIFAR-10、MNIST、noise 的评估、报告和汇总。下列命令保留用于复现/恢复；现有完成结果会被复用，不需要重新启动。实时状态记录保留在 `runs/matrix/status.json`。

```powershell
& $py -u run_matrix.py --amp --seeds 0 1 2 --plot-python C:\Users\zxzx\anaconda3\python.exe
# 对已有评估生成可重建的报告（使用带 NumPy/Matplotlib 的 Python）
& C:\Users\zxzx\anaconda3\python.exe tools/build_report.py --evaluation runs/cifar100-seed0/eval-pilot-all --output-dir report/pilot-seed0-all --bootstrap 300 --label '10-epoch pilot, seed 0'
```

本任务已设置每 30 分钟的续跑。它会检查训练、推进可执行工作并更新进展；电脑和 Codex 必须保持运行。用户的新指令可以随时调整方向或停止。

## SVHN 与失败分析

GPU 环境无需安装 SciPy。使用已有 CPU 环境下载/校验并转换 SVHN test；缓存保留完整 26,032 张原始 RGB 32×32 图片、官方 MD5 和数组 SHA-256。缓存已与官方 MAT 逐元素核验，GPU 环境以 CPU 模式读取通过。后台扩展队列已完成三 seed 的完整 SVHN 评估并退出；状态见 `runs/svhn-extension/status.json`。当前三 seed 基准的 OOD 列表保持不变。

```powershell
& C:\Users\zxzx\anaconda3\python.exe tools/prepare_svhn.py
& C:\Users\zxzx\anaconda3\python.exe tools/analyze_failures.py --evaluation runs/cifar100-seed0/eval-pilot-all --output-dir report/pilot-failure-analysis
```

见 [完整 seed 0 失败分析](report/final-seed0-failure-analysis/REPORT.md) 和 [pilot 失败分析](report/pilot-failure-analysis/REPORT.md)。所有拒绝/接受统计使用已经保存的 ID validation 阈值，不重新选择测试阈值。

```powershell
# 下列队列已完成并退出；命令仅用于明确需要的复现/恢复
& $py -u tools/run_extension.py --analysis-python C:\Users\zxzx\anaconda3\python.exe
# 从原始文件独立审计完整训练结果（仅 CPU）
& C:\Users\zxzx\anaconda3\python.exe tools/audit_evaluation.py --evaluation runs/cifar100-seed0/eval-final --output report/final-seed0/independent-audit.json
```

扩展控制器拒绝未完成/暂停/失败的主实验、不同 seed 集合或不同 checkpoint/源码/环境；校验 ID logits/features 与分数复算差异，并保存 `baseline-comparison.json`。它使用与主评估相同的 ID validation 校准规则。`runs/svhn-extension/STOP` 可暂停扩展；各实验原始结果保持本地。

## 真实失败样本图

[完整 seed 0 本地图册](runs/cifar100-seed0/failure-gallery/REPORT.md) 展示六种方法共 108 个极端样本，分别属于正确 ID 被拒绝、错误 ID 被拒绝、CIFAR-10 OOD 被接受。所有样本保留官方 test index 与固定 validation 阈值，排序/标签/接受决定已经独立核验。图册用于解释失败，整体表现仍以全部 test 样本的指标为准；原图和图册仅保存在忽略的 `runs/`。

```powershell
& C:\Users\zxzx\anaconda3\python.exe tools/build_failure_gallery.py --evaluation runs/cifar100-seed0/eval-final --output-dir runs/cifar100-seed0/failure-gallery
```

## 完整三 seed 汇总审计

CPU 独立审计队列已完成主实验与 SVHN 的三 seed 原始结果/汇总核验并退出。状态见 `runs/audit-queue/status.json`，PID 见 `runs/audit-queue.pid`；已有报告目录保留，重跑会使用时间戳目录。四项队列安全检查、两阶段 CPU smoke 和全部 31 项测试通过。下面保留手动恢复命令，运行前先确认队列已退出，避免重复审计。它重新核验每个 seed 的原始指标、checkpoint/源码/划分和完整训练日程，要求所有预定 seeds 恰好出现一次；以 Python statistics 复算 mean/sample SD，并保存同一 classifier 上各方法相对 MSP 的配对差值。配对差值的训练 seed SD 与单模型 bootstrap CI 分开报告，不作显著性或测试集方法选择声明。

```powershell
& C:\Users\zxzx\anaconda3\python.exe tools/audit_matrix.py --evaluations runs/cifar100-seed0/eval-final runs/cifar100-seed1/eval-final runs/cifar100-seed2/eval-final --summary runs/matrix/summary/summary.json --output-dir report/matrix-audit
```

[目前两个已完成 seed 的观察](report/interim-seeds0-1/OBSERVATIONS.md) 包含全部六方法的逐 seed 指标和来源记录。它不是完整三 seed 基准；现有 OOD 波动不用于改变 k、shrinkage、temperature 或 score 方向。

## 冻结输出分布诊断

[已完成 seeds 0/1 的分布报告](report/output-diagnostics-seeds0-1/REPORT.md) 分开描述 ID validation、ID test 和 OOD 的 confidence、final-feature L2 norm、Energy 与 logit 统计，并提供可导出的 PNG/SVG、源数组 SHA-256 和独立核验记录。它仅解释冻结输出，不拟合 detector、不改变温度，也不选择测试集参数；预定三 seed 汇总仍等待 seed 2。

```powershell
& C:\Users\zxzx\anaconda3\python.exe tools/diagnose_representations.py --evaluations runs/cifar100-seed0/eval-final runs/cifar100-seed1/eval-final --output-dir report/output-diagnostics-seeds0-1
```

## Textures 数据准备（评估仍待执行）

[预先固定的 DTD 扩展方案](docs/TEXTURES_EXTENSION.md) 使用官方 partition 1 的 test split：1,880 张图片，47 类各 40 张。缓存已通过官方 MD5、split 隔离、标签/顺序和全部图片的独立 torchvision 逐像素检查；原始 archive、JPEG 哈希、转换方案及环境版本保持本地。训练/SVHN 完成并退出后才接入 loader/evaluator；正式 Textures 三 seed 评估和独立审计也已全部完成。

```powershell
# CPU 准备；已经完成的兼容缓存仅校验并复用
& C:\Users\zxzx\anaconda3\python.exe tools/prepare_textures.py
# 以原准备环境解码 RGB，再用 torchvision 独立检查全部缩放/裁剪像素
# 原有审计保留；重跑指定新输出路径
$env:CUDA_VISIBLE_DEVICES=''
& C:\Users\zxzx\anaconda3\envs\ece488_clip\python.exe tools/audit_textures.py --decoder-python C:\Users\zxzx\anaconda3\python.exe --output runs/textures-conversion-audit-rerun.json
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

[通过的本地审计](runs/textures-conversion-audit.json) 记录 canonical decoder 与参考环境版本。两个环境的 Pillow/libjpeg 对部分 JPEG 的直接解码存在差异；未放宽像素一致性要求，而是在同一 canonical RGB 输入上核验 torchvision 变换。[最初失败记录](runs/textures-direct-decoder-audit-failed.json) 保留差异证据。未来 evaluator 直接读取冻结的 uint8 缓存，避免重新使用另一套 JPEG 解码器。正式 Textures 性能结果现已保存在 `runs/textures-extension/summary/`，与独立输入审计一并保留。

## CPU 审计队列恢复

当前审计队列已完成并退出，状态保留在 `runs/audit-queue/status.json`。仅在明确需要恢复/重跑审计时检查旧进程状态并执行：

```powershell
& C:\Users\zxzx\anaconda3\python.exe -u tools/run_audit_queue.py --analysis-python C:\Users\zxzx\anaconda3\python.exe
```

队列等待并审计 `runs/matrix/summary/` 和 `runs/svhn-extension/summary/`，默认报告为 `report/matrix-audit/`、`report/svhn-matrix-audit/`。它只读实验结果；自己的 `runs/audit-queue/STOP` 可停止等待或它启动的 CPU 审计，不影响训练或 SVHN controller。主实验暂停/失败、未完成的 seed 集合或独立审计不通过时，它不会宣布完成。

## 完成的四自然 OOD 基准

[合并报告](report/baseline-complete/REPORT.md) 给出全部六方法的逐数据集 mean/sample SD、四自然数据集的等权 macro，以及每组 ID/OOD 数量和 prevalence。保留 [原始主基准](runs/matrix/summary/SUMMARY.md)、[SVHN](runs/svhn-extension/summary/SUMMARY.md)、[Textures](runs/textures-extension/summary/SUMMARY.md) 各自独立的汇总，未覆盖旧结果。所有 controller 已完成并退出，现有 queue 不需要重新启动。

[主基准图](report/matrix-comparison/REPORT.md)、[SVHN 图](report/svhn-comparison/REPORT.md)、[Textures 图](report/textures-comparison/REPORT.md) 提供 PNG/SVG；彩色点是训练 seeds，误差线是 sample SD。完整基准是本地独立协议，不能直接当作 OpenOOD leaderboard 复现。

Textures 新阶段记录实际加载的原始 ID train/validation/test 和 OOD 像素 SHA-256，纯 NumPy/官方 MD5 reference 已逐组独立核验。早期阶段的 input hash 不会被追溯伪装为执行时记录。calibration 已按预先固定的 ID-only 方案完成；温度只使用 ID validation 信息选择，原始基准保持冻结。

```powershell
# 原有结果/审计保留；重建时指定新的输出目录
& C:\Users\zxzx\anaconda3\python.exe tools/build_baseline_overview.py --output-dir report/baseline-complete-rerun
& C:\Users\zxzx\anaconda3\python.exe tools/audit_input_pixels.py --evaluations runs/cifar100-seed0/eval-extension-textures runs/cifar100-seed1/eval-extension-textures runs/cifar100-seed2/eval-extension-textures --output report/textures-matrix-audit/input-pixel-audit-rerun.json
```

## ID-only calibration 的复算与下一步

校准协议见 [CALIBRATION_PHASE2.md](docs/CALIBRATION_PHASE2.md)。原 5,000 张 ID validation 按每类固定 seed 2027 分成 temperature-fit 与 threshold-calibration 各 2,500 张；二者都曾用于 checkpoint selection，不能称作全新 holdout。CPU float64 的 T=1 对照和已拟合 T 使用相同 logits、阈值校准人口，MaxLogit/KNN/Mahalanobis 保持不变。原 GPU float32 分数的精度差异另外记录。

```powershell
# 使用已保存的 split/hash/source guard；完成的 seed 会被复用，不重新拟合
& C:\Users\zxzx\anaconda3\python.exe tools/run_calibration.py
# 独立概率公式、凸 inverse-temperature 求根、sklearn AUROC/AP 和逐阈值复算
& C:\Users\zxzx\anaconda3\python.exe tools/audit_calibration.py
# 汇总报告与科学图；独立核验配对 mean/sample SD 与四数据集 macro
& C:\Users\zxzx\anaconda3\python.exe tools/build_calibration_report.py
& C:\Users\zxzx\anaconda3\python.exe tools/audit_calibration_report.py
```

[80/90/95/99% 操作点分析](report/selective-risk-phase2/REPORT.md) 也已完成：144 个设置、720 行逐 seed/population 计数与全部样本决定通过独立核验，保留 ID 错误率、实际 ID coverage 和 OOD acceptance 各用其正确分母。95% 点与校准报告完全一致。MSP 校准前后 ID retained error 为 21.46% / 21.49%，自然 OOD macro acceptance 为 79.78% / 78.70%；不能从 ECE 降低推导拒绝错误的能力全面提高。

## 固定操作点复算与训练特征几何扩展

```powershell
# 完成的 grid 被 hash/source guard 复用，不改变原始分数
& C:\Users\zxzx\anaconda3\python.exe tools/run_selective_risk.py
& C:\Users\zxzx\anaconda3\python.exe tools/audit_selective_risk.py
& C:\Users\zxzx\anaconda3\python.exe tools/build_selective_risk_report.py
& C:\Users\zxzx\anaconda3\python.exe tools/audit_selective_risk_report.py
```

[训练 ID feature geometry 扩展](report/geometry-phase3/REPORT.md) 已按固定协议完成：三个 frozen classifiers 增加 ViM、Residual、ReAct Energy，所有统计量只拟合原 45,000 ID training features；D=256、clipping quantile=0.99，affine Energy 作为数值对照。原始六个 T=1 control 完全保留，原始 logits/scores/masks、独立数学/指标/计数/配对汇总审计和 PNG/SVG 科学图均已保存。

自然四数据集等权 macro AUROC：Energy 80.97% ± 0.79 pp、ViM 81.26% ± 1.20 pp、ReAct 81.43% ± 0.67 pp、Residual 53.80% ± 2.28 pp。ViM 在 Textures 相对 Energy +9.58 pp，但在 CIFAR-10/SVHN -4.05/-3.79 pp；没有普遍改善或测试集选出的最佳方法。ReAct 中间 logits 会改变少量分类预测，其准确率单独记录；所有 error/coverage 统计使用原始 classifier correctness。

```powershell
# CPU torch 审计需真正屏蔽 CUDA；当前 Windows shell 中空值已验证无效
$env:CUDA_VISIBLE_DEVICES='-1'
# 原始训练 banks/fit 已完成；仅独立复算本地结果和报告
& C:\Users\zxzx\anaconda3\python.exe tools/audit_geometry_evaluation.py
& C:\Users\zxzx\anaconda3\python.exe tools/build_geometry_report.py
& C:\Users\zxzx\anaconda3\python.exe tools/audit_geometry_report.py
```

[固定 posthoc diagnostics](report/geometry-diagnostics/REPORT.md) 现已完成：三个 seed 的全部八人口、108 个 fixed setting 的 transition codes、ID 每类正确/错误计数和完整 signed component 分解均已保存并独立审计。ViM 在同一 95% ID-calibration 目标下相对 Energy 减少 Textures OOD 接受率 22.59 pp，却增加 CIFAR-10 OOD 接受率 5.28 pp；两方法实际 ID coverage 不完全相同。这是已有固定阈值的描述，不能归结为因果或普遍改善。

[RMD 三 seed 阶段](report/rmd-phase4/REPORT.md) 已完成：固定 1% shrinkage，class/background covariance 仅由原 45,000 ID training features 拟合；signed relative distance 保留负值。原有十列控制结果完全一致，另存 matched CPU MD，十二列完整报告含四自然数据集与独立 noise diagnostics。原始矩阵/scores/masks、独立 Cholesky/指标/计数/配对统计审计和 PNG/SVG 均已保存并核验。

四自然数据集等权 macro AUROC：RMD 81.61% ± 1.57 pp，matched MD 63.19% ± 2.20 pp，KNN 82.42% ± 1.11 pp。RMD-minus-matched-MD 的同 seed macro 差值 +18.43 ± 1.72 pp；这不表示 RMD 全面超过其他方法。固定 95% ID-calibration 目标下 RMD 的自然 OOD 接受率 75.24% ± 1.57 pp、实际 ID-test coverage 95.09% ± 0.25 pp；它不是恰好相同 test coverage 的比较。

```powershell
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:CUDA_VISIBLE_DEVICES='-1'
& C:/Users/zxzx/anaconda3/python.exe tools/audit_rmd_evaluation.py
& C:/Users/zxzx/anaconda3/python.exe tools/build_rmd_report.py
& C:/Users/zxzx/anaconda3/python.exe tools/audit_rmd_report.py
```

[RMD 分量/决定诊断](report/rmd-diagnostics/REPORT.md) 已完成：七个人口、完整 M/B/R 向量、132 个 fixed setting 的 transition codes、原 ID 各类正确/错误分母与计数，均经独立审计。CIFAR-10 相对 test ID 的平均 delta M=+15.35、delta B=-50.32，因此 delta R=+65.67；这是 mean identity，不是 AUROC 或因果解释。在固定 95% ID-calibration 目标下，RMD 相对 matched MD 减少 CIFAR-10/SVHN/Textures OOD 接受率 10.34/27.03/14.77 pp；相对 Energy 的 CIFAR-10 接受率反而增加 4.31 pp。完整十一参考、四目标、三 seed 的所有方向保留，实际 ID coverage 差值单独保存。

[ASH-P/B/S 三 seed 阶段](report/ash-phase5/REPORT.md) 已完成：固定 p=65，512D 保留179个坐标；training-ID-only preflight 全部通过后评分。十三个 named methods 加两个数值控制，共十五列，原十二列完全一致。全部原始 logits/intermediates/scores/masks、独立数学/指标/计数/配对汇总审计与 PNG/SVG 均已本地保存。

自然四数据集等权 macro AUROC：ASH-P 81.42 ± 0.71%，ASH-B 69.48 ± 2.21%，ASH-S 83.16 ± 0.72%；S-minus-Energy 同 seed macro 差值 +2.18 ± 0.78 pp。固定95% ID-calibration 目标下 S 的自然 OOD 接受率 72.54 ± 4.43%，实际 ID-test coverage 94.99 ± 1.11%；不是严格相同 test coverage 的比较，也没有用结果选择 percentile 或最佳变体。

ASH 中间分类准确率 P/B/S 为75.66%/71.77%/75.67%，原始分类器75.77%；所有拒绝错误率仍使用原始预测。报告审计的整数计数展示问题已修复并保留失败证据，全部完成检查通过；75 tests 中73通过、2项已在原CPU环境验证的 SciPy optimizer tests 跳过。

```powershell
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:CUDA_VISIBLE_DEVICES='-1'
& C:/Users/zxzx/anaconda3/python.exe tools/audit_ash_evaluation.py
& C:/Users/zxzx/anaconda3/python.exe tools/build_ash_report.py
& C:/Users/zxzx/anaconda3/python.exe tools/audit_ash_report.py
```

[ASH support/scale 与决定诊断](report/ash-diagnostics/REPORT.md) 已完成：七人口、三变体、十二参考、四目标共432个设置，完整raw vectors、classification/acceptance codes、逐类分母与计数均已独立核验。ID平均保留84.22% activation mass，C10为80.17%；这与保留179/512坐标不同。固定95%目标下S-minus-Energy的C10接受率变化-1.24±1.79 pp，包含S-only3.92%与Energy-only5.16%，不能只保留有利样本。

所有640个来源hash不变，15,654个统计记录与1,678个展示字段通过独立审计；PNG已视觉检查。完整CPU-masked suite运行80tests：78通过，2项已验证的原SciPy optimizer skips。浮点macro误用精确一致检查的失败记录已保留，并按原冻结容差修复审计规则；原始分数/参数未改变。

[全基准 paired image bootstrap](report/paired-bootstrap/REPORT.md) 已完成：1,000次固定图像抽样，全部15列/105配对、66个endpoint、三个固定classifier和四操作目标。原始weights/counts/metrics、全部区间与原三seed SD分别保存；6,000个权重向量、126万计数和31,680个统计记录通过独立审计。最终PNG/SVG与完整85tests证据已绑定，83通过、2项已验证旧optimizer skips。

ASH-S自然macro AUROC83.16%，固定模型条件下名义95%图像区间[82.69%,83.56%]，原训练seed SD0.72pp另列。S-minus-Energy macro差值+2.18pp，图像区间[+2.05,+2.31]pp；C10差值区间[-0.165,+0.134]pp仍跨零。区间不覆盖训练随机性、不修复validation reuse，也不用于方法选择/显著性/部署声明。首次图表共享轴排序问题已留存并修复，全部数值未改变。

```powershell
# 只读检查已完成审计/源码/raw/图表/hash绑定，不改写冻结审计文件
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:CUDA_VISIBLE_DEVICES='-1'
& C:/Users/zxzx/anaconda3/python.exe tools/verify_bootstrap_completion.py
```

[共享样本相关性诊断](report/bootstrap-dependence/REPORT.md)已完成：786个固定来源，66个endpoint的45×45矩阵、7,920个三模型和360个四自然人口方差分解，独立scalar/报告/图表审计全通过。三模型自然macro AUROC的observed/diagonal-only图像SD比值在15方法中1.42–1.58；四自然macro的45个seed-method比值为1.38–1.66。这不是训练seed uncertainty或因果解释。

[项目总览与证据边界](report/project-assessment/REPORT.md)按顺序链接11个已完成阶段的报告与完成审计，记录旧validation重复使用、同一已看过OOD test的探索性方法比较和外部benchmark限制；40项来源hash通过独立核验。

[ID validation三路隔离复验](report/validation-isolation/REPORT.md)已完成。固定的每类20/15/15划分使checkpoint选择、temperature拟合、operational threshold校准分别使用2,000/1,500/1,500张互斥ID validation图像。三个seed都完成100epoch且训练轨迹/最终张量通过独立审计；新选中epoch为92/96/98，旧为89/96/92。冻结模型的ID-test accuracy旧75.77±0.30%、新75.80±0.27%，paired差+0.03±0.08pp。

六个预定detector在四个自然OOD集、独立noise诊断、T=1与ID拟合温度及四个固定接受目标上均保存原始logits/features/scores/masks。自然macro的T=1 Energy AUROC旧80.97±0.79%、新80.78±0.87%，paired差-0.20±0.17pp；固定95% ID校准目标下其自然OOD接受率旧78.62±2.60%、新79.74±3.04%。全部72个比较单元和三seed点在[机器可读汇总](report/validation-isolation/summary.json)，独立特征/评分/汇总/图表审计、人工图检与91项回归完成证据在[完成清单](report/validation-isolation/completion.json)。旧新同时改变checkpoint选择与拟合样本数；这些OOD test此前已经看过，结果仍属探索性，不作为新方法排名或独立OOD确认。原工程及旧冻结产物未修改，未发布。

[Places365-Standard 36,500-image extension](report/places365-extension/REPORT.md): three frozen ID-isolated seeds, six original methods, raw logits/features/scores and independent 108-file completion audit; one sequentially selected natural-OOD population, not a combined benchmark.

[Places365 all-scene diagnostic](report/places365-scene-diagnostic/REPORT.md): 365 scenes, complete frozen ID-threshold decision grid, independently checked raw-score reconstruction; post-result descriptive only.

[Oxford-IIIT Pet 3,669-image test extension](report/oxford-pets-extension/REPORT.md): three frozen ID-isolated seeds, original six methods and complete raw logits/features/scores/masks with independent 105-file completion audit; sequential single-population OOD check.
