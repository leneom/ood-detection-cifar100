# 事后 OOD 检测的改进能否在新数据上成立？CIFAR-100 受控研究

本项目固定 CIFAR-100 分类器，比较 13 种 post-hoc 打分方法和一种训练损失的改动，评估范围是 8 个 OOD 数据集和一个同源留出类协议，所有条件都用 3 个训练 seed。探索阶段得出了两条方法结论，之后在从未参与评分的数据集上重新检验；检验前，假设和判定规则已提交到 git。

[English README](README.md) · [研究记录](docs/PROGRESS.md) · [核心协议](docs/PROTOCOL.md) · [概念学习指南](docs/LEARNING_GUIDE.zh-CN.md)

## 主要发现

**1. 探索阶段的改进在新数据集上没有成立。** 在开发用的 4 个数据集上，ASH-S 的 macro AUROC 比 Energy 高 +2.18 pp，图像 bootstrap 区间很窄（[+2.05, +2.31] pp）。换到 4 个新数据集后，平均增益降到约 +0.6 pp；在 Tiny ImageNet 上，3 个 seed 全部为负（−0.24 ± 0.04 pp）。同一次预注册检验确认了 RMD 优于 Mahalanobis：所有数据集、所有 seed 都为正，高出 +20 到 +28 pp。真正决定结论的不确定性来自"选了哪些数据集"，图像 bootstrap 和 seed 方差都衡量不到这一点。

**2. 大幅改善集中在低层统计差异明显的数据源上。** ASH-S 和 LogitNorm 的大幅改善主要出现在灰度数据源上。例如用 LogitNorm 后，Fashion-MNIST 在固定 95% ID 阈值下的最低 OOD 通过率从 54% 降到 15%。在以物体为主体的自然照片（CIFAR-10、Tiny ImageNet、Oxford Pets）上，AUROC 只变化 −3.5 到 +1.6 pp，正负都有。LogitNorm 还使 ID 准确率下降 2.0 pp。

**3. 去掉数据源差异后，各方法几乎没有区别。** 在同源留出类上（3 种类别划分 × 3 个 seed），5 种可用打分的 AUROC 都在 74.5–76.1% 之间；在 95% ID 阈值下，82–85% 的新类别图像仍被放行。

这些结果与 SC-OOD（Yang et al., 2021）的观点一致：常见 OOD 基准奖励的是对数据源低层差异的敏感度。本项目没有提出新方法，贡献在于用预注册、多 seed 的受控实验为这一观点提供 CIFAR 规模上的证据，包括阴性结果。

![两条预注册假设的配对 AUROC 差值](report/posthoc-confirmation/paired_auroc_differences.png)

## 设定

- **ID 数据**：CIFAR-100，固定分层划分（split seed 2026），训练集 45,000 张，validation 5,000 张；官方 test 10,000 张只用于评估。
- **分类器**：适配 32×32 输入的 ResNet-18，从零开始用交叉熵训练。SGD，lr 0.1，momentum 0.9，weight decay 5e-4，batch 128，训练 100 epoch，在第 50/75/90 epoch 后 lr ×0.1。3 个 seed 的 ID test 准确率为 75.8 ± 0.3%。
- **validation 三路隔离**：5,000 张 validation 按类别分成互斥的三份，分别用于选 checkpoint（2,000）、拟合温度（1,500）、校准阈值（1,500）。OOD 图像不参与任何选择或拟合。
- **OOD 数据**：CIFAR-10、MNIST、SVHN、Textures 用于开发；Tiny ImageNet-200、Fashion-MNIST、Places365、Oxford-IIIT Pets 后来才加入，从未用于开发 ASH-S、RMD 或 LogitNorm。
- **同源语义 OOD**：80 个 CIFAR-100 类作为 ID，另外 20 个类不参与训练；3 种类别划分 × 3 个 seed，共 9 个 80 类模型。
- **打分方法**：MSP、Energy、MaxLogit、Entropy、KNN（k=50）、Mahalanobis、ViM、Residual、ReAct、RMD、ASH-P/B/S；训练端的改动是 LogitNorm（τ=0.04）。
- **指标**：AUROC、AUPR、FPR95，以及在 ID validation 上校准的 80/90/95/99% 阈值下的 OOD 通过率。

## 证据如何控制

- **只用 ID 数据做选择**：checkpoint、温度、阈值和特征统计量都只来自 ID 训练集或 validation。
- **预注册**：检验性研究在评分前提交协议和输入 hash 声明（ASH-S/RMD 为 `59e914f`，LogitNorm 为 `3b734e6` 和 `b8f8703`）。判定规则：3 个 seed 的配对差值全正记为 supported，全负记为 contradicted，其余记为 mixed。
- **独立复算**：每项研究都有单独的审计脚本，从保存的原始 logits 和特征重新计算分数与指标；`completion.json` 记录全部输入输出的 SHA-256。
- **保留失败记录**：失败的尝试和修复记录保存在对应报告旁边（`report/*/…attempt-*`）。

各项研究的问题、结果和报告链接见[英文 README 的 Study map](README.md#study-map) 和[研究记录](docs/PROGRESS.md)。

## 运行

实验环境：Python 3.10、PyTorch 2.13（CUDA 12.6）、torchvision 0.28、NumPy 2.2，单张 RTX 4080 Laptop GPU；分析脚本另外需要 SciPy、scikit-learn、Matplotlib。

```bash
pip install -e . scipy scikit-learn matplotlib
python -m unittest discover -s tests -v
python train.py --dataset synthetic --model tiny --epochs 3 --device cpu --output-dir runs/smoke
python evaluate.py --checkpoint runs/smoke/best.pt --ood noise constant --device cpu --output-dir runs/smoke/evaluation
```

完整训练和评估命令见[英文 README](README.md#reproducing)。数据集、checkpoint 和逐图原始输出（`data/`、`runs/`，约 13 GB）不包含在仓库里。实验最初在 Windows 工作站上运行。发布前，文档和记录中与本机相关的绝对路径已换成仓库内相对路径（解释器路径写作 `python`），PowerShell 命令也换成了等价的 shell 命令，所有数值都没有改动。完成清单中的 hash 引用已按依赖顺序同步更新；每个受影响文件的新旧 SHA-256 列在 [RELEASE_PATH_NORMALIZATION.json](docs/RELEASE_PATH_NORMALIZATION.json) 中，原始文件保留在 git 历史里。

## 局限

- 只有一个 ID 数据集、一种架构（32×32 的 ResNet-18），每个条件 3 个 seed，没有 ImageNet 规模的实验或预训练模型。
- 开发用的 4 个数据集在加入新方法的过程中已经看过，在它们上面的比较属于探索性结果；只有预注册的研究属于检验性结果。
- 本项目不是 OpenOOD 的复现：数据划分、预处理和模型都不同，数字不能与排行榜比较。
- 发现 2 中的"低层线索"解释目前还是假设，下一步要用受控的 2×2 实验检验：同源留出类与 ID 类，各自加或不加低层线索（灰度、降分辨率）。

本项目在 AI 编程助手的帮助下完成。
