# OOD Detection：项目复现索引与证据评估

当前工作是同一 CIFAR-100 ID 定义下的完整三 seed 本地实验与事后方法分析。三个 ResNet-18 均按预先固定的 100-epoch 预算训练，checkpoint 仅按 ID validation 选择；OOD test 没有参与训练、checkpoint 或训练 ID feature 统计拟合。下面的后续方法对比是在已看过 OOD 结果后扩展的，属于探索性而非新的独立确认。

原始 ID-test accuracy 为 **75.77% ± 0.30 pp**（三训练 seed 的样本 SD）；各 seed 为 76.11%, 75.56%, 75.64%。旧 continual-learning 项目和所有原 checkpoint/raw logits/scores 保持不变。

## 可复现的证据链

所有数据、checkpoint、hash、审计和报告都留在本地。按顺序从原模型、ID-only calibration、固定操作点、训练特征方法、结果后诊断、图像 bootstrap 到共享样本依赖分析读取；每一项下方均绑定独立完成检查。
- [Three-seed six-method natural OOD baseline](../baseline-complete/REPORT.md) — [完成审计](../baseline-complete/completion-audit.json)，状态 `passed`。
- [ID-only temperature calibration](../calibration-phase2/REPORT.md) — [完成审计](../calibration-phase2/completion-audit.json)，状态 `complete_and_verified`。
- [Four fixed validation operating targets](../selective-risk-phase2/REPORT.md) — [完成审计](../selective-risk-phase2/completion-audit.json)，状态 `complete_and_verified`。
- [Training-ID ViM, Residual and ReAct](../geometry-phase3/REPORT.md) — [完成审计](../geometry-phase3/completion-audit.json)，状态 `complete`。
- [Feature-geometry saved-output diagnostics](../geometry-diagnostics/REPORT.md) — [完成审计](../geometry-diagnostics/completion-audit.json)，状态 `complete`。
- [Training-ID RMD and matched CPU MD](../rmd-phase4/REPORT.md) — [完成审计](../rmd-phase4/completion-audit.json)，状态 `complete`。
- [RMD component/transition diagnostics](../rmd-diagnostics/REPORT.md) — [完成审计](../rmd-diagnostics/completion-audit.json)，状态 `complete`。
- [Fixed ASH-P/B/S extensions](../ash-phase5/REPORT.md) — [完成审计](../ash-phase5/completion-audit.json)，状态 `complete`。
- [ASH support/scale/decision diagnostics](../ash-diagnostics/REPORT.md) — [完成审计](../ash-diagnostics/completion-audit.json)，状态 `complete`。
- [Full paired image-bootstrap intervals](../paired-bootstrap/REPORT.md) — [完成审计](../paired-bootstrap/completion-audit.json)，状态 `passed`。
- [Saved-bootstrap dependence decomposition](../bootstrap-dependence/REPORT.md) — [完成审计](../bootstrap-dependence/completion-audit.json)，状态 `passed`。

[完整实验协议](../../docs/PROTOCOL.md)；[阶段进展](../../docs/PROGRESS.md)；[中文原理与解释](../../docs/LEARNING_GUIDE.zh-CN.md)。`index.json` 保留上面11个报告、完成审计、运行状态及其 SHA-256，并绑定主协议与主要原始汇总。

## 已支持的描述性结果

四自然数据集等权 macro AUROC：Energy 80.97%、KNN 82.42%、RMD 81.61%、matched CPU MD 63.19%、ASH-S 83.16%。完整15列、各人口、AUPR/FPR95-ID和四目标都在[完整 paired-bootstrap 报告](../paired-bootstrap/REPORT.md)，不能只看这一行方法。
ASH-S−Energy 自然 macro AUROC 差 +2.18 pp，固定三模型下的名义95%图像区间 [+2.05, +2.31] pp；CIFAR-10 单独差 -0.01 pp，区间 [-0.17, +0.13] pp。四人口均值的改善不等于所有人口都改善，也不是新 test-set 上的确认性结论。
固定95% ID-calibration 目标下，ASH-S−Energy 自然 OOD acceptance 差 -6.29 pp，同时实际 ID-test coverage 差 -0.16 pp；测试 coverage 未被强制匹配。
[共享样本诊断](../bootstrap-dependence/REPORT.md)中，15方法的三模型自然 macro AUROC SD/忽略跨模型协方差的代数 SD 比值范围为 1.42–1.58；每 seed/方法的四自然人口 macro 比值范围为 1.38–1.66。后者共45个方法-seed单元。共用ID图像采样会让把各 seed/人口当独立图像抽样低估此实验的波动，但这些比值**不**估计重新训练模型的方差。

## 结论的边界

- 模型选择用过全部5,000条ID validation；后来的温度与操作阈值只用其中重新划分的ID validation子集，所以这些阈值并非独立holdout校准。实际ID-test coverage另报；FPR95-ID另用ID test order statistic，只是benchmark点。
- ViM、RMD、ASH 等扩展是在基线OOD结果之后设计/加入。固定后没有用OOD test调单个参数或选checkpoint，但当前同一组OOD test上的后续比较仍是探索性的；上述图像区间不能消除方法发展过程中的重复使用。
- 三个训练 seed 的原始样本SD与1,000次paired image bootstrap的条件区间是两种不同不确定性。所有15列方法和三个模型共用每次图像权重；合成noise仅单独诊断，不进入四自然macro。
- CIFAR-10/MNIST/SVHN/Textures的样本数与OOD占比不同，AUPR不能脱离占比比较。CIFAR语义重叠、单一模型架构与有限数据集使结果不能直接当完整OpenOOD基准或部署保证。
- ID/Textures的执行时像素指纹跨 seed 核对过；较早的CIFAR-10/MNIST/SVHN/noise人口按归档执行loader、文件/cache/顺序和noise RNG对齐，没有伪造当时未捕获的执行像素hash。
- 非因果的score/feature支持、transition和相关性诊断不能证明改进机制。数值控制列 affine Energy、matched CPU MD 保留在15列全量表格，但不算独立新 detector。

## 下一项具体工作

优先补一项**独立ID validation校准复验**：在重新训练/重选checkpoint前，先把原固定5,000条ID validation按类别预定义成互斥的checkpoint、temperature-fit、threshold-calibration三份；保持45,000条train和官方test/OOD完全隔离，预先冻结三个训练 seed、100-epoch预算、十五列/四目标以及原始分数保存/审计规则。这样能检验当前validation reuse的影响。仅在已有OOD数据上重跑仍不构成独立OOD确认；进一步泛化评估要先冻结新增自然OOD集、数据版本/完整性及分类定义，并和探索过程中已看过的OOD人口分开。

本报告是研究进展和复现边界，不是论文新颖性、统计显著性、完整外部基准或公开发布判断。没有推送、发布或对外发送数据。
