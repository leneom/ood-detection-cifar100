# OOD Detection：本项目的学习路线

## 1. 我们到底在判断什么？

classifier 只能从训练时的 100 个 CIFAR-100 类别中选一个。即使输入手写数字，它仍会输出某个类别。OOD detector 先回答：**这张图是否属于 classifier 学过的数据分布？** 通过检测后，才信任分类结果。

ID classification accuracy 和 OOD detection AUROC 是两件事。一个模型可以分类正确率不错，却对陌生图片过度自信。我们因此同时报告分类和检测结果。

## 2. MSP 为什么可以失败？

MSP 取最大 softmax probability。比如 logits `[8, 1, 0]` 给第一类很高的概率，看起来是熟悉输入；`[0, 0, 0]` 比较不确定，看起来像 OOD。

但 softmax 只比较各类别的相对值。`[8, 1, 0]` 和 `[108, 101, 100]` 的 softmax 完全一样。对陌生输入出现大的类别差异，MSP 就会把它当作熟悉输入。

## 3. Energy 多看到了什么？

Energy 是 `E(x) = -T logsumexp(z(x)/T)`。在 T=1 时，把所有 logits 加 100 会让 Energy 减少 100，但不会改变 MSP。Energy 因此保留 logits 的整体尺度信息。

这不是保证。一个训练不足的分类器也可能给陌生输入很大的 logits，导致 Energy 更负、错误地接受 OOD。必须在真实数据上验证，不能凭公式声称任何方法必胜。

我们把所有 score 统一为“越大越像 OOD”：MSP 用 `1 - max probability`，Energy 本身越大越像 OOD，MaxLogit 用 `-max logit`。

## 4. AUROC 与 FPR95 为什么会给不同印象？

AUROC 可以理解为：随机抽一张 OOD 和一张 ID，OOD 的 anomaly score 更大的概率，平手计半分。它评价所有阈值的整体排序。

FPR95-ID 则固定一个要求：至少接受 95% 的 ID，然后看有多少 OOD 也被接受。例：10,000 张 ID 接受 9,500 张，但 10,000 张 OOD 也接受了 9,000 张，FPR95=90%，检测不适合这个工作点，即使 AUROC 不算低。

ID validation 可以提供真正独立的阈值。validation 接受 95%，并不保证 test 也恰好接受 95%；报告 test 实际接受率，才知道阈值是否泛化。

## 5. KNN 与 Mahalanobis 看特征空间

KNN 先保存训练 ID 的 final-layer features，并 L2 normalize。新样本距离第 k 个最近邻越远，越像 OOD。本项目先固定 k=50，不通过 OOD test 挑选 k。

Mahalanobis 用训练 ID 的类均值与共享 covariance 计算到最近类的距离。某个方向本来波动很大，同样的欧氏距离就不一定异常；covariance 对不同方向进行缩放。本项目使用固定 1% isotropic shrinkage，保证有限样本下的数值稳定。它是 final-layer baseline，不是原论文多层组合和 input perturbation 的完整复现。

两者拟合时都不能混入 validation、ID test 或 OOD test。

## 6. 如何读我们自己的结果？

1. 先看训练是否完成、ID accuracy、checkpoint 选择规则和 seed 数。
2. 分开看 CIFAR-10 near-OOD、MNIST far-OOD；不要用均匀噪声代替自然图片。
3. 看 AUROC，再看 FPR95 和 validation 阈值下的接受率。
4. 用 score CDF 解释重叠与尾部，用 ROC 看阈值取舍。
5. 单个模型的 bootstrap interval 只反映 test sample uncertainty；不同 seed 重新训练的 mean ± sample SD 才反映部分训练随机性。
6. 差于随机的结果也保留，不看到 test 结果后翻转 score、换阈值或挑“最佳”方法来包装。

## 7. 用真实失败样本检查直觉

[完整 seed 0 的本地图册](../runs/cifar100-seed0/failure-gallery/REPORT.md) 为六种方法分别保存三行图片：被拒绝但分类正确的 ID、被拒绝且分类错误的 ID、被错误接受的 CIFAR-10 OOD。图册记录官方 test index、原标签、classifier 输出、confidence、score 和阈值。

MSP 的 validation 阈值是 τ=0.66525656。因为 s=1−max p，接受条件 s≤τ 等价于 max p≥1−τ≈0.33474344。这个阈值由 5,000 张 ID validation 校准；在 10,000 张 ID test 上拒绝了 109 张正确分类图片和 322 张错误分类图片，同时接受了 8,218 张 CIFAR-10 OOD。

例如官方 CIFAR-100 test index 8177 的 true/prediction 都是 house，但 confidence 约 9.60%，所以被拒绝。CIFAR-10 test index 8017 的 true label 是 airplane，classifier 却输出 CIFAR-100 的 whale，confidence 接近 100%，因此被 MSP 接受。两者说明：分类是否正确、classifier 是否自信、输入是否 OOD，需要分别判断。图中 confidence 保留四位小数，1.0000 是显示舍入，不声称精确等于 1。

每行展示固定规则选出的六个极端案例：拒绝 ID 按 anomaly score 降序，接受 OOD 按 score 升序，同分按原 test index 排序。它们不是随机样本，也不能代表整体分布；整体表现仍看完整 test population 的 AUROC、FPR95 和 validation 阈值接受率。OOD 标签仅作图注，不参与拟合、参数选择或 OOD 样本排序。

## 8. 从冻结输出分布解释差异

[seeds 0/1 的诊断图](../report/output-diagnostics-seeds0-1/REPORT.md) 每一列固定一个 classifier，每一行分别看 max-softmax、final-feature L2 norm 和 Energy 的 empirical CDF。ID validation 使用虚线，ID test 使用黑色实线，两者没有混成训练或 detector fitting 数据。

在 T=1 时，令 m=max z，p_max=exp(m)/Σ exp(z)。由 log p_max=m−logsumexp(z)，得到 Energy=−m+log p_max。这说明 MSP 只取 p_max，而 Energy 同时与最大 logit 的绝对值相关。工具逐样本检查这一恒等式，并核对原评估保存的 MSP/Energy 分数。

两个完整 classifier 的 ID test confidence 中位数是 0.9821 和 0.9786，固定生成规则的 uniform-noise confidence 中位数却是 0.9528 和 0.1921。它们对应此前观察到的噪声检测差异，但噪声仍只是 synthetic diagnostic。CDF 描述输出分布；校准误差还需要比较相应置信度区间中的 confidence 与实际 accuracy。

同一 classifier 的 feature norm 可以帮助描述不同输入的响应。不同 seed 的特征坐标和尺度可以变化，因此跨 seed 的范数差异不能单独证明几何因果关系。这份图没有拟合新的距离 detector，也没有用 OOD test 调 temperature、k 或 shrinkage。

## 自测

- logits 全部加同一个常数，MSP、Energy、MaxLogit 哪些变化？
- validation 阈值下 ID test 只接受 92%，可以把 OOD 接受率称作 FPR95 吗？
- 一个方法 AUROC 90%，FPR95 80%，该如何解释？
- 用 ID test 特征拟合 KNN 会破坏哪一条隔离要求？
- 三个 seed 的 SD 与 300 次 test sample bootstrap 的 CI 衡量相同的东西吗？

参考：[MSP 原始论文](https://arxiv.org/abs/1610.02136)、[Energy 原始论文](https://arxiv.org/abs/2010.03759)、[KNN 原始论文](https://proceedings.mlr.press/v162/sun22d.html)、[Mahalanobis 作者代码](https://github.com/pokaxpoka/deep_Mahalanobis_detector)。

## 完整原始三 seed 基准怎么读

三个模型都完成固定的 100 epochs，只按 ID validation accuracy 选择 checkpoint。ID test accuracy 为 75.77% ± 0.30%（训练 seed mean ± sample SD）。全部六种方法的逐 seed 指标、自然 OOD macro 与配对差值已通过独立审计；[汇总](../runs/matrix/summary/SUMMARY.md)、[审计](../report/matrix-audit/REPORT.md) 和 [科学图](../report/matrix-comparison/REPORT.md) 保留完整比较，不使用测试结果挑选参数。

| 自然 OOD | MSP AUROC % | Energy AUROC % | MSP FPR95-ID % | Energy FPR95-ID % |
|---|---:|---:|---:|---:|
| CIFAR-10 | 76.86 ± 0.10 | 77.75 ± 0.13 | 81.09 ± 1.04 | 81.46 ± 1.60 |
| MNIST | 75.84 ± 2.69 | 80.02 ± 1.83 | 85.66 ± 1.21 | 85.25 ± 7.25 |

这里每个 ± 都是三个训练 seed 的 sample SD，不是置信区间。Energy 的排序指标 AUROC 有所提高，但 CIFAR-10 的 FPR95-ID 平均反而略高；MNIST 的 FPR95-ID 配对差值为 -0.41 ± 6.66 pp，改善很小且 seed 波动较大。排序改善与高 ID 接受率下的拒绝能力必须分别看，不能从 AUROC 单独推导实际阈值效果。

科学图中的彩色点是各 seed，黑色点与线是 mean/sample SD；它们不同于单个 classifier 上重采样 test 样本的 bootstrap 区间。Noise 的大 seed 波动仍只是合成诊断，不能取代自然 OOD 证据。原始 natural macro 只包含 CIFAR-10 和 MNIST；SVHN/Textures 扩展也已完成，[四自然数据集合并报告](../report/baseline-complete/REPORT.md) 另行保存扩展后的 macro 与人口范围。

## ID-only temperature calibration：概率更准确，OOD 效果要另外量

这次三个 checkpoint 保持不变。把同一套原始 logits 写成 `softmax(z/T)`：正 temperature 保留 logits 的排序，所以预测类别和 ID accuracy 不变；拟合得到的 T 约为 1.36，输出分布更平缓。平均 ID test confidence 从 84.68% 降至 77.08%，而实际 accuracy 仍为 75.77%。

T 只按 2,500 张 ID-validation temperature-fit 样本的 NLL 拟合；另一组互斥的 2,500 张 ID validation 负责操作阈值。它们都来自曾经选择 checkpoint 的 validation，所以我们明确记录复用，不宣称全新 holdout 或覆盖保证。ID test/OOD test 不参与拟合。

| ID test 指标 | matched T=1 | ID-only fitted T | 配对变化 |
|---|---:|---:|---:|
| NLL | 1.0019 ± 0.0159 | 0.9311 ± 0.0137 | -0.0708 ± 0.0070 |
| Multiclass Brier | 0.3551 ± 0.0038 | 0.3428 ± 0.0043 | -0.0123 ± 0.0005 |
| ECE15 % | 9.0156 ± 0.2145 | 3.4807 ± 0.2126 | -5.5349 ± 0.3097 pp |
| Accuracy % | 75.77 ± 0.30 | 75.77 ± 0.30 | 0 |

这里 ± 是三个训练 seed 的 sample SD。配对变化先对同一个 classifier 做 calibrated-minus-T1，再汇总；它不等于两组 SD 直接相减。NLL、Brier、ECE 都下降，支持这个固定 ID test 上的概率校准改善，不能单独证明 OOD 拒绝改善。

四自然 OOD 数据集内先做每 seed 等权平均，再汇总：MSP 的 AUROC +1.2892 ± 0.1556 pp，FPR95-ID -0.9728 ± 0.6642 pp；Energy 的 AUROC -0.1245 ± 0.1264 pp，FPR95-ID +0.0850 ± 0.3957 pp；Entropy 的 AUROC +1.0936 ± 0.3103 pp，FPR95-ID +0.0203 ± 0.2184 pp。变化依 score 而异，排序改善与高 ID 接受率下的拒绝仍应分开解读。MaxLogit、KNN、Mahalanobis 是完全不变的对照。

[完整校准报告](../report/calibration-phase2/REPORT.md) 保留所有 seed/方法/自然 OOD 指标、原始 5,000-validation 阈值、matched 2,500 阈值、精度差异与科学图。T=1 的经典分数在此阶段也用 CPU float64 复算；不能把它与旧 GPU float32 结果误称为逐字节相同。所有变化与阈值已独立核验，没有根据 OOD test 选择 T 或方法。

## 固定 ID 接受目标：实际 coverage、保留错误与 OOD 接受率

[操作点报告](../report/selective-risk-phase2/REPORT.md) 已完成全部 80/90/95/99% 目标。阈值只从固定 2,500 张 ID-calibration 分数选择；目标 95% 不表示在新 ID test 上必定接受 95%。以 MSP 为例，三个 seed 的实际 test coverage 平均为 94.89%。

accepted accuracy 的分母是被接受的 ID，retained error risk 是其中错误 ID 的比例；incorrect rejection fraction 的分母是全部错误 ID。OOD acceptance 则只以各 OOD 人口为分母。这三个数不能互换，也不能把被拒绝的正确 ID 当作正确的 OOD 检测。

| MSP，95% ID calibration 目标 | T=1 | ID-only fitted T |
|---|---:|---:|
| 实际 ID test coverage % | 94.89 ± 0.62 | 94.89 ± 0.73 |
| 保留 ID classification error % | 21.46 ± 0.10 | 21.49 ± 0.17 |
| 正确 ID 被拒绝 % | 1.64 ± 0.20 | 1.68 ± 0.19 |
| 错误 ID 被拒绝 % | 15.96 ± 1.79 | 15.82 ± 2.32 |
| 四自然 OOD 等权平均接受率 % | 79.78 ± 3.25 | 78.70 ± 2.84 |

每个 ± 是三个训练 seed 的 sample SD。概率校准的 ECE 改善很大，但这里保留错误基本不变：模型置信度的数值更贴近准确性，并不自动保证每个操作阈值下的错误排名或 OOD 拒绝改善。最后一行使用 ID-calibration 阈值，不能叫作 benchmark FPR95-ID；其 ID test coverage 并非恰好 95%。报告列出全部四个目标和所有方法，没有从 OOD 结果挑一个有利操作点。

## 已完成的 training-ID geometry：ViM、Residual、ReAct

[三 seed 报告](../report/geometry-phase3/REPORT.md) 使用完全相同的 frozen classifiers。512D pooled feature 不做 L2 normalize；几何空间、alpha 与 clipping 阈值只由原 45,000 张 ID training 拟合。验证/test/OOD 不参与这些拟合，D=256、clipping quantile=0.99 在新结果前已固定。

ViM 的 anomaly score 是 `alpha*r + E`，其中 `r=||(f-u)R||`，`u=-pinv(W)b`，`E=-logsumexp(Wf+b)`。Residual 单独使用 r。ReAct 则把 feature 上截断到训练全坐标的固定 99% 分位数，再通过原分类头得到 scoring logits。affine Energy 保留作 float64 重建数值对照；原六方法的 T=1 scores/metrics 完全不变。

| 四自然 OOD 等权 macro，mean ± seed sample SD | AUROC % | 95% ID-calibration 目标下 OOD 接受率 % |
|---|---:|---:|
| Energy | 80.97 ± 0.79 | 78.83 ± 2.60 |
| ViM | 81.26 ± 1.20 | 76.74 ± 3.04 |
| ReAct Energy | 81.43 ± 0.67 | 78.37 ± 2.37 |
| Residual | 53.80 ± 2.28 | 92.35 ± 0.48 |

这里列出的是解释所需的分数；完整报告仍保留全部九种方法和 affine Energy 数值对照，没有按测试结果挑方法。± 是三个训练 seed 的 sample SD，并非 CI。OOD 接受率使用固定 2,500 ID-validation 阈值，其实际 ID-test coverage 单独报告；不能称它为 benchmark FPR95-ID。

ViM 相对 Energy 的 AUROC 变化：Textures +9.58 ± 1.00 pp，CIFAR-10 -4.05 ± 0.89 pp，SVHN -3.79 ± 1.60 pp，MNIST -0.58 ± 5.49 pp。只看 macro 会掩盖这些方向相反的结果。Residual 单独的分数在此协议下较弱；我们保留预先固定的方向，不因某些 AUROC 接近/低于 50% 而翻转分数。

ReAct 的 clipped logits 在三 seed 上分别改变 86/87/89 个 ID test argmax；这些 scoring intermediates 的准确率为 75.70%/75.46%/75.57%，原始分类器为 76.11%/75.56%/75.64%。不能说 ReAct 的中间分类预测完全不变；所有 retained-error 和 ID rejection 计数仍用原始预测来保证同一个 classifier 的比较。

posthoc 分量与决定变化诊断现已完成，具体例子见下一节。它不能凭相关性宣称因果，也不用于调整 D、quantile 或挑选 OOD test 上最有利的操作点。

## 已完成的分量与决定变化诊断：均值、排序和操作阈值不是同一个问题

[诊断报告](../report/geometry-diagnostics/REPORT.md) 保留全部三 seeds、八个不同人口和完整四目标/三新增方法/三参考的决定变化。`v=alpha*r`，`S=v+E`。每个样本的 score identity、每个人口的 `Var(S)=Var(v)+Var(E)+2Cov(v,E)` 都通过独立验证；variance 的分母是 population n，而跨 seed 汇总的 sample SD 分母是 seed_count-1。

Textures 相对 test ID：平均 delta v=+2.593，delta E=+3.843，所以 delta S=+6.436。CIFAR-10 的 delta v=-0.397、delta E=+3.709，因此 delta S=+3.312。signed 分量说明了平均分数如何相加；不能把这个均值差当作 AUROC，更不能据此宣称特征变化的因果机制。完整分位数和 seed 波动也已保存，未丢弃不利方向。

在预先固定的 95% ID-calibration 目标，Textures 的同一样本接受决定相对 Energy 分成四类（百分比是先在每个 seed 内计算，再取三 seed 均值）：

| 决定 | Textures 占比 |
|---|---:|
| ViM 与 Energy 都接受 | 41.81% |
| 只有 ViM 接受 | 14.29% |
| 只有 Energy 接受 | 36.88% |
| 两者都拒绝 | 7.02% |

所以 ViM 接受约 56.10%，Energy 接受约 78.69%，net change=14.29%-36.88%=-22.59 pp。这里 addition-only OOD acceptance 是不利变化，reference-only acceptance 被新方法拒绝则是有利变化；只数有利样本会夸大效果。

CIFAR-10 的四类均值是 71.27%、15.93%、10.65%、2.15%，net change=+5.28 pp，方向相反。MNIST/SVHN 的 paired changes 分别 +0.90 ± 12.06 pp / +8.07 ± 13.96 pp；训练 seed 波动不能隐藏。两方法的 ID-test coverage 差值为 +0.13 ± 0.84 pp，因而这不是严格相同 test coverage 下的比较，也不隔离纯 ranking 变化。

ID 的同类计数另按原 classifier 的正确/错误划分，分母分别是全部正确 ID 和全部错误 ID；每个原 ID class 的 raw cells 和原始 labels 也保存了。OOD 类别不附加 classifier correctness。所有原 scores、阈值、方向、训练 schedule 和 checkpoint selection 不变；posthoc 描述不授权用 OOD test 调参。

## 已完成的 RMD：保留 signed background subtraction

[RMD 三 seed 报告](../report/rmd-phase4/REPORT.md) 现已完成。`M=min_k (f-mu_k)^T P_w (f-mu_k)`，`B=(f-mu_0)^T P_0 (f-mu_0)`，anomaly score `R=M-B`，越大越像 OOD。class/background covariance 均只用训练 ID，population denominator 为 N=45,000；两者分别沿用固定 1% trace shrinkage。这是公开原公式的本地适配，不能照搬原论文的完整性能结论。

直觉例子（仅演算，不是实验数据）：若某样本 M=500、B=450，R=50；另一样本 M=400、B=100，R=300。MD 对前者分数更高，RMD 的次序相反，因为它扣掉了 label-agnostic background distance。不能从这个例子断言哪类真实样本一定如此；负的 R 也合法，三 seed ID-test 最低 R 约 -82/-78/-80，均未裁剪。

| 四自然 OOD 等权 macro，mean ± seed sample SD | AUROC % | 固定 95% ID-calibration 下 OOD 接受率 % |
|---|---:|---:|
| matched CPU MD（数值对照） | 63.19 ± 2.20 | 90.46 ± 0.33 |
| RMD | 81.61 ± 1.57 | 75.24 ± 1.57 |
| KNN | 82.42 ± 1.11 | 73.19 ± 3.62 |

RMD 相对 matched MD 的 paired macro AUROC +18.43 ± 1.72 pp；每个差值先在同 seed 内计算。单独 RMD AUROC：CIFAR-10 75.37 ± 0.72%、MNIST 76.72 ± 5.06%、SVHN 87.44 ± 0.63%、Textures 86.91 ± 0.14%。MNIST 的 seed 波动较大，macro 不能掩盖它。全部十二列和十一参考的配对统计仍在完整报告，无测试集挑选。

RMD 实际 ID-test coverage 95.09 ± 0.25%，保留 ID 错误率 22.20 ± 0.15%；matched MD 为 95.30 ± 0.29% / 24.10 ± 0.32%。同一个 ID-calibration 目标并不等于相同 ID-test coverage。自然 OOD 接受率仍高，特别是 CIFAR-10 86.23% 和 MNIST 88.96%；不能仅由 AUROC 改善称项目已经适合部署。Noise 保持独立诊断，不进入 macro。

独立审计检查全部 class-distance matrices、540 项 AUROC/AP、1,008 个 masks、2,205 个汇总统计及 333 个展示字段；原有十列分数和指标未变。首次 wrapper 把原 split 的 `val` 键写成 `validation`，在拟合前停止；失败来源/manifest 已保存，修复字段后重跑，数据/方法未改。

## 已完成的 RMD 诊断：mean subtraction 与样本决定变化

[完整诊断报告](../report/rmd-diagnostics/REPORT.md) 现已完成。对七个人口独立核验 `R=M-B` 和 `Var(R)=Var(M)+Var(B)-2Cov(M,B)`，同时保存所有分位数、完整向量与 scatter sampled row indices。三个 seed 的 ID-test Pearson(M,B) 为平均 0.922；强相关描述了两个距离一起波动，不能据此声称 background subtraction 改善所有分布的因果机制。

CIFAR-10 相对 test ID 的 mean difference：delta M=+15.354、delta B=-50.317，因此 delta R=+65.672。Textures 则是 +249.693 - 138.366 = +111.327。相同公式的两个 signed 贡献可以方向不同；mean shift 不是排序指标，不能仅由这些均值推导 AUROC。

以原先固定的 95% ID-calibration 目标，RMD 与 matched CPU MD 在 CIFAR-10 上的四种决定比例为：

| 决定 | 三 seed 平均 CIFAR-10 占比 |
|---|---:|
| 两者都接受 | 85.09% |
| 只有 RMD 接受 | 1.13% |
| 只有 matched MD 接受 | 11.48% |
| 两者都拒绝 | 2.30% |

因此 OOD 接受率 net change=1.13%-11.48%≈-10.34 pp（用未舍入数据计算）。这同时保留了有利与不利变化，而非只数新拒绝的 OOD。RMD-minus-matched-MD：MNIST -8.73 ± 8.15 pp、SVHN -27.03 ± 1.74 pp、Textures -14.77 ± 0.33 pp，ID-test coverage -0.21 ± 0.38 pp。每个 ± 是同 seed 配对后跨 seed 的 sample SD；不同方法自己的固定阈值仍不等于严格相同 test coverage。

相对 Energy，RMD 在 CIFAR-10 OOD 接受率 +4.31 ± 1.37 pp、Textures -20.35 ± 3.38 pp；MNIST +2.98 ± 15.16 pp、SVHN -1.27 ± 15.01 pp。方向相反与大 seed 波动说明不能把相对 MD 的改善推广到每个参考/数据集。完整报告保留全部十一参考及 80/90/95/99% 目标，ID 逐类计数另存显式正确/错误分母；OOD 没有 classifier-correctness 标签。

## 已完成的 ASH：保留坐标、等值化与 exponential scale

[完整三 seed 报告](../report/ash-phase5/REPORT.md) 现已完成。我们固定作者函数默认 p=65，而非复现论文最佳参数搜索。512D 最后 pooled feature 每行保留179个最大 activation；同值按原坐标升序。原模型、head、输入与checkpoint均不变，不做温度或percentile拟合。135,000条训练ID特征先通过独立数值检查，之后才计算OOD分数。

设原 feature 为 f，h 是保留坐标上的原值，S1=sum(f)、S2=sum(h)：

- ASH-P 使用 h；它只移除较小坐标。
- ASH-B 把保留的179个坐标全部改成 S1/179，因此 sum(f_B)=S1；总和相同不表示head logits或分类预测相同。
- ASH-S 使用 h*exp(S1/S2)。这里必须有 exp，不能替换成只乘S1/S2；scale是当前样本定义，并非拟合系数。

三者经原head z=W f_shaped+b，再用 E=-logsumexp(z)，T=1，越大越OOD。例子仅演算：如果S1=100、S2=80，ratio=1.25，S的scale=exp(1.25)≈3.49；B的填充值≈0.559。它们保留相同support，但数值和head输出不同。

| 四自然 OOD 等权 macro，mean ± seed sample SD | AUROC % | 固定95% ID-calibration 下 OOD接受率 % |
|---|---:|---:|
| Energy | 80.97 ± 0.79 | 78.83 ± 2.60 |
| KNN | 82.42 ± 1.11 | 73.19 ± 3.62 |
| ASH-P | 81.42 ± 0.71 | 77.24 ± 2.73 |
| ASH-B | 69.48 ± 2.21 | 90.95 ± 4.41 |
| ASH-S | 83.16 ± 0.72 | 72.54 ± 4.43 |

S-minus-Energy 的同seed macro AUROC +2.18 ± 0.78 pp；CIFAR-10约-0.01±0.14 pp，MNIST/SVHN/Textures分别+3.39/+2.04/+3.31 pp。完整报告保留所有变体对十二参考的配对结果，不能把macro提升称为每个人口都改善，也不能据此选择更有利的percentile。

固定95%校准目标下，S OOD接受率仍有72.54 ± 4.43%，实际ID覆盖率94.99 ± 1.11%；相对Energy的接受率变化-6.29 ± 4.11 pp，coverage变化-0.16±0.70 pp。相同校准目标不等于相同test coverage，接受率也不是benchmark FPR95-ID。

原始classifier accuracy为75.77%；shaped P/B/S accuracy分别75.66%/71.77%/75.67%。B改变约1,300条ID预测，P/S各约160–180条；这些中间预测准确率单列，所有risk/coverage仍按原始预测。

S缩放P特征不保证P/S的argmax完全相同：含原bias时 z_S=c*z_P+(1-c)*b，c=exp(r)。实际P/S在三个seed上有19/9/23条ID argmax不同。忽略bias会错误声称缩放不改变预测；即使预测几乎不变，样本相关scale仍可改变跨样本Energy排序。

本阶段独立核验675项AUROC/AP、1,260个masks、720个旧控制数组和3,655个汇总统计，全部完成检查通过。一次报告计数被显示成浮点列表导致展示检查失败；已保留失败来源、只修复整数格式，再通过完整审计，未放宽数值容差或修改原始scores。

## 已完成的 ASH 诊断：coordinate count、activation mass 与决定变化

[完整诊断报告](../report/ash-diagnostics/REPORT.md) 已完成。保留179/512≈34.96%的坐标，不表示仅保留34.96%的activation总和：最大的坐标携带更多mass。test ID平均S2/S1=84.22±0.06%，CIFAR-10为80.17±0.12%；平均exp(S1/S2)约3.306/3.509。一般mean(exp(r))不等于exp(mean(r))，所以所有逐样本scale与population统计都单独保存。均值差和相关性不能推出AUROC或因果。

ASH-B中间预测在test ID修正约2.24%的原错误，却破坏约6.24%的原正确，net accuracy change=-4.00 pp。只数修正的错误会掩盖更大的损失。完整四态correctness codes还包含“两者均正确/均错误”，argmax改变也可能仍然预测错；全部100原ID classes有显式分母。

固定95% ID-calibration目标，ASH-S与Energy在CIFAR-10的决定比例（三seed均值）为：

| 决定 | C10占比 |
|---|---:|
| 两者均接受 | 76.76% |
| 只有S接受 | 3.92% |
| 只有Energy接受 | 5.16% |
| 两者都拒绝 | 14.16% |

S的OOD接受率变化=3.92%-5.16%=-1.24 pp（paired sample SD1.79 pp）。这里S-only是不利的新接受，Energy-only被S拒绝是有利变化；不能只展示后者。MNIST/SVHN/Textures的paired变化分别-7.94±6.64/-10.69±6.07/-5.28±2.38 pp；ID coverage变化-0.16±0.70 pp，原classifier retained-error-risk变化-0.25±0.36 pp。因此不是严格相同test coverage下的比较，且seed差异必须保留。

三变体对十二参考的432个设置全部保存，没有选有利比较。独立核验378个component vectors、2,592个接受决定数组、345,600个ID逐类cell及63个完整shaped head；15,654个统计记录/1,678个展示字段通过审计。一次新macro被误用逐字一致检查的审计错误已保留；独立计算差值最大1.11e-16，按原冻结atol/rtol校验后通过，原复制值仍要求完全一致。

[完整paired image bootstrap](../report/paired-bootstrap/REPORT.md)现已完成：固定1,000次image draws共享给全部15列方法和三个模型，所有105个pair与四操作目标完整保留。它描述固定模型、固定阈值条件下的样本抽样波动，不能替代训练seed波动，也不能修复validation reuse。三seed sample SD不是confidence interval；bootstrap的名义95%percentile interval没有自动赋予总体coverage或显著性保证，不用于选参数/方法。

例如ASH-S自然macro AUROC=83.1554%，原三seed SD=0.7206pp；固定三模型均值的图像interval=[82.6876%,83.5626%]。S-minus-Energy的paired macro差值=+2.1820pp，其图像interval=[+2.0545,+2.3103]pp，原paired-seed SD=0.7821pp。三个数回答不同的问题，不能把图像interval当成重新训练的模型差异范围。CIFAR-10差值仅-0.0074pp，图像interval=[-0.1651,+0.1340]pp；保留这个方向和跨零区间，不把四人口平均改善说成每个人口都有改善。

固定95%操作点的S-minus-Energy自然OOD acceptance差值=-6.2868pp，图像interval=[-6.5566,-6.0195]pp；同时ID coverage差值=-0.1567pp、interval=[-0.3434,+0.0267]pp。两个方法实际coverage不同，所以这个差值不是严格相同ID coverage的检测比较。ID risk始终依据原始classifier correctness，而不是ASH改变后的中间预测。

独立raw审计核验6,000个draw-count向量、126万精确计数与全部操作率、前10个expanded resamples的13,500个ranking字段。Python statistics/手算linear quantile独立核验31,680个统计记录，图表最终视觉检查通过。共享y轴重复invert导致第一次图的排序问题已留存，修正后的每个数值记录与之前完全一致。85tests有83通过，2项原optimizer skips已在原CPU环境验证。

[bootstrap dependence](../report/bootstrap-dependence/REPORT.md)已完成：同一个图像bootstrap draw里，全部模型/方法共享ID和相应OOD人口的抽样计数。三模型均值的方差由各模型方差与三个cross-covariance之和决定；如果只保留对角项，相当于一个不成立的“模型图像抽样独立”代数对照。15方法的自然macro AUROC observed/diagonal-only图像SD比值范围1.42–1.58；四自然人口macro的45个seed-method比值1.38–1.66。共保存66个45×45矩阵、7,920个三模型分解、360个四人口分解，所有值和undefined correlation mask都经独立scalar核验。相关性是固定已训练模型的图像波动，**不是**不同训练seed的泛化方差，更不能证明某feature导致检测改善。小方差原表格保留了展示失败版本，现以科学记数法显示，原数值未变。

[项目复现索引和证据评估](../report/project-assessment/REPORT.md)汇总11个完成阶段与各自审计。主结论是本地协议和数值可复核，但后续方法是在看过OOD test后发展的，校准又重用了checkpoint选择的ID validation子集；单一CIFAR-100 ResNet-18与四个自然OOD人口也不等于完整外部基准。

## 已完成的三路ID validation隔离复验

[预先冻结的协议](VALIDATION_ISOLATION.md)和[完整结果](../report/validation-isolation/REPORT.md)现在形成一个独立的探索性复验。原5,000张validation被重新分成互斥的checkpoint 2,000、temperature 1,500和threshold 1,500；这修正了旧实验“同一批validation既选checkpoint又做后续校准”的设计问题。新三seed均完成原固定100epoch，没有用ID test/OOD test选择checkpoint或超参数。

| 量 | 旧实验 | 三路隔离复验 |
|---|---:|---:|
| checkpoint选择epoch（seed 0/1/2） | 89/96/92 | 92/96/98 |
| ID-test accuracy（三seed均值±样本SD） | 75.77±0.30% | 75.80±0.27% |
| T=1 Energy自然OOD macro AUROC | 80.97±0.79% | 80.78±0.87% |
| 固定95% ID校准目标下Energy自然OOD接受率 | 78.62±2.60% | 79.74±3.04% |

ID-test accuracy的同seed配对差是+0.03±0.08pp，Energy AUROC配对差-0.20±0.17pp。新Energy实际ID-test接受率是95.18±0.75%，不能把OOD接受率变化说成严格相同ID coverage下的检测差异。seed 1在新旧实验选中了同一epoch、全部122个模型张量完全一致；但其temperature/threshold ID拟合样本不同。seed 0和2还改变了选中epoch，因此不能把所有差异单独归因于validation角色重叠。

六个预定detector、T=1/ID拟合温度、四个自然OOD集及独立noise诊断的完整三seed值、AUPR、FPR95-ID、四个operating target和人口数保存在[机器可读汇总](../report/validation-isolation/summary.json)。原始逐图logits/features/scores/masks留在本地runs/validation-isolation；训练、训练ID特征重算、评分与统计/图表均经独立审计，91项回归为89通过、2项旧optimizer测试跳过，证据见[完成清单](../report/validation-isolation/completion.json)。这四个OOD test早已用于方法开发，因此隔离ID validation不等于获得“未见过的OOD确认”；下一步若要确认，须先固定新的未观察OOD人口和完整数据诚信协议。

## 新增未评分自然 OOD 人口：Tiny ImageNet-200

在三路 ID validation 隔离复验完成后，[新人口协议](TINYIMAGENET_CONFIRMATION.md)先于数据下载冻结；预处理布局错误也在首次图像解码之前以[增补协议](TINYIMAGENET_CONFIRMATION_AMENDMENT_1.md)修正并保留原文。三个原已训练模型的 checkpoint、温度、KNN/Mahalanobis 训练 ID 特征拟合和 80/90/95/99 ID 阈值均不使用 Tiny ImageNet。官方 validation 的 10,000 张图像全数评分，没有挑选子集或 OOD validation。

[完整报告](../report/tinyimagenet-confirmation/REPORT.md)展示六个主要 detector 的 T=1 与 ID 温度版本。例子：T=1 Energy 的 Tiny ImageNet AUROC 为 78.95±0.33%（三训练 seed 均值±样本 SD），KNN-50 为 80.17±0.28%。在固定95% ID-validation 目标下，两者 OOD 接受率分别为 78.80±2.25% 和 78.65±2.75%；实际 ID-test 接受率分别为 95.18±0.75% 和 95.53±0.72%。不要把“固定 ID 目标”误读成两方法实际 test ID coverage 严格相等，更不能据此选最佳方法。

[机器可读结果](../report/tinyimagenet-confirmation/summary.json)保留全部 seed 点、AUPR、FPR95-ID、四目标接受率、计数和阈值；本地原始 logits/features/scores/masks 及 [70 文件完成校验](../report/tinyimagenet-confirmation/completion.json)支持复核。缓存的 10,000 张解码像素、三模型特征评分、全部分数/阈值掩码、汇总和展示数值经独立审计，完整回归为 89 通过、2 项原环境已知跳过。这个人口可能与 CIFAR-100 有语义重叠，也混合了不同图源和64→32分辨率变化；单一新人口不能推出普适 OOD 泛化或与 OpenOOD 排行榜直接比较。

## Tiny ImageNet 的类别间差异只是描述性诊断

[固定类别诊断协议](TINYIMAGENET_CLASS_HETEROGENEITY.md)在查看逐类别指标前写定：官方 validation 的200个synset每类50张，所有类别都保留；每类的50张OOD都与同一组10,000张ID test比较，checkpoint、分数与阈值不变。[完整报告](../report/tinyimagenet-class-heterogeneity/REPORT.md)给出六方法和两温度版本的类别分布，原始7,200个AUROC及28,800个接受计数见[class_values.npz](../report/tinyimagenet-class-heterogeneity/class_values.npz)。

例如T=1 Energy的200个“类别三seed均值”AUROC中位数为80.23%，四分位区间75.75–84.28%；固定95% ID阈值下OOD接受率中位数80.00%，四分位区间73.83–85.33%。这些是已观察类别之间的离散程度，不是置信区间，也不是换一个训练seed或新OOD数据集的预期范围。单seed每类只有50张，接受率只能按2个百分点变化；跨三seed平均后是2/3个百分点的网格。

独立审计用sorted-ID成对比较公式重算全部类别AUROC，并核对类别等权平均与原总体AUROC一致、全部类别接受数与原总体计数一致。两张图经目视修复与复审，[完成清单](../report/tinyimagenet-class-heterogeneity/completion.json)绑定24个本地文件。不能根据类别极值重新挑选方法，也不能从synset编号推断“语义近/远”的因果机制。


## 顺序选择的 Fashion-MNIST 人口扩展

[冻结协议](FASHIONMNIST_EXTENSION.md)在访问 Fashion-MNIST 测试图像前写定，三颗隔离训练的 CIFAR-100 模型、六个原方法、ID-only 温度与 80/90/95/99 接受阈值全部复用。这个人口是在看过 Tiny ImageNet 结果后才选择的，所以它是额外的顺序扩展，不是预先规划的多数据集确认套件，也不能并入原四自然 OOD 的 macro。

[完整报告](../report/fashionmnist-extension/REPORT.md)覆盖官方 test 全 10,000 张，并与同一 10,000 张 CIFAR-100 ID test 比较。例子：T=1 Energy AUROC 为 90.42±0.16%，KNN-50 为 90.48±0.44%，Pooled Mahalanobis 为 60.51±2.72%（三训练 seed 均值±样本 SD）；固定 95% ID-validation 阈值下，它们的 Fashion OOD 接受率分别是 58.76±5.65%、54.00±1.20%、97.72±0.78%。这是对固定方法的描述，不据此选择赢家；实际 ID-test 接受率、所有方法/温度/阈值与每颗 seed 见[机器可读结果](../report/fashionmnist-extension/summary.json)。

这组结果强烈混合了语义与图像来源差异：[逐像素来源审计](../runs/fashionmnist-extension/source-shift-descriptive.json)显示，重复 RGB 后全部 10,000 张 Fashion 图像严格为灰度，CIFAR-100 ID test 只有 153/10,000 张严格为灰度。均值亮度也不同。检测分数不能单独归因于“衣物类别语义”。全部 raw logits/features/scores/masks、三 seed 独立特征与评分审计、图表人工检查、91 项回归（89 通过、2 项既有跳过）和[76 文件完成校验](../report/fashionmnist-extension/completion.json)均已保存。首次 GPU 拥堵尝试与后来按四项负载门槛顺序恢复的证据保留，不算额外 seed 或模型结果。


## CIFAR-100 同源 held-out-class：完整三 seed 结果

[预先冻结的协议](CIFAR100_CLASS_HOLDOUT.md)先从官方训练标签确定每个 coarse group 留出一个 fine class，形成 80 个 ID 类与 20 个 OOD 类；held-out-class 的训练图像完全排除。三个新 80 类 ResNet-18 各训练完整 100 epochs，checkpoint 只由 1,600 张 ID 验证图选择，温度和接受阈值分别由互斥的 1,200 张 ID 图拟合。官方 CIFAR-100 test 的 8,000 张 ID 与 2,000 张 OOD 全数评分。

[完整三 seed 报告](../report/cifar100-class-holdout/SUMMARY.md)同时展示六个固定方法、T=1/ID 温度、AUROC/AUPR/FPR95-ID 和四个 ID-validation 接受目标。ID-test 分类准确率为 **76.96±0.38%**。例如 T=1 Energy AUROC 为 **73.57±0.73%**，固定 95% ID-validation 目标下实际 ID-test 接受率 **95.87±0.14%**、OOD 接受率 **86.47±0.06%**；KNN-50 的对应 AUROC 为 **73.09±1.23%**、OOD 接受率 **88.35±0.93%**。这些均为三训练 seed 均值±样本 SD，不用于事后挑选方法。FPR95-ID 使用 ID **test** 次序统计量，是独立的基准操作点，不能与 ID-validation 固定阈值的接受率混为一谈。

本阶段使用同一官方 CIFAR-100 图像来源和 32×32 格式，减少了 Fashion-MNIST 中明显的灰度/来源混杂；但同时改变了分类器和 ID 标签空间，所以两阶段数值差异不能被解释为“同源性”的因果效应。项目此前已经使用过 CIFAR-100 图像，这也不是未触碰的外部确认。[机器可读结果](../report/cifar100-class-holdout/summary.json)、全部原始逐图数组、各阶段独立审计、91 项回归（89 通过、2 项既有跳过）和[160 文件完成清单](../report/cifar100-class-holdout/completion.json)均保留本地。Seed 2 的同一冻结 checkpoint 在原训练 CUDA 路径重算为 1,253/1,600，在 CPU 路径为 1,254/1,600；只有一张边界图预测不同，两套 ID-only 原始 logits 均保存，见[运行记录](CLASS_HOLDOUT_RUNBOOK.md)。


## 留出类别间差异：只描述，不重新挑选方法

[事先写定的逐类诊断协议](CLASS_HOLDOUT_HETEROGENEITY.md)对上节同一 OOD test 的全部 20 个 held-out fine class 各保留 100 张图，并用同一完整 8,000 张 ID test 作 AUROC 参考。六个方法、两种温度、三颗已冻结模型和四个 ID-only 阈值原样复用。[完整报告](../report/cifar100-class-holdout-classes/REPORT.md)列出各方法逐类三 seed 均值的 min/Q1/median/Q3/max；例如 T=1 Energy 的逐类 AUROC 范围为 **57.79%–87.79%**，中位数 **73.78%**，显示总体 73.57% AUROC 掩盖了较宽的类别差异。这里的 20 个类别共享相同 ID 参考与测试来源；这组范围不是置信区间，也不支持从极值类别挑选新方法。

[完整逐类 AUROC 表](../report/cifar100-class-holdout-classes/class_auroc.csv)保留 240 行全部方法/温度/类别的三 seed 值及样本 SD；[固定阈值接受表](../report/cifar100-class-holdout-classes/class_acceptance.csv)保留 960 行接受计数和实际 ID-test coverage。每类只有 100 张 OOD，一颗 seed 的接受率按 1 个百分点跳变。720 个 AUROC 和 2,880 个接受计数经独立 raw-score/原阈值复算；每个 seed 的 20 类均值或计数总和与原完整 OOD 人口严格对应。[36 文件完成证据](../report/cifar100-class-holdout-classes/completion.json)还重新验证了底层 160 文件基准。逐类 AUPR 被省略，因为 100:8,000 的类别子集 prevalence 与原 2,000:8,000 不同。

## 同源留出类的配对图像 bootstrap：区分两种波动

在已冻结的[抽样协议](CLASS_HOLDOUT_BOOTSTRAP.md)中，1,000 次重复每次都从**每个已观察 fine class 的 100 张 test 图**有放回抽 100 张，因此 ID/OOD 总数始终为 8,000/2,000。每次的同一组图像权重用于全部三颗模型、六方法和两温度。每颗模型的原始 AUROC、AUPR 和 FPR95-ID 在抽样下重新计算；FPR95-ID 的阈值来自该次抽样的 ID **test** 次序统计量，而固定 95% 操作目标仍使用原先互斥 ID validation 拟合的阈值。

[全方法报告](../report/cifar100-class-holdout-bootstrap/REPORT.md)把三训练 seed 的原始样本 SD 与固定三模型均值的图像抽样 2.5/50/97.5% 分位数分开列示。例如 T=1 Energy 的完整 test AUROC 为 **73.57±0.73%**（均值±训练 seed SD）；图像抽样的描述性 2.5–97.5% 范围为 **72.58–74.53%**。固定 95% ID-validation 目标的 OOD 接受率原值 **86.47±0.06%**，图像抽样范围 **85.33–87.58%**。[72 行数值表](../report/cifar100-class-holdout-bootstrap/statistics.csv)包含所有六个指标/操作量和全部方法，而原始 1,000 次整数权重与指标保存在本地 blocks。独立审计重新生成每次分层抽样、全部 144,000 个 ranking 字段和 72,000 个固定阈值接受计数，并用展开图像的 scikit-learn 核对前十次；[67 文件完成证据](../report/cifar100-class-holdout-bootstrap/completion.json)通过只读复核。

这个抽样只刻画**固定类别、固定类别划分、固定三颗模型**时图像样本变化。它不能估计换留出类别或重训模型的不确定性，也不是有标称覆盖率的置信区间。原图像和 OOD test 已用于先前报告，所以不构成独立验证或挑选最优方法的依据。

## 已冻结 80 类模型在 CIFAR-10 上的扩展

[事先写定的扩展协议](CLASS_HOLDOUT_CIFAR10_EXTENSION.md)把上面的三颗完整 100-epoch 80 类模型、36k 张优化训练 ID 的特征库、互斥 ID 验证角色拟合的温度和四组接受阈值固定，再读取完整 10,000 张官方 CIFAR-10 test。CIFAR-10 的原始行/标签/像素与官方数据校验值先通过独立审计；三颗模型随后逐个通过 GPU 门控提取 80 维 logits、512 维特征及六种分数，并经 CPU 独立重拟合训练 ID 特征探针、NumPy/scikit-learn 指标复算和全部固定阈值掩码检查。没有用 CIFAR-10 图像重新训练、拟合、调阈值或挑选方法。

[完整三 seed 报告](../report/class-holdout-cifar10-extension/REPORT.md)展示全部六方法、两温度，且给出与原留出类人口共用 ID test/阈值的 AUROC 与固定 95% OOD 接受率对照。T=1 Energy 在 CIFAR-10 上的 AUROC 为 **76.49±0.58%**、固定 95% ID-validation 目标的 OOD 接受率为 **85.38±0.59%**；对应原 2,000 张同源留出类为 **73.57±0.73%** 与 **86.47±0.06%**。数值均为三训练 seed 均值±样本 SD，不能仅凭人口差异推断来源或语义的因果效应。[48 行指标](../report/class-holdout-cifar10-extension/metrics.csv)和[144 行操作量](../report/class-holdout-cifar10-extension/operating.csv)保留全部方法/目标/种子值；91 项回归通过，[125 文件完成清单](../report/class-holdout-cifar10-extension/completion.json)可只读复核。

CIFAR-10 OOD 对比的实际比例是 10,000/(8,000+10,000)=**55.56%**，同源留出类是 2,000/(8,000+2,000)=**20%**；AUPR 强烈依赖这个比例，不能把两组 AUPR 当成同条件比较。CIFAR-10 在本项目早期已用过，而且这项扩展在查看同源结果后才选定，因此它是事后探索，不是新的未接触外部确认，也不用于选出最佳检测器。

## 固定 OOD 比例后再看 AUPR

AUPR 的 precision 分母同时含 ID 和 OOD 数量，所以原同源留出类的 20% OOD 比例与 CIFAR-10 扩展的 55.56% OOD 比例下的 AUPR 不能直接比较。[预先写定的标准化协议](CLASS_HOLDOUT_PRIOR_STANDARDIZATION.md)把两个人口都按假设性 **20% OOD / 80% ID** 比例重新加权：每张 ID 图的权重为 0.8/8000，每张 OOD 图的权重为 0.2/n_OOD。全部原始逐图分数保留，未抽子集、重训、重新校准或修改阈值；AUPR-out 按 OOD 为正例，AUPR-in 把 ID 为正例并反转分数。

[完整全方法报告](../report/class-holdout-prior-standardization/REPORT.md)列出三 seed、六方法、两温度下的两个人口 AUPR-out/in。T=1 Energy 的 CIFAR-10 AUPR-out 原为 **76.26±0.43%**（自然 55.56% OOD 比例），固定到 20% 后为 **39.94±0.59%**；同一假设比例下的留出类为 **37.38±1.00%**。三 seed 均值±样本 SD 与全部原始点在[48 行数值表](../report/class-holdout-prior-standardization/statistics.csv)。144 个标准化 AP、144 个自然比例 AP 由独立 scikit-learn 加权复算；72 个留出类值与其原始自然 20% 比例结果一致，[42 文件完成证据](../report/class-holdout-prior-standardization/completion.json)通过只读核对，91 项回归通过。

这个 20% 比例是为了展示同一条件下的 precision-recall 算术，不是估计出来的部署比例。它假定各人口内部已观察到的分数分布不变，因此不能消除来源、语义、类别划分或此前看过测试集的限制，也不用于挑选最佳方法。

## 预声明对比检验：探索出来的"赢家"能否在新人口上复现？

ASH-S 和 RMD 都是在看过原四个 OOD 测试集之后才被认为"更好"的方法。在同一批数据上反复试方法、再挑最好的，结果天然偏乐观（"分岔花园"）。这一步的做法是：**先写死假设和判定规则，再算分**。协议在 git 提交 59e914f 中冻结，冻结时 ASH/RMD 从未在三颗隔离模型或 Tiny ImageNet、Fashion-MNIST、Places365、Oxford Pets 上计算过。

判定规则同样事先固定：

- 某个人口上，三颗 seed 的配对 AUROC 差值全部为正，才算"支持"；全部为负，算"反驳"；其余都算"混合"。
- 四个人口全部"支持"，才算"确认"。

| 假设 | Tiny ImageNet | Fashion-MNIST | Places365 | Oxford Pets | 结论 |
|---|---|---|---|---|---|
| H1 ASH-S > Energy | −0.24 反驳 | +1.23 支持 | +1.09 支持 | +0.22 混合 | **未确认** |
| H2 RMD > 匹配 MD | +20.1 支持 | +27.6 支持 | +19.9 支持 | +22.0 支持 | **确认** |

怎么读这个结果：

1. **H2 确认，但信息量有限。**RMD 修好了 Mahalanobis，这一点在所有数据上都很稳定；可 RMD 并不比最简单的 Energy 好（差值分别是 +0.00、−2.35、+3.41、−3.05 pp）。"修好了一个坏方法"和"成为更好的方法"是两回事。
2. **H1 未确认。**探索阶段 ASH-S 比 Energy 平均高 2.18 pp，到了新人口上，有增益的地方只剩一半左右，在 Tiny ImageNet 上三颗 seed 还全部变差。
3. **事后观察，不参与判定。**ASH-S 在图像来源差别大的人口上有增益（灰度、数字、纹理、场景），在物体类自然照片上几乎为零。这只能作为下一个待检验的假设。
4. **整体看**，在 Tiny ImageNet、Places365、Pets 上，没有任何方法能在 95% ID 通过时把 OOD 通过率压到约 76% 以下。这再次说明瓶颈在表示本身，而不是打分公式。

这就是"探索"与"确认"的区别：探索阶段看起来稳定的 +2 pp 增益，经过事先写死规则的检验后，只剩下有条件的一部分。完整数字见 [报告](../report/posthoc-confirmation/REPORT.md)。
