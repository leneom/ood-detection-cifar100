"""Reproducibility/evidence map across already audited local OOD phases."""
from pathlib import Path
import shutil
from geometry_io import ROOT,read,write,digest,require,now

REPORT=ROOT/'report/project-assessment'
PHASES=[
 ('baseline-complete','matrix','Three-seed six-method natural OOD baseline'),
 ('calibration-phase2','calibration-phase2','ID-only temperature calibration'),
 ('selective-risk-phase2','selective-risk-phase2','Four fixed validation operating targets'),
 ('geometry-phase3','geometry-phase3','Training-ID ViM, Residual and ReAct'),
 ('geometry-diagnostics','geometry-diagnostics','Feature-geometry saved-output diagnostics'),
 ('rmd-phase4','rmd-phase4','Training-ID RMD and matched CPU MD'),
 ('rmd-diagnostics','rmd-diagnostics','RMD component/transition diagnostics'),
 ('ash-phase5','ash-phase5','Fixed ASH-P/B/S extensions'),
 ('ash-diagnostics','ash-diagnostics','ASH support/scale/decision diagnostics'),
 ('paired-bootstrap','paired-bootstrap','Full paired image-bootstrap intervals'),
 ('bootstrap-dependence','bootstrap-dependence','Saved-bootstrap dependence decomposition')]

def main():
    require(not REPORT.exists(),'assessment output exists; preserve prior version before rebuilding')
    inputs={};phase_rows=[]
    for report_name,run_name,title in PHASES:
        paths={'report':ROOT/'report'/report_name/'REPORT.md','completion_audit':ROOT/'report'/report_name/'completion-audit.json','run_status':ROOT/'runs'/run_name/'status.json'}
        require(all(p.is_file() for p in paths.values()),'missing completed phase evidence '+report_name)
        audit=read(paths['completion_audit']);status=read(paths['run_status'])
        require(audit['state'] in ['passed','complete','complete_and_verified'] and status['state']=='complete','phase not verified '+report_name)
        relative={k:str(v.relative_to(ROOT)).replace('\\','/') for k,v in paths.items()}
        hashes={k:digest(v) for k,v in paths.items()}
        for k,n in relative.items():inputs[n]=hashes[k]
        phase_rows.append({'phase':report_name,'title':title,'paths':relative,'sha256':hashes,'audit_state':audit['state'],'run_state':status['state']})
    more=['docs/PROTOCOL.md','docs/PAIRED_BOOTSTRAP.md','docs/BOOTSTRAP_DEPENDENCE.md','report/baseline-complete/overview.json',
          'runs/matrix/summary/summary.json','report/paired-bootstrap/summary.json','report/bootstrap-dependence/plot-data.json']
    for n in more:inputs[n]=digest(ROOT/n)
    original=read(ROOT/'report/baseline-complete/overview.json');first=read(ROOT/'runs/matrix/summary/summary.json');boot=read(ROOT/'report/paired-bootstrap/summary.json');depend=read(ROOT/'report/bootstrap-dependence/plot-data.json')
    require(original['seeds']==first['seeds']==[0,1,2] and abs(original['id_test_accuracy']['mean']-first['id_test_accuracy']['mean'])<1e-12,'classifier provenance differs')
    require(boot['replicates']==1000 and len(boot['methods'])==15 and len(boot['pairs'])==105 and len(boot['endpoints'])==66,'complete fixed benchmark scope changed')
    key='ranking/natural_macro/auroc';s=boot['endpoints'][key];id_acc=first['id_test_accuracy']
    metrics={
      'id_test_accuracy_mean':id_acc['mean'],'id_test_accuracy_seed_sd':id_acc['sample_sd'],'id_test_accuracy_by_seed':id_acc['per_seed'],
      'energy_macro_auroc':s['methods']['energy']['fixed_three'],
      'knn_macro_auroc':s['methods']['knn']['fixed_three'],
      'rmd_macro_auroc':s['methods']['rmd']['fixed_three'],
      'matched_cpu_md_macro_auroc':s['methods']['mahalanobis_cpu']['fixed_three'],
      'ash_s_macro_auroc':s['methods']['ash_s_energy']['fixed_three'],
      'ash_s_minus_energy_macro_auroc':s['pairs']['ash_s_energy__minus__energy']['fixed_three'],
      'ash_s_minus_energy_cifar10_auroc':boot['endpoints']['ranking/cifar10/auroc']['pairs']['ash_s_energy__minus__energy']['fixed_three'],
      'ash_s_minus_energy_natural_acceptance_95':boot['endpoints']['operating/0.95/natural_ood_acceptance_macro']['pairs']['ash_s_energy__minus__energy']['fixed_three'],
      'ash_s_minus_energy_id_coverage_95':boot['endpoints']['operating/0.95/id_coverage']['pairs']['ash_s_energy__minus__energy']['fixed_three']}
    ratios=depend['three_model_ratios'][key];natural=depend['four_natural_ratios']
    metrics['three_model_sd_ratio_range_natural_macro_auroc']=[min(ratios.values()),max(ratios.values())]
    metrics['four_natural_sd_ratio_range_natural_macro_auroc']=[min(v for row in natural.values() for v in row),max(v for row in natural.values() for v in row)]
    for n,h in inputs.items():require(digest(ROOT/n)==h,'input changed during assessment '+n)
    REPORT.mkdir(parents=True);source=Path(__file__);snap=REPORT/'executed-source';snap.mkdir(exist_ok=False);shutil.copy2(source,snap/source.name)
    write(REPORT/'inputs.json',{'state':'bound_before_assessment_serialization','prepared_at':now(),'input_sha256':inputs,'source_sha256':digest(source),'source_snapshot':'report/project-assessment/executed-source/build_project_assessment.py','phase_count':11})
    write(REPORT/'index.json',{'state':'report_pending_independent_audit','created_at':now(),'phase_count':11,'phases':phase_rows,'metrics':metrics,
                               'boundary':'Local reproducibility/evidence index of previously frozen phases; not new experiments, benchmark certification or publication claim'})
    percent=lambda x:f'{100*x:.2f}%';point=lambda x:f"{100*x['original_point']:.2f}%";interval=lambda x:f"[{100*x['percentile_lower']:.2f}%, {100*x['percentile_upper']:.2f}%]"
    diff=lambda x:f"{100*x['original_point']:+.2f} pp";diff_interval=lambda x:f"[{100*x['percentile_lower']:+.2f}, {100*x['percentile_upper']:+.2f}] pp"
    lines=['# OOD Detection：项目复现索引与证据评估','',
           '当前工作是同一 CIFAR-100 ID 定义下的完整三 seed 本地实验与事后方法分析。三个 ResNet-18 均按预先固定的 100-epoch 预算训练，checkpoint 仅按 ID validation 选择；OOD test 没有参与训练、checkpoint 或训练 ID feature 统计拟合。下面的后续方法对比是在已看过 OOD 结果后扩展的，属于探索性而非新的独立确认。','',
           f"原始 ID-test accuracy 为 **{percent(metrics['id_test_accuracy_mean'])} ± {100*metrics['id_test_accuracy_seed_sd']:.2f} pp**（三训练 seed 的样本 SD）；各 seed 为 "+', '.join(percent(v) for v in metrics['id_test_accuracy_by_seed'])+'。旧 continual-learning 项目和所有原 checkpoint/raw logits/scores 保持不变。','',
           '## 可复现的证据链','',
           '所有数据、checkpoint、hash、审计和报告都留在本地。按顺序从原模型、ID-only calibration、固定操作点、训练特征方法、结果后诊断、图像 bootstrap 到共享样本依赖分析读取；每一项下方均绑定独立完成检查。']
    for row in phase_rows:lines.append(f"- [{row['title']}](../{row['phase']}/REPORT.md) — [完成审计](../{row['phase']}/completion-audit.json)，状态 `{row['audit_state']}`。")
    lines+=['','[完整实验协议](../../docs/PROTOCOL.md)；[阶段进展](../../docs/PROGRESS.md)；[中文原理与解释](../../docs/LEARNING_GUIDE.zh-CN.md)。`index.json` 保留上面11个报告、完成审计、运行状态及其 SHA-256，并绑定主协议与主要原始汇总。','',
            '## 已支持的描述性结果','',
            f"四自然数据集等权 macro AUROC：Energy {point(metrics['energy_macro_auroc'])}、KNN {point(metrics['knn_macro_auroc'])}、RMD {point(metrics['rmd_macro_auroc'])}、matched CPU MD {point(metrics['matched_cpu_md_macro_auroc'])}、ASH-S {point(metrics['ash_s_macro_auroc'])}。完整15列、各人口、AUPR/FPR95-ID和四目标都在[完整 paired-bootstrap 报告](../paired-bootstrap/REPORT.md)，不能只看这一行方法。",
            f"ASH-S−Energy 自然 macro AUROC 差 {diff(metrics['ash_s_minus_energy_macro_auroc'])}，固定三模型下的名义95%图像区间 {diff_interval(metrics['ash_s_minus_energy_macro_auroc'])}；CIFAR-10 单独差 {diff(metrics['ash_s_minus_energy_cifar10_auroc'])}，区间 {diff_interval(metrics['ash_s_minus_energy_cifar10_auroc'])}。四人口均值的改善不等于所有人口都改善，也不是新 test-set 上的确认性结论。",
            f"固定95% ID-calibration 目标下，ASH-S−Energy 自然 OOD acceptance 差 {diff(metrics['ash_s_minus_energy_natural_acceptance_95'])}，同时实际 ID-test coverage 差 {diff(metrics['ash_s_minus_energy_id_coverage_95'])}；测试 coverage 未被强制匹配。",
            f"[共享样本诊断](../bootstrap-dependence/REPORT.md)中，15方法的三模型自然 macro AUROC SD/忽略跨模型协方差的代数 SD 比值范围为 {metrics['three_model_sd_ratio_range_natural_macro_auroc'][0]:.2f}–{metrics['three_model_sd_ratio_range_natural_macro_auroc'][1]:.2f}；每 seed/方法的四自然人口 macro 比值范围为 {metrics['four_natural_sd_ratio_range_natural_macro_auroc'][0]:.2f}–{metrics['four_natural_sd_ratio_range_natural_macro_auroc'][1]:.2f}。后者共45个方法-seed单元。共用ID图像采样会让把各 seed/人口当独立图像抽样低估此实验的波动，但这些比值**不**估计重新训练模型的方差。",
            '', '## 结论的边界','',
            '- 模型选择用过全部5,000条ID validation；后来的温度与操作阈值只用其中重新划分的ID validation子集，所以这些阈值并非独立holdout校准。实际ID-test coverage另报；FPR95-ID另用ID test order statistic，只是benchmark点。',
            '- ViM、RMD、ASH 等扩展是在基线OOD结果之后设计/加入。固定后没有用OOD test调单个参数或选checkpoint，但当前同一组OOD test上的后续比较仍是探索性的；上述图像区间不能消除方法发展过程中的重复使用。',
            '- 三个训练 seed 的原始样本SD与1,000次paired image bootstrap的条件区间是两种不同不确定性。所有15列方法和三个模型共用每次图像权重；合成noise仅单独诊断，不进入四自然macro。',
            '- CIFAR-10/MNIST/SVHN/Textures的样本数与OOD占比不同，AUPR不能脱离占比比较。CIFAR语义重叠、单一模型架构与有限数据集使结果不能直接当完整OpenOOD基准或部署保证。',
            '- ID/Textures的执行时像素指纹跨 seed 核对过；较早的CIFAR-10/MNIST/SVHN/noise人口按归档执行loader、文件/cache/顺序和noise RNG对齐，没有伪造当时未捕获的执行像素hash。',
            '- 非因果的score/feature支持、transition和相关性诊断不能证明改进机制。数值控制列 affine Energy、matched CPU MD 保留在15列全量表格，但不算独立新 detector。',
            '', '## 下一项具体工作','',
            '优先补一项**独立ID validation校准复验**：在重新训练/重选checkpoint前，先把原固定5,000条ID validation按类别预定义成互斥的checkpoint、temperature-fit、threshold-calibration三份；保持45,000条train和官方test/OOD完全隔离，预先冻结三个训练 seed、100-epoch预算、十五列/四目标以及原始分数保存/审计规则。这样能检验当前validation reuse的影响。仅在已有OOD数据上重跑仍不构成独立OOD确认；进一步泛化评估要先冻结新增自然OOD集、数据版本/完整性及分类定义，并和探索过程中已看过的OOD人口分开。',
            '', '本报告是研究进展和复现边界，不是论文新颖性、统计显著性、完整外部基准或公开发布判断。没有推送、发布或对外发送数据。','']
    (REPORT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    for n,h in inputs.items():require(digest(ROOT/n)==h,'input changed during assessment output '+n)
    write(REPORT/'build.json',{'state':'built_pending_independent_audit','built_at':now(),'inputs_sha256':digest(REPORT/'inputs.json'),'index_sha256':digest(REPORT/'index.json'),'report_sha256':digest(REPORT/'REPORT.md'),'source_sha256':digest(source),'phase_count':11})
    print({'state':'built_pending_independent_audit','phase_count':11,'input_count':len(inputs)},flush=True)

if __name__=='__main__':main()
