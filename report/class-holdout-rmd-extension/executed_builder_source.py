"""Build complete eight-method descriptive class-holdout RMD extension report."""
from __future__ import annotations
import csv,hashlib,json,shutil,statistics,sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"runs/class-holdout-rmd-extension"
REPORT=ROOT/"report/class-holdout-rmd-extension"
sys.path.insert(0,str(ROOT))
from tools.verify_class_holdout_replication_completion import validate as original_validate
PARTS=(2031,2032,2033);SEEDS=(0,1,2)
CONTROLS=("msp","energy","maxlogit","entropy","knn","mahalanobis")
METHODS=CONTROLS+("rmd","matched_cpu_md")
FIELDS=("auroc","aupr_out","aupr_in","fpr95_id",
        "id_test_acceptance_at_validation_threshold",
        "ood_acceptance_at_validation_threshold")

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def require(ok,msg):
 if not ok:raise ValueError(msg)
def run(p):
 return ROOT/("runs/cifar100-class-holdout" if p==2031 else
              f"runs/cifar100-class-holdout-replication/split{p}")
def stats(values):
 return {"per_seed":values,"mean":statistics.mean(values),
         "sample_sd":statistics.stdev(values)}
def main():
 require(not REPORT.exists(),"write-once report exists")
 d=read(BASE/"evaluation-declaration.json")
 require(d["state"]=="frozen_after_nine_ID_fit_audits_before_RMD_OOD_scoring" and
         d["frozen_controls"]==list(CONTROLS) and
         d["new_methods"]==["rmd","matched_cpu_md"],"frozen eight-method grid")
 for rel,digest in d["inputs_sha256"].items():
  require(sha(ROOT/rel)==digest,"frozen input changed: "+rel)
 old_completion=read(ROOT/"report/cifar100-class-holdout-replication/completion.json")
 require(old_completion["state"]=="complete" and
         old_completion["inputs_sha256"]==original_validate(),
         "original nine-model six-method completion")
 bound={"runs/class-holdout-rmd-extension/evaluation-declaration.json":
        sha(BASE/"evaluation-declaration.json")}
 rows=[]
 for p in PARTS:
  for seed in SEEDS:
   ext=BASE/f"split{p}"/f"seed{seed}"/"evaluation"
   result=read(ext/"results.json")
   audit=read(ext/"audit.json")
   require(result["state"]=="rmd_scored_pending_independent_audit" and
           audit["state"]=="passed" and audit["results_sha256"]==sha(ext/"results.json") and
           audit["class_seed"]==result["class_seed"]==p and
           audit["seed"]==result["seed"]==seed,
           f"{p}/{seed} independent RMD audit")
   original_path=run(p)/f"score-seed{seed}/manifest.json"
   original=read(original_path)
   require(original["methods"]==list(CONTROLS) and
           result["original_six_method_manifest_sha256"]==sha(original_path),
           f"{p}/{seed} six frozen controls")
   for path in (ext/"results.json",ext/"audit.json",original_path):
    bound[path.relative_to(ROOT).as_posix()]=sha(path)
   for method in METHODS:
    metric=(original["method_results"]["T1"][method]["metrics"]
            if method in CONTROLS else result["results"][method]["metrics"])
    row={"partition":p,"seed":seed,"method":method}
    row.update({k:float(metric[k]) for k in FIELDS})
    rows.append(row)
 require(len(rows)==72,"complete 8 x 9 grid")
 cells=[]
 for method in METHODS:
  fields={}
  for field in FIELDS:
   parts={}
   means=[]
   for p in PARTS:
    values=[next(r[field] for r in rows if r["partition"]==p and
                 r["seed"]==seed and r["method"]==method) for seed in SEEDS]
    parts[str(p)]=stats(values); means.append(parts[str(p)]["mean"])
   fields[field]={"partitions":parts,"equal_partition_grand_mean":statistics.mean(means),
                  "partition_mean_sample_sd":statistics.stdev(means)}
  cells.append({"method":method,"fields":fields})
 REPORT.mkdir(parents=True)
 csvpath=REPORT/"per-model.csv"
 with csvpath.open("w",encoding="utf-8",newline="") as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
  writer.writeheader();writer.writerows(rows)
 summary=REPORT/"summary.json"
 summary.write_text(json.dumps({"state":"rmd_extension_report_pending_independent_audit",
   "partitions":list(PARTS),"seeds":list(SEEDS),"methods":list(METHODS),
   "fields":list(FIELDS),"model_rows":len(rows),"cells":cells,
   "boundary":"Post-hoc exploratory method extension after OOD results viewed; descriptive partition sensitivity only."},
   indent=2,allow_nan=False)+"\n",encoding="utf-8")
 lines=["# Class-holdout RMD extension: all eight fixed T=1 columns","",
        "Exploratory post-hoc extension on the same nine frozen models and official test populations. RMD and matched CPU MD were fitted on 36,000 optimization-ID features only with the previously fixed 1% covariance shrinkage; all nine fits and scores passed independent audits. The six original T=1 methods are unchanged controls. Values are equal-weight means over three fixed partition means ± sample SD of those partition means, not confidence intervals.","",
        "| Method | AUROC % | AUPR-out % | AUPR-in % | FPR95-ID % | ID accepted at fixed95 % | OOD accepted at fixed95 % |",
        "|---|---:|---:|---:|---:|---:|---:|"]
 for cell in cells:
  x=cell["fields"]
  lines.append("| "+cell["method"]+" | "+" | ".join(
   f'{100*x[k]["equal_partition_grand_mean"]:.2f} ± {100*x[k]["partition_mean_sample_sd"]:.2f}'
   for k in FIELDS)+" |")
 lines+=["","The fixed95 operational threshold is the separate 1,200-row ID-validation order statistic for each model and method. FPR95-ID instead uses an ID **test** threshold and is a separate benchmark operating point. The full nine-model points and per-partition means/SDs are in per-model.csv and summary.json. RMD is not promoted as a selected winner; prior OOD exposure and shared test populations preclude a new confirmation claim.",""]
 report_text=REPORT/"REPORT.md";report_text.write_text("\n".join(lines),encoding="utf-8")
 source=REPORT/"executed_builder_source.py";shutil.copyfile(__file__,source)
 manifest={"state":"rmd_extension_report_pending_independent_audit",
           "created_at":datetime.now(timezone.utc).isoformat(),
           "evaluation_declaration_sha256":sha(BASE/"evaluation-declaration.json"),
           "original_completion_sha256":sha(ROOT/"report/cifar100-class-holdout-replication/completion.json"),
           "inputs_sha256":bound,"source_sha256":sha(source),
           "outputs_sha256":{x.name:sha(x) for x in (csvpath,summary,report_text,source)}}
 (REPORT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({"state":manifest["state"],"model_rows":len(rows),"methods":len(cells)}))
if __name__=="__main__":main()
