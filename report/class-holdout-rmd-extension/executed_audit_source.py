"""Independent full-grid and display audit of the class-holdout RMD extension."""
from __future__ import annotations
import csv,hashlib,json,math,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"runs/class-holdout-rmd-extension"
REPORT=ROOT/"report/class-holdout-rmd-extension"
sys.path.insert(0,str(ROOT))
from tools.verify_class_holdout_replication_completion import validate as old_validate
PARTS=(2031,2032,2033);SEEDS=(0,1,2)
METHODS=("msp","energy","maxlogit","entropy","knn","mahalanobis","rmd","matched_cpu_md")
FIELDS=("auroc","aupr_out","aupr_in","fpr95_id",
        "id_test_acceptance_at_validation_threshold",
        "ood_acceptance_at_validation_threshold")

def require(ok,msg):
 if not ok:raise AssertionError(msg)
def close(a,b,msg):
 require(math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12),msg)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def oldrun(p):
 return ROOT/("runs/cifar100-class-holdout" if p==2031 else
              f"runs/cifar100-class-holdout-replication/split{p}")
def main():
 audit_path=REPORT/"audit.json";copy=REPORT/"executed_audit_source.py"
 require(not audit_path.exists() and not copy.exists(),"write-once report audit already exists")
 declaration=read(BASE/"evaluation-declaration.json")
 manifest=read(REPORT/"manifest.json")
 summary=read(REPORT/"summary.json")
 require(manifest["state"]==summary["state"]=="rmd_extension_report_pending_independent_audit" and
         summary["partitions"]==list(PARTS) and summary["seeds"]==list(SEEDS) and
         summary["methods"]==list(METHODS) and summary["fields"]==list(FIELDS) and
         manifest["evaluation_declaration_sha256"]==sha(BASE/"evaluation-declaration.json") and
         manifest["source_sha256"]==sha(ROOT/"tools/build_class_holdout_rmd_report.py")==
         sha(REPORT/"executed_builder_source.py") and
         manifest["outputs_sha256"]=={name:sha(REPORT/name) for name in
          ("per-model.csv","summary.json","REPORT.md","executed_builder_source.py")},
         "report source/grid/output provenance")
 for rel,digest in declaration["inputs_sha256"].items():
  require(sha(ROOT/rel)==digest,"frozen original drift: "+rel)
 old=read(ROOT/"report/cifar100-class-holdout-replication/completion.json")
 require(old["inputs_sha256"]==old_validate() and
         manifest["original_completion_sha256"]==
         sha(ROOT/"report/cifar100-class-holdout-replication/completion.json"),
         "original nine-model completion")
 with (REPORT/"per-model.csv").open("r",encoding="utf-8",newline="") as f:
  records=list(csv.DictReader(f))
 require(len(records)==summary["model_rows"]==72 and len(summary["cells"])==8,
         "all 72 model rows")
 index={(int(r["partition"]),int(r["seed"]),r["method"]):r for r in records}
 require(len(index)==72,"unique model rows")
 for p in PARTS:
  for seed in SEEDS:
   output=BASE/f"split{p}"/f"seed{seed}"/"evaluation"
   result=read(output/"results.json");audit=read(output/"audit.json")
   oldpath=oldrun(p)/f"score-seed{seed}/manifest.json"
   control=read(oldpath)
   require(audit["state"]=="passed" and
           audit["results_sha256"]==sha(output/"results.json") and
           result["original_six_method_manifest_sha256"]==sha(oldpath) and
           result["evaluation_declaration_sha256"]==sha(BASE/"evaluation-declaration.json"),
           f"{p}/{seed} audit/control binding")
   for name,digest in result["output_sha256"].items():
    require(sha(output/name)==digest,f"{p}/{seed}/{name} raw output")
   for path in (output/"results.json",output/"audit.json",oldpath):
    rel=path.relative_to(ROOT).as_posix()
    require(manifest["inputs_sha256"][rel]==sha(path),"report manifest "+rel)
   for method in METHODS:
    src=(control["method_results"]["T1"][method]["metrics"]
         if method in METHODS[:6] else result["results"][method]["metrics"])
    row=index[p,seed,method]
    for field in FIELDS:close(row[field],src[field],f"{p}/{seed}/{method}/{field}")
 cells={x["method"]:x for x in summary["cells"]}
 require(len(cells)==8,"unique eight methods")
 expected_rows=[]
 for method in METHODS:
  cell=cells[method]
  display=[]
  for field in FIELDS:
   x=cell["fields"][field]
   means=[]
   for p in PARTS:
    values=np.asarray([float(index[p,seed,method][field]) for seed in SEEDS])
    item=x["partitions"][str(p)]
    require(np.allclose(item["per_seed"],values,rtol=0,atol=1e-12),
            f"{method}/{field}/{p} three points")
    close(item["mean"],np.mean(values),f"{method}/{field}/{p} mean")
    close(item["sample_sd"],np.std(values,ddof=1),f"{method}/{field}/{p} sample sd")
    means.append(float(np.mean(values)))
   close(x["equal_partition_grand_mean"],np.mean(means),f"{method}/{field} grand")
   close(x["partition_mean_sample_sd"],np.std(means,ddof=1),f"{method}/{field} partition sd")
   display.append(f'{100*np.mean(means):.2f} ± {100*np.std(means,ddof=1):.2f}')
  expected_rows.append("| "+method+" | "+" | ".join(display)+" |")
 text=(REPORT/"REPORT.md").read_text(encoding="utf-8").splitlines()
 actual=[line for line in text if line.startswith("| ") and
         line.split("|")[1].strip() in METHODS]
 require(actual==expected_rows,"all eight displayed method rows")
 shutil.copyfile(__file__,copy)
 audit={"state":"passed","audited_at":datetime.now(timezone.utc).isoformat(),
        "model_rows":72,"methods":8,"metric_fields_checked":72*len(FIELDS),
        "partition_statistics_checked":8*len(FIELDS)*3,
        "display_rows_checked":8,"manifest_sha256":sha(REPORT/"manifest.json"),
        "summary_sha256":sha(REPORT/"summary.json"),
        "csv_sha256":sha(REPORT/"per-model.csv"),
        "display_sha256":sha(REPORT/"REPORT.md"),
        "source_sha256":sha(copy),
        "boundary":"All frozen controls and new methods reported; post-hoc descriptive, no method selection."}
 audit_path.write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({"state":"passed","model_rows":72,"methods":8}))
if __name__=="__main__":main()
