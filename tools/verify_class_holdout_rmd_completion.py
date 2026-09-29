"""Write-once and read-only source/raw completion gate for class-holdout RMD."""
from __future__ import annotations
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"runs/class-holdout-rmd-extension"
REPORT=ROOT/"report/class-holdout-rmd-extension"
sys.path.insert(0,str(ROOT))
from tools.verify_class_holdout_replication_completion import validate as old_validate

def require(ok,msg):
 if not ok:raise AssertionError(msg)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def validate():
 first=read(BASE/"declaration.json")
 second=read(BASE/"evaluation-declaration.json")
 report=read(REPORT/"manifest.json")
 audit=read(REPORT/"audit.json")
 old=read(ROOT/"report/cifar100-class-holdout-replication/completion.json")
 require(first["state"]=="frozen_before_class_holdout_rmd_fit" and
         second["state"]=="frozen_after_nine_ID_fit_audits_before_RMD_OOD_scoring" and
         old["inputs_sha256"]==old_validate() and
         report["state"]=="rmd_extension_report_pending_independent_audit" and
         audit["state"]=="passed" and audit["manifest_sha256"]==sha(REPORT/"manifest.json"),
         "frozen/old/report gates")
 bound={}
 def bind(path,expected=None):
  digest=sha(path)
  require(expected is None or digest==expected,"source/output changed: "+str(path))
  bound[path.relative_to(ROOT).as_posix()]=digest
  return digest
 for path in (BASE/"declaration.json",BASE/"evaluation-declaration.json",
              ROOT/"report/cifar100-class-holdout-replication/completion.json",
              ROOT/"docs/CLASS_HOLDOUT_RMD_EXTENSION.md",
              ROOT/"tools/rmd_core.py",
              ROOT/"tools/fit_class_holdout_rmd.py",
              ROOT/"tools/audit_class_holdout_rmd_fit.py",
              ROOT/"tools/evaluate_class_holdout_rmd.py",
              ROOT/"tools/audit_class_holdout_rmd_evaluation.py",
              ROOT/"tools/build_class_holdout_rmd_report.py",
              ROOT/"tools/audit_class_holdout_rmd_report.py",Path(__file__)):
  bind(path)
 for declaration in (first,second):
  for rel,digest in declaration["inputs_sha256"].items():bind(ROOT/rel,digest)
 for p in (2031,2032,2033):
  for seed in (0,1,2):
   folder=BASE/f"split{p}"/f"seed{seed}"
   fit=read(folder/"fit.json");fa=read(folder/"fit-audit.json")
   require(fa["state"]=="passed" and
           fa["fit_json_sha256"]==sha(folder/"fit.json") and
           fa["fit_state_sha256"]==sha(folder/"fit-state.npz") and
           fit["fit_source_sha256"]==sha(folder/"executed_fit_source.py") and
           fa["source_sha256"]==sha(folder/"executed_fit_audit_source.py"),
           f"{p}/{seed} fit audit")
   for name in ("fit.json","fit-state.npz","fit-audit.json",
                "executed_fit_source.py","executed_fit_audit_source.py"):
    bind(folder/name)
   output=folder/"evaluation"
   ev=read(output/"results.json");ea=read(output/"audit.json")
   require(ea["state"]=="passed" and
           ea["results_sha256"]==sha(output/"results.json") and
           ev["fit_audit_sha256"]==sha(folder/"fit-audit.json") and
           ea["source_sha256"]==sha(output/"executed_evaluation_audit_source.py"),
           f"{p}/{seed} evaluation audit")
   for name,digest in ev["output_sha256"].items():bind(output/name,digest)
   for name in ("results.json","audit.json","executed_evaluation_audit_source.py"):
    bind(output/name)
 for rel,digest in report["inputs_sha256"].items():bind(ROOT/rel,digest)
 for name,digest in report["outputs_sha256"].items():bind(REPORT/name,digest)
 for name in ("manifest.json","audit.json","executed_audit_source.py"):
  bind(REPORT/name)
 require(audit["source_sha256"]==sha(REPORT/"executed_audit_source.py"),
         "report audit source")
 return bound
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--write",action="store_true")
 args=parser.parse_args()
 path=REPORT/"completion.json"
 hashes=validate()
 if args.write:
  require(not path.exists(),"write-once completion exists")
  obj={"state":"complete","completed_at":datetime.now(timezone.utc).isoformat(),
       "models":9,"methods":8,"files":len(hashes),
       "source_sha256":sha(Path(__file__)),"inputs_sha256":hashes,
       "boundary":"Post-hoc RMD extension, all nine fits/scores/report audited, no OOD-based tuning or untouched confirmation."}
  path.write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")
 else:
  saved=read(path)
  require(saved["state"]=="complete" and saved["models"]==9 and
          saved["methods"]==8 and saved["files"]==len(hashes) and
          saved["source_sha256"]==sha(Path(__file__)) and
          saved["inputs_sha256"]==hashes,"saved completion changed")
 print(json.dumps({"state":"verified_complete","models":9,"methods":8,"files":len(hashes)}))
if __name__=="__main__":main()
