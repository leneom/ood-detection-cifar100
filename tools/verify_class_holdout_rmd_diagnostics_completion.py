"""Source/raw completion gate for the post-result class-holdout RMD diagnostic."""
from __future__ import annotations
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"runs/class-holdout-rmd-diagnostics"
REPORT=ROOT/"report/class-holdout-rmd-diagnostics"
sys.path.insert(0,str(ROOT))
from tools.verify_class_holdout_rmd_completion import validate as extension_validate

def require(ok,msg):
 if not ok:raise AssertionError(msg)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def validate():
 d=read(BASE/"declaration.json");m=read(REPORT/"manifest.json");a=read(REPORT/"audit.json")
 e=read(ROOT/"report/class-holdout-rmd-extension/completion.json")
 require(d["state"]=="frozen_before_posthoc_diagnostic_calculation" and
         m["state"]=="diagnostics_pending_independent_audit" and
         a["state"]=="passed" and
         e["inputs_sha256"]==extension_validate() and
         m["declaration_sha256"]==sha(BASE/"declaration.json") and
         m["rmd_completion_sha256"]==sha(ROOT/"report/class-holdout-rmd-extension/completion.json") and
         m["builder_source_sha256"]==sha(ROOT/"tools/build_class_holdout_rmd_diagnostics.py")==
         sha(REPORT/"executed_builder_source.py") and
         a["source_sha256"]==sha(ROOT/"tools/audit_class_holdout_rmd_diagnostics.py")==
         sha(REPORT/"executed_audit_source.py") and
         a["manifest_sha256"]==sha(REPORT/"manifest.json"),
         "completed extension/diagnostic source chain")
 hashes={}
 def bind(path,expected=None):
  digest=sha(path)
  require(expected is None or digest==expected,"changed file "+str(path))
  hashes[path.relative_to(ROOT).as_posix()]=digest
 for path in (BASE/"declaration.json",
              ROOT/"report/class-holdout-rmd-extension/completion.json",
              ROOT/"docs/CLASS_HOLDOUT_RMD_DIAGNOSTICS.md",
              ROOT/"tools/build_class_holdout_rmd_diagnostics.py",
              ROOT/"tools/audit_class_holdout_rmd_diagnostics.py",
              Path(__file__)):
  bind(path)
 for rel,digest in d["inputs_sha256"].items():bind(ROOT/rel,digest)
 for name,digest in m["outputs_sha256"].items():bind(REPORT/name,digest)
 for name in ("manifest.json","audit.json","executed_audit_source.py"):bind(REPORT/name)
 require(a["transition_rows"]==504 and a["component_rows"]==54 and
         a["summary_comparisons"]==28,"full audit dimensions")
 return hashes
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--write",action="store_true")
 args=parser.parse_args()
 path=REPORT/"completion.json"
 hashes=validate()
 if args.write:
  require(not path.exists(),"write-once completion exists")
  obj={"state":"complete","completed_at":datetime.now(timezone.utc).isoformat(),
       "transition_rows":504,"component_rows":54,"comparisons":28,
       "files":len(hashes),"source_sha256":sha(Path(__file__)),
       "inputs_sha256":hashes,
       "boundary":"Post-result exhaustive paired-decision diagnostic, no method selection or causal claim."}
  path.write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")
 else:
  obj=read(path)
  require(obj["state"]=="complete" and obj["files"]==len(hashes) and
          obj["source_sha256"]==sha(Path(__file__)) and
          obj["inputs_sha256"]==hashes,"saved completion drift")
 print(json.dumps({"state":"verified_complete","files":len(hashes),
                   "transition_rows":504,"component_rows":54}))
if __name__=="__main__":main()
