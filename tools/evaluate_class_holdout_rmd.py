"""Score frozen class-holdout features with train-ID-only RMD and matched CPU MD."""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"runs/class-holdout-rmd-extension"
sys.path.insert(0,str(ROOT/"tools"))
sys.path.insert(0,str(ROOT/"src"))
from rmd_core import rmd_scores
from selective_risk_core import threshold_for_target,id_decisions,ood_decisions
from ood.metrics import detection_metrics

ROLES=("checkpoint_id","temperature_id","threshold_id","id_test","ood_test")
METHODS=("rmd","matched_cpu_md")
TARGETS=(80,90,95,99)

def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(path.read_text(encoding="utf-8"))
def source_dir(part):
 return ROOT/("runs/cifar100-class-holdout" if part==2031 else
              f"runs/cifar100-class-holdout-replication/split{part}")

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--class-seed",type=int,choices=(2031,2032,2033),required=True)
 parser.add_argument("--seed",type=int,choices=(0,1,2),required=True)
 a=parser.parse_args()
 declpath=BASE/"evaluation-declaration.json"
 d=read(declpath)
 require(d["state"]=="frozen_after_nine_ID_fit_audits_before_RMD_OOD_scoring" and
         d["new_methods"]==list(METHODS) and d["targets"]==list(TARGETS) and
         d["test_counts"]=={"id":8000,"ood":2000},"frozen evaluation grid")
 for rel,digest in d["inputs_sha256"].items():
  require(sha(ROOT/rel)==digest,f"frozen evaluation input changed: {rel}")
 for part in d["partitions"]:
  for seed in d["seeds"]:
   fit=BASE/f"split{part}"/f"seed{seed}"
   record=read(fit/"fit.json"); audit=read(fit/"fit-audit.json")
   require(audit["state"]=="passed" and
           audit["fit_json_sha256"]==sha(fit/"fit.json") and
           audit["fit_state_sha256"]==sha(fit/"fit-state.npz") and
           record["state"]=="fit_pending_independent_audit",
           f"all nine fit audits before any OOD scoring: {part}/{seed}")
 output=BASE/f"split{a.class_seed}"/f"seed{a.seed}"/"evaluation"
 require(not output.exists(),"write-once RMD evaluation already exists")
 fit=BASE/f"split{a.class_seed}"/f"seed{a.seed}"
 with np.load(fit/"fit-state.npz",allow_pickle=False) as z:
  state={k:z[k].copy() for k in z.files}
 src=source_dir(a.class_seed)
 scores={}; middles={}; diagnostics={}; role_meta={}
 for role in ROLES:
  with np.load(src/f"extract-seed{a.seed}/{role}_raw.npz",allow_pickle=False) as z:
   feature=z["features"].copy()
   indices=z["original_indices"].copy()
   fine=z["original_fine_labels"].copy()
   if role=="id_test":
    correct=z["predictions"]==z["id_class_targets"]
  raw_score,middle,diag=rmd_scores(feature,state)
  scores[role]={"rmd":raw_score["rmd"],"matched_cpu_md":raw_score["mahalanobis_cpu"]}
  middles[role]={"class_distances":middle["class_distances"],
                 "nearest_class":middle["nearest_class"],
                 "background_distance":middle["background_distance"],
                 "matched_cpu_md":middle["mahalanobis_cpu"],
                 "rmd":middle["rmd"],
                 "original_indices":indices,"original_fine_labels":fine}
  diagnostics[role]=diag
  role_meta[role]={"n":len(feature),"raw_sha256":sha(src/f"extract-seed{a.seed}/{role}_raw.npz")}
 require({k:role_meta[k]["n"] for k in ROLES}==
         {"checkpoint_id":1600,"temperature_id":1200,"threshold_id":1200,
          "id_test":8000,"ood_test":2000},"fixed populations")
 original=read(src/f"score-seed{a.seed}/manifest.json")
 require(original["methods"]==d["frozen_controls"] and
         original["n_id"]==8000 and original["n_ood"]==2000,
         "all six T1 controls retained")
 results={}; masks={}
 for method in METHODS:
  inside=scores["id_test"][method]
  outside=scores["ood_test"][method]
  calibration=scores["threshold_id"][method]
  metrics=detection_metrics(inside,outside,calibration)
  grid={}
  for target in TARGETS:
   threshold,cal,cal_mask=threshold_for_target(calibration,target/100)
   id_state,id_mask=id_decisions(inside,correct,threshold)
   ood_state,ood_mask=ood_decisions(outside,threshold)
   grid[f"target{target}"]={"threshold":threshold,"calibration":cal,
                            "id_test":id_state,"ood_test":ood_state}
   masks[f"{method}_target{target}_threshold_ID"]=cal_mask.astype(np.uint8)
   masks[f"{method}_target{target}_test_ID"]=id_mask.astype(np.uint8)
   masks[f"{method}_target{target}_OOD"]=ood_mask.astype(np.uint8)
  results[method]={"metrics":metrics,"operating_grid":grid}
 output.mkdir(parents=True)
 hashes={}
 for role in ROLES:
  path=output/f"{role}_rmd.npz"
  np.savez(path,**middles[role])
  hashes[path.name]=sha(path)
 path=output/"acceptance_masks.npz"
 np.savez(path,**masks);hashes[path.name]=sha(path)
 source=output/"executed_evaluation_source.py"
 shutil.copyfile(__file__,source);hashes[source.name]=sha(source)
 result={"state":"rmd_scored_pending_independent_audit",
         "class_seed":a.class_seed,"seed":a.seed,
         "evaluated_at":datetime.now(timezone.utc).isoformat(),
         "evaluation_declaration_sha256":sha(declpath),
         "fit_json_sha256":sha(fit/"fit.json"),"fit_audit_sha256":sha(fit/"fit-audit.json"),
         "fit_state_sha256":sha(fit/"fit-state.npz"),
         "checkpoint_sha256":sha(src/f"seed{a.seed}/final-best.pt"),
         "original_six_method_manifest_sha256":sha(src/f"score-seed{a.seed}/manifest.json"),
         "methods":list(METHODS),"frozen_controls":d["frozen_controls"],
         "target_role":"ID_threshold_1200_only","score_direction":"higher_is_ood",
         "role_metadata":role_meta,"diagnostics":diagnostics,
         "results":results,"output_sha256":hashes,
         "boundary":d["boundary"]}
 (output/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
 print(json.dumps({"state":result["state"],"class_seed":a.class_seed,"seed":a.seed,
                   "populations":role_meta}))
if __name__=="__main__":main()
