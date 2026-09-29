"""Independent direct-distance, sklearn metric and mask audit of class-holdout RMD."""
from __future__ import annotations
import argparse,hashlib,json,math,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"runs/class-holdout-rmd-extension"
ROLES=("checkpoint_id","temperature_id","threshold_id","id_test","ood_test")
METHODS=("rmd","matched_cpu_md")
TARGETS=(80,90,95,99)

def require(ok,msg):
 if not ok:raise AssertionError(msg)
def close(a,b,msg,tol=1e-10):
 require(math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol),f"{msg}: {a} versus {b}")
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(path.read_text(encoding="utf-8"))
def run_dir(part):
 return ROOT/("runs/cifar100-class-holdout" if part==2031 else
              f"runs/cifar100-class-holdout-replication/split{part}")
def threshold(x,target):
 return float(np.sort(x)[int(np.ceil(target*len(x)))-1])

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--class-seed",type=int,choices=(2031,2032,2033),required=True)
 parser.add_argument("--seed",type=int,choices=(0,1,2),required=True)
 a=parser.parse_args()
 out=BASE/f"split{a.class_seed}"/f"seed{a.seed}"/"evaluation"
 audit_path=out/"audit.json"; source_copy=out/"executed_evaluation_audit_source.py"
 require(not audit_path.exists() and not source_copy.exists(),"write-once evaluation audit exists")
 result=read(out/"results.json")
 decl=read(BASE/"evaluation-declaration.json")
 fit=BASE/f"split{a.class_seed}"/f"seed{a.seed}"
 src=run_dir(a.class_seed)
 require(result["state"]=="rmd_scored_pending_independent_audit" and
         result["class_seed"]==a.class_seed and result["seed"]==a.seed and
         result["methods"]==list(METHODS) and
         result["evaluation_declaration_sha256"]==sha(BASE/"evaluation-declaration.json") and
         result["fit_json_sha256"]==sha(fit/"fit.json") and
         result["fit_audit_sha256"]==sha(fit/"fit-audit.json") and
         result["fit_state_sha256"]==sha(fit/"fit-state.npz") and
         result["original_six_method_manifest_sha256"]==sha(src/f"score-seed{a.seed}/manifest.json"),
         "source-bound RMD result")
 for rel,digest in decl["inputs_sha256"].items():
  require(sha(ROOT/rel)==digest,f"frozen input drift: {rel}")
 for filename,digest in result["output_sha256"].items():
  require(sha(out/filename)==digest,"raw output hash: "+filename)
 require(result["output_sha256"]["executed_evaluation_source.py"]==
         sha(ROOT/"tools/evaluate_class_holdout_rmd.py"),"executed source drift")
 with np.load(fit/"fit-state.npz",allow_pickle=False) as z:
  means=z["class_means"].copy(); origin=z["background_mean"].copy()
  pw=z["within_precision"].copy(); pb=z["background_precision"].copy()
 arrays={}
 for role in ROLES:
  with (
       np.load(src/f"extract-seed{a.seed}/{role}_raw.npz",allow_pickle=False) as raw,
       np.load(out/f"{role}_rmd.npz",allow_pickle=False) as scored,
  ):
   require(np.array_equal(raw["original_indices"],scored["original_indices"]) and
           np.array_equal(raw["original_fine_labels"],scored["original_fine_labels"]),
           role+" original row/label binding")
   f=raw["features"]
   n=len(f)
   cls=scored["class_distances"]
   bg=scored["background_distance"]
   md=scored["matched_cpu_md"]
   signed=scored["rmd"]
   nearest=scored["nearest_class"]
   require(cls.shape==(n,80) and bg.shape==(n,) and
           np.isfinite(cls).all() and np.isfinite(bg).all() and
           np.array_equal(cls.min(axis=1),md) and
           np.array_equal(cls.argmin(axis=1),nearest) and
           np.allclose(signed,md-bg,atol=1e-12,rtol=1e-12) and
           (cls>=0).all() and (bg>=0).all(),role+" all-row signed component identity")
   probes=np.unique(np.linspace(0,n-1,min(17,n),dtype=int))
   for i in probes:
    diff=f[i].astype(np.float64)[None,:]-means
    direct=np.einsum("kd,df,kf->k",diff,pw,diff)
    global_diff=f[i].astype(np.float64)-origin
    direct_bg=float(global_diff@pb@global_diff)
    require(np.allclose(cls[i],direct,atol=1e-8,rtol=1e-9),
            role+" direct class distance")
    close(bg[i],direct_bg,role+" direct background",tol=1e-8)
   arrays[role]={"rmd":signed.copy(),"matched_cpu_md":md.copy()}
   if role=="id_test":
    correct=(raw["predictions"]==raw["id_class_targets"]).copy()
 with np.load(out/"acceptance_masks.npz",allow_pickle=False) as masks:
  for method in METHODS:
   inside=arrays["id_test"][method]; outside=arrays["ood_test"][method]
   cal=arrays["threshold_id"][method]
   saved=result["results"][method]
   met=saved["metrics"]
   labels=np.r_[np.zeros(len(inside),dtype=int),np.ones(len(outside),dtype=int)]
   joined=np.r_[inside,outside]
   expected={"auroc":roc_auc_score(labels,joined),
             "aupr_out":average_precision_score(labels,joined),
             "aupr_in":average_precision_score(1-labels,-joined),
             "fpr95_id":np.mean(outside<=threshold(inside,.95)),
             "id_tpr_at_fpr95":np.mean(inside<=threshold(inside,.95)),
             "benchmark_threshold":threshold(inside,.95),
             "validation_threshold":threshold(cal,.95),
             "id_test_acceptance_at_validation_threshold":np.mean(inside<=threshold(cal,.95)),
             "ood_acceptance_at_validation_threshold":np.mean(outside<=threshold(cal,.95)),
             "calibration_id_acceptance":np.mean(cal<=threshold(cal,.95))}
   for field,value in expected.items():close(met[field],value,method+" "+field)
   require(met["n_id"]==8000 and met["n_ood"]==2000 and
           met["ood_prevalence"]==.2,method+" counts/prevalence")
   for target in TARGETS:
    grid=saved["operating_grid"][f"target{target}"]
    cut=threshold(cal,target/100)
    ca=cal<=cut; ia=inside<=cut; oa=outside<=cut
    close(grid["threshold"],cut,method+f" target{target} threshold")
    require(np.array_equal(ca,masks[f"{method}_target{target}_threshold_ID"].astype(bool)) and
            np.array_equal(ia,masks[f"{method}_target{target}_test_ID"].astype(bool)) and
            np.array_equal(oa,masks[f"{method}_target{target}_OOD"].astype(bool)),
            method+f" target{target} masks")
    require(grid["calibration"]["accepted"]==int(ca.sum()) and
            grid["id_test"]["accepted"]==int(ia.sum()) and
            grid["ood_test"]["accepted"]==int(oa.sum()) and
            grid["id_test"]["correct_accepted"]==int((correct&ia).sum()) and
            grid["id_test"]["incorrect_accepted"]==int((~correct&ia).sum()),
            method+f" target{target} decisions")
 shutil.copyfile(__file__,source_copy)
 record={"state":"passed","class_seed":a.class_seed,"seed":a.seed,
         "audited_at":datetime.now(timezone.utc).isoformat(),
         "results_sha256":sha(out/"results.json"),"fit_audit_sha256":sha(fit/"fit-audit.json"),
         "declaration_sha256":sha(BASE/"evaluation-declaration.json"),
         "source_sha256":sha(source_copy),
         "direct_feature_probes_per_population":17,
         "populations":len(ROLES),"new_methods":len(METHODS),
         "targets_per_method":len(TARGETS),
         "boundary":"Descriptive RMD extension, all OOD metrics independently checked; no OOD tuning."}
 audit_path.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({"state":"passed","class_seed":a.class_seed,"seed":a.seed,
                   "populations":len(ROLES),"methods":len(METHODS)}))
if __name__=="__main__":main()
