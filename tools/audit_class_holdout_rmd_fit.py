"""Independent training-ID fit and direct-distance audit for one class-holdout RMD model."""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"runs/class-holdout-rmd-extension"
sys.path.insert(0,str(ROOT/"tools"))
from rmd_core import rmd_scores

def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(path.read_text(encoding="utf-8"))
def require(ok,msg):
 if not ok:raise AssertionError(msg)
def close(a,b,msg,atol=1e-8,rtol=1e-9):
 require(np.allclose(a,b,atol=atol,rtol=rtol),msg)
def source_dir(part):
 return ROOT/("runs/cifar100-class-holdout" if part==2031 else
              f"runs/cifar100-class-holdout-replication/split{part}")

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--class-seed",type=int,choices=(2031,2032,2033),required=True)
 parser.add_argument("--seed",type=int,choices=(0,1,2),required=True)
 a=parser.parse_args()
 out=BASE/f"split{a.class_seed}"/f"seed{a.seed}"
 audit_path=out/"fit-audit.json"
 copy_path=out/"executed_fit_audit_source.py"
 require(not audit_path.exists() and not copy_path.exists(),"write-once fit audit exists")
 decl=read(BASE/"declaration.json")
 fit=read(out/"fit.json")
 raw=source_dir(a.class_seed)/f"extract-seed{a.seed}/id_train_raw.npz"
 require(fit["state"]=="fit_pending_independent_audit" and
         fit["class_seed"]==a.class_seed and fit["seed"]==a.seed and
         fit["declaration_sha256"]==sha(BASE/"declaration.json") and
         fit["training_bank_sha256"]==sha(raw) and
         fit["fit_state_sha256"]==sha(out/"fit-state.npz") and
         fit["fit_source_sha256"]==sha(ROOT/"tools/fit_class_holdout_rmd.py")==
         sha(out/"executed_fit_source.py") and
         fit["rmd_core_sha256"]==sha(ROOT/"tools/rmd_core.py"),
         "source-bound fit manifest")
 for rel,digest in decl["inputs_sha256"].items():
  require(sha(ROOT/rel)==digest,f"frozen input changed: {rel}")
 with np.load(raw,allow_pickle=False) as z:
  x=z["features"].astype(np.float64)
  y=z["id_class_targets"].copy()
  idx=z["original_indices"].copy()
 with np.load(source_dir(a.class_seed)/"partition.npz",allow_pickle=False) as p:
  require(np.array_equal(idx,p["train_indices"]) and
          not np.intersect1d(idx,p["validation_indices"]).size and
          np.array_equal(p["id_fine_labels"][y],p["train_original_fine_labels"]),
          "training ID role only")
 require(x.shape==(36000,512) and
         np.array_equal(np.bincount(y,minlength=80),np.full(80,450)),
         "balanced 80-class bank")
 with np.load(out/"fit-state.npz",allow_pickle=False) as z:
  state={k:z[k].copy() for k in z.files}
 require(np.array_equal(state["classes"],np.arange(80)) and
         np.array_equal(state["class_counts"],np.full(80,450)),
         "state class identities/counts")
 means=np.vstack([x[y==c].sum(axis=0)/450 for c in range(80)])
 overall=x.sum(axis=0)/len(x)
 close(state["class_means"],means,"class means",atol=1e-10)
 close(state["background_mean"],overall,"background mean",atol=1e-10)
 within=np.zeros((512,512),dtype=np.float64)
 background=np.zeros_like(within)
 for start in range(0,len(x),2000):
  rows=slice(start,min(start+2000,len(x)))
  centered=x[rows]-means[y[rows]]
  global_centered=x[rows]-overall
  within+=centered.T@centered
  background+=global_centered.T@global_centered
 for name,cov in (("within",within/len(x)),("background",background/len(x))):
  close(state[name+"_covariance_raw"],cov,name+" raw covariance")
  scale=max(float(np.trace(cov)/512),1e-12)
  expected=.99*cov+.01*scale*np.eye(512)
  close(state[name+"_covariance"],expected,name+" fixed shrinkage")
  eigen=np.linalg.eigvalsh(expected)
  require(eigen[0]>0 and np.isfinite(eigen).all(),name+" SPD")
  close(state[name+"_eigenvalues"],eigen,name+" eigenvalues")
  precision=state[name+"_precision"]
  close(precision,precision.T,name+" symmetric precision",atol=1e-10)
  close(precision@expected,np.eye(512),name+" inverse identity")
 probes=x[np.r_[0:5,900:905,30000:30005]]
 scores,mid,diagnostic=rmd_scores(probes,state)
 for row,f in enumerate(probes):
  diff=f[None,:]-means
  squared=np.einsum("kd,df,kf->k",diff,state["within_precision"],diff)
  bg=f-overall
  squared_bg=float(bg@state["background_precision"]@bg)
  close(mid["class_distances"][row],squared,f"probe {row} 80 class distances")
  close(mid["background_distance"][row],squared_bg,f"probe {row} background")
  close(scores["rmd"][row],squared.min()-squared_bg,f"probe {row} signed RMD")
 shutil.copyfile(__file__,copy_path)
 record={"state":"passed","class_seed":a.class_seed,"seed":a.seed,
         "audited_at":datetime.now(timezone.utc).isoformat(),
         "fit_json_sha256":sha(out/"fit.json"),
         "fit_state_sha256":sha(out/"fit-state.npz"),
         "declaration_sha256":sha(BASE/"declaration.json"),
         "source_sha256":sha(copy_path),"probe_rows":len(probes),
         "checked_classes":80,"checked_features":512,
         "boundary":"Training-ID fit and ID feature probes only; no OOD score evaluated."}
 audit_path.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({"state":"passed","class_seed":a.class_seed,"seed":a.seed,
                   "probes":len(probes)}))
if __name__=="__main__":main()
