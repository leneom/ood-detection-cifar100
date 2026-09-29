"""Fit fixed class-holdout RMD on one frozen optimization-ID feature bank only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"runs/class-holdout-rmd-extension"
sys.path.insert(0,str(ROOT/"tools"))
from rmd_core import fit_rmd

def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(path.read_text(encoding="utf-8"))
def require(ok,why):
 if not ok:raise ValueError(why)
def run_dir(part):
 return ROOT/("runs/cifar100-class-holdout" if part==2031 else
              f"runs/cifar100-class-holdout-replication/split{part}")

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--class-seed",type=int,choices=(2031,2032,2033),required=True)
 parser.add_argument("--seed",type=int,choices=(0,1,2),required=True)
 args=parser.parse_args()
 out=BASE/f"split{args.class_seed}"/f"seed{args.seed}"
 require(not out.exists(),"write-once fit output already exists")
 declaration_path=BASE/"declaration.json"
 decl=read(declaration_path)
 require(decl["state"]=="frozen_before_class_holdout_rmd_fit" and
         decl["shrinkage"]==0.01 and decl["background_weight"]==1.0 and
         decl["classes"]==80 and decl["rows_per_class"]==450 and
         decl["feature_dimensions"]==512, "fixed RMD grid")
 for rel,digest in decl["inputs_sha256"].items():
  require(sha(ROOT/rel)==digest,f"frozen input changed: {rel}")
 source_dir=run_dir(args.class_seed)
 raw_path=source_dir/f"extract-seed{args.seed}/id_train_raw.npz"
 with np.load(source_dir/"partition.npz",allow_pickle=False) as split:
  train_indices=split["train_indices"].copy()
  original_fine=split["train_original_fine_labels"].copy()
  id_fine=split["id_fine_labels"].copy()
 with np.load(raw_path,allow_pickle=False) as bank:
  features=bank["features"].copy()
  target=bank["id_class_targets"].copy()
  indices=bank["original_indices"].copy()
  fine=bank["original_fine_labels"].copy()
 require(features.shape==(36000,512) and features.dtype==np.float32 and
         target.shape==(36000,) and
         np.array_equal(np.bincount(target,minlength=80),np.full(80,450)) and
         np.array_equal(indices,train_indices) and
         np.array_equal(fine,original_fine) and
         np.array_equal(id_fine[target],fine) and
         not np.intersect1d(indices,split_role(source_dir,"validation_indices")).size,
         "optimization-ID-only balanced source bank")
 meta,state=fit_rmd(features,target)
 require(meta["training_count"]==36000 and meta["class_count"]==80 and
         meta["feature_dimensions"]==512 and meta["shrinkage"]==0.01,
         "frozen fit metadata")
 out.mkdir(parents=True)
 state_path=out/"fit-state.npz"
 np.savez(state_path,**state)
 source_path=out/"executed_fit_source.py"
 shutil.copyfile(__file__,source_path)
 record={"state":"fit_pending_independent_audit","class_seed":args.class_seed,
         "seed":args.seed,"fitted_at":datetime.now(timezone.utc).isoformat(),
         "declaration_sha256":sha(declaration_path),"training_bank_sha256":sha(raw_path),
         "checkpoint_sha256":sha(source_dir/f"seed{args.seed}/final-best.pt"),
         "partition_sha256":sha(source_dir/"partition.npz"),
         "rmd_core_sha256":sha(ROOT/"tools/rmd_core.py"),
         "fit_source_sha256":sha(source_path),"fit_state_sha256":sha(state_path),
         "metadata":meta}
 (out/"fit.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
 print(json.dumps({"state":record["state"],"class_seed":args.class_seed,"seed":args.seed,
                   "within_condition":meta["within_condition_number"],
                   "background_condition":meta["background_condition_number"]}))
def split_role(source_dir,key):
 with np.load(source_dir/"partition.npz",allow_pickle=False) as split:
  return split[key].copy()
if __name__=="__main__":main()
