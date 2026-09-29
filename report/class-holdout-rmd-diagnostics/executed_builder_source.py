"""Build exhaustive fixed-target RMD component and transition diagnostics."""
from __future__ import annotations
import csv,hashlib,json,shutil,statistics,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"runs/class-holdout-rmd-diagnostics"
REPORT=ROOT/"report/class-holdout-rmd-diagnostics"
PARTS=(2031,2032,2033);SEEDS=(0,1,2)
TARGETS=(80,90,95,99)
REFS=("msp","energy","maxlogit","entropy","knn","mahalanobis","matched_cpu_md")
sys.path.insert(0,str(ROOT))
from tools.verify_class_holdout_rmd_completion import validate as extension_validate

def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(path.read_text(encoding="utf-8"))
def require(ok,msg):
 if not ok:raise ValueError(msg)
def oldrun(p):
 return ROOT/("runs/cifar100-class-holdout" if p==2031 else
              f"runs/cifar100-class-holdout-replication/split{p}")
def metrics(x):
 q=np.percentile(x,[0,25,50,75,100],method="linear")
 return {"n":int(len(x)),"mean":float(np.mean(x)),"population_sd":float(np.std(x,ddof=0)),
         "min":float(q[0]),"q25":float(q[1]),"median":float(q[2]),
         "q75":float(q[3]),"max":float(q[4])}
def main():
 require(not REPORT.exists(),"write-once diagnostics report exists")
 d=read(BASE/"declaration.json")
 require(d["state"]=="frozen_before_posthoc_diagnostic_calculation" and
         d["partitions"]==list(PARTS) and d["seeds"]==list(SEEDS) and
         d["targets"]==list(TARGETS) and d["references"]==list(REFS),
         "fixed exhaustive diagnostic grid")
 for rel,digest in d["inputs_sha256"].items():
  require(sha(ROOT/rel)==digest,"frozen input changed: "+rel)
 completion=read(ROOT/"report/class-holdout-rmd-extension/completion.json")
 require(completion["state"]=="complete" and
         completion["inputs_sha256"]==extension_validate(),"RMD extension completion")
 rows=[]; components=[]; arrays={}
 for p in PARTS:
  for seed in SEEDS:
   score=oldrun(p)/f"score-seed{seed}"
   rmd=ROOT/f"runs/class-holdout-rmd-extension/split{p}/seed{seed}/evaluation"
   original=read(score/"manifest.json")
   new=read(rmd/"results.json")
   with np.load(score/"id_test_correctness.npz",allow_pickle=False) as z:
    correct=z["correct"].astype(bool).copy()
   require(correct.shape==(8000,),"frozen ID correctness")
   population={}
   for role in ("id","ood"):
    with np.load(rmd/f"{role}_test_rmd.npz",allow_pickle=False) as z:
     m=z["matched_cpu_md"].copy(); b=z["background_distance"].copy(); r=z["rmd"].copy()
    require(np.array_equal(m-b,r),f"{p}/{seed}/{role} exact signed RMD")
    population[role]={"matched_cpu_md":m,"background_distance":b,"rmd":r}
    for label,value in population[role].items():
     components.append({"partition":p,"seed":seed,"role":role,"component":label,
                        **metrics(value)})
   id_codes=np.empty((len(REFS),len(TARGETS),8000),dtype=np.uint8)
   ood_codes=np.empty((len(REFS),len(TARGETS),2000),dtype=np.uint8)
   with (
       np.load(rmd/"acceptance_masks.npz",allow_pickle=False) as new_masks,
       np.load(score/"acceptance_masks.npz",allow_pickle=False) as old_masks,
       np.load(score/"id_test_scores.npz",allow_pickle=False) as old_id,
       np.load(score/"ood_test_scores.npz",allow_pickle=False) as old_ood,
   ):
    for ri,reference in enumerate(REFS):
     for ti,target in enumerate(TARGETS):
      rmd_point=new["results"]["rmd"]["operating_grid"][f"target{target}"]
      if reference=="matched_cpu_md":
       ref_point=new["results"][reference]["operating_grid"][f"target{target}"]
      else:
       ref_point=original["method_results"]["T1"][reference]["operating_grid"][f"target{target}"]
      for role,n,codes in (("id",8000,id_codes),("ood",2000,ood_codes)):
       suffix="test_ID" if role=="id" else "OOD"
       a=new_masks[f"rmd_target{target}_{suffix}"].astype(bool)
       if reference=="matched_cpu_md":
        b=new_masks[f"{reference}_target{target}_{suffix}"].astype(bool)
        ref_values=population[role][reference]
       else:
        b=old_masks[f"T1_{reference}_target{target}_{suffix}"].astype(bool)
        ref_values=(old_id if role=="id" else old_ood)[f"T1_{reference}"]
       require(a.shape==b.shape==(n,) and
               np.array_equal(a,population[role]["rmd"]<=rmd_point["threshold"]) and
               np.array_equal(b,ref_values<=ref_point["threshold"]),
               f"{p}/{seed}/{reference}/{target}/{role} original ID threshold masks")
       code=np.where(a,np.where(b,0,1),np.where(b,2,3)).astype(np.uint8)
       codes[ri,ti]=code
       count=np.bincount(code,minlength=4)
       entry={"partition":p,"seed":seed,"reference":reference,"target":target,
              "role":role,"n":n,"both_accept":int(count[0]),
              "rmd_only_accepts":int(count[1]),
              "reference_only_accepts":int(count[2]),
              "neither_accepts":int(count[3]),
              "rmd_acceptance":float(np.mean(a)),
              "reference_acceptance":float(np.mean(b)),
              "delta_acceptance":float((int(count[1])-int(count[2]))/n),
              "correct_by_code":"","incorrect_by_code":""}
       if role=="id":
        entry["correct_by_code"]=json.dumps([int(np.sum((code==i)&correct)) for i in range(4)])
        entry["incorrect_by_code"]=json.dumps([int(np.sum((code==i)&(~correct))) for i in range(4)])
       rows.append(entry)
   arrays[p,seed]={"id_codes":id_codes,"ood_codes":ood_codes}
 require(len(rows)==9*7*4*2 and len(components)==9*2*3,"complete transition/component grid")
 summary=[]
 for reference in REFS:
  for target in TARGETS:
   entry={"reference":reference,"target":target}
   for role in ("id","ood"):
    partition_means={}
    for p in PARTS:
     values=[next(r["delta_acceptance"] for r in rows if
                  r["partition"]==p and r["seed"]==seed and
                  r["reference"]==reference and r["target"]==target and r["role"]==role)
             for seed in SEEDS]
     partition_means[str(p)]={"per_seed":values,"mean":statistics.mean(values),
                               "sample_sd":statistics.stdev(values)}
    means=[partition_means[str(p)]["mean"] for p in PARTS]
    entry[role]={"partitions":partition_means,
                 "equal_partition_grand_mean":statistics.mean(means),
                 "partition_mean_sample_sd":statistics.stdev(means)}
   summary.append(entry)
 REPORT.mkdir(parents=True)
 for (p,seed),data in arrays.items():
  np.savez(REPORT/f"transitions-split{p}-seed{seed}.npz",**data)
 def csvout(name,data):
  with (REPORT/name).open("w",encoding="utf-8",newline="") as f:
   writer=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator="\n")
   writer.writeheader();writer.writerows(data)
 csvout("transitions.csv",rows);csvout("components.csv",components)
 summary_path=REPORT/"summary.json"
 summary_path.write_text(json.dumps({"state":"diagnostics_pending_independent_audit",
   "partitions":list(PARTS),"seeds":list(SEEDS),"references":list(REFS),
   "targets":list(TARGETS),"transition_rows":len(rows),
   "component_rows":len(components),"comparisons":summary,"boundary":d["boundary"]},
   indent=2,allow_nan=False)+"\n",encoding="utf-8")
 lines=["# Class-holdout RMD components and paired fixed-threshold decisions","",
        "Post-result descriptive diagnostic over every fixed partition/model, all seven references and all four original ID-validation targets. Positive deltas mean RMD accepts more rows than the named reference. Both ID and OOD deltas are required because each method has its own validation threshold; these comparisons do not isolate ranking or establish causality.","",
        "| Reference | Target ID validation % | ID acceptance change, pp | OOD acceptance change, pp |",
        "|---|---:|---:|---:|"]
 for item in summary:
  lines.append(f'| {item["reference"]} | {item["target"]} | {100*item["id"]["equal_partition_grand_mean"]:+.2f} ± {100*item["id"]["partition_mean_sample_sd"]:.2f} | {100*item["ood"]["equal_partition_grand_mean"]:+.2f} ± {100*item["ood"]["partition_mean_sample_sd"]:.2f} |')
 lines+=["","Values are equal-weight means of three fixed partition means ± sample SD of partition means, not confidence intervals. Every one of the 504 model/reference/target/population transition rows, complete per-row uint8 transition codes, ID correctness splits, 54 component-distribution rows and per-partition values are saved in transitions.csv, transitions-split*.npz, components.csv and summary.json. RMD is exactly matched class distance minus background distance. Original scores, fit, thresholds and checkpoints are unchanged.",""]
 (REPORT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
 copy=REPORT/"executed_builder_source.py";shutil.copyfile(__file__,copy)
 outputs={p.name:sha(p) for p in REPORT.iterdir() if p.is_file()}
 manifest={"state":"diagnostics_pending_independent_audit",
           "created_at":datetime.now(timezone.utc).isoformat(),
           "declaration_sha256":sha(BASE/"declaration.json"),
           "rmd_completion_sha256":sha(ROOT/"report/class-holdout-rmd-extension/completion.json"),
           "builder_source_sha256":sha(copy),"outputs_sha256":outputs}
 (REPORT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({"state":manifest["state"],"transition_rows":len(rows),
                   "component_rows":len(components),"comparisons":len(summary)}))
if __name__=="__main__":main()
