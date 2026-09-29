"""Independent all-row RMD component and paired-transition audit."""
from __future__ import annotations
import csv,hashlib,json,math,shutil,sys
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

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def require(ok,msg):
 if not ok:raise AssertionError(msg)
def close(a,b,msg,tol=1e-12):
 require(math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol),f"{msg}: {a} vs {b}")
def oldrun(p):
 return ROOT/("runs/cifar100-class-holdout" if p==2031 else
              f"runs/cifar100-class-holdout-replication/split{p}")
def csvrows(path):
 with path.open("r",encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def main():
 audit_path=REPORT/"audit.json";source_copy=REPORT/"executed_audit_source.py"
 require(not audit_path.exists() and not source_copy.exists(),"write-once diagnostic audit exists")
 decl=read(BASE/"declaration.json");manifest=read(REPORT/"manifest.json")
 summary=read(REPORT/"summary.json")
 require(decl["state"]=="frozen_before_posthoc_diagnostic_calculation" and
         manifest["state"]==summary["state"]=="diagnostics_pending_independent_audit" and
         manifest["declaration_sha256"]==sha(BASE/"declaration.json") and
         manifest["builder_source_sha256"]==sha(ROOT/"tools/build_class_holdout_rmd_diagnostics.py")==
         sha(REPORT/"executed_builder_source.py") and
         manifest["outputs_sha256"]=={n:sha(REPORT/n) for n in manifest["outputs_sha256"]},
         "frozen diagnostic source/output manifest")
 for rel,digest in decl["inputs_sha256"].items():
  require(sha(ROOT/rel)==digest,"frozen diagnostic input drift: "+rel)
 completed=read(ROOT/"report/class-holdout-rmd-extension/completion.json")
 require(completed["inputs_sha256"]==extension_validate() and
         manifest["rmd_completion_sha256"]==
         sha(ROOT/"report/class-holdout-rmd-extension/completion.json"),
         "completed RMD extension")
 transitions=csvrows(REPORT/"transitions.csv")
 components=csvrows(REPORT/"components.csv")
 require(len(transitions)==summary["transition_rows"]==504 and
         len(components)==summary["component_rows"]==54,"full saved grids")
 keyed={(int(r["partition"]),int(r["seed"]),r["reference"],int(r["target"]),r["role"]):r
        for r in transitions}
 component_key={(int(r["partition"]),int(r["seed"]),r["role"],r["component"]):r
                for r in components}
 require(len(keyed)==504 and len(component_key)==54,"unique saved rows")
 recomputed={}
 for p in PARTS:
  for seed in SEEDS:
   rmd=ROOT/f"runs/class-holdout-rmd-extension/split{p}/seed{seed}/evaluation"
   old=oldrun(p)/f"score-seed{seed}"
   original=read(old/"manifest.json");new=read(rmd/"results.json")
   with np.load(old/"id_test_correctness.npz",allow_pickle=False) as z:
    correct=z["correct"].astype(bool).copy()
   require(correct.shape==(8000,),"original ID correctness")
   with (
      np.load(rmd/"id_test_rmd.npz",allow_pickle=False) as ir,
      np.load(rmd/"ood_test_rmd.npz",allow_pickle=False) as oraw,
      np.load(old/"id_test_scores.npz",allow_pickle=False) as oi,
      np.load(old/"ood_test_scores.npz",allow_pickle=False) as oo,
      np.load(REPORT/f"transitions-split{p}-seed{seed}.npz",allow_pickle=False) as saved,
   ):
    require(saved["id_codes"].shape==(7,4,8000) and
            saved["ood_codes"].shape==(7,4,2000),"full transition arrays")
    for role,z in (("id",ir),("ood",oraw)):
     m=z["matched_cpu_md"];b=z["background_distance"];r=z["rmd"]
     require(np.array_equal(m-b,r),f"{p}/{seed}/{role} all-row RMD identity")
     for label,array in (("matched_cpu_md",m),("background_distance",b),("rmd",r)):
      item=component_key[p,seed,role,label]
      q=np.quantile(array,[0,.25,.5,.75,1],method="linear")
      require(int(item["n"])==len(array),"component count")
      for name,value in (("mean",np.mean(array)),("population_sd",np.std(array,ddof=0)),
                         ("min",q[0]),("q25",q[1]),("median",q[2]),("q75",q[3]),("max",q[4])):
       close(item[name],value,f"{p}/{seed}/{role}/{label}/{name}")
    for ri,ref in enumerate(REFS):
     for ti,target in enumerate(TARGETS):
      rmd_cut=new["results"]["rmd"]["operating_grid"][f"target{target}"]["threshold"]
      if ref=="matched_cpu_md":
       ref_cut=new["results"][ref]["operating_grid"][f"target{target}"]["threshold"]
      else:
       ref_cut=original["method_results"]["T1"][ref]["operating_grid"][f"target{target}"]["threshold"]
      for role,z,oldz in (("id",ir,oi),("ood",oraw,oo)):
       aa=np.less_equal(z["rmd"],rmd_cut)
       bb=np.less_equal(z[ref],ref_cut) if ref=="matched_cpu_md" else np.less_equal(oldz[f"T1_{ref}"],ref_cut)
       expected=np.zeros(len(aa),dtype=np.uint8)
       expected[aa & ~bb]=1
       expected[~aa & bb]=2
       expected[~aa & ~bb]=3
       actual=saved[f"{role}_codes"][ri,ti]
       require(np.array_equal(expected,actual),f"{p}/{seed}/{ref}/{target}/{role} every transition")
       item=keyed[p,seed,ref,target,role]
       counts=np.bincount(expected,minlength=4)
       require(int(item["n"])==len(expected) and
               [int(item[k]) for k in ("both_accept","rmd_only_accepts",
                                        "reference_only_accepts","neither_accepts")]==counts.tolist(),
               f"{p}/{seed}/{ref}/{target}/{role} counts")
       close(item["rmd_acceptance"],np.mean(aa),"RMD acceptance")
       close(item["reference_acceptance"],np.mean(bb),"reference acceptance")
       delta=(int(counts[1])-int(counts[2]))/len(expected)
       close(item["delta_acceptance"],delta,"paired change")
       recomputed[p,seed,ref,target,role]=delta
       if role=="id":
        require(json.loads(item["correct_by_code"])==
                [int(np.sum((expected==i)&correct)) for i in range(4)] and
                json.loads(item["incorrect_by_code"])==
                [int(np.sum((expected==i)&(~correct))) for i in range(4)],
                "ID correct/incorrect transitions")
       else:
        require(item["correct_by_code"]==item["incorrect_by_code"]=="",
                "OOD has no classifier correctness")
 require(len(recomputed)==504,"every paired comparison recomputed")
 comparisons={(x["reference"],x["target"]):x for x in summary["comparisons"]}
 require(len(comparisons)==28,"all fixed reference/target comparisons")
 display=[]
 for ref in REFS:
  for target in TARGETS:
   item=comparisons[ref,target]
   cells=[]
   for role in ("id","ood"):
    partition_means=[]
    for p in PARTS:
     vals=np.asarray([recomputed[p,seed,ref,target,role] for seed in SEEDS])
     part=item[role]["partitions"][str(p)]
     require(np.allclose(vals,part["per_seed"],rtol=0,atol=1e-12),"three seed deltas")
     close(part["mean"],np.mean(vals),"partition mean")
     close(part["sample_sd"],np.std(vals,ddof=1),"partition sample sd")
     partition_means.append(float(np.mean(vals)))
    mean=float(np.mean(partition_means));sd=float(np.std(partition_means,ddof=1))
    close(item[role]["equal_partition_grand_mean"],mean,"grand mean")
    close(item[role]["partition_mean_sample_sd"],sd,"partition sample sd")
    cells.append(f"{100*mean:+.2f} ± {100*sd:.2f}")
   display.append(f"| {ref} | {target} | {cells[0]} | {cells[1]} |")
 report=(REPORT/"REPORT.md").read_text(encoding="utf-8").splitlines()
 actual=[line for line in report if line.startswith("| ") and
         line.split("|")[1].strip() in REFS]
 require(actual==display,"full 28-row displayed grid")
 shutil.copyfile(__file__,source_copy)
 audit={"state":"passed","audited_at":datetime.now(timezone.utc).isoformat(),
        "transition_rows":504,"component_rows":54,"summary_comparisons":28,
        "all_row_transition_vectors":18,
        "declaration_sha256":sha(BASE/"declaration.json"),
        "manifest_sha256":sha(REPORT/"manifest.json"),
        "source_sha256":sha(source_copy),
        "boundary":"Post-result descriptive diagnostic; no score, threshold or model changed."}
 audit_path.write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({"state":"passed","transition_rows":504,"component_rows":54,"comparisons":28}))
if __name__=="__main__":main()
