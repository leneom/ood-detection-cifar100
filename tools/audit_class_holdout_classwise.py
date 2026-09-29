"""Independent saved-score audit of every classwise diagnostic row."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"runs/cifar100-class-holdout-replication"
DEST=ROOT/"report/cifar100-class-holdout-classwise"
DECL=BASE/"classwise-declaration.json"
sys.path.insert(0,str(ROOT))
from tools.verify_class_holdout_replication_completion import validate as verify_completion

def require(ok,why):
    if not ok:raise AssertionError(why)

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def read(p):return json.loads(p.read_text(encoding="utf-8"))

def run(part):
    return ROOT/("runs/cifar100-class-holdout" if part==2031 else
                 f"runs/cifar100-class-holdout-replication/split{part}")

def close(x,y,why):
    require(abs(float(x)-float(y))<=1e-12,why)

def main():
    audit_path=DEST/"audit.json"
    source_copy=DEST/"executed_audit_source.py"
    require(not audit_path.exists() and not source_copy.exists(), "write-once audit exists")
    decl=read(DECL)
    manifest=read(DEST/"manifest.json")
    summary=read(DEST/"summary.json")
    require(decl["state"]=="frozen_before_fine_class_results" and
            manifest["state"]=="classwise_pending_independent_audit" and
            summary["state"]=="classwise_pending_independent_audit" and
            manifest["declaration_sha256"]==sha(DECL) and
            manifest["source_sha256"]==sha(ROOT/"tools/build_class_holdout_classwise.py")==
            sha(DEST/"executed_builder_source.py") and
            manifest["outputs_sha256"]=={name:sha(DEST/name) for name in
                    ("classwise.csv","summary.json","executed_builder_source.py")},
            "frozen classwise source/output chain")
    for rel,digest in decl["inputs_sha256"].items():
        require(sha(ROOT/rel)==digest,f"source drift: {rel}")
    completion=read(ROOT/"report/cifar100-class-holdout-replication/completion.json")
    require(completion["inputs_sha256"]==verify_completion() and
            manifest["completion_sha256"]==sha(ROOT/"report/cifar100-class-holdout-replication/completion.json"),
            "complete nine-model source chain")
    with (ROOT/"data/cifar-100-python/test").open("rb") as f:
        official=pickle.load(f,encoding="bytes")
    labels=np.asarray(official[b"fine_labels"],dtype=np.int64)
    with (ROOT/"data/cifar-100-python/meta").open("rb") as f:
        names=[b.decode("utf-8") for b in pickle.load(f,encoding="bytes")[b"fine_label_names"]]
    with (DEST/"classwise.csv").open("r",encoding="utf-8",newline="") as f:
        saved=list(csv.DictReader(f))
    require(len(saved)==decl["expected_rows"]==10800 and len(names)==100,"all 10,800 rows")
    keys=[(int(r["partition"]),int(r["seed"]),r["variant"],r["method"],r["role"],int(r["class_id"])) for r in saved]
    table=dict(zip(keys,saved))
    require(len(table)==len(saved),"unique full classwise keys")
    grouped={}
    checked=0
    for part in decl["partitions"]:
        folder=run(part)
        with np.load(folder/"test-population.npz",allow_pickle=False) as z:
            source={role:(z[f'{role.lower()}_test_indices'].copy(),
                          z[f'{role.lower()}_original_fine_labels'].copy())
                    for role in ("ID","OOD")}
        require(np.array_equal(np.sort(np.r_[source["ID"][0],source["OOD"][0]]),np.arange(10000)),
                "full official test rows")
        for role in ("ID","OOD"):
            idx,fine=source[role]
            require(np.array_equal(labels[idx],fine) and
                    len(np.unique(fine))==(80 if role=="ID" else 20),
                    f"{part} official fine labels")
        for seed in decl["seeds"]:
            scored=folder/f"score-seed{seed}"
            extract=folder/f"extract-seed{seed}"
            score_manifest=read(scored/"manifest.json")
            raw={}
            for role in ("ID","OOD"):
                with np.load(extract/f'{role.lower()}_test_raw.npz',allow_pickle=False) as z:
                    idx,fine=source[role]
                    require(np.array_equal(z["original_indices"],idx) and
                            np.array_equal(z["original_fine_labels"],fine),
                            f"{part}/{seed}/{role} raw rows")
                    raw[role]={"fine":fine,
                               "correct":(z["predictions"]==z["id_class_targets"])
                                         if role=="ID" else None}
            with (
                np.load(scored/"id_test_scores.npz",allow_pickle=False) as id_scores,
                np.load(scored/"ood_test_scores.npz",allow_pickle=False) as ood_scores,
                np.load(scored/"acceptance_masks.npz",allow_pickle=False) as masks,
            ):
                for variant in decl["variants"]:
                    for method in decl["methods"]:
                        k=f"{variant}_{method}"
                        op=score_manifest["method_results"][variant][method]["operating_grid"]["target95"]
                        cutoff=op["threshold"]
                        for role,scores in (("ID",id_scores[k]),("OOD",ood_scores[k])):
                            fine=raw[role]["fine"]
                            accept=np.less_equal(scores,cutoff)
                            mask=masks[f"{k}_target95_{'test_ID' if role=='ID' else 'OOD'}"]
                            require(np.array_equal(accept,mask.astype(bool)) and
                                    int(accept.sum())==op[f"{role.lower()}_accepted" if role=="OOD" else "id_test_accepted"],
                                    f"{part}/{seed}/{k}/{role} independent mask")
                            for cls in sorted(map(int,np.unique(fine))):
                                pick=fine==cls
                                key=(part,seed,variant,method,role,cls)
                                r=table[key]
                                n=int(pick.sum()); a=int(np.count_nonzero(accept&pick))
                                require(n==100 and
                                        r["class_name"]==names[cls] and
                                        int(r["count"])==n and int(r["accepted"])==a,
                                        f"{key} count/name")
                                close(r["acceptance_rate"],a/n,f"{key} rate")
                                if role=="ID":
                                    correct=raw[role]["correct"]
                                    c=int(np.count_nonzero(correct&pick))
                                    ac=int(np.count_nonzero(correct&accept&pick))
                                    require(int(r["correct"])==c and
                                            int(r["accepted_correct"])==ac and
                                            int(r["accepted_error"])==a-ac,
                                            f"{key} retained classification")
                                else:
                                    require(r["correct"]==r["accepted_correct"]==r["accepted_error"]=="",
                                            f"{key} OOD has no correctness")
                                grouped.setdefault((part,variant,method,role),{}).setdefault(cls,{})[seed]=a/n
                                checked+=1
    require(checked==10800 and len(grouped)==72,"full independent grid")
    summaries={(x["partition"],x["variant"],x["method"],x["role"]):x
               for x in summary["groups"]}
    require(len(summary["groups"])==len(summaries)==72,"unique full summary")
    for key,classes in grouped.items():
        part,variant,method,role=key
        vals=np.asarray([[seeds[s] for s in decl["seeds"]]
                         for cls,seeds in sorted(classes.items())])
        require(vals.shape==((80 if role=="ID" else 20),3),"class/seed summary shape")
        means=vals.mean(axis=1)
        q=np.percentile(means,[0,25,50,75,100],method="linear")
        item=summaries[key]
        require(item["class_count"]==len(classes) and
                item["examples_per_class"]==100,"class summary count")
        for field,value in zip(("class_mean_min","class_mean_q25","class_mean_median",
                                "class_mean_q75","class_mean_max"),q):
            close(item[field],value,f"{key} {field}")
        close(item["overall_acceptance"],vals.mean(),f"{key} overall")
        model_rates=[read(run(part)/f"score-seed{s}/manifest.json")
                     ["method_results"][variant][method]["operating_grid"]["target95"]
                     ["id_test_acceptance" if role=="ID" else "ood_acceptance"]
                     for s in decl["seeds"]]
        close(vals.mean(),np.mean(model_rates),f"{key} model aggregate")
    shutil.copyfile(__file__,source_copy)
    record={"state":"passed","audited_at":datetime.now(timezone.utc).isoformat(),
            "row_count":checked,"summary_groups":len(grouped),
            "declaration_sha256":sha(DECL),"manifest_sha256":sha(DEST/"manifest.json"),
            "classwise_sha256":sha(DEST/"classwise.csv"),
            "summary_sha256":sha(DEST/"summary.json"),"audit_source_sha256":sha(source_copy),
            "boundary":"Post-hoc descriptive classwise analysis, no threshold or method selection."}
    audit_path.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"state":"passed","rows":checked,"groups":len(grouped)}))
if __name__=="__main__":main()
