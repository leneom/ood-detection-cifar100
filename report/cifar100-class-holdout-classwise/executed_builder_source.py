"""Build the frozen, post-hoc all-fine-class acceptance diagnostic."""
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

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/cifar100-class-holdout-replication"
DEST = ROOT / "report/cifar100-class-holdout-classwise"
DECL = BASE / "classwise-declaration.json"
sys.path.insert(0, str(ROOT))
from tools.verify_class_holdout_replication_completion import validate as verify_completion

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def run_dir(part):
    return ROOT / ("runs/cifar100-class-holdout" if part == 2031 else
                   f"runs/cifar100-class-holdout-replication/split{part}")

def main():
    require(not DEST.exists(), "write-once classwise report already exists")
    d = load_json(DECL)
    require(d["state"] == "frozen_before_fine_class_results" and
            d["partitions"] == [2031,2032,2033] and
            d["seeds"] == [0,1,2] and
            d["variants"] == ["T1","calibrated"] and
            d["methods"] == ["msp","energy","maxlogit","entropy","knn","mahalanobis"] and
            d["target"] == 95 and d["expected_rows"] == 10800, "declaration grid")
    for rel, digest in d["inputs_sha256"].items():
        require(sha(ROOT / rel) == digest, f"frozen input changed: {rel}")
    completion = load_json(ROOT / "report/cifar100-class-holdout-replication/completion.json")
    require(completion["state"] == "complete" and completion["inputs_sha256"] == verify_completion(),
            "nine-model completion chain")
    with (ROOT / "data/cifar-100-python/test").open("rb") as f:
        official = pickle.load(f, encoding="bytes")
    labels = np.asarray(official[b"fine_labels"], dtype=np.int64)
    with (ROOT / "data/cifar-100-python/meta").open("rb") as f:
        meta = pickle.load(f, encoding="bytes")
    names = [x.decode("utf-8") for x in meta[b"fine_label_names"]]
    require(labels.shape == (10000,) and len(names) == 100 and
            np.array_equal(np.bincount(labels, minlength=100), np.full(100,100)),
            "official CIFAR-100 fine labels")
    rows = []
    for part in d["partitions"]:
        folder = run_dir(part)
        with np.load(folder / "test-population.npz", allow_pickle=False) as pop:
            id_idx = pop["id_test_indices"].copy()
            ood_idx = pop["ood_test_indices"].copy()
            id_label = pop["id_original_fine_labels"].copy()
            ood_label = pop["ood_original_fine_labels"].copy()
            id_targets = pop["id_class_targets"].copy()
        require(id_idx.shape == (8000,) and ood_idx.shape == (2000,) and
                np.array_equal(np.sort(np.r_[id_idx,ood_idx]), np.arange(10000)) and
                np.array_equal(labels[id_idx], id_label) and
                np.array_equal(labels[ood_idx], ood_label), f"{part} population")
        id_classes = set(map(int, np.unique(id_label)))
        ood_classes = set(map(int, np.unique(ood_label)))
        require(len(id_classes) == 80 and len(ood_classes) == 20 and
                id_classes.isdisjoint(ood_classes) and
                id_classes | ood_classes == set(range(100)), f"{part} role classes")
        for seed in d["seeds"]:
            extract = folder / f"extract-seed{seed}"
            scored = folder / f"score-seed{seed}"
            manifest = load_json(scored / "manifest.json")
            require(manifest["seed"] == seed and
                    manifest["state"] == "scored_pending_independent_audit", "score manifest")
            with np.load(extract / "id_test_raw.npz", allow_pickle=False) as raw:
                require(np.array_equal(raw["original_indices"],id_idx) and
                        np.array_equal(raw["original_fine_labels"],id_label) and
                        np.array_equal(raw["id_class_targets"],id_targets), "ID raw labels/rows")
                correct = raw["predictions"] == raw["id_class_targets"]
            with np.load(extract / "ood_test_raw.npz", allow_pickle=False) as raw:
                require(np.array_equal(raw["original_indices"],ood_idx) and
                        np.array_equal(raw["original_fine_labels"],ood_label), "OOD raw labels/rows")
            with (
                np.load(scored / "acceptance_masks.npz", allow_pickle=False) as masks,
                np.load(scored / "id_test_scores.npz", allow_pickle=False) as id_scores,
                np.load(scored / "ood_test_scores.npz", allow_pickle=False) as ood_scores,
            ):
                for variant in d["variants"]:
                    for method in d["methods"]:
                        key = f"{variant}_{method}"
                        op = manifest["method_results"][variant][method]["operating_grid"]["target95"]
                        cutoff = op["threshold"]
                        id_accept = id_scores[key] <= cutoff
                        ood_accept = ood_scores[key] <= cutoff
                        require(np.array_equal(id_accept, masks[f"{key}_target95_test_ID"].astype(bool)) and
                                np.array_equal(ood_accept, masks[f"{key}_target95_OOD"].astype(bool)) and
                                int(id_accept.sum()) == op["id_test_accepted"] and
                                int(ood_accept.sum()) == op["ood_accepted"], f"{part}/{seed}/{key} mask and manifest")
                        for role, fine, accepted in (("ID",id_label,id_accept),("OOD",ood_label,ood_accept)):
                            classes = sorted(id_classes if role == "ID" else ood_classes)
                            for cls in classes:
                                select = fine == cls
                                count = int(select.sum())
                                require(count == 100, f"{part}/{seed}/{role}/{cls} count")
                                n_accept = int(accepted[select].sum())
                                entry = {"partition":part,"seed":seed,"variant":variant,
                                         "method":method,"role":role,"class_id":cls,
                                         "class_name":names[cls],"count":count,
                                         "accepted":n_accept,"acceptance_rate":n_accept/count,
                                         "correct":"","accepted_correct":"","accepted_error":""}
                                if role == "ID":
                                    entry["correct"] = int(correct[select].sum())
                                    entry["accepted_correct"] = int((correct & accepted & select).sum())
                                    entry["accepted_error"] = n_accept-entry["accepted_correct"]
                                rows.append(entry)
    require(len(rows) == d["expected_rows"], "full classwise grid")
    fields = list(rows[0])
    groups = []
    for part in d["partitions"]:
        for variant in d["variants"]:
            for method in d["methods"]:
                for role in ("ID","OOD"):
                    cells = [r for r in rows if r["partition"] == part and
                             r["variant"] == variant and r["method"] == method and r["role"] == role]
                    classes = sorted({r["class_id"] for r in cells})
                    values = np.asarray([
                        [next(r["acceptance_rate"] for r in cells if
                              r["class_id"] == cls and r["seed"] == seed)
                         for seed in d["seeds"]]
                        for cls in classes
                    ])
                    require(values.shape == ((80 if role=="ID" else 20),3), "summary class/seed grid")
                    class_means = values.mean(axis=1)
                    q = np.percentile(class_means,[0,25,50,75,100],method="linear")
                    groups.append({"partition":part,"variant":variant,"method":method,"role":role,
                                   "class_count":len(classes),"examples_per_class":100,
                                   "class_mean_min":float(q[0]),"class_mean_q25":float(q[1]),
                                   "class_mean_median":float(q[2]),"class_mean_q75":float(q[3]),
                                   "class_mean_max":float(q[4]),"overall_acceptance":float(values.mean())})
    require(len(groups)==72, "full group grid")
    DEST.mkdir(parents=True)
    table=DEST/"classwise.csv"
    with table.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    summary=DEST/"summary.json"
    summary.write_text(json.dumps({"state":"classwise_pending_independent_audit",
        "partitions":d["partitions"],"seeds":d["seeds"],"variants":d["variants"],
        "methods":d["methods"],"target":95,"row_count":len(rows),"groups":groups,
        "boundary":d["boundary"]},indent=2,allow_nan=False)+"\n",encoding="utf-8")
    source=DEST/"executed_builder_source.py"
    shutil.copyfile(__file__,source)
    manifest={"state":"classwise_pending_independent_audit",
              "created_at":datetime.now(timezone.utc).isoformat(),
              "declaration_sha256":sha(DECL),
              "completion_sha256":sha(ROOT/"report/cifar100-class-holdout-replication/completion.json"),
              "inputs_sha256":d["inputs_sha256"],"source_sha256":sha(source),
              "outputs_sha256":{p.name:sha(p) for p in (table,summary,source)}}
    (DEST/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"state":"pending_independent_audit","rows":len(rows),"groups":len(groups)}))

if __name__=="__main__":
    main()
