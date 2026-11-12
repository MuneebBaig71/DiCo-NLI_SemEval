"""English-track experiments (outside Assignment 2): compare systems and ensembles.

Usage:
    python 08_exp_summary.py                       # reads results_exp/
    python 08_exp_summary.py results_exp results_a2   # several result folders together

For every system (runs that share a tag up to "_seed<N>") it reports, under argmax
and struct_full decoding:
  - validation W-F1 / HardCons (mean over seeds, and the seed ensemble)
  - dev W-F1 / HardCons        (mean +- std over seeds, and the seed ensemble)
It then builds ensembles across systems and ranks them by VALIDATION score.

Choose what to keep from the validation columns. Dev is the reporting set: look at
it, but do not pick a system because of its dev number.
"""
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data import load_split, LABELS
from src.decoding import partner_index, decode

DIRS = [Path(a) for a in sys.argv[1:]] or [Path("results_exp")]
MODES = ["argmax", "struct_full"]

# gold labels and pair structure for both evaluation sets
train, dev = load_split("train"), load_split("dev")
GOLD = {"val": train.set_index("instance_id"), "dev": dev.set_index("instance_id")}


def load(path):
    df = pd.read_csv(path).set_index("instance_id")
    return df.index.tolist(), df[LABELS].to_numpy(dtype=float)


def evaluate(ids, probs, split, mode):
    """W-F1 over all rows and HardCons over reversible pairs (same as the official scorer)."""
    meta = GOLD[split].loc[ids].reset_index()
    partner = partner_index(ids, meta)
    pred = np.array(decode(probs, partner, mode))
    gold = meta.label.to_numpy()
    wf1 = f1_score(gold, pred, labels=LABELS, average="weighted", zero_division=0)
    a = np.array([i for i, j in enumerate(partner) if j > i])
    b = partner[a]
    hard = float(np.mean((gold[a] == pred[a]) & (gold[b] == pred[b])))
    return wf1, hard


# ---- collect runs: system -> seed -> {split: (ids, probs)}
runs = {}
for d in DIRS:
    for f in sorted((d / "predictions").glob("*_dev_probs.csv")):
        tag = f.name.replace("_dev_probs.csv", "")
        m = re.match(r"(.+)_seed(\d+)$", tag)
        if not m:
            continue
        val = f.with_name(f"{tag}_val_probs.csv")
        if not val.exists():
            continue
        runs.setdefault(m.group(1), {})[int(m.group(2))] = {"dev": load(f), "val": load(val)}

if not runs:
    sys.exit(f"No runs found in {[str(d) for d in DIRS]}")


def ensemble(members, split):
    """Average probabilities of (system, seed) members on one split."""
    ids = runs[members[0][0]][members[0][1]][split][0]
    pos = {i: k for k, i in enumerate(ids)}
    stack = []
    for s, seed in members:
        i2, p2 = runs[s][seed][split]
        stack.append(p2[[{i: k for k, i in enumerate(i2)}[i] for i in ids]])
    return ids, np.mean(stack, axis=0)


rows = []
for name, seeds in runs.items():
    r = {"system": name, "seeds": len(seeds)}
    for mode in MODES:
        for split in ("val", "dev"):
            per = np.array([evaluate(*seeds[s][split], split, mode) for s in seeds])
            ens = evaluate(*ensemble([(name, s) for s in seeds], split), split, mode)
            r[f"{split} {mode} W-F1"] = per[:, 0].mean()
            if split == "dev":
                r[f"dev {mode} std"] = per[:, 0].std(ddof=1) if len(seeds) > 1 else np.nan
            r[f"{split} {mode} Hard"] = per[:, 1].mean()
            r[f"{split} {mode} ens W-F1"] = ens[0]
            r[f"{split} {mode} ens Hard"] = ens[1]
    rows.append(r)

res = pd.DataFrame(rows).sort_values("val struct_full ens W-F1", ascending=False)
out = DIRS[0]
res.to_csv(out / "exp_summary.csv", index=False)

pd.set_option("display.width", 220)
show = ["system", "seeds", "val struct_full W-F1", "val struct_full ens W-F1",
        "dev struct_full W-F1", "dev struct_full std", "dev struct_full Hard",
        "dev struct_full ens W-F1", "dev struct_full ens Hard", "dev argmax W-F1"]
print("\nPer system (sorted by validation score of the seed ensemble)\n")
print(res[show].round(3).to_string(index=False))

# ---- ensembles across systems (all seeds of each chosen system)
names = list(runs)
combos = []
for k in range(2, min(4, len(names)) + 1):
    for combo in itertools.combinations(names, k):
        members = [(s, seed) for s in combo for seed in runs[s]]
        v = evaluate(*ensemble(members, "val"), "val", "struct_full")
        dv = evaluate(*ensemble(members, "dev"), "dev", "struct_full")
        combos.append({"ensemble": " + ".join(combo), "models": len(members),
                       "val W-F1": v[0], "val Hard": v[1], "dev W-F1": dv[0], "dev Hard": dv[1]})
if combos:
    c = pd.DataFrame(combos).sort_values("val W-F1", ascending=False)
    c.to_csv(out / "exp_ensembles.csv", index=False)
    print("\nEnsembles across systems, struct_full (sorted by validation W-F1)\n")
    print(c.head(10).round(3).to_string(index=False))

print("\nNote: the validation split has 304 rows (128 pairs), so differences under about "
      "0.02 there are within noise.")
