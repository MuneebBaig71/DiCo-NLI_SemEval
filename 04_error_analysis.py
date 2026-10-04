"""Error analysis for any dev prediction file.

Run:  python 04_error_analysis.py                       (analyses the default models below)
      python 04_error_analysis.py results/predictions/X.csv [more files ...]
Out:  results/error_analysis/<name>/  confusion.pdf, summary.txt, errors_sample.csv
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_split, REVERSE
from src.features import pair_features

ORDER = ["FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "EQUIVALENCE", "NEGATIVE_OTHER"]
SHORT = {"FORWARD_ENTAILMENT": "FORWARD", "BACKWARD_ENTAILMENT": "BACKWARD",
         "EQUIVALENCE": "EQUIV", "NEGATIVE_OTHER": "NEG_OTHER"}

dev = load_split("dev")
dev = pd.concat([dev, pair_features(dev, use_pos=False)], axis=1)


def analyse(pred_path):
    pred_path = Path(pred_path)
    name = pred_path.stem
    out = Path("results/error_analysis") / name
    out.mkdir(parents=True, exist_ok=True)
    d = dev.merge(pd.read_csv(pred_path).rename(columns={"label": "pred"}), on="instance_id")
    d["correct"] = d.label == d.pred
    log = [f"Model: {name}   dev accuracy {d.correct.mean():.3f}"]

    # 1) confusion matrix (row-normalised = recall view)
    cm = pd.crosstab(d.label, d.pred).reindex(index=ORDER, columns=ORDER, fill_value=0)
    cmn = cm.div(cm.sum(1), axis=0)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(cmn.values, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmn.values[i, j]:.2f}\n({cm.values[i, j]})", ha="center", va="center",
                    color="white" if cmn.values[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(4), [SHORT[l] for l in ORDER])
    ax.set_yticks(range(4), [SHORT[l] for l in ORDER])
    ax.set(title=f"Dev Confusion Matrix ({name})", xlabel="Predicted label", ylabel="Gold label")
    fig.colorbar(im, ax=ax, label="Share of gold label (row)")
    fig.tight_layout(); fig.savefig(out / "confusion.pdf"); plt.close(fig)
    log += ["\nConfusion matrix (counts, rows = gold):", cm.to_string()]

    # 2) pair-level consistency on reversible pairs
    rev = d[d.reverse_pair_id != ""].set_index("instance_id")
    seen, cats = set(), []
    for iid, r in rev.iterrows():
        if iid in seen:
            continue
        o = rev.loc[r.reverse_pair_id]
        seen |= {iid, r.reverse_pair_id}
        p1, p2 = r.pred, o.pred
        both_right = p1 == r.label and p2 == o.label
        consistent = p1 in REVERSE and p2 == REVERSE[p1]
        if both_right:
            c = "both directions correct"
        elif consistent:
            c = "consistent but wrong"
        elif p1 == p2 and p1 in ("FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"):
            c = "direction-blind (same entailment label both ways)"
        elif "NEGATIVE_OTHER" in (p1, p2):
            c = "one direction predicted NEG_OTHER"
        else:
            c = "other inconsistency"
        cats.append({"gold": "EQUIV pair" if r.label == "EQUIVALENCE" else "FWD/BWD pair", "category": c})
    cats = pd.DataFrame(cats)
    log += ["\nReversible-pair outcomes (gold shown as pair type):",
            pd.crosstab(cats.category, cats.gold, margins=True).to_string()]

    # 3) heuristic error tags at instance level
    err = d[~d.correct].copy()

    def tag(r):
        ent = ("FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT")
        if r.pred in ent and ((r.pred == ent[0]) == (r.len_diff > 0)) and r.len_diff != 0:
            return "followed length cue (longer side -> entails)"
        if {r.label, r.pred} <= {"EQUIVALENCE", *ent}:
            return "EQUIV vs entailment confusion"
        if r.label == "NEGATIVE_OTHER":
            return "unrelated/contrastive pair predicted as related" + (" (high overlap)" if r.jaccard >= 0.3 else "")
        if r.pred == "NEGATIVE_OTHER":
            return "related pair rejected as NEG_OTHER"
        return "other"
    err["error_type"] = err.apply(tag, axis=1)
    log += ["\nError types (heuristic tags, all dev errors):", err.error_type.value_counts().to_string()]

    # 4) sample of errors for manual reading (5 per type, fixed seed)
    sample = err.sample(frac=1, random_state=0).groupby("error_type").head(5).sort_values("error_type")
    sample[["instance_id", "text1", "text2", "label", "pred", "len_diff", "jaccard", "error_type"]].to_csv(
        out / "errors_sample.csv", index=False)

    (out / "summary.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))
    print(f"\n-> {out}/")


if __name__ == "__main__":
    files = sys.argv[1:]
    if not files:
        cands = ["results/predictions/tfidf_plus_features_lr.csv",
                 "results/predictions/deberta-v3-base_seed42.csv",
                 "results/predictions/bert-base-uncased_seed42.csv"]
        files = [f for f in cands if Path(f).exists()]
    for f in files:
        analyse(f)
        print("=" * 70)
