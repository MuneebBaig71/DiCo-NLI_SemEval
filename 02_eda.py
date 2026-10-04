"""Selective, task-driven EDA. Produces ONE PDF of plots + a text file of stats.

Run:  python 02_eda.py
Out:  results/eda_plots.pdf   (all figures, no captions, self-explanatory titles/axes)
      results/eda_stats.txt   (dataset checks and numbers quoted in the report)

Semantic similarity needs sentence-transformers. If it is not installed locally,
run the Colab notebook first: it saves results/semantic_similarity.csv, which
this script picks up automatically.
"""
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from src.data import load_split
from src.features import pair_features, pos_proportions, tokens, mtld

OUT = Path("results")
OUT.mkdir(exist_ok=True)
ORDER = ["FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "EQUIVALENCE", "NEGATIVE_OTHER"]
SHORT = {"FORWARD_ENTAILMENT": "FORWARD", "BACKWARD_ENTAILMENT": "BACKWARD",
         "EQUIVALENCE": "EQUIV", "NEGATIVE_OTHER": "NEG_OTHER"}
COLORS = {"FORWARD_ENTAILMENT": "#2a6fdb", "BACKWARD_ENTAILMENT": "#e07b24",
          "EQUIVALENCE": "#2b9e5e", "NEGATIVE_OTHER": "#8a8a8a"}
plt.rcParams.update({"figure.figsize": (8, 4.8), "axes.grid": True, "grid.alpha": 0.3})

train, dev = load_split("train"), load_split("dev")
train["split"], dev["split"] = "train", "dev"
df = pd.concat([train, dev], ignore_index=True)
F = pair_features(df)
df = pd.concat([df, F], axis=1)
log = []


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    log.append(line)


# ------------------------------------------------------------------ checks
say("=== 1. Dataset-level checks ===")
for name, d in [("train", train), ("dev", dev)]:
    say(f"{name}: {len(d)} instances, {d.pair_id.nunique()} source pairs")
say("pair_id overlap train/dev:", len(set(train.pair_id) & set(dev.pair_id)))
say("missing values:", int(df[["text1", "text2", "label"]].isna().sum().sum()))
say("empty texts:", int(((df.text1.str.strip() == "") | (df.text2.str.strip() == "")).sum()))
say("identical text1==text2 (case-insensitive):", int(df.identical.sum()))
say("1-word phrases (either side):", int(((df.len1 == 1) | (df.len2 == 1)).sum()),
    f"({((df.len1 == 1) | (df.len2 == 1)).mean():.1%})")
say("max phrase length (tokens):", int(max(df.len1.max(), df.len2.max())))
dup = df.duplicated(["text1", "text2"], keep=False)
say("duplicated (text1,text2) rows:", int(dup.sum()))
conflict = df[dup].groupby(["text1", "text2"]).label.nunique()
say("  of which duplicated with DIFFERENT labels:", int((conflict > 1).sum()), "pairs")
tr_pairs = set(zip(train.text1, train.text2))
say("dev rows whose exact (text1,text2) also appears in train:",
    sum((a, b) in tr_pairs for a, b in zip(dev.text1, dev.text2)))
say("\nrows per pair_id by label (NEGATIVE_OTHER has no flipped copy):")
say(pd.crosstab(df.label, df.groupby("pair_id").pair_id.transform("size")).to_string())
say("\nlabel distribution (rows):")
say(pd.crosstab(df.label, df.split, normalize="columns").round(3).to_string())

say("\n=== 2. Length ===")
say(df.groupby("label")[["len1", "len2", "len_diff"]].mean().round(2).to_string())
say("share of FORWARD with text1 longer:", round((df[df.label == ORDER[0]].len_diff > 0).mean(), 3))
say("share of BACKWARD with text1 shorter:", round((df[df.label == ORDER[1]].len_diff < 0).mean(), 3))
say("share of EQUIV with equal length:", round((df[df.label == ORDER[2]].len_diff == 0).mean(), 3))

say("\n=== 3. Lexical overlap / containment ===")
say(df.groupby("label")[["jaccard", "contains_1_in_2", "contains_2_in_1"]].mean().round(3).to_string())

sem_path = OUT / "semantic_similarity.csv"
if not sem_path.exists():
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        ph = list(pd.unique(pd.concat([df.text1, df.text2])))
        E = dict(zip(ph, m.encode(ph, batch_size=128, normalize_embeddings=True)))
        pd.DataFrame({"instance_id": df.instance_id,
                      "cos": [float(E[a] @ E[b]) for a, b in zip(df.text1, df.text2)]}).to_csv(sem_path, index=False)
    except Exception as e:
        say(f"(semantic similarity skipped: {type(e).__name__}; run the Colab notebook to create {sem_path})")
if sem_path.exists():
    df = df.merge(pd.read_csv(sem_path), on="instance_id", how="left")
    say("\n=== 4. Semantic similarity (MiniLM cosine) ===")
    say(df.groupby("label").cos.describe()[["mean", "50%", "std"]].round(3).to_string())

say("\n=== 5. Vocabulary ===")
uniq = lambda d: pd.unique(pd.concat([d.text1, d.text2]))
tr_tok = [w for t in uniq(train) for w in tokens(t)]
dv_tok = [w for t in uniq(dev) for w in tokens(t)]
c_tr = Counter(tr_tok)
say(f"train: {len(tr_tok)} tokens, {len(c_tr)} types, hapax share {sum(v == 1 for v in c_tr.values()) / len(c_tr):.3f}")
say(f"dev OOV token rate w.r.t. train vocab: {np.mean([w not in c_tr for w in dv_tok]):.3f}")
for lab in ORDER:
    sub = df[df.label == lab]
    say(f"MTLD {lab:20s}: {mtld([w for t in uniq(sub) for w in tokens(t)]):.1f}")

# ------------------------------------------------------------------ plots
with PdfPages(OUT / "eda_plots.pdf") as pdf:
    def done(fig):
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # P1 label distribution per split (rows)
    fig, ax = plt.subplots()
    tab = pd.crosstab(df.label, df.split, normalize="columns").loc[ORDER] * 100
    x = np.arange(len(ORDER))
    ax.bar(x - 0.2, tab["train"], 0.4, label=f"train (n={len(train)})", color="#2a6fdb")
    ax.bar(x + 0.2, tab["dev"], 0.4, label=f"dev (n={len(dev)})", color="#e07b24")
    ax.set_xticks(x, [SHORT[l] for l in ORDER])
    ax.set(title="Label Distribution by Split", xlabel="Gold label", ylabel="Share of instances (%)")
    ax.legend()
    done(fig)

    # P2 signed length difference by label  (the key directional cue)
    fig, ax = plt.subplots()
    bins = np.arange(-6.5, 7.5, 1)
    for lab in ORDER:
        v = df[df.label == lab].len_diff.clip(-6, 6)
        ax.hist(v, bins=bins, histtype="step", linewidth=2, density=True, label=SHORT[lab], color=COLORS[lab])
    ax.set(title="Signed Length Difference (text1 − text2) by Label",
           xlabel="Token count difference (clipped to ±6)", ylabel="Density")
    ax.legend()
    done(fig)

    # P3 phrase length per side and label
    fig, ax = plt.subplots()
    data, pos, cols = [], [], []
    for i, lab in enumerate(ORDER):
        sub = df[df.label == lab]
        data += [sub.len1, sub.len2]
        pos += [i * 3, i * 3 + 1]
    bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True, showfliers=False)
    for j, b in enumerate(bp["boxes"]):
        b.set_facecolor("#9ec5f8" if j % 2 == 0 else "#f8c79e")
    ax.set_xticks([i * 3 + 0.5 for i in range(4)], [SHORT[l] for l in ORDER])
    ax.legend([bp["boxes"][0], bp["boxes"][1]], ["text1", "text2"])
    ax.set(title="Phrase Length by Label and Side", xlabel="Gold label", ylabel="Phrase length (tokens)")
    done(fig)

    # P4 Jaccard overlap
    fig, ax = plt.subplots()
    ax.violinplot([df[df.label == l].jaccard for l in ORDER], showmedians=True)
    ax.set_xticks(range(1, 5), [SHORT[l] for l in ORDER])
    ax.set(title="Word Overlap (Jaccard) by Label", xlabel="Gold label", ylabel="Jaccard similarity (0–1)")
    done(fig)

    # P5 containment pattern
    cont = np.select([df.contains_1_in_2.astype(bool) & df.contains_2_in_1.astype(bool),
                      df.contains_2_in_1.astype(bool), df.contains_1_in_2.astype(bool)],
                     ["same word set", "text1 ⊇ text2", "text2 ⊇ text1"], "neither")
    ct = pd.crosstab(df.label, cont, normalize="index").loc[ORDER]
    ct = ct[[c for c in ["text1 ⊇ text2", "text2 ⊇ text1", "same word set", "neither"] if c in ct]]
    fig, ax = plt.subplots()
    im = ax.imshow(ct.values, cmap="Blues", vmin=0, vmax=1)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax.text(j, i, f"{ct.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if ct.values[i, j] > 0.5 else "black")
    ax.set_xticks(range(ct.shape[1]), ct.columns)
    ax.set_yticks(range(4), [SHORT[l] for l in ORDER])
    ax.grid(False)
    ax.set(title="Word-Set Containment Pattern by Label (row share)", xlabel="Containment pattern", ylabel="Gold label")
    fig.colorbar(im, ax=ax, label="Share of label's instances")
    done(fig)

    # P6 POS of the non-shared words, text1 side vs text2 side
    cols = ["NOUN", "PROPN", "ADJ", "VERB", "ADP", "NUM"]
    if "extra1_NOUN" in df:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
        for ax, side in zip(axes, ["1", "2"]):
            m = df.groupby("label")[[f"extra{side}_{c}" for c in cols]].mean().loc[ORDER]
            x = np.arange(len(cols))
            for k, lab in enumerate(ORDER):
                ax.bar(x + (k - 1.5) * 0.2, m.loc[lab].values, 0.2, label=SHORT[lab], color=COLORS[lab])
            ax.set_xticks(x, cols)
            ax.set(title=f"Words only in text{side}", xlabel="POS tag")
        axes[0].set_ylabel("Mean count per pair")
        axes[1].legend()
        fig.suptitle("POS of Non-Shared Words by Label")
        done(fig)

    # P7 overall POS proportions per label
    pp = pos_proportions(df)
    if pp is not None:
        pp = pp[ORDER]
        top = pp.mean(axis=1).sort_values(ascending=False).index[:8]
        fig, ax = plt.subplots()
        bottom = np.zeros(4)
        cmap = plt.get_cmap("tab10")
        for k, tag in enumerate(top):
            ax.bar([SHORT[l] for l in ORDER], pp.loc[tag].values * 100, bottom=bottom, label=tag, color=cmap(k))
            bottom += pp.loc[tag].values * 100
        ax.set(title="POS Composition of Phrases by Label (top 8 tags)", xlabel="Gold label", ylabel="Share of tokens (%)")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        done(fig)

    # P8 semantic similarity
    if "cos" in df:
        fig, ax = plt.subplots()
        ax.violinplot([df[df.label == l].cos.dropna() for l in ORDER], showmedians=True)
        ax.set_xticks(range(1, 5), [SHORT[l] for l in ORDER])
        ax.set(title="Sentence-Embedding Similarity by Label (all-MiniLM-L6-v2)",
               xlabel="Gold label", ylabel="Cosine similarity")
        done(fig)

    # P9 Zipf / vocabulary
    c_all = Counter(w for t in uniq(df) for w in tokens(t))
    f = np.array(sorted(c_all.values(), reverse=True))
    fig, ax = plt.subplots()
    ax.loglog(np.arange(1, len(f) + 1), f, ".", markersize=3, color="#2a6fdb")
    ax.set(title="Rank–Frequency of Words Across All Unique Phrases",
           xlabel="Word rank (log scale)", ylabel="Frequency (log scale)")
    done(fig)

(OUT / "eda_stats.txt").write_text("\n".join(log) + "\n")
print("\nwrote results/eda_plots.pdf and results/eda_stats.txt")
