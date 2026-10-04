"""Combine classical + transformer results into one table (CSV + LaTeX booktabs).

Run after 01_classical_baselines.py and after unzipping the Colab results into results/.
Out: results/all_results.csv, results/results_table.tex
"""
from pathlib import Path
import pandas as pd

R = Path("results")
NAMES = {
    "majority": "Majority class",
    "lr_length_only": "LR (length features only)",
    "lr_overlap_only": "LR (symmetric overlap only)",
    "lr_engineered_features": "LR (all engineered features)",
    "tfidf_lr": "TF-IDF + LR",
    "tfidf_plus_features_lr": "TF-IDF + features + LR",
}
cols = ["weighted_f1", "soft_cons", "hard_cons", "macro_f1", "accuracy"]
rows = []
cl = pd.read_csv(R / "classical_results.csv", index_col=0)
for k, r in cl.iterrows():
    rows.append({"group": "Classical (A)", "model": NAMES.get(k, k), **{c: f"{r[c]:.3f}" for c in cols}})

tf = R / "transformer_results_per_seed.csv"
if tf.exists():
    t = pd.read_csv(tf)
    for m, g in t.groupby("model"):
        cell = lambda c: f"{g[c].mean():.3f} $\\pm$ {g[c].std(ddof=0):.3f}" if len(g) > 1 else f"{g[c].mean():.3f}"
        rows.append({"group": "Pretrained, fine-tuned (B)", "model": f"{m} ({len(g)} seeds)", **{c: cell(c) for c in cols}})
else:
    print("(no transformer results yet: run the Colab notebook and unzip into results/)")

df = pd.DataFrame(rows)
df.to_csv(R / "all_results.csv", index=False)
print(df.to_string(index=False))

head = ["Model", "W-F1", "SoftCons", "HardCons", "Macro-F1", "Acc."]
lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lccccc}", r"\toprule",
         " & ".join(head) + r" \\", r"\midrule"]
for g, sub in df.groupby("group", sort=False):
    lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{g}}}}} \\")
    for _, r in sub.iterrows():
        lines.append(" & ".join([r.model.replace("_", r"\_")] + [r[c] for c in cols]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Dev-set results (Track 1, English) with the official scorer. W-F1, SoftCons and HardCons are the official metrics.}",
          r"\label{tab:results}", r"\end{table}"]
(R / "results_table.tex").write_text("\n".join(lines) + "\n")
print("\nwrote results/results_table.tex")
