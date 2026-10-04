"""Baseline A: classical ML baselines, all scored with the official scorer on dev.

Run:  python 01_classical_baselines.py
Writes predictions, score reports, a results CSV/LaTeX table and LR coefficients
into results/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_split, write_predictions, score, headline
from src.features import pair_features

SEED = 42
OUT = Path("results")
(OUT / "predictions").mkdir(parents=True, exist_ok=True)
(OUT / "scores").mkdir(parents=True, exist_ok=True)

train, dev = load_split("train"), load_split("dev")
print(f"train={len(train)}  dev={len(dev)}")

print("computing features ...")
F_tr, F_dev = pair_features(train), pair_features(dev)
F_tr.assign(label=train["label"].values).to_csv(OUT / "train_features.csv", index=False)
F_dev.assign(label=dev["label"].values).to_csv(OUT / "dev_features.csv", index=False)

LENGTH = ["len1", "len2", "len_diff", "char_diff"]
SYMMETRIC = ["jaccard", "identical"]  # cannot see direction by construction
ALL = list(F_tr.columns)

results, models = {}, {}


def run(name, preds):
    path = write_predictions(dev["instance_id"], preds, OUT / "predictions" / f"{name}.csv")
    rep = score(path, "dev", OUT / "scores" / name)
    results[name] = headline(rep)
    print(f"{name:28s} " + "  ".join(f"{k}={v:.3f}" for k, v in results[name].items()))


def lr():
    return LogisticRegression(max_iter=5000, C=1.0, random_state=SEED)


# 1) Majority class: the floor any model must beat.
majority = train["label"].value_counts().idxmax()
run("majority", [majority] * len(dev))

# 2) Length-only LR: tests the "longer phrase entails shorter phrase" shortcut.
m = make_pipeline(StandardScaler(), lr()).fit(F_tr[LENGTH], train["label"])
run("lr_length_only", m.predict(F_dev[LENGTH]))

# 3) Symmetric-overlap-only LR: should fail to split FORWARD vs BACKWARD.
m = make_pipeline(StandardScaler(), lr()).fit(F_tr[SYMMETRIC], train["label"])
run("lr_overlap_only", m.predict(F_dev[SYMMETRIC]))

# 4) All engineered features (the interpretable Baseline A).
feat_lr = make_pipeline(StandardScaler(), lr()).fit(F_tr[ALL], train["label"])
run("lr_engineered_features", feat_lr.predict(F_dev[ALL]))
models["feat_lr"] = feat_lr

# 5) TF-IDF + LR. text1 and text2 get separate blocks, so word order of the
#    pair (direction) is preserved: [tfidf(text1) | tfidf(text2)].
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
vec.fit(pd.concat([train["text1"], train["text2"]]))


def tfidf_pair(df):
    return hstack([vec.transform(df["text1"]), vec.transform(df["text2"])]).tocsr()


X_tr, X_dev = tfidf_pair(train), tfidf_pair(dev)
m = lr().fit(X_tr, train["label"])
run("tfidf_lr", m.predict(X_dev))

# 6) TF-IDF + engineered features together.
sc = StandardScaler().fit(F_tr[ALL])
Xc_tr = hstack([X_tr, csr_matrix(sc.transform(F_tr[ALL]))]).tocsr()
Xc_dev = hstack([X_dev, csr_matrix(sc.transform(F_dev[ALL]))]).tocsr()
m = lr().fit(Xc_tr, train["label"])
run("tfidf_plus_features_lr", m.predict(Xc_dev))

# ---- Save results table ----
res = pd.DataFrame(results).T
res.to_csv(OUT / "classical_results.csv")
print("\n", res.round(3))

# ---- Coefficients of the engineered-feature LR (standardised, so comparable) ----
clf = feat_lr[-1]
coef = pd.DataFrame(clf.coef_.T, index=ALL, columns=clf.classes_)
coef.to_csv(OUT / "feature_lr_coefficients.csv")
print("\nTop features per class (standardised LR coefficients):")
for c in coef.columns:
    top = coef[c].sort_values(ascending=False)
    print(f"  {c:20s} +: {', '.join(top.index[:4])}   -: {', '.join(top.index[-3:])}")
