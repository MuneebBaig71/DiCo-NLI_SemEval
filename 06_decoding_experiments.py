"""A2 step 1: test decoding rules on saved dev probabilities (no training).

Usage:
    python 06_decoding_experiments.py                      # all *_probs.csv in results/predictions
    python 06_decoding_experiments.py path/to/model_probs.csv

Writes results_a2/decoding/<model>__<mode>.csv and results_a2/decoding_results.csv
"""
import sys
from pathlib import Path

import pandas as pd

from src.data import split_path, score, headline
from src.decoding import MODES, load_probs, partner_index, decode, one_sided_neg

OUT = Path("results_a2/decoding")
OUT.mkdir(parents=True, exist_ok=True)

files = [Path(a) for a in sys.argv[1:]] or sorted(Path("results/predictions").glob("*_probs.csv"))
participant = pd.read_csv(split_path("dev", "participant_labeled"), keep_default_na=False)

rows = []
for f in files:
    model = f.stem.replace("_probs", "")
    ids, probs = load_probs(f)
    partner = partner_index(ids, participant)
    for mode in MODES:
        preds = decode(probs, partner, mode)
        out = OUT / f"{model}__{mode}.csv"
        pd.DataFrame({"instance_id": ids, "label": preds}).to_csv(out, index=False)
        h = headline(score(out, "dev"))
        rows.append({"model": model, "mode": mode, **h,
                     "one_sided_neg": one_sided_neg(preds, partner)})

res = pd.DataFrame(rows)
res.to_csv("results_a2/decoding_results.csv", index=False)
pd.set_option("display.width", 160)
print(res.round(3).to_string(index=False))
