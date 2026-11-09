"""Score every multilingual run on dev with the official scorer.

For each saved `<tag>__<track>_dev_probs.csv` it
  1. decodes labels with argmax, symmetric and struct_pair (src/decoding.py),
  2. writes the two-column prediction file and scores it against that track's
     reference file with the organizers' scorer,
  3. collects W-F1 / SoftCons / HardCons / Macro-F1 / Acc and per-class F1.

Extra tables:
  - aggregates: monolingual average (EN/ES/EU) and all-track average (+MIX)
    for runs that were evaluated on those tracks
  - Track 4 breakdown: W-F1 and accuracy per language combination

Run from the repo root:   python -m multilingual.ml_eval
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.data import LABELS, ROOT
from src.decoding import decode
from multilingual.ml_data import TRACK_NAMES, load, partner_index, track_path

MODES = ["argmax", "symmetric", "struct_pair"]
SHORT = {"EQUIVALENCE": "f1_eq", "FORWARD_ENTAILMENT": "f1_fwd",
         "BACKWARD_ENTAILMENT": "f1_bwd", "NEGATIVE_OTHER": "f1_neg"}


def _official(ref_path, pred_path):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from evaluation_functions import score_files
    return score_files(ref_path, pred_path, output_dir=None).to_dict()


def setting(train_tracks, eval_track):
    if eval_track in train_tracks:
        return "in-language"
    if eval_track == "mix":
        return "zero-shot (mixed pairs unseen)"
    return "zero-shot cross-lingual"


def score_all(out_dir="results_multilingual", modes=MODES):
    out = ROOT / out_dir
    dec_dir = out / "decoded"
    dec_dir.mkdir(parents=True, exist_ok=True)
    rows, breakdown = [], []

    for f in sorted((out / "predictions").glob("*__*_dev_probs.csv")):
        tag, rest = f.stem.split("__", 1)
        track = rest[: -len("_dev_probs")]
        meta_path = out / "runs" / f"{tag}.json"
        cfg = json.loads(meta_path.read_text())["cfg"] if meta_path.exists() else {}
        train_tracks = cfg.get("train_tracks", [])

        probs_df = pd.read_csv(f)
        dev = load(track, "dev").set_index("instance_id").loc[probs_df.instance_id].reset_index()
        partner = partner_index(dev)
        probs = probs_df[LABELS].to_numpy(dtype=float)

        for mode in modes:
            preds = decode(probs, partner, mode)
            pred_path = dec_dir / f"{tag}__{track}__{mode}.csv"
            pd.DataFrame({"instance_id": dev.instance_id, "label": preds}).to_csv(pred_path, index=False)
            rep = _official(track_path(track, "dev"), pred_path)
            c = rep["classification"]
            row = dict(run=tag, model=cfg.get("model", "?").split("/")[-1],
                       train_on="+".join(train_tracks), eval_on=track,
                       setting=setting(train_tracks, track), decoding=mode,
                       wf1=rep["weighted_f1"], soft_cons=rep["soft_cons"],
                       hard_cons=rep["hard_cons"], macro_f1=c["macro_f1"],
                       acc=c["accuracy"])
            for lab, key in SHORT.items():
                row[key] = c["per_label"][lab]["f1"]
            rows.append(row)

            if track == "mix" and mode in ("argmax", "struct_pair"):
                for lp, g in dev.assign(pred=preds).groupby("lang_pair"):
                    breakdown.append(dict(run=tag, decoding=mode, lang_pair=lp,
                                          wf1=f1_score(g.label, g.pred, average="weighted"),
                                          acc=accuracy_score(g.label, g.pred), n=len(g)))

    res = pd.DataFrame(rows)
    if res.empty:
        print("No prediction files found in", out / "predictions")
        return res, pd.DataFrame(), pd.DataFrame()
    res.to_csv(out / "ml_results.csv", index=False)

    agg = []
    for (run, mode), g in res.groupby(["run", "decoding"]):
        g = g.set_index("eval_on")
        if {"en", "es", "eu"} <= set(g.index):
            mono = g.loc[["en", "es", "eu"]]
            entry = dict(run=run, decoding=mode,
                         mono_wf1=mono.wf1.mean(), mono_soft=mono.soft_cons.mean(),
                         mono_hard=mono.hard_cons.mean())
            if "mix" in g.index:
                allt = g.loc[["en", "es", "eu", "mix"]]
                entry.update(all_wf1=allt.wf1.mean(), all_soft=allt.soft_cons.mean(),
                             all_hard=allt.hard_cons.mean())
            agg.append(entry)
    agg = pd.DataFrame(agg)
    if not agg.empty:
        agg.to_csv(out / "ml_aggregates.csv", index=False)

    bd = pd.DataFrame(breakdown)
    if not bd.empty:
        bd.to_csv(out / "ml_mix_by_lang_pair.csv", index=False)
    return res, agg, bd


def main(out_dir="results_multilingual"):
    res, agg, bd = score_all(out_dir)
    if res.empty:
        return
    pd.set_option("display.width", 200)
    cols = ["run", "train_on", "eval_on", "setting", "wf1", "soft_cons", "hard_cons", "macro_f1", "acc"]
    for mode in MODES:
        print(f"\n=== decoding: {mode} ===")
        print(res[res.decoding == mode][cols].round(3).to_string(index=False))
    print("\n=== per-class F1 (argmax) ===")
    print(res[res.decoding == "argmax"][["run", "eval_on", *SHORT.values()]].round(3).to_string(index=False))
    if not agg.empty:
        print("\n=== averages over tracks ===")
        print(agg.round(3).to_string(index=False))
    if not bd.empty:
        print("\n=== Track 4 by language combination ===")
        print(bd.pivot_table(index=["run", "decoding"], columns="lang_pair", values="wf1").round(3).to_string())
    print("\nTrack names:", TRACK_NAMES)


if __name__ == "__main__":
    main(*sys.argv[1:])
