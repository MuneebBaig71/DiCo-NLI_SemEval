"""Multilingual pilot runs for DiCo-NLI (Tracks 1-4).

One `run(...)` call = train one model (one seed) on one or more tracks, then
write dev-set probabilities for every track in `eval_tracks`.

Same training recipe as src/train_a2.py (fp32, AdamW, linear warmup, best epoch
picked on a held-out validation split, optional consistency loss and Neg
weighting), only made track-aware. Scoring is done separately in ml_eval.py.

Example:
    from multilingual.ml_train import run
    run(train_tracks=["en", "es", "eu"], eval_tracks=["en", "es", "eu", "mix"],
        tag="X3_joint")
"""
import json
import time

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from src.data import LABELS, LABEL2ID, ROOT
from src.train_a2 import (PairCollator, consistency_loss, device, predict_probs,
                          set_seed, wf1)
from multilingual.ml_data import TRACKS, check_data, load, make_units, val_pair_ids

NEG_ID = LABEL2ID["NEGATIVE_OTHER"]

DEFAULTS = dict(
    model="microsoft/mdeberta-v3-base",
    seed=42,
    epochs=6,
    lr=2e-5,
    batch_pairs=8,          # units per batch -> about 16 rows, like A1/A2
    max_len=64,
    warmup=0.1,
    weight_decay=0.01,
    lambda_cons=0.0,
    neg_weight=1.0,
    val_frac=0.1,
    split_seed=2027,        # same validation pair_ids as the A2 English runs
    max_train_rows=None,    # smoke tests only
    out_dir="results_multilingual",
    tag=None,
)


def run(train_tracks=("en",), eval_tracks=("en",), **overrides):
    check_data()
    train_tracks, eval_tracks = list(train_tracks), list(eval_tracks)
    for t in train_tracks + eval_tracks:
        if t not in TRACKS:
            raise ValueError(f"unknown track {t!r}; use one of {list(TRACKS)}")
    cfg = {**DEFAULTS, **overrides,
           "train_tracks": train_tracks, "eval_tracks": eval_tracks}
    short = cfg["model"].split("/")[-1]
    tag = cfg["tag"] or f"{short}_{'+'.join(train_tracks)}_seed{cfg['seed']}"
    if "__" in tag:
        raise ValueError("tag may not contain '__' (used as a separator in file names)")
    cfg["tag"] = tag

    out = ROOT / cfg["out_dir"]
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ data
    set_seed(cfg["seed"])
    dev_ = device()
    train_all = pd.concat([load(t, "train") for t in train_tracks], ignore_index=True)
    is_val = train_all.pair_id.isin(val_pair_ids(cfg["val_frac"], cfg["split_seed"]))
    tr = train_all[~is_val].reset_index(drop=True)
    va = train_all[is_val].reset_index(drop=True)
    if cfg["max_train_rows"]:
        ids = tr.pair_id.drop_duplicates()
        sizes = tr.groupby("pair_id").size().loc[ids]
        keep = set(ids[(sizes.cumsum() <= cfg["max_train_rows"]).to_numpy()])
        tr = tr[tr.pair_id.isin(keep)].reset_index(drop=True)
    print(f"[{tag}] train on {train_tracks}: {len(tr)} rows, val {len(va)} rows, "
          f"device {dev_}", flush=True)

    # ------------------------------------------------------------ model
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"], num_labels=len(LABELS)).to(dev_).float()  # fp32: mDeBERTa NaNs in fp16

    loader = DataLoader(make_units(tr), batch_size=cfg["batch_pairs"], shuffle=True,
                        collate_fn=PairCollator(tr, tok, cfg["max_len"]))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    total = len(loader) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup"] * total), total)
    cw = torch.ones(len(LABELS))
    cw[NEG_ID] = cfg["neg_weight"]
    cw = cw.to(dev_)

    # ------------------------------------------------------------ train
    best, best_ep, best_state, history = -1.0, None, None, []
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        n = ce_sum = cons_sum = 0.0
        for batch in loader:
            pa, pb = batch.pop("pair_a"), batch.pop("pair_b")
            labels = batch.pop("labels").to(dev_)
            batch = {k: v.to(dev_) for k, v in batch.items()}
            logits = model(**batch).logits
            ce = F.cross_entropy(logits, labels, weight=cw)
            cons = (consistency_loss(logits, pa.to(dev_), pb.to(dev_))
                    if cfg["lambda_cons"] > 0 else logits.new_zeros(()))
            loss = ce + cfg["lambda_cons"] * cons
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {ep}; lower lr or check fp32.")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            n += 1
            ce_sum += ce.item()
            cons_sum += cons.item()
        val_f1 = wf1(va, predict_probs(model, va, tok, cfg, dev_))
        history.append(dict(epoch=ep, ce=ce_sum / n, cons=cons_sum / n, val_wf1=val_f1))
        print(f"[{tag}] epoch {ep}  ce {ce_sum/n:.4f}  cons {cons_sum/n:.4f}  "
              f"val W-F1 {val_f1:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if val_f1 > best:
            best, best_ep = val_f1, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ------------------------------------------------------------ predict dev
    model.load_state_dict(best_state)
    for t in eval_tracks:
        dev_df = load(t, "dev")
        p = predict_probs(model, dev_df, tok, cfg, dev_)
        probs = pd.DataFrame(p, columns=LABELS)
        probs["instance_id"] = dev_df.instance_id.values
        probs.to_csv(out / "predictions" / f"{tag}__{t}_dev_probs.csv", index=False)
        print(f"[{tag}] wrote dev probabilities for {t} ({len(dev_df)} rows)", flush=True)

    meta = dict(cfg=cfg, tag=tag, best_epoch=best_ep, best_val_wf1=best,
                history=history, seconds=round(time.time() - t0))
    (out / "runs" / f"{tag}.json").write_text(json.dumps(meta, indent=2))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta
