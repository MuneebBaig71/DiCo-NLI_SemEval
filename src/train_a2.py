"""Assignment 2 training: cross-encoder fine-tuning with optional
reversal-consistency loss and Neg class weighting.

One `run(cfg)` call = one model, one seed. It
  1. carves a validation split out of train, grouped by pair_id (fixed seed,
     identical for every run, so all models see the same split),
  2. fine-tunes, keeping the epoch with the best validation weighted-F1,
  3. writes dev probabilities + argmax predictions, scores them with the
     official scorer, and returns the headline numbers.

Loss = CE(weighted)  +  lambda_cons * symmetric-KL(p(t1,t2) || swap(p(t2,t1)))
The consistency term is applied only to pairs whose two orientations are both
in the batch (non-Neg pairs); Neg singles get CE only.
"""
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from src.data import load_split, LABELS, LABEL2ID, ROOT
from src.decoding import SWAP

SWAP_T = torch.tensor(SWAP)
NEG_ID = LABEL2ID["NEGATIVE_OTHER"]

DEFAULTS = dict(
    model="microsoft/deberta-v3-base",
    seed=42,
    epochs=6,
    lr=2e-5,
    batch_pairs=8,          # pairs per batch -> about 16 rows, like A1
    max_len=64,
    warmup=0.1,
    weight_decay=0.01,
    lambda_cons=0.0,        # 0 = no consistency loss
    neg_weight=1.0,         # 1 = no class weighting
    val_frac=0.1,
    split_seed=2027,        # fixed: same validation split for every run
    max_train_rows=None,    # for quick smoke tests only
    out_dir="results_a2",
    tag=None,
    save_checkpoint=False,  # True: write best epoch to <out_dir>/checkpoints/<tag>/
    reinit_head=False,      # True: start from a checkpoint with a different label set (NLI-pretrained)
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- data

def group_split(df, val_frac, seed):
    """Split train by pair_id so a pair and its reverse stay together.
    Paired (non-Neg) and single (Neg) groups are sampled separately so the
    validation set keeps the train label mix."""
    rng = np.random.RandomState(seed)
    sizes = df.groupby("pair_id").size()
    val_ids = []
    for n in (1, 2):
        ids = sizes[sizes == n].index.to_numpy()
        rng.shuffle(ids)
        val_ids += list(ids[: int(round(len(ids) * val_frac))])
    is_val = df.pair_id.isin(set(val_ids))
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def make_units(df):
    """List of row-index groups: [i, j] for a reversible pair, [i] for a single."""
    return [list(ix) for ix in df.groupby("pair_id", sort=False).indices.values()]


class PairCollator:
    def __init__(self, df, tok, max_len):
        self.df, self.tok, self.max_len = df, tok, max_len

    def __call__(self, units):
        rows = [i for u in units for i in u]
        sub = self.df.iloc[rows]
        enc = self.tok(list(sub.text1), list(sub.text2), truncation=True,
                       max_length=self.max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([LABEL2ID[l] for l in sub.label])
        # positions (within this batch) of the two orientations of each pair
        a, b, k = [], [], 0
        for u in units:
            if len(u) == 2:
                a.append(k); b.append(k + 1)
            k += len(u)
        enc["pair_a"], enc["pair_b"] = torch.tensor(a, dtype=torch.long), torch.tensor(b, dtype=torch.long)
        return enc


# ---------------------------------------------------------------- loss

def consistency_loss(logits, pair_a, pair_b):
    """Symmetric KL between p(t1,t2) and the label-swapped p(t2,t1)."""
    if len(pair_a) == 0:
        return logits.new_zeros(())
    lp = F.log_softmax(logits, dim=-1)
    la = lp[pair_a]
    lb = lp[pair_b][:, SWAP_T.to(logits.device)]
    kl_ab = F.kl_div(lb, la, log_target=True, reduction="batchmean")  # KL(a || swap b)
    kl_ba = F.kl_div(la, lb, log_target=True, reduction="batchmean")  # KL(swap b || a)
    return 0.5 * (kl_ab + kl_ba)


# ---------------------------------------------------------------- predict

@torch.no_grad()
def predict_probs(model, df, tok, cfg, dev):
    model.eval()
    out = []
    for s in range(0, len(df), 64):
        sub = df.iloc[s:s + 64]
        enc = tok(list(sub.text1), list(sub.text2), truncation=True,
                  max_length=cfg["max_len"], padding=True, return_tensors="pt").to(dev)
        out.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


def wf1(df, probs):
    gold = df.label.map(LABEL2ID).to_numpy()
    return f1_score(gold, probs.argmax(1), average="weighted")


# ---------------------------------------------------------------- run

def run(**overrides):
    cfg = {**DEFAULTS, **overrides}
    short = cfg["model"].split("/")[-1]
    tag = cfg["tag"] or (f"{short}_cons{cfg['lambda_cons']}_neg{cfg['neg_weight']}"
                         f"_seed{cfg['seed']}")
    out = ROOT / cfg["out_dir"]
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(parents=True, exist_ok=True)

    set_seed(cfg["seed"])
    dev_ = device()
    train_all = load_split("train")
    tr, va = group_split(train_all, cfg["val_frac"], cfg["split_seed"])
    if cfg["max_train_rows"]:
        keep = tr.pair_id.drop_duplicates().iloc[: cfg["max_train_rows"] // 2]
        tr = tr[tr.pair_id.isin(keep)].reset_index(drop=True)
    dev_df = load_split("dev")

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    # reinit_head=True replaces a checkpoint's own classifier (e.g. a 3-class NLI head)
    # with a fresh 4-class one; the default leaves loading exactly as in the A2 runs.
    extra = dict(ignore_mismatched_sizes=True) if cfg["reinit_head"] else {}
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"], num_labels=len(LABELS), **extra).to(dev_).float()  # fp32 on purpose (DeBERTa NaN fix)

    loader = DataLoader(make_units(tr), batch_size=cfg["batch_pairs"], shuffle=True,
                        collate_fn=PairCollator(tr, tok, cfg["max_len"]))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = len(loader) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup"] * total), total)
    cw = torch.ones(len(LABELS)); cw[NEG_ID] = cfg["neg_weight"]
    cw = cw.to(dev_)

    best, best_state, history = -1, None, []
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        tot = ce_sum = cons_sum = 0.0
        for batch in loader:
            pa, pb = batch.pop("pair_a"), batch.pop("pair_b")
            labels = batch.pop("labels").to(dev_)
            batch = {k: v.to(dev_) for k, v in batch.items()}
            logits = model(**batch).logits
            ce = F.cross_entropy(logits, labels, weight=cw)
            cons = consistency_loss(logits, pa.to(dev_), pb.to(dev_)) if cfg["lambda_cons"] > 0 else logits.new_zeros(())
            loss = ce + cfg["lambda_cons"] * cons
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {ep}; lower lr or check fp32.")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += 1; ce_sum += ce.item(); cons_sum += cons.item()
        val_f1 = wf1(va, predict_probs(model, va, tok, cfg, dev_))
        history.append(dict(epoch=ep, ce=ce_sum / tot, cons=cons_sum / tot, val_wf1=val_f1))
        print(f"[{tag}] epoch {ep}  ce {ce_sum/tot:.4f}  cons {cons_sum/tot:.4f}  "
              f"val W-F1 {val_f1:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if val_f1 > best:
            best, best_ep = val_f1, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    if cfg["save_checkpoint"]:
        save_checkpoint(model, tok, cfg, tag, best_ep, out / "checkpoints" / tag)
    for name, df in (("dev", dev_df), ("val", va)):
        p = predict_probs(model, df, tok, cfg, dev_)
        probs = pd.DataFrame(p, columns=LABELS)
        probs["instance_id"] = df.instance_id.values
        probs.to_csv(out / "predictions" / f"{tag}_{name}_probs.csv", index=False)
        if name == "dev":  # official two-column file (argmax decoding)
            pd.DataFrame({"instance_id": df.instance_id.values,
                          "label": [LABELS[k] for k in p.argmax(1)]}
                         ).to_csv(out / "predictions" / f"{tag}_dev.csv", index=False)

    meta = dict(cfg=cfg, tag=tag, best_epoch=best_ep, best_val_wf1=best,
                history=history, seconds=round(time.time() - t0))
    (out / "runs" / f"{tag}.json").write_text(json.dumps(meta, indent=2))
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return meta


# ---------------------------------------------------------------- checkpoints

def save_checkpoint(model, tok, cfg, tag, best_epoch, path):
    """Save the selected (best-validation) model, its tokenizer and the run config."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    (path / "a2_config.json").write_text(json.dumps(dict(cfg=cfg, tag=tag, best_epoch=best_epoch), indent=2))
    print(f"[{tag}] checkpoint saved to {path}", flush=True)


def load_checkpoint(path):
    """Load a checkpoint written by save_checkpoint. Returns (model, tokenizer, cfg)."""
    path = Path(path)
    cfg = json.loads((path / "a2_config.json").read_text())["cfg"]
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device()).float()
    model.eval()
    return model, tok, cfg


def predict_from_checkpoint(path, split="dev"):
    """Reload a saved model and return (instance_ids, probabilities) for a split."""
    model, tok, cfg = load_checkpoint(path)
    df = load_split(split)
    probs = predict_probs(model, df, tok, cfg, device())
    return df.instance_id.tolist(), probs
