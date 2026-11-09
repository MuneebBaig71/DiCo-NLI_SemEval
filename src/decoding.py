"""Inference-time decoding rules for DiCo-NLI (Assignment 2).

All rules take a model's class probabilities and only change how labels are
picked. No retraining is needed, so they work on any saved model.

Partner detection uses only `pair_id` from the participant file (the file we
get at test time), never the gold reference.

  argmax      plain highest-probability label (the A1 behaviour)
  symmetric   average p(t1,t2) with the label-swapped p(t2,t1) of its partner,
              so both orientations always get mutually consistent labels
  struct_pair symmetric, and rows that have a partner may not be Neg
              (in train and dev, every row with a partner is non-Neg)
  struct_full struct_pair, and rows without a partner are forced to Neg
              (pure data-structure rule; reported as an artifact, not a model)
"""
import numpy as np
import pandas as pd

LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
NEG = LABELS.index("NEGATIVE_OTHER")
# Index permutation that maps a reversed-pair distribution onto this orientation:
# Eq->Eq, Fwd<-Bwd, Bwd<-Fwd, Neg->Neg
SWAP = [0, 2, 1, 3]

MODES = ["argmax", "symmetric", "struct_pair", "struct_full"]


def load_probs(path):
    """Read a *_probs.csv (columns = LABELS + instance_id)."""
    df = pd.read_csv(path)
    return df["instance_id"].tolist(), df[LABELS].to_numpy(dtype=float)


def partner_index(ids, participant_df):
    """For each row, the row index of its reversed partner, or -1 if none.

    Partners are the two rows that share a pair_id.
    """
    meta = participant_df.set_index("instance_id").loc[ids]
    pos = {iid: i for i, iid in enumerate(ids)}
    groups = meta.groupby("pair_id").groups
    partner = np.full(len(ids), -1)
    for _, members in groups.items():
        members = list(members)
        if len(members) == 2:
            a, b = pos[members[0]], pos[members[1]]
            partner[a], partner[b] = b, a
    return partner


def decode(probs, partner, mode="argmax"):
    """Return label strings for every row under the chosen decoding rule."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    p = probs.copy()
    has = partner >= 0

    if mode != "argmax":
        # symmetrise: mean of own probs and partner's label-swapped probs
        p[has] = 0.5 * (probs[has] + probs[partner[has]][:, SWAP])

    if mode in ("struct_pair", "struct_full"):
        p[has, NEG] = -1.0  # paired rows can never be Neg

    pred = p.argmax(axis=1)

    if mode == "struct_full":
        pred[~has] = NEG

    return [LABELS[k] for k in pred]


def one_sided_neg(preds, partner):
    """Number of pairs where exactly one orientation is predicted Neg."""
    count = 0
    for i, j in enumerate(partner):
        if j > i:
            a = preds[i] == "NEGATIVE_OTHER"
            b = preds[j] == "NEGATIVE_OTHER"
            count += a != b
    return count
