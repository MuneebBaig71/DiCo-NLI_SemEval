"""Track-aware data loading for DiCo-NLI Tracks 1-4.

Track keys used everywhere in this folder:
    en  -> track1 (English)        es  -> track2 (Spanish)
    eu  -> track3 (Basque)         mix -> track4 (mixed cross-language pairs)

Partners (the reversed orientation of a row) are found from instance_id alone,
so this also works on the unlabeled test file in January:
    dico_phrasis_0000002__en-es__original  <->  dico_phrasis_0000002__es-en__flipped
In Track 4 one pair_id has 12 rows (6 language combos x 2 directions), so the
A2 rule "two rows with the same pair_id are partners" does not hold there.
"""
import numpy as np
import pandas as pd

from src.data import DATA_DIR, load_split as load_track1
from src.train_a2 import group_split

TRACKS = {"en": "track1", "es": "track2", "eu": "track3", "mix": "track4"}
TRACK_NAMES = {"en": "English", "es": "Spanish", "eu": "Basque", "mix": "Mixed"}


def track_path(track, split, kind="reference"):
    """kind: reference | participant_labeled | submission_template."""
    return DATA_DIR / split / f"dico_nli_{split}_{TRACKS[track]}_{kind}.csv"


def check_data():
    """Raise a clear error if the Track 2-4 files are not in data/ yet."""
    missing = [str(track_path(t, s)) for t in TRACKS for s in ("train", "dev")
               if not track_path(t, s).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing data files (run the data cell in multilingual_colab.ipynb, or "
            "copy final_data/ from the organizers' repo):\n  " + "\n  ".join(missing))


def load(track, split):
    """Load one track/split from the reference file (same labels as
    participant_labeled, plus reverse_pair_id for the scorer)."""
    df = pd.read_csv(track_path(track, split), keep_default_na=False)
    df["direction"] = df.instance_id.str.rsplit("__", n=1).str[-1]
    df["lang_pair"] = df.text1_lang + "-" + df.text2_lang
    df["track"] = track
    return df


def partner_id(instance_id):
    """The instance_id of the reversed orientation (it may not exist: Neg rows
    only have an 'original' row)."""
    pid, langs, direction = instance_id.split("__")
    a, b = langs.split("-")
    flip = "original" if direction == "flipped" else "flipped"
    return f"{pid}__{b}-{a}__{flip}"


def partner_index(df):
    """Row position of each row's reversed partner, or -1 if it has none."""
    pos = {iid: i for i, iid in enumerate(df.instance_id)}
    return np.array([pos.get(partner_id(i), -1) for i in df.instance_id])


def make_units(df):
    """Batching units: [i, j] for a reversible pair, [i] for a single.
    Same format src.train_a2.PairCollator expects, so the consistency loss
    keeps working on Tracks 2-4."""
    p = partner_index(df)
    return [[i, int(j)] if j >= 0 else [i]
            for i, j in enumerate(p) if j < 0 or i < j]


def val_pair_ids(val_frac=0.1, seed=2027):
    """pair_ids held out for validation. Computed on Track 1 with the exact A2
    function and seed, so every language uses the same held-out source pairs
    as the A2 English runs (no source pair leaks between train and val
    through a translation)."""
    _, va = group_split(load_track1("train"), val_frac, seed)
    return set(va.pair_id)
