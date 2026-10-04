"""Shared data loading and official scoring for DiCo-NLI Track 1 (English).

Everyone imports from here so we all read the same files and score the same way.
"""
import os
import sys
from pathlib import Path

import pandas as pd

# Project root. data/ holds the Track 1 CSVs, evaluation_functions/ is the
# organizers' official scorer (copied unchanged from their GitHub repo).
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DICO_DATA", ROOT / "data"))

LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
REVERSE = {
    "EQUIVALENCE": "EQUIVALENCE",
    "FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT",
    "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT",
}


def split_path(split, kind="participant_labeled"):
    """split: 'train' or 'dev'. kind: participant_labeled | reference | submission_template."""
    return DATA_DIR / split / f"dico_nli_{split}_track1_{kind}.csv"


def load_split(split):
    """Load Track 1 for a split. Uses the reference file because it also has
    reverse_pair_id, which we need for consistency analysis (labels are identical)."""
    df = pd.read_csv(split_path(split, "reference"), keep_default_na=False)
    df["direction"] = df["instance_id"].str.rsplit("__", n=1).str[-1]  # original / flipped
    return df


def write_predictions(ids, preds, path):
    """Write a prediction file in the official two-column format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"instance_id": list(ids), "label": list(preds)}).to_csv(path, index=False)
    return path


def score(pred_path, split="dev", out_dir=None):
    """Run the organizers' official scorer and return its full report as a dict."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from evaluation_functions import score_files

    report = score_files(split_path(split, "reference"), pred_path, output_dir=out_dir)
    return report.to_dict()


def headline(report):
    """The three official numbers, plus accuracy and macro-F1 as diagnostics."""
    c = report["classification"]
    return {
        "weighted_f1": report["weighted_f1"],
        "soft_cons": report["soft_cons"],
        "hard_cons": report["hard_cons"],
        "macro_f1": c["macro_f1"],
        "accuracy": c["accuracy"],
    }
