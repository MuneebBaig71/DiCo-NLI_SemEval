"""Hand-crafted, interpretable features for each (text1, text2) pair.

These are used twice: in the EDA (to understand the data) and as the input
of the feature-based Logistic Regression baseline (Baseline A).

Key design point: FORWARD vs BACKWARD entailment only differ by direction,
so symmetric features (Jaccard, overlap) cannot tell them apart. We therefore
also include *signed / directional* features (text1 minus text2, coverage of
text1 by text2 and vice versa).
"""
import re

import numpy as np
import pandas as pd

NEGATIONS = {"not", "no", "never", "n't", "without", "none", "nobody", "nothing"}
_TOKEN = re.compile(r"\w+|'\w+|[^\w\s]")


def tokens(text):
    return [t.lower() for t in _TOKEN.findall(str(text))]


def _pos_tagger():
    try:
        import spacy
        return spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
    except Exception:
        return None


def pair_features(df, use_pos=True):
    """Return a DataFrame of numeric features, one row per instance in df."""
    rows = []
    for t1, t2 in zip(df["text1"], df["text2"]):
        a, b = tokens(t1), tokens(t2)
        sa, sb = set(a), set(b)
        inter = sa & sb
        union = sa | sb
        rows.append({
            # length / structure (signed difference is the directional cue)
            "len1": len(a),
            "len2": len(b),
            "len_diff": len(a) - len(b),
            "char_diff": len(str(t1)) - len(str(t2)),
            # symmetric lexical overlap
            "jaccard": len(inter) / len(union) if union else 0.0,
            "identical": float(str(t1).lower().strip() == str(t2).lower().strip()),
            # directional coverage: how much of one side appears in the other
            "cov_1_in_2": len(sa & sb) / len(sa) if sa else 0.0,
            "cov_2_in_1": len(sa & sb) / len(sb) if sb else 0.0,
            "contains_1_in_2": float(bool(sa) and sa <= sb),
            "contains_2_in_1": float(bool(sb) and sb <= sa),
            "extra1": len(sa - sb),  # words only in text1 (added specificity?)
            "extra2": len(sb - sa),
            # task-specific cues
            "neg_mismatch": float(bool(sa & NEGATIONS) != bool(sb & NEGATIONS)),
            "num_mismatch": float({t for t in sa if t.isdigit()} != {t for t in sb if t.isdigit()}),
        })
    feats = pd.DataFrame(rows, index=df.index)

    nlp = _pos_tagger() if use_pos else None
    if nlp is not None:
        # POS of the words that are NOT shared: which kind of word makes the
        # phrases differ (e.g. an extra adjective often makes a phrase more specific).
        texts = pd.unique(pd.concat([df["text1"], df["text2"]]).astype(str))
        pos_of = {}
        for text, doc in zip(texts, nlp.pipe(texts, batch_size=256)):
            pos_of[text] = [(tok.text.lower(), tok.pos_) for tok in doc]
        for side, (own, other) in {"1": ("text1", "text2"), "2": ("text2", "text1")}.items():
            for tag in ["NOUN", "PROPN", "ADJ", "VERB", "ADP", "NUM"]:
                vals = []
                for t_own, t_other in zip(df[own].astype(str), df[other].astype(str)):
                    other_words = {w for w, _ in pos_of[t_other]}
                    vals.append(sum(1 for w, p in pos_of[t_own] if p == tag and w not in other_words))
                feats[f"extra{side}_{tag}"] = vals
    return feats


def pos_proportions(df):
    """Corpus-level POS proportions per label (for the EDA)."""
    nlp = _pos_tagger()
    if nlp is None:
        return None
    out = []
    for label, sub in df.groupby("label"):
        texts = list(sub["text1"].astype(str)) + list(sub["text2"].astype(str))
        tags = [tok.pos_ for doc in nlp.pipe(texts, batch_size=256) for tok in doc]
        s = pd.Series(tags).value_counts(normalize=True)
        s.name = label
        out.append(s)
    return pd.concat(out, axis=1).fillna(0.0)


def mtld(toks, threshold=0.72):
    """Measure of Textual Lexical Diversity (length-robust, unlike raw TTR)."""
    def one_pass(seq):
        factors, types, count = 0.0, set(), 0
        for t in seq:
            count += 1
            types.add(t)
            if len(types) / count <= threshold:
                factors += 1
                types, count = set(), 0
        if count:
            factors += (1 - len(types) / count) / (1 - threshold)
        return len(seq) / factors if factors else np.nan
    return (one_pass(toks) + one_pass(toks[::-1])) / 2
