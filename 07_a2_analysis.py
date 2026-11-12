"""A2 analysis on saved dev predictions: bootstrap CIs, subgroups, confusion matrix.

Usage: python 07_a2_analysis.py   (writes results_a2/analysis/)
"""
import json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
from src.data import split_path, load_split, LABELS, score, headline
from src.decoding import partner_index, decode

A = Path('results_a2/analysis'); A.mkdir(parents=True, exist_ok=True)
part = pd.read_csv(split_path('dev', 'participant_labeled'), keep_default_na=False)
ids = pd.read_csv(split_path('dev', 'submission_template'), keep_default_na=False).instance_id.tolist()
ref = pd.read_csv(split_path('dev', 'reference'), keep_default_na=False).set_index('instance_id').loc[ids]
meta = part.set_index('instance_id').loc[ids]
gold = ref.label.to_numpy(); pi = partner_index(ids, part)
SEEDS = (13, 42, 87)
SYS = {'M0': 'M0_roberta', 'M1': 'M1_deberta', 'M2': 'M2_deberta_cons', 'M3': 'M3_deberta_cons_neg'}
def probs(tag): return pd.read_csv(f'results_a2/predictions/{tag}_dev_probs.csv').set_index('instance_id').loc[ids, LABELS].to_numpy()
P = {(k, s): probs(f'{v}_seed{s}') for k, v in SYS.items() for s in SEEDS}
def seeds(k, m): return [np.array(decode(P[(k, s)], pi, m)) for s in SEEDS]
def ens(k, m): return np.array(decode(np.mean([P[(k, s)] for s in SEEDS], 0), pi, m))

pa = np.array([i for i, j in enumerate(pi) if j > i]); pb = pi[pa]
def wf1(y, p): return f1_score(y, p, labels=LABELS, average='weighted', zero_division=0)

# ---- sanity: local metrics equal official scorer
x = seeds('M1', 'argmax')[1]
o = A / '_check.csv'; pd.DataFrame({'instance_id': ids, 'label': x}).to_csv(o, index=False)
h = headline(score(o, 'dev')); o.unlink()
assert abs(wf1(gold, x) - h['weighted_f1']) < 1e-9
assert abs(np.mean((gold[pa] == x[pa]) & (gold[pb] == x[pb])) - h['hard_cons']) < 1e-9
print('local metrics match official scorer')

# ---- cluster bootstrap over pair_id groups (pairs resampled whole)
groups = meta.pair_id.to_numpy(); ug = np.unique(groups)
rows_of = {g: np.where(groups == g)[0] for g in ug}
pair_of_group = {groups[i]: n for n, i in enumerate(pa)}
rng = np.random.default_rng(2027); B = 2000
draws = []
for _ in range(B):
    gs = rng.choice(ug, len(ug), replace=True)
    idx = np.concatenate([rows_of[g] for g in gs])
    w = np.zeros(len(pa))
    for g in gs:
        if g in pair_of_group: w[pair_of_group[g]] += 1
    draws.append((idx, w))

def metric_vals(PL, idx=None, w=None):
    """Mean over seed predictions of (W-F1, HardCons) on a resample."""
    out = []
    for p in PL:
        f = wf1(gold, p) if idx is None else wf1(gold[idx], p[idx])
        ok = ((gold[pa] == p[pa]) & (gold[pb] == p[pb])).astype(float)
        hc = ok.mean() if w is None else (w * ok).sum() / w.sum()
        out.append((f, hc))
    return np.mean(out, 0)

def compare(name, PA, PB):
    d0 = metric_vals(PA) - metric_vals(PB)
    ds = np.array([metric_vals(PA, i, w) - metric_vals(PB, i, w) for i, w in draws])
    lo, hi = np.percentile(ds, [2.5, 97.5], axis=0)
    p = [(ds[:, k] <= 0).mean() if d0[k] > 0 else (ds[:, k] >= 0).mean() for k in range(2)]
    r = dict(comparison=name, dWF1=d0[0], WF1_lo=lo[0], WF1_hi=hi[0], p_WF1=p[0],
             dHard=d0[1], Hard_lo=lo[1], Hard_hi=hi[1], p_Hard=p[1])
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    return r

C = [
    ('M1 vs M0 (argmax)', seeds('M1', 'argmax'), seeds('M0', 'argmax')),
    ('M1 symmetric vs argmax', seeds('M1', 'symmetric'), seeds('M1', 'argmax')),
    ('M1 struct_full vs argmax', seeds('M1', 'struct_full'), seeds('M1', 'argmax')),
    ('M2 vs M1 (argmax)', seeds('M2', 'argmax'), seeds('M1', 'argmax')),
    ('M2 vs M1 (struct_full)', seeds('M2', 'struct_full'), seeds('M1', 'struct_full')),
    ('M3 vs M2 (argmax)', seeds('M3', 'argmax'), seeds('M2', 'argmax')),
    ('M3 vs M2 (struct_full)', seeds('M3', 'struct_full'), seeds('M2', 'struct_full')),
    ('M3 vs M1 (struct_full)', seeds('M3', 'struct_full'), seeds('M1', 'struct_full')),
    ('M3 ens vs M3 seeds (struct_full)', [ens('M3', 'struct_full')], seeds('M3', 'struct_full')),
]
pd.DataFrame([compare(*c) for c in C]).to_csv(A / 'bootstrap_comparisons.csv', index=False)

# ---- single-system 95% CIs (M3 ensemble, struct_full; M1 seeds argmax)
def ci(PL):
    v = np.array([metric_vals(PL, i, w) for i, w in draws])
    return metric_vals(PL), np.percentile(v, [2.5, 97.5], axis=0)
cis = {}
for name, PL in [('M0 argmax', seeds('M0', 'argmax')), ('M1 argmax', seeds('M1', 'argmax')),
                 ('M3 argmax', seeds('M3', 'argmax')), ('M3 struct_full', seeds('M3', 'struct_full')),
                 ('M3 ens struct_full', [ens('M3', 'struct_full')])]:
    m, (lo, hi) = ci(PL); cis[name] = dict(WF1=m[0], WF1_lo=lo[0], WF1_hi=hi[0], Hard=m[1], Hard_lo=lo[1], Hard_hi=hi[1])
pd.DataFrame(cis).T.to_csv(A / 'bootstrap_cis.csv'); print(pd.DataFrame(cis).T.round(3))

# ---- subgroups (mean over seeds)
w1 = ref.text1.str.split().str.len(); w2 = ref.text2.str.split().str.len()
tot = (w1 + w2).to_numpy()
def toks(s): return set(s.lower().split())
jac = np.array([len(toks(a) & toks(b)) / max(1, len(toks(a) | toks(b))) for a, b in zip(ref.text1, ref.text2)])
paired = pi >= 0
qL = np.quantile(tot, [1/3, 2/3]); qJ = np.quantile(jac, [1/3, 2/3])
grp = {
    'paired rows': paired, 'single rows': ~paired,
    f'short (<= {int(qL[0])} words)': tot <= qL[0],
    f'medium ({int(qL[0])+1}-{int(qL[1])})': (tot > qL[0]) & (tot <= qL[1]),
    f'long (> {int(qL[1])} words)': tot > qL[1],
    f'low overlap (Jaccard <= {qJ[0]:.2f})': jac <= qJ[0],
    f'mid overlap': (jac > qJ[0]) & (jac <= qJ[1]),
    f'high overlap (> {qJ[1]:.2f})': jac > qJ[1],
}
sub = []
for gname, mask in grp.items():
    r = dict(group=gname, n=int(mask.sum()))
    for k, m in [('M0', 'argmax'), ('M1', 'argmax'), ('M3', 'argmax'), ('M3', 'struct_full')]:
        r[f'{k} {m} acc'] = np.mean([np.mean(p[mask] == gold[mask]) for p in seeds(k, m)])
    sub.append(r)
sub = pd.DataFrame(sub); sub.to_csv(A / 'subgroups.csv', index=False); print(sub.round(3).to_string(index=False))
print('pair words: mean', tot.mean().round(1), 'max', tot.max(), '| phrase words max', max(w1.max(), w2.max()))

# ---- per-class precision/recall (mean over seeds) and confusion matrix
pr = []
for k, m in [('M0', 'argmax'), ('M1', 'argmax'), ('M3', 'argmax'), ('M3', 'struct_full')]:
    v = np.mean([precision_recall_fscore_support(gold, p, labels=LABELS, zero_division=0)[:3] for p in seeds(k, m)], 0)
    for li, lab in enumerate(LABELS):
        pr.append(dict(system=k, mode=m, label=lab, precision=v[0][li], recall=v[1][li], f1=v[2][li]))
pd.DataFrame(pr).to_csv(A / 'per_class_prf.csv', index=False)
print(pd.DataFrame(pr).pivot_table(index=['system', 'mode'], columns='label', values=['precision', 'recall']).round(3).to_string())
for m in ['argmax', 'struct_full']:
    cm = confusion_matrix(gold, ens('M3', m), labels=LABELS)
    pd.DataFrame(cm, index=[f'gold {l}' for l in LABELS], columns=LABELS).to_csv(A / f'confusion_M3ens_{m}.csv')
    print(m); print(pd.DataFrame(cm, index=LABELS, columns=LABELS))

# ---- train/dev duplicates
tr = load_split('train'); dv = load_split('dev')
key = lambda d: d.text1.str.strip().str.lower() + ' ||| ' + d.text2.str.strip().str.lower()
dup = dv[key(dv).isin(set(key(tr)))]
print('train/dev duplicate dev rows:', len(dup))
e = dict(zip(ids, ens('M3', 'struct_full')))
print('M3 ens struct_full correct on duplicates:', sum(e[i] == l for i, l in zip(dup.instance_id, dup.label)), '/', len(dup))
print('train label counts', tr.label.value_counts().to_dict()); print('dev label counts', dv.label.value_counts().to_dict())
