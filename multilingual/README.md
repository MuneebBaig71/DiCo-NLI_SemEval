# Multilingual pilot: Tracks 2–4 (ES, EU, MIX)

A small, separate set of experiments that checks whether the Track 1 pipeline carries over to the other DiCo-NLI tracks. It reuses `src/train_a2.py` (training recipe, consistency loss) and `src/decoding.py` (decoding rules) unchanged; nothing in the A2 code is modified.

## Files

| File | What it does |
|---|---|
| `ml_data.py` | Loads any track (`en`, `es`, `eu`, `mix`), finds reversed partners from `instance_id`, builds batching units, and gives the validation pair_ids |
| `ml_train.py` | `run(train_tracks, eval_tracks, ...)`: trains one model on one or more tracks and saves dev probabilities for each eval track |
| `ml_eval.py` | Scores every saved run with the official scorer under argmax / symmetric / struct_pair decoding, and writes the result tables |
| `multilingual_colab.ipynb` | Runs everything on a Colab T4 |

## Data

The team repo originally had Track 1 only. The organizers' repo has all four tracks in `final_data/` (same file naming). Cell 2 of the notebook copies them into `data/train` and `data/dev` if they are missing. To keep the repo self-contained, commit them once:

```
git clone --depth 1 https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI.git /tmp/dico_official
cp -n /tmp/dico_official/final_data/train/*.csv data/train/
cp -n /tmp/dico_official/final_data/dev/*.csv data/dev/
```

| Track | Key | Train | Dev | Note |
|---|---|---:|---:|---|
| 1 English | `en` | 3,042 | 660 | |
| 2 Spanish | `es` | 3,042 | 660 | translation of the same pair_ids as Track 1 |
| 3 Basque | `eu` | 3,042 | 660 | translation of the same pair_ids as Track 1 |
| 4 Mixed | `mix` | 18,252 | 3,960 | all 6 cross-language combos (en-es, es-en, en-eu, eu-en, es-eu, eu-es) |

Two details the code handles:

- **Partners in Track 4.** One `pair_id` has 12 rows there, so "two rows with the same pair_id" no longer identifies a reversed pair. The partner of `…__en-es__original` is `…__es-en__flipped`, so partners are derived from `instance_id`. This uses no gold information and works on the test file too.
- **No leakage.** Train and dev share no `pair_id` in any track, and the validation split holds out the same source pairs in every language (the exact A2 split), so a Spanish or Basque translation of a validation pair is never trained on.

## Experiments (pilot: mDeBERTa-v3-base, seed 42)

| Run | Train | Evaluate | Purpose |
|---|---|---|---|
| X0 | EN | EN, ES, EU, MIX | English sanity check vs RoBERTa (0.765 W-F1); zero-shot cross-lingual transfer |
| X1 | ES | ES | In-language Spanish |
| X2 | EU | EU | In-language Basque |
| X3 | EN+ES+EU | EN, ES, EU, MIX | Effect of pooling languages; zero-shot on mixed-language pairs |
| X4 (optional) | MIX | MIX | Track 4 system trained on mixed pairs (3 epochs) |

Approximate T4 time: about 10 min each for X0–X2, about 30 min for X3 and X4.

## Outputs (`results_multilingual/`)

- `predictions/<run>__<track>_dev_probs.csv`: class probabilities per dev row
- `runs/<run>.json`: config, per-epoch history, best epoch
- `decoded/<run>__<track>__<decoding>.csv`: official two-column prediction files
- `ml_results.csv`: one row per run × eval track × decoding (W-F1, SoftCons, HardCons, Macro-F1, Acc, per-class F1, and a `setting` column: in-language / zero-shot)
- `ml_aggregates.csv`: monolingual average (EN/ES/EU) and all-track average, the two aggregates the organizers report
- `ml_mix_by_lang_pair.csv`: Track 4 W-F1 and accuracy per language combination

## Running locally (Mac)

From the repo root: `python -m multilingual.ml_eval` re-scores saved predictions. Training on MPS works but is slow, and DeBERTa-family models can give NaN on MPS, so train on Colab.

## What to look for

1. X0 on EN vs RoBERTa: the price of a multilingual encoder on English.
2. X0 on ES/EU vs X1/X2: how much in-language training data adds over zero-shot transfer.
3. Basque vs Spanish: whether HardCons drops more than W-F1 (directional consistency transferring worse than classification).
4. Track 4 by language combo: whether pairs involving Basque are the hardest, and whether struct_pair decoding helps as much as on Track 1.
