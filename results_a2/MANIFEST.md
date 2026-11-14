# Assignment 2 experiment manifest

Every number in the A2 report can be traced to a file listed here. All scores were
recomputed from the saved dev probabilities with the official scorer
(`verified_results.csv`); they match the numbers produced on Colab to within 1e-16.

## Shared setup (all runs)

| Setting | Value |
|---|---|
| Training data | Track 1 train, minus a 10% validation split |
| Validation split | grouped by `pair_id`, paired and single groups sampled separately, `split_seed=2027`, identical for every run |
| Optimiser | AdamW, lr 2e-5, weight decay 0.01, linear warm-up 10%, gradient clipping 1.0 |
| Batching | 8 pairs per batch (both orientations together), about 16 rows |
| Max length / precision | 64 tokens, fp32 |
| Epochs / selection | 6 epochs; keep the epoch with the best **validation** argmax weighted-F1 (dev never used for selection) |
| Seeds | 13, 42, 87 |
| Hardware | Google Colab T4 GPU (about 2 min per RoBERTa run, 3.3 min per DeBERTa run) |
| Code | `src/train_a2.py` (training), `src/decoding.py` (decoding rules), `06_decoding_experiments.py` (scoring), `a2_colab.ipynb` (driver) |

## Systems

| ID | Backbone | Consistency λ | Neg weight | Seeds | Best epoch (13/42/87) | Best val W-F1 (13/42/87) |
|---|---|---|---|---|---|---|
| M0 | roberta-base | 0 | 1.0 | 13, 42, 87 | 6 / 5 / 6 | 0.856 / 0.835 / 0.816 |
| M1 | microsoft/deberta-v3-base | 0 | 1.0 | 13, 42, 87 | 3 / 2 / 6 | 0.849 / 0.868 / 0.864 |
| M2 | microsoft/deberta-v3-base | 1.0 | 1.0 | 13, 42, 87 | 6 / 4 / 6 | 0.871 / 0.867 / 0.856 |
| M2b | microsoft/deberta-v3-base | 0.5 | 1.0 | 13 only | 4 | 0.868 |
| M3 | microsoft/deberta-v3-base | 1.0 | 2.0 | 13, 42, 87 | 6 / 5 / 6 | 0.856 / 0.874 / 0.862 |

M2b is a one-seed sensitivity check on λ, not a full system.

## Decoding rules (inference only, `src/decoding.py`)

| Mode | Rule |
|---|---|
| argmax | highest-probability label per row |
| symmetric | average each row's probabilities with its `pair_id` partner's label-swapped probabilities |
| struct_pair | symmetric, and rows with a partner cannot be Neg |
| struct_full | struct_pair, and rows without a partner are set to Neg |

Partners come only from `pair_id` in the participant file, never from gold labels.
`struct_pair`/`struct_full` rely on a property observed in the released train and dev
data (paired rows are never Neg, single rows are always Neg); it is not guaranteed for test.

## Files

| What | Path |
|---|---|
| Per-run config, best epoch, training history | `runs/<tag>.json` |
| Dev probabilities (per run) | `predictions/<tag>_dev_probs.csv` |
| Validation probabilities (per run) | `predictions/<tag>_val_probs.csv` |
| Dev predictions, argmax (official format) | `predictions/<tag>_dev.csv` |
| Dev predictions under each decoding rule | `decoding/<tag>__<mode>.csv` |
| All runs x all decoding rules, official scorer | `verified_results.csv` |

`<tag>` is e.g. `M3_deberta_cons_neg_seed42`. `decoding/` also holds A1 reanalysis files
(`bert-base-uncased_seed*`, `roberta-base_seed*`, from `results_v2` probabilities).

## Aggregation

- **Per-system results:** mean ± sample std over the three seeds, each seed scored separately.
- **Ensembles (`<system>_ens3` in `verified_results.csv`):** average the three seeds'
  dev probabilities row by row, then decode once. This is a single system, not a mean.

## Codabench development-phase submission (8 Nov 2026)

| Field | Value |
|---|---|
| User / submission ID | mbaig / 1042178 |
| System | M3 3-seed probability ensemble + struct_full decoding |
| Tracks | Track 1 (EN) only |
| Codabench score | W-F1 0.935965, SoftCons 1.0, HardCons 0.924188 |
| Local official-scorer score | W-F1 0.9360, SoftCons 1.0, HardCons 0.9242 (matches) |
| Rebuild | `a2_colab.ipynb` cell 9 (verified: reproduces the Codabench scores exactly) |

## Verification notes

- M1 seeds 13 and 42 give identical struct_full W-F1. This is not a bug: their
  probabilities differ (max abs difference 0.92) and 24 dev rows get different labels,
  but each seed gets exactly 12 of those 24 right, in mirrored Fwd/Bwd vs Eq swaps, so
  the confusion counts coincide. Symmetric decoding makes errors come in mirrored pairs,
  which makes such ties more likely.

## Checkpoint save/load check (12 Nov 2026, `a2_colab.ipynb` cell 10)

M3 was retrained at seed 42 with `save_checkpoint=True`, the checkpoint was reloaded from disk with
`predict_from_checkpoint`, and dev was predicted again. The reloaded predictions
(`checkpoint_check/M3_ckpt_check_seed42_reloaded_dev.csv`) give W-F1 0.8292, SoftCons 0.8736,
HardCons 0.8195, and all 660 labels are identical to the original M3 seed-42 run
(`predictions/M3_deberta_cons_neg_seed42_dev.csv`). The checkpoint itself (about 0.74 GB) is kept
on Google Drive (`MyDrive/DiCo_A2_ckpt_check`), not in the repository.

## Neg-both-ways counts (report Section 9, consistency loss)

Pairs (of 277) whose two orientations are both predicted Neg under argmax decoding, from
`decoding/<tag>__argmax.csv`, seeds 13/42/87: M1 11, 10, 23 (mean 14.7); M2 16, 28, 21 (mean 21.7);
M3 20, 25, 24 (mean 23.0). Strictly one-sided Neg pairs: M1 mean 9.0, M2 6.7, M3 6.3.
