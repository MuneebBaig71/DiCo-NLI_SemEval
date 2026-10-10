# DiCo-NLI (SemEval-2027 Task 2) — Assignment 1: Problem formulation, EDA, baselines

Track 1 (English). Official train/dev split, scored with the organizers' official scorer.

## Run on Colab 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MuneebBaig71/DiCo-NLI_SemEval/blob/main/assignment1_dico_nli_v2_colab.ipynb)

Click the badge → **File → Save a copy in Drive** (so you edit your own copy) → Runtime → Change runtime type → **T4 GPU** → **Run all**.
The first cell clones this repo; results download as `results_colab.zip` at the end.

The Mac results used in the report are already in `results_v2/` (3 seeds) and `results/` (v1), and the report source is in `report/`.

## Main file

**`assignment1_dico_nli.ipynb`** contains the whole pipeline in order: data checks, selective EDA
(saved as one PDF), classical baselines, fine-tuned BERT/DeBERTa, results table (LaTeX), and error analysis.
It runs locally on Apple Silicon (MPS), on CPU, or on Colab (CUDA). All outputs go to `results/`.

```bash
uv sync
uv run python -m spacy download en_core_web_sm
# open assignment1_dico_nli.ipynb in VS Code / Jupyter, select the .venv kernel, Run All
```

Set `RUN_TRANSFORMERS = False` in the first cell for a fast run (about 1 minute) without fine-tuning.
Set `SEEDS = [13, 42, 87]` to report mean ± std over 3 seeds.

## Layout

```
assignment1_dico_nli.ipynb   Main notebook, v1 (1 seed)
assignment1_dico_nli_v2.ipynb   Main notebook, v2 (3 seeds), results in results_v2/
assignment1_dico_nli_v2_colab.ipynb   Same as v2, set up for Colab
src/data.py                  Shared loading, prediction writing, official scoring
src/features.py              Hand-crafted pair features (used by the EDA and the feature baseline)
data/                        Track 1 train/dev CSVs (unchanged from the organizers' repo)
evaluation_functions/        Official scorer (unchanged from the organizers' repo)
notebooks/03_finetune_transformers_colab.ipynb   Colab-only fallback for the fine-tuning step
results/                     All outputs: eda_plots.pdf, results_table.tex, predictions, scores, error analysis
```

## Design choices

- **No re-splitting.** The organizers' split is grouped by `pair_id`, so a pair and its flipped copy never
  cross train/dev.
- **Directional features.** FORWARD and BACKWARD differ only in direction, so symmetric features
  (Jaccard) cannot separate them. Signed length difference and one-sided containment/coverage are added.
- **Shortcut probes.** Length-only and overlap-only LRs measure how far shallow cues go.
- **No tuning on dev.** Transformers use fixed hyperparameters and a fixed epoch count.
