# DiCo-NLI (SemEval-2027 Task 2): Assignment 1

CS-878 Deep Learning, Assignment 1: task formulation, selective EDA and baselines for DiCo-NLI, Track 1 (English).

Group 1: Muhammad Muneeb Baig, Syed Kamran Ali Shah, Kashaf Ali Haroon.

All scores are on the official dev split, computed with the organizers' official scorer.

## Report

- `report/G1_DL_report.pdf` is the submitted report.
- `report/main.tex` and `report/references.bib` are the LaTeX source (Overleaf, BibTeX). The figures are in `report/figures/`.
- `results_v2/eda_plots.pdf` is the EDA plots PDF submitted with the report.

The numbers in the report come from `results_v2/`.

## How to run

The environment is managed with [uv](https://docs.astral.sh/uv/) (Python 3.12). A `requirements.txt` is also provided.

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

The full pipeline is in **`assignment1_dico_nli_v2.ipynb`**: data checks, EDA, classical baselines, BERT/RoBERTa fine-tuning with 3 seeds (13, 42, 87), the results table and error analysis. Outputs are written to `results_v2/`.

- `RUN_TRANSFORMERS = False` in the first cell gives a quick run (about 1 minute) without fine-tuning.
- `assignment1_dico_nli_v2_colab.ipynb` is the same notebook set up for Colab (CUDA):

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MuneebBaig71/DiCo-NLI_SemEval/blob/main/assignment1_dico_nli_v2_colab.ipynb)

The individual steps are also available as scripts, run from the repo root:

| Script | What it does |
|---|---|
| `02_eda.py` | Dataset checks and EDA, writes `eda_plots.pdf` and `eda_stats.txt` |
| `01_classical_baselines.py` | Majority, length-only, overlap-only, engineered-feature, TF-IDF and TF-IDF + feature LR baselines |
| `04_error_analysis.py` | Confusion matrices, error categories and error samples for prediction files |
| `05_results_table.py` | Combines all scores into a CSV and a LaTeX table |

The fine-tuning step runs inside the notebook. `notebooks/03_finetune_transformers_colab.ipynb` is a standalone Colab version of it.

## Layout

```
assignment1_dico_nli_v2.ipynb         Main notebook (3 seeds), results in results_v2/
assignment1_dico_nli_v2_colab.ipynb   Same notebook, set up for Colab
assignment1_dico_nli.ipynb            First version (1 seed), results in results/
01_ ... 05_*.py                       Individual pipeline steps as scripts
src/data.py                           Data loading, prediction writing, official scoring
src/features.py                       Hand-crafted pair features (EDA and feature baseline)
data/                                 Track 1 train/dev CSVs (unchanged from the organizers' repo)
evaluation_functions/                 Official scorer (unchanged from the organizers' repo)
literature/                           Task proposal, pilot paper, BibTeX
results_v2/                           Final results used in the report
results/                              First run (1 seed), kept for reference
report/                               Submitted report, LaTeX source and figures
```

`results/` holds the first single-seed run. It includes a DeBERTa-v3 run whose loss went to NaN on Apple MPS, so the model collapsed (W-F1 0.110). That is why DeBERTa-v3 is not among the reported models.

## Design choices

- **No re-splitting.** The organizers' split is grouped by `pair_id`, so a pair and its flipped copy never end up on different sides of train/dev.
- **Directional features.** FORWARD and BACKWARD differ only in direction, so symmetric features such as Jaccard cannot separate them. We add signed length difference and one-sided containment/coverage.
- **Shortcut probes.** Length-only and overlap-only LRs measure how far shallow cues go.
- **No tuning on dev.** The transformers use fixed hyperparameters (`results_v2/transformer_config.json`) and a fixed number of epochs.
