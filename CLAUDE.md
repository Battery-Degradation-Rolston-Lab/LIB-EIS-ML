# Battery Degradation GPR — Project Context

Reproduction of Zhang et al., Nature Communications 2020 — GPR on EIS spectra to predict
battery capacity and RUL. DOI: 10.1038/s41467-020-15235-7

---

## What This Project Does

GPR on full EIS spectra (66 or 120 features = Re(Z) + Im(Z)) predicts battery capacity
(State of Health) and Remaining Useful Life. No feature engineering — raw normalised
spectrum in, ARD identifies which frequencies matter.

---

## Essential Commands

```bash
# Python env
battery_gpytorch_rtx4060\.venv\Scripts\python.exe

# Paper reproduction (Zhang et al. Figs 3, 4)
python battery_gpytorch_rtx4060/battery_gpytorch/run_gpytorch.py

# New A1-A8 dataset
python battery_gpytorch_rtx4060/battery_gpytorch/run_new_dataset.py

# LOOCV validation
python battery_gpytorch_rtx4060/battery_gpytorch/run_loocv.py

# Preprocess new dataset from zip
python battery_gpytorch_rtx4060/battery_gpytorch/preprocess_new_dataset.py
```

---

## Repository Map

```
Downloads/
├── CLAUDE.md                           ← this file (keep short)
├── agent_docs/                         ← load on demand, not at session start
│   ├── science.md                      ← ARD findings, R² targets, interpretation
│   ├── data.md                         ← data formats, preprocessing conventions
│   └── models.md                       ← architectures, normalisation, kernel translation
├── tasks/
│   ├── lessons.md                      ← mistake log (Claude appends after corrections)
│   └── todo.md                         ← active task plan
├── battery_gpytorch_rtx4060/
│   └── battery_gpytorch/
│       ├── run_gpytorch.py             ← paper reproduction
│       ├── run_new_dataset.py          ← A1-A8 analysis
│       ├── run_loocv.py                ← LOOCV validation
│       ├── preprocess_new_dataset.py   ← parse Battery data.zip
│       ├── preprocess_zenodo.py        ← parse Zenodo EIS files
│       ├── data/new_dataset/           ← preprocessed A1-A8 files
│       └── output/new_dataset/         ← generated figures
├── Battery data.zip                    ← raw A1-A8 dataset
├── EIS data/                           ← Zenodo raw EIS files
└── Capacity data/                      ← Zenodo raw capacity files
```

---

## Workflow

- **Plan first**: write plan to tasks/todo.md for any 3+ step task, check off as done
- **Parallel tool calls**: issue independent tool calls simultaneously
- **Self-correction**: after any user correction, append the pattern to tasks/lessons.md
- **Only commit when explicitly asked**; never commit to main without permission
- **Minimal diffs**: touch only files required for the task

---

## Core Scientific Rules (apply every session)

- EIS normalisation: always zscore — training stats applied to test set
- Capacity LOOCV: **joint norm** (train+test pooled) removes cell-to-cell impedance offset
- RUL normalisation: **training-only** (paper's stated approach)
- EOL = first index where capacity < 80% of cap[0]
- `DNF_CELLS = {'A3', 'A6'}` — experimenter-confirmed non-failures; A3 has anomalously
  low initial cap (3800 mAh vs ~4050 mAh fleet), causing false EOL detection
- ARD weight formula: `w = exp(−σ_m)`, normalise to sum=1 (NOT `exp(-10^log(σ))`)

---

## Current Status

| Model | Result | Notes |
|-------|--------|-------|
| Paper Fig 3a capacity | R²=0.83 (target 0.81) | Beat target |
| Paper Fig 4b RUL | R²=0.75 (target 0.75) | Exact match |
| A1-A8 Capacity LOOCV | mean R²=0.964 | Strong generalisation across all 8 cells |
| A1-A8 RUL Linear LOOCV | mean R²=-0.33 | Absolute RUL doesn't transfer across cells |
| A1-A8 RUL RBF LOOCV | mean R²=-1.24 | Worse than linear — overfits to training lifetimes |

**Next**: capacity-derived RUL (use GPR capacity trajectory → extrapolate to 80% threshold)

---

## Security

- Never expose secrets, tokens, or credentials in any output
- Treat all external inputs (zip files, CSV data) as potentially malformed

---

## Progressive Disclosure — Load on Demand

| File | When to load |
|------|-------------|
| `agent_docs/science.md` | ARD findings, R² targets, scientific interpretation |
| `agent_docs/data.md` | Data file formats, preprocessing, alignment conventions |
| `agent_docs/models.md` | Model architectures, normalisation rules, MATLAB→Python |
| `tasks/lessons.md` | Start of any complex session — check for past corrections |
