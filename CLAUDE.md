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
# Python env (activate first)
source battery_gpytorch_rtx4060/.venv/Scripts/activate

# Paper reproduction (Zhang et al. Figs 1–4)
python battery_gpytorch_rtx4060/battery_gpytorch/run_gpytorch.py

# New A1-A8 dataset (fixed train/test)
python battery_gpytorch_rtx4060/battery_gpytorch/run_new_dataset.py

# CA1-CA8 LOOCV (complete lifecycle)
python battery_gpytorch_rtx4060/battery_gpytorch/run_loocv.py

# Capacity-derived RUL (CA1-CA8)
python battery_gpytorch_rtx4060/battery_gpytorch/run_cap_rul.py

# Multi-temp CB dataset (capacity + RUL)
python battery_gpytorch_rtx4060/battery_gpytorch/run_multitemp_approaches.py
python battery_gpytorch_rtx4060/battery_gpytorch/run_multitemp_rul.py

# Single-T fixed DOE per temperature group (Zhang Fig 1/2 equivalent)
python battery_gpytorch_rtx4060/battery_gpytorch/run_ca_zhang.py

# Multi-temp replication on Zhang data
python battery_gpytorch_rtx4060/battery_gpytorch/run_multitemp_zhang.py

# Preprocess datasets
python battery_gpytorch_rtx4060/battery_gpytorch/preprocess_new_dataset.py
python battery_gpytorch_rtx4060/battery_gpytorch/preprocess_ca_dataset.py
python battery_gpytorch_rtx4060/battery_gpytorch/preprocess_multitemp_dataset.py
```

---

## Repository Map

```
LIB-EIS-ML/
├── CLAUDE.md                           ← this file (keep short)
├── README.md                           ← full experiment documentation
├── reproducibility.md                  ← data compatibility checklist per experiment
├── agent_docs/                         ← load on demand, not at session start
│   ├── science.md                      ← ARD findings, R² targets, interpretation
│   ├── data.md                         ← data formats, preprocessing conventions
│   └── models.md                       ← architectures, normalisation, kernel translation
├── tasks/
│   └── lessons.md                      ← mistake log (Claude appends after corrections)
├── docs/
│   └── paper_text.txt                  ← paper reference text
├── presentation/                       ← slides + generation scripts (gitignored)
├── reference/                          ← original MATLAB code, paper PDF, reference data
├── battery_gpytorch_rtx4060/
│   └── battery_gpytorch/
│       ├── run_gpytorch.py             ← paper reproduction (Cambridge)
│       ├── run_new_dataset.py          ← A1-A8 fixed train/test
│       ├── run_loocv.py                ← CA1-CA8 LOOCV (complete lifecycle)
│       ├── run_cap_rul.py              ← capacity-derived RUL (CA1-CA8)
│       ├── run_freq_subset_loocv.py    ← frequency band LOOCV (A1-A8)
│       ├── run_coupled_ard_loocv.py    ← coupled ARD (A1-A8)
│       ├── run_ca_zhang.py             ← single-T fixed DOE per temp group (Zhang Fig 1/2)
│       ├── run_multitemp_approaches.py ← multi-temp CB dataset approaches
│       ├── run_multitemp_rul.py        ← multi-temp CB dataset RUL
│       ├── run_multitemp_zhang.py      ← multi-temp replication on Zhang data
│       ├── preprocess_new_dataset.py   ← parse Battery data.zip → A1-A8
│       ├── preprocess_ca_dataset.py    ← parse Battery data/ .mpt → CA1-CA8
│       ├── preprocess_multitemp_dataset.py ← parse CB cells → multi-temp dataset
│       ├── preprocess_zenodo.py        ← parse Zenodo EIS files → Cambridge
│       ├── data/new_dataset/           ← preprocessed A1-A8 (66 features)
│       ├── data/ca_dataset/            ← preprocessed CA1-CA8 (66 features)
│       ├── data/multitemp_dataset/     ← preprocessed CB multi-temp data
│       ├── output/                     ← paper reproduction figures (fig1a–fig4c)
│       ├── output/new_dataset/         ← A1-A8 and CA1-CA8 figures
│       ├── output/multitemp/           ← CB multi-temp figures
│       └── output/multitemp_zhang/     ← Zhang multi-temp figures
├── Battery data/                       ← raw CA1-CA8 + CB complete lifecycle (.mpt)
├── raw_data/Battery data.zip           ← raw A1-A8 partial export
├── raw_data/Code-Matlab.zip            ← original MATLAB code archive
└── raw_data/zenodo_eis|capacity/       ← Zenodo raw Cambridge files
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
| Paper Fig 3a capacity (35°C) | R²=0.91 (target 0.81) | Beat target |
| Paper Fig 4b RUL (35°C) | R²=0.85 (target 0.75) | Beat target |
| A1-A8 Capacity LOOCV | mean R²=0.964 | Strong generalisation across all 8 cells |
| A1-A8 RUL Linear LOOCV | mean R²=−0.33 | Absolute RUL doesn't transfer — expected |
| CA1-CA8 Capacity LOOCV | run `run_loocv.py` | Complete lifecycle dataset |
| RT Cap-derived RUL (LOOCV) | mean R²=0.893 (7 EOL cells) | Best RT RUL approach — `run_cap_rul.py` |
| −10°C Cap-derived RUL (DOE) | R²=−0.82 (N10_CB4) | Fails — cap model too weak (R²=0.68) |
| −20°C Cap-derived RUL (DOE) | R²=0.970 (N20_CB4) | Excellent — pred EOL=16.1 vs actual 17 |
| −10°C Cap-derived RUL (LOOCV) | mean R²=−0.23 | High cell variability (EOL 71–114) |
| −20°C Cap-derived RUL (LOOCV) | mean R²=0.658 | CB3/CB4 excellent, CB1/CB2 weaker |
| CB Multi-temp Capacity (baseline) | N20_CB1-4 R²≈−8.7 | Train RT+-10°C only — -20°C out of distribution |
| CB Multi-temp Capacity (Zhang DOE) | N10_CB4 R²=0.375 / N20_CB4 R²=0.949 | All temps in training — Zhang-faithful |
| CB Multi-temp RUL (Zhang DOE) | N10_CB4 R²=0.226 / N20_CB4 R²=−120 | -20°C RUL range 0-17 vs RT 0-214 — scale mismatch |
| CB Coupled ARD (Zhang DOE) | top: 1.33 Hz (w=0.71) | 33 ls, Re+Im paired per frequency |
| RT single-T cap (Coupled ARD) | CA7=0.996, CA8=0.991 | Zhang Fig 1a equivalent — `run_ca_zhang.py` |
| −10°C single-T cap (Coupled ARD) | N10_CB4 R²=0.676 | Zhang Fig 1a equivalent — `run_ca_zhang.py` |
| −20°C single-T cap (Coupled ARD) | N20_CB4 R²=0.937 | Zhang Fig 1a equivalent — `run_ca_zhang.py` |
| RT single-T RUL | R²=−4.3 (fails) | Zhang Fig 2 equivalent — absolute RUL scale mismatch |
| −10°C single-T RUL | R²=0.734 | Zhang Fig 2 equivalent — works |
| −20°C single-T RUL | R²=0.459 | Zhang Fig 2 equivalent |
| Coupled ARD −10°C (single-T) | 13.3 Hz (w=1.0) | Single dominant frequency |

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
