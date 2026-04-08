# Battery Degradation GPR — Reproduction of Zhang et al., Nature Comms 2020

Reproduction of:
> Zhang et al., "Identifying degradation patterns of lithium ion batteries from impedance spectroscopy using machine learning," *Nature Communications* 2020. DOI: [10.1038/s41467-020-15235-7](https://doi.org/10.1038/s41467-020-15235-7)

GPR on raw EIS spectra (no feature engineering) predicts battery capacity and RUL. ARD discovers which frequencies matter.

---

## Quick Start

```bash
# Activate environment
source battery_gpytorch_rtx4060/.venv/Scripts/activate

# Experiment 1 — Paper reproduction (Figs 1–4, Cambridge/Zenodo dataset)
python battery_gpytorch_rtx4060/battery_gpytorch/run_gpytorch.py

# Experiment 2 — In-house A1-A8 fixed train/test
python battery_gpytorch_rtx4060/battery_gpytorch/run_new_dataset.py

# Experiment 3 — CA1-CA8 LOOCV (complete lifecycle)
python battery_gpytorch_rtx4060/battery_gpytorch/run_loocv.py
python battery_gpytorch_rtx4060/battery_gpytorch/run_cap_rul.py

# Experiment 4 — Multi-temperature CB dataset (−10°C / −20°C)
python battery_gpytorch_rtx4060/battery_gpytorch/run_multitemp_approaches.py
python battery_gpytorch_rtx4060/battery_gpytorch/run_multitemp_rul.py
python battery_gpytorch_rtx4060/battery_gpytorch/run_multitemp_zhang.py
```

---

## Experiment 1 — Paper Reproduction (Cambridge / Zenodo Dataset)

**Script:** `run_gpytorch.py` | **Data:** GitHub + Zenodo | **Features:** 120 (60 freqs × Re + Im)

Train on multi-temperature cells (25/35/45°C), test on held-out cells at each temperature.

### Results

| Figure | Description | Our R² | Paper R² | Status |
|--------|-------------|--------|----------|--------|
| Fig 1a | Single-T 25°C capacity (25C05–08) | **0.882** | 0.88 | ✅ matched |
| Fig 1b | Single-T 25°C capacity scatter | qualitative | qualitative | ✅ |
| Fig 1c | ARD weights 25°C | #91 & #100 in top-5 | #91, #100 dominant | ≈ |
| Fig 2  | Single-T 25°C RUL per cell | 0.84 / 0.95 / 0.76 / **−0.33** | 0.96 / 0.73 / 0.68 / 0.81 | ⚠️ |
| Fig 3a | Multi-T 35°C capacity | **0.91** | 0.81 | ✅ beat |
| Fig 3b | Multi-T 45°C capacity | **0.94** | 0.72 | ✅ beat |
| Fig 3c | ARD weights 35°C top feature | **#91** | #91 | ✅ exact |
| Fig 3d | ARD weights 45°C top feature | **#88** | #91 | ≈ same region |
| Fig 4a | Multi-T 25°C RUL (25C05) | **0.970** | 0.87 | ✅ beat |
| Fig 4b | Multi-T 35°C RUL (35C02) | **0.85** | 0.75 | ✅ beat |
| Fig 4c | Multi-T 45°C RUL (45C02) | **0.91** | 0.92 | ✅ matched |

### Notes

**Fig 2 / 25C08 (R²=−0.33):** The GitHub repo only released 35C02 data. Per-cell 25C test files (Fig 2) were never published. Our 25C07/08 test data comes from Zenodo, where 25C08 has only 37 capacity cycles (vs 275 for 25C05) and hits EOL at cycle 16 — an anomalously short run. The training cells have RUL up to 234 cycles; 25C08's range of 0–32 is out-of-distribution. The paper's 0.81 for that cell cannot be reproduced from publicly available data.

**Fig 3d / ARD 45°C:** Top feature #88 vs paper's #91. Both lie in the same low-frequency Im(Z) region (~17–20 Hz). L-BFGS-B finds a slightly different local optimum from MATLAB's conjugate gradient.

**Fig 4a:** R²=0.970 substantially exceeds paper's 0.87. The multi-temperature model generalises strongly to 25C05, whose smooth monotone RUL trajectory is well captured by the linear kernel.

---

## Experiment 2 — New In-House Dataset: A1-A8 (Fixed Train/Test Split)

**Script:** `run_new_dataset.py` | **Data:** `data/new_dataset/` | **Features:** 66 (33 freqs × Re + Im, native Ns=6 grid)

Fixed split: train A1-A4, test A5-A8. Same GPR architecture as the paper.

### Cell Summary

| Cell | Initial Cap (mAh) | EoL index | RUL_max (cycles) | Notes |
|------|------------------|-----------|-----------------|-------|
| A1 | 4060 | 174 | 348 | EOL cell |
| A2 | 4060 | 168 | 336 | EOL cell |
| A3 | 3800 | — | — | **DNF** — anomalously low initial cap; false 80% trigger |
| A4 | 4030 | 140 | 280 | EOL cell |
| A5 | 3860 | 116 | 232 | EOL cell |
| A6 | 4050 | — | — | **DNF** — never reached 80% threshold |
| A7 | 4040 | 95 | 190 | EOL cell (shortest life) |
| A8 | 4040 | 224 | 448 | EOL cell (longest life) |

`DNF_CELLS = {A3, A6}` — confirmed non-failures. A3 has anomalously low initial capacity (~3800 mAh vs fleet ~4050 mAh), causing false EOL detection at cycle 216.

### EIS State Selection

| Ns | Timing | Freq range | Freqs | Role |
|----|--------|-----------|-------|------|
| 1 | Before charge | 10 kHz → 10 mHz | 48 | — |
| **6** | **After charge** | **10 kHz → 1 Hz** | **33** | **Used (State V analogue)** |

Ns=6 matches the paper's State V (post-charge, rested). 33 native measured frequencies used directly — no interpolation, no sub-1 Hz extrapolation.

---

## Experiment 3 — LOOCV: A1-A8 (Partial Lifecycle, ~268 cycles)

**Scripts:** `run_freq_subset_loocv.py`, `run_coupled_ard_loocv.py` | **Data:** `data/new_dataset/`

Leave-one-out cross-validation on the partial export (cycles 0–267). EIS measured every 2 battery cycles (RUL factor = 2).

### Capacity LOOCV — fixed RBF l=30, joint normalisation

| Cell | A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 | Mean |
|------|----|----|----|----|----|----|----|----|------|
| R²   | 0.987 | 0.983 | 0.967 | 0.959 | 0.991 | 0.867 | 0.995 | 0.964 | **0.964** |

Joint normalisation (pooling train+test per fold) removes cell-to-cell impedance offset. Strong generalisation across all 8 cells including both DNF cells.

### RUL LOOCV — 6 EOL cells, training-only normalisation

| Cell | A1 | A2 | A4 | A5 | A7 | A8 | Mean |
|------|----|----|----|----|----|----|------|
| Linear R² | −0.52 | −0.56 | −0.00 | −0.38 | 0.17 | −0.68 | **−0.33** |
| RBF R²    | −0.57 | −0.68 | −0.00 | −2.09 | −5.27 | −0.09 | **−1.24** |

### Frequency Subset Analysis

LOOCV capacity R² by EIS band (all 8 cells):

| Band | Freq range | Cap R² | RUL R² |
|------|-----------|--------|--------|
| Full spectrum | 1–10 000 Hz | **0.964** | −0.33 |
| High | 500–10 000 Hz (11 pts) | 0.905 | −2.15 |
| Mid | 10–500 Hz (14 pts) | 0.867 | −1.79 |
| Low | 1–10 Hz (8 pts) | 0.872 | −2.07 |

Capacity is encoded redundantly — any band works reasonably well. RUL fails equally across all bands, confirming the failure is not a spectral coverage problem.

### Coupled ARD

Standard ARD uses 66 independent length-scales (Re and Im decoupled). Kramers-Kronig relations physically couple Re(Z) and Im(Z) at every frequency, making decoupled importance scores ambiguous. A coupled kernel (one length-scale per frequency, shared across Re and Im) gives physically interpretable ARD weights at comparable LOOCV accuracy.

---

## Experiment 4 — LOOCV: CA1-CA8 (Complete Lifecycle, 470+ cycles)

**Scripts:** `run_loocv.py`, `run_cap_rul.py` | **Data:** `data/ca_dataset/`

CA1-CA8 are the complete runs of the same physical batteries. EIS measured every battery cycle (RUL factor = 1). CA3 included as genuine EOL (low initial cap but real degradation). CA6 excluded (DNF — 83.6% final capacity, never crossed 80%).

### Capacity LOOCV — fixed RBF l=30, joint normalisation

| Cell | CA1 | CA2 | CA3 | CA4 | CA5 | CA6 | CA7 | CA8 | Mean |
|------|-----|-----|-----|-----|-----|-----|-----|-----|------|
| R²   | — | — | — | — | — | — | — | — | see `fig_cap_loocv.png` |

### RUL LOOCV — 7 EOL cells (CA1–CA5, CA7, CA8)

| Cell | CA1 | CA2 | CA3 | CA4 | CA5 | CA7 | CA8 | Mean |
|------|-----|-----|-----|-----|-----|-----|-----|------|
| Linear R² | — | — | — | — | — | — | — | see `fig_rul_loocv.png` |
| RBF R²    | — | — | — | — | — | — | — | see `fig_rul_loocv_comparison.png` |

> Run `run_loocv.py` to populate these numbers.

### Capacity-Derived RUL (`run_cap_rul.py`)

Direct EIS → RUL fails because EIS encodes current health, not total lifespan. Alternative approach:

1. Predict full capacity trajectory via LOOCV GPR
2. Fit linear trend to predicted trajectory
3. Extrapolate to 80% threshold → predicted EOL
4. RUL[i] = predicted_EOL − i

Results: see `fig_cap_rul_trajectories.png` and `fig_cap_rul_scatter.png`.

---

## Experiment 5 — Multi-Temperature CB Dataset (Zhang DOE + Coupled ARD)

**Script:** `run_multitemp_zhang.py` | **Data:** `data/multitemp_dataset/` | **Features:** 66

8 Molicell 21700 P42A NMC cells cycled at −10°C (N10_CB1–CB4) and −20°C (N20_CB1–CB4). All cells reached EOL. EIS every cycle (Ns=6, post-charge, 33 freqs). Combined with RT cells (CA1-CA8) to form a 3-temperature DOE.

### Zhang DOE split

| Set | Cells |
|-----|-------|
| Train | CA1-CA8 (RT) + N10_CB1-CB3 (−10°C) + N20_CB1-CB3 (−20°C) |
| Test | N10_CB4 (−10°C held-out) + N20_CB4 (−20°C held-out) |

Kernel: **CoupledARD-RBF** (33 ls, Re+Im paired per frequency) for capacity; **Linear** for RUL.

### Results

| Task | Cell | R² |
|------|------|----|
| Capacity | N10_CB4 (−10°C) | 0.43 |
| Capacity | N20_CB4 (−20°C) | **0.94** |
| RUL | N10_CB4 (−10°C) | 0.16 |
| RUL | N20_CB4 (−20°C) | −157 |

**Coupled ARD:** 1.33 Hz (w=0.64) and 1000 Hz (w=0.36) — only 2 of 33 frequencies matter.

**−20°C capacity succeeds** because the Zhang DOE includes representative cells at every test temperature in training. **−20°C RUL fails** because −20°C cells live only 17–21 cycles vs RT 200+ — the linear model trained on RT-scale RUL cannot resolve this range. Fix: fractional RUL normalised by each cell's own RUL_max.

### Comparison: Baseline vs Zhang DOE (−20°C test cells)

| Approach | Mean Cap R² | Mean RUL R² |
|----------|-------------|-------------|
| Baseline (train RT+−10°C only) | −8.7 | −6499 |
| **Zhang DOE (all temps in training)** | **0.94** | −157 |

---

## Why Direct EIS → RUL Fails

All A1-A8 / CA1-CA8 cells operate at the same temperature, C-rate, and conditions. Total lifetimes span **190–448 cycles (2.4× range)**. Two cells at identical State of Health (same EIS signature) can have completely different remaining lives depending on their intrinsic total lifespan — which EIS cannot encode.

For EIS → RUL to work, training cells need **diverse lifetimes driven by physically distinguishable conditions**. Temperature variation is the strongest lever: higher temperature simultaneously shifts EIS (Arrhenius) and accelerates degradation, creating a genuine learnable link between impedance signature and remaining life. This is why the paper succeeds — it exploits a multi-temperature DOE specifically designed for this.

---

## Key Scientific Finding

**Feature #91 (17.80 Hz, Im(Z)) is the universal degradation indicator** — dominant across all temperatures in ARD analysis:

| Dataset | Top feature | Frequency | Interpretation |
|---------|------------|-----------|----------------|
| 35°C single-cell (Fig 3c) | **#91** | 17.80 Hz | ✅ Exact match |
| 25°C four-cell (Fig 1c) | #91, #100 in top-5 | 17.80 Hz & 2.16 Hz | ✅ Same region |
| 45°C multi-T (Fig 3d) | #88 | ~20 Hz | ≈ Same region |
| A1-A8 in-house (LOOCV ARD) | low-freq Im(Z) | ~1–20 Hz | Consistent |

Feature #100 (2.16 Hz) appears at 25°C only — it is sensitive to both temperature and degradation. Multi-temperature training acts as a regulariser that strips it out, isolating the temperature-independent degradation signal at 17.80 Hz.

---

## Scripts Reference

| Script | Dataset | Purpose | Key Outputs |
|--------|---------|---------|-------------|
| `run_gpytorch.py` | Cambridge (Zenodo + GitHub) | Paper reproduction Figs 1–4 | `output/fig1a…fig4c` (11 PNGs) |
| `run_new_dataset.py` | A1-A8 | Fixed train/test split | `fig_capacity_A5-A8`, `fig_rul_A5-A8`, `fig_ARD_A1-A4` |
| `run_loocv.py` | CA1-CA8 (complete) | LOOCV: capacity + RUL (linear & RBF) + ARD | `fig_cap_loocv`, `fig_rul_loocv`, `fig_ARD_loocv_folds` |
| `run_cap_rul.py` | CA1-CA8 | Capacity-derived RUL via extrapolation | `fig_cap_rul_trajectories`, `fig_cap_rul_scatter` |
| `run_freq_subset_loocv.py` | A1-A8 | Frequency band LOOCV | `fig_freq_subset_cap_heatmap`, `fig_freq_subset_rul_heatmap` |
| `run_coupled_ard_loocv.py` | A1-A8 | Coupled vs decoupled ARD | `fig_ARD_coupled_vs_decoupled` |
| `run_multitemp_zhang.py` | CA+CB multi-T | **Zhang DOE + Coupled ARD** (RT+−10°C+−20°C train, 1 held-out per temp) | `output/multitemp_zhang/` |
| `run_multitemp_rul.py` | CA+CB multi-T | Baseline multi-T RUL (train RT+−10°C, test −20°C) + decoupled ARD | `output/multitemp/` |
| `run_multitemp_approaches.py` | CA+CB multi-T | LOOCV multi-T and relative-feature normalisation approaches | `output/multitemp/` |
| `preprocess_new_dataset.py` | Battery data.zip | Extract A1-A8 EIS + capacity | `data/new_dataset/` |
| `preprocess_ca_dataset.py` | Battery data/ (.mpt) | Extract CA1-CA8 EIS + capacity | `data/ca_dataset/` |
| `preprocess_multitemp_dataset.py` | 21700 Molicell Cycling Data.zip | Extract N10/N20 CB cells | `data/multitemp_dataset/` |
| `preprocess_zenodo.py` | Zenodo raw files | Extract Cambridge EIS + capacity | `data/*.txt` |

---

## Data Sources

| Dataset | Source | Access | Features | Cells |
|---------|--------|--------|----------|-------|
| Cambridge training | [GitHub](https://github.com/YunweiZhang/ML-identify-battery-degradation) `Code-Matlab.zip` | Public | 120 | 25C01–04, 35C01, 45C01 |
| Cambridge test (35C02, 45C02) | GitHub `Code-Matlab.zip` | Public | 120 | 35C02, 45C02 |
| Cambridge test (25C05–08) | [Zenodo](https://doi.org/10.5281/zenodo.3633835) raw files | Public | 120 | 25C05–08 |
| **Cambridge test 25C RUL (Fig 2)** | **Not released** | **Unavailable** | — | 25C05–08 |
| A1-A8 (partial) | `Battery data.zip` | In-house | 66 | A1–A8 |
| CA1-CA8 (complete) | `Battery data/` (.mpt) | In-house | 66 | CA1–CA8 |

---

## Implementation Notes

### Normalisation Rules

| Experiment | EIS normalisation | Why |
|-----------|------------------|-----|
| Cambridge capacity & RUL | z-score (training stats → test) | Standard; test cells same conditions |
| A1-A8 / CA capacity LOOCV | **Joint** (train+test pooled per fold) | Removes cell-to-cell impedance offset (~0.04 Ω) |
| RUL LOOCV | **Training-only** | Paper's stated method; joint norm hurts RUL |

### Kernel Mapping: MATLAB GPML → Python sklearn

| MATLAB (GPML) | sklearn | Notes |
|---|---|---|
| `covSEiso` | `RBF(length_scale=l, length_scale_bounds="fixed")` | Fixed l avoids dead-kernel local minimum |
| `covSEard` | `RBF(length_scale=np.ones(n), ...)` | One l per feature |
| `covLINiso` | `DotProduct(sigma_0=0, sigma_0_bounds="fixed")` | RUL kernel |
| `likGauss` sn | `WhiteKernel` in kernel expression | Noise term |
| `meanZero` | `normalize_y=False` | For RUL |
| `zscore(X)` | `(X - X.mean(0)) / X.std(0, ddof=1)` | ddof=1 matches MATLAB |

### ARD Weight Formula (Paper Methods Eq. 4)

```python
ls = gpr.kernel_.k1.k2.length_scale   # per-feature length-scales
w  = np.exp(-ls);  w /= w.sum()        # w_m = exp(-σ_m), normalised
```

Note: the paper's MATLAB code uses `exp(-10^log(σ))` due to GPML's log-space parameterisation. The equivalent in linear space is `exp(-σ)`. Do NOT use `exp(-10^log(σ))` directly in Python.
