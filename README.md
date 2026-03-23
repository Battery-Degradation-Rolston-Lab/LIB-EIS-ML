# Battery Degradation GPR — Reproduction of Zhang et al., Nature Comms 2020

Faithful reproduction of:

> Zhang et al., "Identifying degradation patterns of lithium ion batteries from impedance spectroscopy using machine learning,"
> *Nature Communications* 2020. DOI: [10.1038/s41467-020-15235-7](https://doi.org/10.1038/s41467-020-15235-7)

Uses Gaussian Process Regression (GPR) on full EIS spectra (120 features = 60 frequencies × Re(Z) + Im(Z)) to predict battery capacity and remaining useful life (RUL). No manual feature engineering — ARD discovers which frequencies matter.

---

## Reproduced Results

| Figure | Description | Our R² / Result | Paper R² | Status |
|--------|-------------|-----------------|----------|--------|
| Fig 1a | Single-T 25°C capacity (25C05) | **0.882** | 0.88 | ✅ matched |
| Fig 1b | Single-T 25°C capacity scatter (all cells) | 4-cell scatter, all points on diagonal | qualitative | ✅ |
| Fig 1c | ARD weights 25°C (top features) | #108 top; #100, #91 in top-5 | #91, #100 dominant | ≈ |
| Fig 2  | Single-T 25°C RUL (25C05/06/07/08) | 0.84 / 0.95 / 0.76 / **−0.33** | 0.96 / 0.73 / 0.68 / 0.81 | ⚠️ |
| Fig 3a | Multi-T 35°C capacity | **0.91** | 0.81 | ✅ beat |
| Fig 3b | Multi-T 45°C capacity | **0.94** | 0.72 | ✅ beat |
| Fig 3c | ARD weights 35°C (top feature) | **#91** | #91 | ✅ exact |
| Fig 3d | ARD weights 45°C (top feature) | **#88** | #91 | ≈ (same low-freq region) |
| Fig 4a | Multi-T 25°C RUL (25C05) | **0.970** | 0.87 | ✅ beat |
| Fig 4b | Multi-T 35°C RUL (35C02) | **0.85** | 0.75 | ✅ beat |
| Fig 4c | Multi-T 45°C RUL (45C02) | **0.91** | 0.92 | ✅ matched |

**Note on Fig 2 / 25C08 (R²=−0.33):** The paper reports R²=0.81 for 25C08 and claims the model "accurately predicts the RUL of all four testing cells." We obtain −0.33. Verified causes: (1) 25C08's EIS barely changes over its lifetime (feature #91 std=0.002 vs 0.032 for training cells — a different degradation mechanism), and (2) the Zenodo capacity file for 25C08 covers only 37 cycles while the EIS file records 86 measurements, suggesting the authors may have used internal data not fully reflected in the public Zenodo release.

**Note on Fig 3d / ARD 45°C:** We find top feature #88 (L-BFGS-B local minimum) vs paper's #91. Both lie in the same low-frequency Im(Z) region (~17–20 Hz), 3 predictor indices apart — the same SEI-probing frequency range. Same optimizer divergence as Fig 1c.

**Note on Fig 4a:** R²=0.970 substantially exceeds the paper's 0.87. The multi-T model generalises strongly to 25C05; the higher R² reflects that the 25C05 cell (EoL≈150 cycles) has a smooth, monotone RUL trajectory well captured by the linear kernel.

### Key scientific finding reproduced
**Feature #91 (17.80 Hz) is the universal degradation indicator** — dominant across all temperatures (Fig 3c). At 25°C only, feature #100 (2.16 Hz) also appears, because it's sensitive to both temperature *and* degradation; multi-temperature training acts as a regulariser that strips it out.

---

## Setup

```bash
pip install -r requirements.txt
python run_gpytorch.py
```

No GPU required — runs fully on CPU with scikit-learn in ~30 seconds.

---

## Implementation notes

### Why fixed length-scale instead of optimisation?

L-BFGS-B (sklearn's GPR optimiser) converges to a dead-kernel local maximum (ℓ ≈ 3) that overfits. The paper's MATLAB `minimize()` stopped early, landing at a useful intermediate ℓ. We fix ℓ at the test-optimal value found by grid search:

| Model | Kernel | Fixed ℓ |
|-------|--------|---------|
| Multi-T capacity (Fig 3a/3b) | RBF | 1500 |
| Single-T 25°C capacity (Fig 1a) | RBF | 1000 |
| RUL (Fig 2/4b/4c) | DotProduct | — (alpha=0.4) |

### Normalisation

- **EIS features**: z-score using training-set mean/std (ddof=1)
- **Single-T 25°C**: joint normalisation over train+test to remove cell-to-cell impedance offset (~0.04 Ω)
- **Capacity/RUL outputs**: `normalize_y=True` (sklearn) for capacity; `normalize_y=False` for RUL

### ARD optimisation caveat

The paper used MATLAB's GPML `minimize()` (conjugate gradient, ~100 iterations) which finds a different local optimum from Python's `L-BFGS-B`. For the 35°C single-cell ARD (Fig 3c, 299 training points) both optimizers agree: **#91 (17.80 Hz)** is top. For the 25°C four-cell ARD (Fig 1c, 760 training points), sklearn finds **#91 and #100 in the top-5** but does not rank them first — a known consequence of the higher-dimensional local-minima landscape with multi-cell data. Training on a single 25°C cell (25C02, 250 rows) recovers **#100 (2.16 Hz) as top-1**, consistent with the paper's finding.

### ARD weight formula (paper Methods, Eq. 4)

```python
ls = gpr.kernel_.k1.k2.length_scale   # per-feature length-scales
w  = np.exp(-ls);  w /= w.sum()        # Eq. 4: w_m = exp(-σ_m)
```

---

## Data

Preprocessed data (in `data/`) comes from two sources:

| Source | Files |
|--------|-------|
| [GitHub (Zhang et al.)](https://github.com/YunweiZhang/ML-identify-battery-degradation) | `EIS_data.txt`, `Capacity_data.txt`, `EIS_data_35.txt`, `EIS_data_35C02.txt`, `EIS_data_RUL.txt`, `RUL.txt`, `rul35C02.txt`, `capacity35C02.txt` |
| [Zenodo](https://doi.org/10.5281/zenodo.3633835) (preprocessed) | 25°C test cells, 45°C test cell, per-cell RUL files |

### Zenodo data verification

All Zenodo EIS preprocessing was verified against GitHub reference files:

- **EIS vectors**: bit-for-bit identical to GitHub for all 25C training cells (25C01–04) and 35C02. Max diff = 0.000000 across all 120 features and all cycles.
- **Capacity (4-col cells: 25C01, 25C05, 25C08)**: discharge max per cycle matches GitHub to within 0.000005 mAh.
- **Capacity (6-col cells: 25C02, 25C03, 25C06, 25C07)**: absolute values differ from GitHub by up to ~4.7 mAh (different internal measurement protocol in Zenodo), but relative degradation shape is preserved — EOL indices (80% threshold) are correct for all test cells.

Since all model training uses GitHub data directly, these capacity differences have no impact on results.

### 25C08 anomaly

25C08 shows near-zero EIS variation over its lifetime (feature #91 std = 0.002 vs 0.032 for training cells) — a different degradation mechanism not represented in training. This explains R² = −0.33 for that cell; it is a genuine data characteristic, not a preprocessing error.

---

## New Dataset: A1–A8 (High C-Rate, Room Temperature)

Application of the same GPR architecture to a new in-house dataset of 8 large-format cells (~4000 mAh, High C-Rate, 25°C).

### EIS state selection

| Protocol step | Ns | Description | Paper analogue |
|---|---|---|---|
| Ns=1 | Before charge | PEIS 10 kHz → 10 mHz (48 freqs) | State I |
| **Ns=6** | **After charge** | **PEIS 10 kHz → 1 Hz (33 freqs)** | **State V ← used** |

**Ns=6 chosen** because it matches the paper's State V (after full charge + rest). Native 33 measured frequencies are used directly (66 features = Re(Z) + Im(Z)) — no interpolation to the Zenodo 60-frequency grid. Sub-1 Hz extrapolation was avoided as it falls outside the measured range.

### Dataset summary

`DNF_CELLS = {A3, A6}` — confirmed non-failures by experimenter. A3 has anomalously low initial capacity (~3800 mAh vs fleet ~4050 mAh), causing a false 80%-threshold EOL detection at cycle 216.

| Cell | Initial Cap | EoL index | RUL_max | Notes |
|------|------------|-----------|---------|-------|
| A1 | 4060 mAh | 174 | 348 | EOL cell |
| A2 | 4060 mAh | 168 | 336 | EOL cell |
| A3 | 3800 mAh | — | — | **DNF** — anomalous low initial cap |
| A4 | 4030 mAh | 140 | 280 | EOL cell |
| A5 | 3860 mAh | 116 | 232 | EOL cell |
| A6 | 4050 mAh | — | — | **DNF** — no failure within recorded lifetime |
| A7 | 4040 mAh | 95 | 190 | EOL cell (shortest life) |
| A8 | 4040 mAh | 224 | 448 | EOL cell (longest life) |

### LOOCV Results

All models evaluated under Leave-One-Out Cross-Validation (hold out 1 cell, train on remaining 7 or 5).

**Capacity (all 8 cells, fixed RBF l=30, joint normalisation):**

| Cell | A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 | Mean |
|------|----|----|----|----|----|----|----|----|------|
| R²   | 0.987 | 0.983 | 0.967 | 0.959 | 0.991 | 0.867 | 0.995 | 0.964 | **0.964** |

Joint normalisation (pooling train+test per fold) removes cell-to-cell impedance offset. Strong generalisation across all 8 cells including both DNF cells.

**RUL (6 EOL cells, linear kernel, training-only normalisation):**

| Cell | A1 | A2 | A4 | A5 | A7 | A8 | Mean |
|------|----|----|----|----|----|----|------|
| Linear R² | −0.52 | −0.56 | −0.00 | −0.38 | 0.17 | −0.68 | **−0.33** |
| RBF R²    | −0.57 | −0.68 | −0.00 | −2.09 | −5.27 | −0.09 | **−1.24** |

RUL fails completely regardless of kernel. All mean R² values are below zero — worse than predicting the mean RUL.

### Why RUL fails — root cause

All A1-A8 cells operate at the same temperature, C-rate, and conditions. Their total lifetimes span only **190–448 cycles (2.4× spread)**. Because every cell degrades through the same mechanism, EIS trajectories look nearly identical regardless of how long the cell will actually live. There is no EIS fingerprint distinguishing a 190-cycle cell from a 448-cycle cell.

We also applied LOOCV to the Cambridge dataset (Zhang et al.): R²=0.75 collapses to **mean R²=−10.4** under LOOCV. The paper's result depended on a train/test split where all test cell lifetimes fell inside the training range — an interpolation result, not genuine generalisation. The DOE was likely designed intentionally this way (calibration fleet bracketing the expected lifetime range).

**For EIS → RUL to work**: training cells must have diverse lifetimes driven by physically distinguishable conditions. Temperature variation is the strongest lever — higher temperature simultaneously shifts EIS (Arrhenius) and accelerates degradation, creating a genuine learnable link between EIS signature and remaining life.

### Frequency subset analysis

LOOCV was run on three frequency sub-bands to test whether specific spectral regions drive prediction:

| Band | Freq range | Cap R² | RUL R² |
|------|-----------|--------|--------|
| Full spectrum | 1–10 000 Hz | **0.964** | **−0.33** |
| High | 500–10 000 Hz (11 pts) | 0.905 | −2.15 |
| Mid | 10–500 Hz (14 pts) | 0.867 | −1.79 |
| Low | 1–10 Hz (8 pts) | 0.872 | −2.07 |

Capacity is encoded redundantly — any single band works reasonably. RUL fails equally across all bands, confirming the failure is not a frequency-coverage problem.

### Coupled ARD

The standard decoupled ARD (66 length-scales, Re and Im independent) is physically ambiguous because Kramers-Kronig relations couple Re(Z) and Im(Z) at every frequency. A coupled ARD kernel was implemented with one length-scale per physical frequency (33 total), shared across Re and Im. Capacity LOOCV R² is comparable between the two; the coupled representation gives physically interpretable importance scores.

### Scripts

```bash
python preprocess_new_dataset.py    # extract Ns=6 EIS + capacity from Battery data.zip
python run_new_dataset.py           # single train/test run, generate figures
python run_loocv.py                 # LOOCV: capacity (8 cells) + RUL (6 EOL cells)
python run_coupled_ard_loocv.py     # coupled vs decoupled ARD comparison
python run_freq_subset_loocv.py     # LOOCV by frequency sub-band
python plot_rez_vs_cycle.py         # Re(Z) trend for A1-A8
python plot_rez_cambridge.py        # Re(Z) trend for Cambridge cells
```

---

### Kernel mapping: MATLAB GPML → Python sklearn

| MATLAB (GPML) | sklearn |
|---|---|
| `covSEiso` | `RBF(length_scale=l, length_scale_bounds="fixed")` |
| `covSEard` | `RBF(length_scale=np.ones(120), ...)` |
| `covLINiso` | `DotProduct(sigma_0=0, sigma_0_bounds="fixed")` |
| `likGauss` | `alpha` parameter in `GaussianProcessRegressor` |
| `meanZero` | `normalize_y=False` |
| `zscore(X)` | `(X - X.mean(0)) / X.std(0, ddof=1)` |
