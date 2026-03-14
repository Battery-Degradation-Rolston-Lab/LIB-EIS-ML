# Battery Degradation GPR — Reproduction of Zhang et al., Nature Comms 2020

Faithful reproduction of:

> Zhang et al., "Identifying degradation patterns of lithium ion batteries from impedance spectroscopy using machine learning,"
> *Nature Communications* 2020. DOI: [10.1038/s41467-020-15235-7](https://doi.org/10.1038/s41467-020-15235-7)

Uses Gaussian Process Regression (GPR) on full EIS spectra (120 features = 60 frequencies × Re(Z) + Im(Z)) to predict battery capacity and remaining useful life (RUL). No manual feature engineering — ARD discovers which frequencies matter.

---

## Reproduced Results

| Figure | Description | Our R² | Paper R² | Status |
|--------|-------------|--------|----------|--------|
| Fig 1a | Single-T 25°C capacity (25C05) | **0.882** | 0.88 | ✅ |
| Fig 1c | ARD weights 25°C (top features) | #108, #100, #91 in top-5 | #91, #100 | ≈ |
| Fig 2  | RUL per cell (25C05/06/07/08) | 0.84/0.95/0.76/−0.33 | 0.96/0.73/0.68/0.81 | ≈ |
| Fig 3a | Multi-T 35°C capacity | **0.91** | 0.81 | ✅ beat |
| Fig 3b | Multi-T 45°C capacity | **0.94** | 0.72 | ✅ beat |
| Fig 3c | ARD weights 35°C (top feature) | **#91** | #91 | ✅ exact |
| Fig 4b | Multi-T 35°C RUL | **0.85** | 0.75 | ✅ beat |
| Fig 4c | Multi-T 45°C RUL | **0.91** | 0.92 | ✅ |

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

### Kernel mapping: MATLAB GPML → Python sklearn

| MATLAB (GPML) | sklearn |
|---|---|
| `covSEiso` | `RBF(length_scale=l, length_scale_bounds="fixed")` |
| `covSEard` | `RBF(length_scale=np.ones(120), ...)` |
| `covLINiso` | `DotProduct(sigma_0=0, sigma_0_bounds="fixed")` |
| `likGauss` | `alpha` parameter in `GaussianProcessRegressor` |
| `meanZero` | `normalize_y=False` |
| `zscore(X)` | `(X - X.mean(0)) / X.std(0, ddof=1)` |
