# Science Reference — Battery Degradation GPR

## Paper

Zhang et al., Nature Communications 2020
"Identifying degradation patterns of lithium ion batteries from impedance spectroscopy using machine learning"
DOI: 10.1038/s41467-020-15235-7
GitHub: https://github.com/YunweiZhang/ML-identify-battery-degradation
Zenodo: https://doi.org/10.5281/zenodo.3633835

---

## R² Targets Per Figure

| Figure | Model | Train | Test | Kernel | R² target |
|--------|-------|-------|------|--------|-----------|
| Fig 1a | Single-T 25°C capacity | 25C01–04 | 25C05–08 | ARD-SE | **0.88** |
| Fig 2  | Single-T 25°C RUL | 25C01–04 | 25C05–08 | Linear | 0.68–0.96 per cell |
| Fig 3a | Multi-T 35°C capacity | 25C01–04+35C01+45C01 | 35C02 | SE/RBF | **0.81** |
| Fig 3b | Multi-T 45°C capacity | same | 45C02 | SE/RBF | 0.72 |
| Fig 3c | ARD 35°C | 35C01 only | — | ARD-SE | top=#91 |
| Fig 4b | Multi-T 35°C RUL | EIS_data_RUL | 35C02 | Linear | **0.75** |
| Fig 4c | Multi-T 45°C RUL | same | 45C02 | Linear | 0.92 |

**R²=0.88 is Fig 1 (single-T), NOT Fig 3a (multi-T). Do not confuse them.**

---

## Reproduced Results

| Figure | Our Result | Target | Status |
|--------|-----------|--------|--------|
| Fig 3a | R²=0.83 | 0.81 | Beat target |
| Fig 4b | R²=0.75 | 0.75 | Exact match |
| Fig 3c | top=#89–91 | #91 | Correct |
| A1-A8 Capacity LOOCV | mean R²=0.964 | — | Strong |
| A1-A8 RUL Linear LOOCV | mean R²=-0.33 | — | Fails |

---

## ARD Feature Importance (Key Findings)

### Zenodo 25°C/35°C/45°C cells (120-feature model)
- **Feature #91 (17.80 Hz, Im(Z))** — universal degradation feature, survives ALL temperatures
- **Feature #100 (2.16 Hz, Im(Z))** — appears only in single-T 25°C model, disappears under multi-T training
- Both are in the **low-frequency SEI region**

### New A1-A8 dataset (66-feature native grid)
- **Feature #2 (7500 Hz, Re(Z))** — w=0.556 (high-frequency bulk resistance)
- **Feature #66 (0.999 Hz, Im(Z))** — w=0.393 (measurement floor — diffusion region just below 1 Hz)
- **Feature #56 (17.8 Hz, Im(Z))** — w=0.045 (SEI feature, analogous to #91)

### CB Multi-temperature dataset (66-feature, Coupled ARD — Zhang DOE)
- **1.33 Hz (w=0.64)** — dominant: lowest measured frequency, SEI/diffusion boundary
- **1000 Hz (w=0.36)** — secondary: bulk resistance / charge-transfer transition
- All 31 other frequencies suppressed (length-scale → upper bound)
- Sharper than decoupled ARD because Re+Im paired: 33 ls instead of 66

### Physical Interpretation
- 17.80 Hz probes the solid-electrolyte interphase (SEI) layer — grows irreversibly with cycling
- SEI growth is the dominant LCO/graphite degradation mechanism, discovered purely from data
- Multi-T training acts as regulariser: strips temperature-confounded features, keeps degradation-robust ones
- Coupled ARD (one ls per frequency) is physically correct: Kramers-Kronig relations couple Re and Im — treating them independently gives ambiguous importance scores

---

## ARD Weight Formula (CRITICAL — paper-correct)

Paper Methods defines: `w_m = exp(−σ_m)`, where σ_m is the length-scale in linear space.

```python
# sklearn
ls = gpr.kernel_.k1.k2.length_scale   # array of length-scales
weights = np.exp(-ls)
weights /= weights.sum()

# GPyTorch
ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
weights = np.exp(-ls)
weights /= weights.sum()
```

**Wrong formula** (MATLAB approximation): `exp(-10^log(σ_m))` — do not use.

---

## Why Capacity Works but RUL Fails in LOOCV

**Capacity**: EIS → current health. Stable relationship across cells. Joint normalisation
removes impedance offset. R²=0.964 LOOCV.

**RUL** (absolute cycles remaining): EIS at cycle 50 of a 200-cycle cell looks similar to
cycle 50 of a 450-cycle cell. The EIS encodes current health, not total remaining life.
A1-A8 span RUL_max 190–448 (2.4× range) — model trained on RUL_max~300 can't generalise
to A7 (RUL_max=190) or A8 (RUL_max=448).

**Solution under investigation**: capacity-derived RUL — predict capacity trajectory with
GPR (already R²=0.964), fit degradation curve, extrapolate to 80% threshold.

**Multi-T RUL failure (CB dataset):** −20°C cells live only 17–21 cycles vs RT 200+ cycles.
Even with Zhang DOE (all temps in training), the absolute RUL range is completely mismatched.
**Fix**: fractional RUL — normalise each cell's RUL by its own RUL_max before training,
so all cells map to [0, 1]. This removes the absolute scale mismatch across temperatures.
