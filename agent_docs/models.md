# Model Reference — Battery Degradation GPR

## MATLAB → Python Kernel Translation

| MATLAB (GPML) | sklearn | GPyTorch |
|---|---|---|
| `covSEiso` | `RBF(length_scale=1.0)` | `ScaleKernel(RBFKernel())` |
| `covSEard` | `RBF(length_scale=np.ones(N))` | `ScaleKernel(RBFKernel(ard_num_dims=N))` |
| `covLINiso` | `DotProduct(sigma_0=0)` | `ScaleKernel(LinearKernel())` |
| `likGauss` | `WhiteKernel` | `GaussianLikelihood()` |
| `meanZero` | `normalize_y=False` | `ZeroMean()` |
| `minimize(...,-10000)` | L-BFGS-B | Adam 3000–5000 steps |
| `zscore(X)` | `(X - X.mean(0)) / X.std(0, ddof=1)` | same |

---

## Normalisation Rules

```python
def zscore(X):
    mu = X.mean(0); sig = X.std(0, ddof=1)
    sig = np.where(sig == 0, 1, sig)
    return (X - mu) / sig, mu, sig

def apply_norm(X, mu, sig):
    return (X - mu) / sig
```

- **EIS features (paper Zenodo model)**: zscore using `EIS_data.txt` training stats
- **EIS features (new A1-A8 capacity model)**: joint norm (train+test pooled per LOOCV fold)
- **EIS features (new A1-A8 RUL model)**: training-only norm (paper's stated approach)
- **Capacity output**: `normalize_y=True` in GaussianProcessRegressor handles this
- **RUL output**: `normalize_y=False` for linear kernel (paper convention)

---

## Paper Reproduction Models (run_gpytorch.py)

### Model 1 — Multi-T Capacity GPR (Fig 3a)
- Kernel: `ScaleKernel(RBFKernel())`
- Train: 1358 rows of `EIS_data.txt` (do NOT subsample — causes R² variance 0.34–0.93)
- normalize_y: True
- GPyTorch: 3000 Adam steps, lr=0.05
- Target: R²=0.81 on 35C02, achieved 0.83

### Model 2 — Multi-T RUL GPR (Fig 4b)
- Kernel: `ScaleKernel(LinearKernel())` — paper eq.(5) k_LIN = Σ xim·xjm
- Train: 525 rows of `EIS_data_RUL.txt`
- EIS normalisation: use mu_x/sig_x from `EIS_data.txt` (not EIS_data_RUL.txt)
- Test: first 127 rows of `EIS_data_35C02.txt`
- normalize_y: False
- GPyTorch: 3000 Adam steps, lr=0.05
- Target: R²=0.75 on 35C02 RUL, achieved 0.75

### Model 3 — ARD GPR (Fig 3c)
- Kernel: `ScaleKernel(RBFKernel(ard_num_dims=120))`
- Train: `EIS_data_35.txt` (299 rows, 35°C cell only)
- Init: all length-scales = 1.0
- GPyTorch: 5000 Adam steps, lr=0.05
- Top feature: #91 (17.80 Hz, Im(Z))

---

## New A1-A8 Dataset Models (run_new_dataset.py / run_loocv.py)

### Capacity GPR
- Kernel: `ConstantKernel(fixed) * RBF(l=30, fixed) + WhiteKernel`
- l=30 calibrated by grid search on joint-normalised 66-feature native data
- normalize_y: True, alpha=0.1
- LOOCV result: mean R²=0.964 across all 8 cells

### RUL GPR — Linear (paper method)
- Kernel: `DotProduct(sigma_0=0, fixed) + WhiteKernel`
- normalize_y: False, alpha=0.4, n_restarts_optimizer=3
- LOOCV result: mean R²=-0.33 (absolute RUL doesn't transfer across heterogeneous cells)

### RUL GPR — RBF (extended)
- Kernel: `ConstantKernel * RBF(l=30, bounds=(1,1000)) + WhiteKernel`
- normalize_y: True, alpha=0.4, n_restarts_optimizer=3
- LOOCV result: mean R²=-1.24 (overfits to training lifetimes, worse than linear)

### ARD GPR (feature importance)
- Kernel: `ConstantKernel * RBF(length_scale=np.ones(66)) + WhiteKernel`
- Train: A1-A4 joint-normalised, normalize_y=True
- Top features: #2 (7500 Hz Re(Z), w=0.556), #66 (0.999 Hz Im(Z), w=0.393), #56 (17.8 Hz Im(Z), w=0.045)

---

## Next Model: Capacity-Derived RUL

**Approach**: use the working capacity GPR (R²=0.964) to derive RUL instead of
predicting absolute cycles directly.

1. Predict full capacity trajectory with capacity GPR
2. Fit degradation curve (linear or exponential) to predicted sequence
3. Extrapolate to find when capacity crosses 80% of initial → RUL

This avoids the fundamental problem: EIS encodes current health, not remaining life.
