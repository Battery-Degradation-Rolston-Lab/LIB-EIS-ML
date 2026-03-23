"""
Coupled ARD LOOCV — pairing (Re(Z), Im(Z)) per frequency under one length-scale.

Current (decoupled):
  Feature vector: [Re(ω₁)..Re(ω₃₃), Im(ω₁)..Im(ω₃₃)]  — 66 features, 66 length-scales
  ARD attribution is physically ambiguous: K-K couples Re and Im.

Coupled (this script):
  One length-scale lᵢ per frequency ωᵢ.
  k(x,x') = σ² · exp(−Σᵢ [(Re_i−Re_i')² + (Im_i−Im_i')²] / (2·lᵢ²))
  Importance = exp(−lᵢ) / Σ exp(−lⱼ)  — one weight per physical frequency.

Compares capacity LOOCV R² and ARD frequency importance: decoupled vs coupled.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Kernel, ConstantKernel, RBF, WhiteKernel
)
from sklearn.metrics import r2_score

DATA = Path(__file__).parent / "data" / "new_dataset"
OUT  = Path(__file__).parent / "output" / "new_dataset"
OUT.mkdir(parents=True, exist_ok=True)

NATIVE_FREQS = np.array([
    10000.0, 7500.0, 5620.0, 4220.0, 3160.0, 2370.0, 1780.0, 1330.0,
    1000.0,  750.0,  564.0,  422.0,  316.0,  237.0,  178.0,  135.0,
    102.0,   75.0,   56.2,   42.2,   31.6,   23.7,   17.8,   13.3,
    10.0,    7.5,    5.62,   4.22,   3.16,   2.37,   1.78,   1.33, 0.999
])  # 33 freqs high → low (same order as feature vector Re then Im)

N_FREQ    = 33
ALL_CELLS = [f'A{i}' for i in range(1, 9)]
DNF_CELLS = {'A3', 'A6'}
ARD_MAX_N = 300   # subsample training set for ARD fits (K is N²; 300→90k vs 1900→3.6M)
RNG       = np.random.default_rng(42)


# ===========================================================================
# Custom Coupled ARD-RBF kernel
# ===========================================================================
class CoupledARDRBF(Kernel):
    """
    ARD-RBF kernel with one length-scale per frequency, shared across Re and Im.

    Feature layout assumed: X[:, :N_FREQ] = Re(Z), X[:, N_FREQ:] = Im(Z).

    k(x,x') = exp(−Σᵢ [(Re_i−Re_i')² + (Im_i−Im_i')²] / (2·lᵢ²))
    """

    def __init__(self, length_scale=None,
                 length_scale_bounds=(1e-5, 1e5)):
        if length_scale is None:
            length_scale = np.ones(N_FREQ)
        self.length_scale = np.asarray(length_scale, dtype=float)
        self.length_scale_bounds = length_scale_bounds

    @property
    def hyperparameter_length_scale(self):
        from sklearn.gaussian_process.kernels import Hyperparameter
        return Hyperparameter("length_scale", "numeric",
                              self.length_scale_bounds,
                              n_elements=N_FREQ)

    def __call__(self, X, Y=None, eval_gradient=False):
        X_re = X[:, :N_FREQ]
        X_im = X[:, N_FREQ:]
        if Y is None:
            Y_re, Y_im = X_re, X_im
        else:
            Y_re, Y_im = Y[:, :N_FREQ], Y[:, N_FREQ:]

        ls = self.length_scale   # (33,)
        N, M = X_re.shape[0], Y_re.shape[0]

        # Build K by accumulating per-frequency to avoid large (N,M,33) tensor
        dist2 = np.zeros((N, M))
        for i in range(N_FREQ):
            dr = X_re[:, i:i+1] - Y_re[:, i]   # (N, M)
            di = X_im[:, i:i+1] - Y_im[:, i]
            dist2 += (dr ** 2 + di ** 2) / (2.0 * ls[i] ** 2)
        K = np.exp(-dist2)

        if eval_gradient:
            # dK/d(log lᵢ) = K · (dr_i² + di_i²) / lᵢ²  — compute per freq
            dK = np.empty((N, M, N_FREQ))
            for i in range(N_FREQ):
                dr = X_re[:, i:i+1] - Y_re[:, i]
                di = X_im[:, i:i+1] - Y_im[:, i]
                dK[:, :, i] = K * (dr ** 2 + di ** 2) / ls[i] ** 2
            return K, dK

        return K

    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return True

    def __repr__(self):
        return f"CoupledARDRBF(ls={self.length_scale.round(3)})"


# ===========================================================================
# Helpers
# ===========================================================================
def zscore(X):
    mu  = X.mean(0)
    sig = X.std(0, ddof=1)
    sig = np.where(sig == 0, 1.0, sig)
    return (X - mu) / sig, mu, sig

def apply_norm(X, mu, sig):
    return (X - mu) / sig

def ard_weights(ls):
    """Convert length-scales to normalised ARD weights."""
    w = np.exp(-ls)
    return w / w.sum()


# ===========================================================================
# Load data
# ===========================================================================
print("Loading per-cell data ...")
cell_eis = {c: np.loadtxt(DATA / f"EIS_{c}.txt") for c in ALL_CELLS}
cell_cap = {c: np.loadtxt(DATA / f"cap_{c}.txt") for c in ALL_CELLS}


# ===========================================================================
# LOOCV — Decoupled ARD (66 length-scales, baseline)
# ===========================================================================
print("\n" + "="*60)
print("DECOUPLED ARD LOOCV  (66 length-scales, Re and Im independent)")
print("="*60)

r2_decoupled       = {}
fold_ls_re_dec     = []   # (8, 33)
fold_ls_im_dec     = []

for test_cell in ALL_CELLS:
    train_cells = [c for c in ALL_CELLS if c != test_cell]

    EIS_tr = np.vstack([cell_eis[c] for c in train_cells])
    Cap_tr = np.concatenate([cell_cap[c] for c in train_cells])
    EIS_te = cell_eis[test_cell]
    Cap_te = cell_cap[test_cell]

    EIS_all = np.vstack([EIS_tr, EIS_te])
    _, mu_x, sig_x = zscore(EIS_all)
    X_tr_n = apply_norm(EIS_tr, mu_x, sig_x)
    X_te_n = apply_norm(EIS_te, mu_x, sig_x)

    # Subsample for ARD (K is N²; 300 pts → fast, importance pattern stable)
    idx = RNG.choice(len(X_tr_n), min(ARD_MAX_N, len(X_tr_n)), replace=False)
    X_sub, y_sub = X_tr_n[idx], Cap_tr[idx]

    k = (ConstantKernel(1.0) *
         RBF(length_scale=np.ones(66)) +
         WhiteKernel(noise_level=1.0))
    gpr = GaussianProcessRegressor(kernel=k, normalize_y=True,
                                   alpha=0.1, n_restarts_optimizer=0)
    gpr.fit(X_sub, y_sub)
    r2_decoupled[test_cell] = r2_score(Cap_te, gpr.predict(X_te_n))

    ls = gpr.kernel_.k1.k2.length_scale   # (66,)
    fold_ls_re_dec.append(ls[:N_FREQ])
    fold_ls_im_dec.append(ls[N_FREQ:])
    print(f"  leave-{test_cell}: R²={r2_decoupled[test_cell]:.4f}")

mean_r2_dec = np.mean(list(r2_decoupled.values()))
print(f"  Mean R²: {mean_r2_dec:.4f}")

# Fold-averaged weights
ls_re_dec = np.array(fold_ls_re_dec)   # (8, 33)
ls_im_dec = np.array(fold_ls_im_dec)

# Compute weights fold-by-fold, then average
w_re_dec_folds = np.array([ard_weights(np.concatenate([ls_re_dec[i], ls_im_dec[i]]))[:N_FREQ]
                            for i in range(8)])
w_im_dec_folds = np.array([ard_weights(np.concatenate([ls_re_dec[i], ls_im_dec[i]]))[N_FREQ:]
                            for i in range(8)])
mean_w_re_dec = w_re_dec_folds.mean(0)
std_w_re_dec  = w_re_dec_folds.std(0)
mean_w_im_dec = w_im_dec_folds.mean(0)
std_w_im_dec  = w_im_dec_folds.std(0)


# ===========================================================================
# LOOCV — Coupled ARD (33 length-scales, one per frequency)
# ===========================================================================
print("\n" + "="*60)
print("COUPLED ARD LOOCV  (33 length-scales, one per frequency)")
print("="*60)

r2_coupled     = {}
fold_ls_coupled = []   # (8, 33)

for test_cell in ALL_CELLS:
    train_cells = [c for c in ALL_CELLS if c != test_cell]

    EIS_tr = np.vstack([cell_eis[c] for c in train_cells])
    Cap_tr = np.concatenate([cell_cap[c] for c in train_cells])
    EIS_te = cell_eis[test_cell]
    Cap_te = cell_cap[test_cell]

    EIS_all = np.vstack([EIS_tr, EIS_te])
    _, mu_x, sig_x = zscore(EIS_all)
    X_tr_n = apply_norm(EIS_tr, mu_x, sig_x)
    X_te_n = apply_norm(EIS_te, mu_x, sig_x)

    idx = RNG.choice(len(X_tr_n), min(ARD_MAX_N, len(X_tr_n)), replace=False)
    X_sub, y_sub = X_tr_n[idx], Cap_tr[idx]

    k = (ConstantKernel(1.0) *
         CoupledARDRBF(length_scale=np.ones(N_FREQ)) +
         WhiteKernel(noise_level=1.0))
    gpr = GaussianProcessRegressor(kernel=k, normalize_y=True,
                                   alpha=0.1, n_restarts_optimizer=0)
    gpr.fit(X_sub, y_sub)
    r2_coupled[test_cell] = r2_score(Cap_te, gpr.predict(X_te_n))

    ls = gpr.kernel_.k1.k2.length_scale   # (33,)
    fold_ls_coupled.append(ls)

    top_freq = NATIVE_FREQS[np.argmin(ls)]   # smallest ls = highest importance
    print(f"  leave-{test_cell}: R²={r2_coupled[test_cell]:.4f}  "
          f"top freq={top_freq:.1f} Hz")

mean_r2_cou = np.mean(list(r2_coupled.values()))
print(f"  Mean R²: {mean_r2_cou:.4f}")

# Fold-averaged weights
ls_coupled = np.array(fold_ls_coupled)   # (8, 33)
w_cou_folds = np.array([ard_weights(ls_coupled[i]) for i in range(8)])
mean_w_cou  = w_cou_folds.mean(0)
std_w_cou   = w_cou_folds.std(0)


# ===========================================================================
# Plot — side by side comparison
# ===========================================================================
# Frequencies plotted low → high (left → right)
freqs_plot = NATIVE_FREQS[::-1]

def _flip(arr):
    return arr[::-1]

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)

# ── Left: decoupled ───────────────────────────────────────────────────────
ax = axes[0]
mr, sr = _flip(mean_w_re_dec), _flip(std_w_re_dec)
mi, si = _flip(mean_w_im_dec), _flip(std_w_im_dec)
ax.semilogx(freqs_plot, mr, 'r-^', ms=5, lw=1.5, label='Re(Z)')
ax.fill_between(freqs_plot, mr - sr, mr + sr, color='red',    alpha=0.15)
ax.semilogx(freqs_plot, mi, color='orange', ls='--', marker='v',
            ms=5, lw=1.5, label='-Im(Z)')
ax.fill_between(freqs_plot, mi - si, mi + si, color='orange', alpha=0.15)
ax.axhline(0, color='black', lw=0.6)
ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('ARD weight', fontsize=12)
ax.set_title(f'Decoupled ARD  (66 length-scales)\n'
             f'mean R² = {mean_r2_dec:.3f}', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ── Right: coupled ────────────────────────────────────────────────────────
ax = axes[1]
mc, sc = _flip(mean_w_cou), _flip(std_w_cou)
ax.semilogx(freqs_plot, mc, 'b-o', ms=5, lw=1.8,
            label='|Z(ω)| — Re & Im paired')
ax.fill_between(freqs_plot, mc - sc, mc + sc, color='blue', alpha=0.15)
ax.axhline(0, color='black', lw=0.6)
ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('ARD weight', fontsize=12)
ax.set_title(f'Coupled ARD  (33 length-scales, one per ω)\n'
             f'mean R² = {mean_r2_cou:.3f}', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Annotate top frequency for coupled
top_idx = np.argmax(mc)
ax.annotate(f'{freqs_plot[top_idx]:.1f} Hz',
            xy=(freqs_plot[top_idx], mc[top_idx]),
            xytext=(freqs_plot[top_idx] * 2, mc[top_idx] + 0.01),
            fontsize=9, color='blue',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1))

fig.suptitle('ARD LOOCV — Decoupled (Re, Im separate) vs Coupled (Re, Im paired per ω)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
out_path = OUT / 'fig_ARD_coupled_vs_decoupled.png'
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nSaved: {out_path}")

# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\n{'Cell':<6}  {'Decoupled R²':>14}  {'Coupled R²':>12}")
for c in ALL_CELLS:
    tag = ' [DNF]' if c in DNF_CELLS else ''
    print(f"{c+tag:<10}  {r2_decoupled[c]:>14.4f}  {r2_coupled[c]:>12.4f}")
print(f"{'Mean':<10}  {mean_r2_dec:>14.4f}  {mean_r2_cou:>12.4f}")

# Top coupled frequency per fold
print("\nTop frequency (highest coupled ARD weight) per fold:")
for i, cell in enumerate(ALL_CELLS):
    top_i = np.argmin(ls_coupled[i])
    print(f"  leave-{cell}: {NATIVE_FREQS[top_i]:.2f} Hz  (w={w_cou_folds[i,top_i]:.3f})")
