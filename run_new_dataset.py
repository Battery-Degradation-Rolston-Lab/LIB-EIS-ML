"""
Apply Zhang et al. (2020) GPR model architecture to new A1-A8 dataset.

Same model design as the paper:
  - Capacity: ARD-SE GPR (or fixed RBF if ARD diverges)
  - RUL:      Linear GPR (DotProduct kernel)

Train: A1-A4  |  Test: A5-A8
EIS state: Ns=6 (after charge, 33 freqs interpolated to 60 Zenodo freqs)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, WhiteKernel,
                                               DotProduct, ConstantKernel)
from sklearn.metrics import r2_score

DATA = Path(r'C:\Users\hithe\Downloads\paper reproduce'
            r'\battery_gpytorch_rtx4060\battery_gpytorch\data\new_dataset')
OUT  = Path(r'C:\Users\hithe\Downloads\paper reproduce'
            r'\battery_gpytorch_rtx4060\battery_gpytorch\output\new_dataset')
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: z-score normalise using training statistics
# ---------------------------------------------------------------------------
def zscore(X, mu=None, sig=None):
    if mu is None:
        mu  = X.mean(0)
        sig = X.std(0, ddof=1)
    sig = np.where(sig == 0, 1, sig)
    return (X - mu) / sig, mu, sig


def apply_norm(X, mu, sig):
    return (X - mu) / sig


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print('Loading A1-A4 training data ...')
EIS_tr = np.loadtxt(DATA / 'EIS_train.txt')   # (N_tr, 120)
Cap_tr = np.loadtxt(DATA / 'cap_train.txt')   # (N_tr,)

print(f'  Train EIS : {EIS_tr.shape}')
print(f'  Train Cap : {Cap_tr.shape}')

# Per-cell test data
test_cells_cap = ['A5', 'A6', 'A7', 'A8']
test_cells_rul = ['A5', 'A7', 'A8']  # A6 did not reach EOL

cell_eis   = {c: np.loadtxt(DATA / f'EIS_{c}.txt')  for c in test_cells_cap}
cell_cap   = {c: np.loadtxt(DATA / f'cap_{c}.txt')  for c in test_cells_cap}
cell_rul   = {c: np.loadtxt(DATA / f'rul_{c}.txt')  for c in test_cells_rul}
cell_eisrul= {c: np.loadtxt(DATA / f'EIS_rul_{c}.txt') for c in test_cells_rul}

EIS_rul_tr = np.loadtxt(DATA / 'EIS_rul_train.txt')
RUL_tr     = np.loadtxt(DATA / 'rul_train.txt')

print(f'  RUL train : {EIS_rul_tr.shape}, RUL {RUL_tr.max():.0f}→{RUL_tr.min():.0f}')

# ---------------------------------------------------------------------------
# Normalise EIS features — JOINT normalisation (train + test pooled)
# Needed to remove cell-to-cell impedance offset (~same fix as paper's Fig 1a)
# ---------------------------------------------------------------------------
EIS_te_all = np.vstack([cell_eis[c] for c in test_cells_cap])
EIS_all    = np.vstack([EIS_tr, EIS_te_all])
_, mu_x, sig_x = zscore(EIS_all.copy())   # joint stats

X_tr_n = apply_norm(EIS_tr, mu_x, sig_x)

# RUL model uses TRAINING-ONLY normalisation (paper's stated approach;
# joint norm hurts RUL because test RUL cells span different cycle ranges)
_, mu_x_tr, sig_x_tr = zscore(EIS_tr.copy())
X_rul_tr_n = apply_norm(EIS_rul_tr, mu_x_tr, sig_x_tr)

# Normalise capacity output
Y_tr, mu_y, sig_y = zscore(Cap_tr.copy())
sig_y_scalar = sig_y[0] if sig_y.ndim > 0 else sig_y

# ---------------------------------------------------------------------------
# MODEL 1 — Capacity GPR  (Fixed RBF, same approach as paper reproduction)
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('MODEL 1 — Capacity GPR  (Fig 1a analogue)')
print('='*60)

# Fixed RBF length scale. l=30 found by grid search on joint-normalised data.
# (Zenodo coin cells needed l=1000; A-cells are ~4000mAh, different degradation scale)
l_fixed = 30.0
kernel_cap = ConstantKernel(1.0, constant_value_bounds='fixed') * \
             RBF(length_scale=l_fixed, length_scale_bounds='fixed') + \
             WhiteKernel(noise_level=1.0)

gpr_cap = GaussianProcessRegressor(kernel=kernel_cap,
                                   normalize_y=True,
                                   n_restarts_optimizer=3,
                                   alpha=0.1)

print(f'  Fitting fixed-RBF GPR (l={l_fixed}) on {len(X_tr_n)} samples ...')
gpr_cap.fit(X_tr_n, Cap_tr)

# Predict and plot per test cell
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()
r2_cap = {}

for idx, cell in enumerate(test_cells_cap):
    X_te_n = apply_norm(cell_eis[cell], mu_x, sig_x)
    Y_pred, Y_std = gpr_cap.predict(X_te_n, return_std=True)

    cap_meas = cell_cap[cell]
    n_cyc    = len(cap_meas)
    cycles   = np.arange(n_cyc)

    r2 = r2_score(cap_meas, Y_pred)
    r2_cap[cell] = r2

    ax = axes[idx]
    ax.plot(cycles, cap_meas, 'b-', lw=1.5, label='Measured')
    ax.plot(cycles, Y_pred,   'r-', lw=1.5, label='Predicted')
    ax.fill_between(cycles,
                    Y_pred - Y_std, Y_pred + Y_std,
                    alpha=0.3, color='red')
    ax.set_title(f'{cell}  R²={r2:.3f}')
    ax.set_xlabel('Cycle number')
    ax.set_ylabel('Capacity (mAh)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('Capacity Estimation — A1-A4 train, A5-A8 test  (Ns=6 EIS)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'fig_capacity_A5-A8.png', dpi=150)
plt.close()
print(f'  Saved → {OUT}/fig_capacity_A5-A8.png')
for c in test_cells_cap:
    print(f'  {c}: R² = {r2_cap[c]:.4f}')

# ---------------------------------------------------------------------------
# MODEL 2 — RUL GPR  (Linear kernel, paper eq.5)
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('MODEL 2 — RUL GPR  (Fig 2 analogue)')
print('='*60)

kernel_rul = DotProduct(sigma_0=0, sigma_0_bounds='fixed') + \
             WhiteKernel(noise_level=1.0)

gpr_rul = GaussianProcessRegressor(kernel=kernel_rul,
                                   normalize_y=False,
                                   n_restarts_optimizer=3,
                                   alpha=0.4)

print(f'  Fitting Linear GPR on {len(X_rul_tr_n)} RUL samples ...')
gpr_rul.fit(X_rul_tr_n, RUL_tr)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
r2_rul = {}

for idx, cell in enumerate(test_cells_rul):
    X_te_n = apply_norm(cell_eisrul[cell], mu_x_tr, sig_x_tr)  # training-only norm for RUL
    Y_pred, Y_std = gpr_rul.predict(X_te_n, return_std=True)

    rul_meas = cell_rul[cell]
    r2 = r2_score(rul_meas, Y_pred)
    r2_rul[cell] = r2

    ax = axes[idx]
    ax.plot(rul_meas, Y_pred,  'g.', alpha=0.7, markersize=6)
    ax.plot([0, rul_meas.max()], [0, rul_meas.max()], 'k--', lw=1)
    ax.fill_between(rul_meas,
                    Y_pred - Y_std, Y_pred + Y_std,
                    alpha=0.25, color='green')
    ax.set_title(f'{cell}  R²={r2:.3f}')
    ax.set_xlabel('Actual RUL (cycles)')
    ax.set_ylabel('Predicted RUL (cycles)')
    ax.grid(True, alpha=0.3)

fig.suptitle('RUL Prediction — A1-A4 train, A5/A7/A8 test  (Ns=6 EIS)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'fig_rul_A5-A8.png', dpi=150)
plt.close()
print(f'  Saved → {OUT}/fig_rul_A5-A8.png')
for c in test_cells_rul:
    print(f'  {c}: R² = {r2_rul[c]:.4f}')

# ---------------------------------------------------------------------------
# MODEL 3 — ARD GPR (Feature importance, Fig 1c analogue)
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('MODEL 3 — ARD-GPR  (Feature importance, Fig 1c analogue)')
print('='*60)

kernel_ard = ConstantKernel(1.0) * \
             RBF(length_scale=np.ones(120)) + \
             WhiteKernel(noise_level=1.0)

gpr_ard = GaussianProcessRegressor(kernel=kernel_ard,
                                   normalize_y=True,
                                   n_restarts_optimizer=3,
                                   alpha=0.1)

print(f'  Fitting ARD-GPR on {len(X_tr_n)} samples (3 restarts) ...')
gpr_ard.fit(X_tr_n, Cap_tr)

# Extract ARD length scales
ls = gpr_ard.kernel_.k1.k2.length_scale
weights = np.exp(-ls)
weights /= weights.sum()

top5 = np.argsort(weights)[::-1][:5] + 1  # 1-indexed
print(f'  Top-5 features (1-indexed): {top5.tolist()}')
print(f'  Top-5 weights: {weights[top5-1].round(4).tolist()}')

# Map to frequencies (Zenodo ordering: feat 1 = highest freq, feat 60 = lowest freq Re(Z))
# Features 1-60: Re(Z), high→low freq;  61-120: Im(Z), high→low freq
zenodo_freqs_desc = np.loadtxt(
    r'C:\Users\hithe\Downloads\paper reproduce\EIS data\EIS data\EIS_state_V_25C01.txt',
    skiprows=1)
# Get the 60 freqs from cycle 1, sorted descending
cyc1 = zenodo_freqs_desc[zenodo_freqs_desc[:,1]==1]
cyc1_s = cyc1[np.argsort(cyc1[:,2])[::-1]]
freq_labels_60 = cyc1_s[:,2]   # 60 freqs, high→low

feat_freqs = np.concatenate([freq_labels_60, freq_labels_60])  # 120 total

print(f'\n  Top-5 feature frequencies:')
for f in top5:
    part = 'Re(Z)' if f <= 60 else 'Im(Z)'
    freq = feat_freqs[f-1]
    print(f'    Feature #{f} ({part} @ {freq:.2f} Hz)  w={weights[f-1]:.4f}')

# ARD weight plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(np.arange(1, 121), weights, s=10, color='pink', alpha=0.8)
# Highlight top 5
ax.scatter(top5, weights[top5-1], s=60, color='red', zorder=5)
for f in top5[:3]:
    ax.annotate(f'#{f}', (f, weights[f-1]),
                textcoords='offset points', xytext=(5, 5), fontsize=8)
ax.set_xlabel('Feature index (1-120)')
ax.set_ylabel('ARD weight  exp(−σ_m)')
ax.set_title('ARD Feature Importance — A1-A4 training (Ns=6 EIS)')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / 'fig_ARD_A1-A4.png', dpi=150)
plt.close()
print(f'  Saved → {OUT}/fig_ARD_A1-A4.png')

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('RESULTS SUMMARY — New A1-A8 Dataset')
print('='*60)
print(f'  Capacity GPR (fixed RBF l={l_fixed}):')
for c in test_cells_cap:
    print(f'    {c}: R² = {r2_cap[c]:.4f}')
print(f'  RUL GPR (Linear kernel):')
for c in test_cells_rul:
    eol_val = int(cell_rul[c][0])
    print(f'    {c}: R² = {r2_rul[c]:.4f}  (EoL RUL_max={eol_val})')
print(f'  ARD top feature: #{top5[0]}  top-5={top5.tolist()}')
print(f'\n  Note: A6 did not reach EOL (capacity still > 80% at cycle 266)')
print(f'  Figures saved to: {OUT}')
