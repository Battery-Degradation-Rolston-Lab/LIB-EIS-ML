"""
Leave-One-Out Cross-Validation (LOOCV) for CA1-CA8 battery dataset.

CA1-CA8 are the COMPLETE runs of the same physical batteries as A1-A8.
The A dataset (Battery data.zip) was an early partial export (cycles 0-267);
the CA dataset (.mpt files) continues to near-zero capacity (470+ cycles).

RUL factor: 1  (EIS measured every battery cycle, confirmed from cycle number alignment)
            A1-A8 used factor=2 in error — CA data corrects this.

Capacity GPR : all 8 cells, fixed RBF l=30, joint normalisation
RUL GPR      : 7 EOL cells (CA1-CA5, CA7, CA8)  — CA3 included (genuine failure)
               CA6 excluded (DNF — never reached 80% threshold)
               - Paper method : Linear (DotProduct) kernel  — faithful to Zhang et al.
               - Extended     : RBF kernel                  — our adaptation

LOOCV gives an honest, unbiased performance estimate:
  train on 7 cells (or 6 for RUL), test on the held-out cell, rotate.
No cell is ever in both train and test simultaneously.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, ConstantKernel
from sklearn.metrics import r2_score

DATA = Path(__file__).parent / "data" / "ca_dataset"
OUT  = Path(__file__).parent / "output" / "new_dataset"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def zscore(X):
    mu  = X.mean(0)
    sig = X.std(0, ddof=1)
    sig = np.where(sig == 0, 1, sig)
    return (X - mu) / sig, mu, sig


def apply_norm(X, mu, sig):
    return (X - mu) / sig


# ---------------------------------------------------------------------------
# Load all cell data
# ---------------------------------------------------------------------------
ALL_CELLS = [f'CA{i}' for i in range(1, 9)]
DNF_CELLS = {'CA6'}                               # never reached 80% threshold
EOL_CELLS = [c for c in ALL_CELLS if c not in DNF_CELLS]  # 7 cells (CA3 included)

print('Loading per-cell data ...')
cell_eis    = {c: np.loadtxt(DATA / f'EIS_{c}.txt') for c in ALL_CELLS}
cell_cap    = {c: np.loadtxt(DATA / f'cap_{c}.txt') for c in ALL_CELLS}
cell_eisrul = {c: np.loadtxt(DATA / f'EIS_rul_{c}.txt') for c in EOL_CELLS}
cell_rul    = {c: np.loadtxt(DATA / f'rul_{c}.txt')     for c in EOL_CELLS}

for c in EOL_CELLS:
    eol = len(cell_rul[c]) - 1
    print(f'  {c}: {cell_eis[c].shape[0]} cap cycles, '
          f'EoL index={eol}, RUL_max={int(cell_rul[c][0])}')
for c in DNF_CELLS:
    print(f'  {c}: {cell_eis[c].shape[0]} cap cycles, DNF (no RUL)')

l_cap = 30.0   # calibrated length scale for capacity RBF

# ===========================================================================
# MODEL 1 — Capacity LOOCV  (all 8 cells)
# ===========================================================================
print('\n' + '='*60)
print('MODEL 1 — Capacity LOOCV  (all 8 cells, fixed RBF l=30)')
print('='*60)

r2_cap   = {}
pred_cap = {}
meas_cap = {}

for test_cell in ALL_CELLS:
    train_cells = [c for c in ALL_CELLS if c != test_cell]

    EIS_tr = np.vstack([cell_eis[c] for c in train_cells])
    Cap_tr = np.concatenate([cell_cap[c] for c in train_cells])
    EIS_te = cell_eis[test_cell]
    Cap_te = cell_cap[test_cell]

    # Joint normalisation — removes cell-to-cell impedance offset
    EIS_all = np.vstack([EIS_tr, EIS_te])
    _, mu_x, sig_x = zscore(EIS_all)
    X_tr_n = apply_norm(EIS_tr, mu_x, sig_x)
    X_te_n = apply_norm(EIS_te, mu_x, sig_x)

    kernel = (ConstantKernel(1.0, constant_value_bounds='fixed') *
              RBF(length_scale=l_cap, length_scale_bounds='fixed') +
              WhiteKernel(noise_level=1.0))
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                   alpha=0.1, n_restarts_optimizer=0)
    gpr.fit(X_tr_n, Cap_tr)
    Y_pred = gpr.predict(X_te_n)

    r2 = r2_score(Cap_te, Y_pred)
    r2_cap[test_cell]   = r2
    pred_cap[test_cell] = Y_pred
    meas_cap[test_cell] = Cap_te
    print(f'  {test_cell}: R² = {r2:.4f}  ({"DNF" if test_cell in DNF_CELLS else "EOL"})')

mean_cap = np.mean(list(r2_cap.values()))
print(f'  Mean R² (all 8): {mean_cap:.4f}')

# Plot — capacity LOOCV scatter per cell
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.ravel()
colors = {'CA3': 'gold', 'CA6': 'orange'}    # CA3=low cap0 anomaly, CA6=DNF
for idx, cell in enumerate(ALL_CELLS):
    ax = axes[idx]
    col = colors.get(cell, 'steelblue')
    ax.scatter(meas_cap[cell], pred_cap[cell], s=8, alpha=0.6, color=col)
    lims = [min(meas_cap[cell].min(), pred_cap[cell].min()),
            max(meas_cap[cell].max(), pred_cap[cell].max())]
    ax.plot(lims, lims, 'k--', lw=1)
    dnf_tag = ' (DNF)' if cell in DNF_CELLS else ''
    ax.set_title(f'{cell}{dnf_tag}  R²={r2_cap[cell]:.3f}', fontsize=9)
    ax.set_xlabel('Measured (mAh)', fontsize=8)
    ax.set_ylabel('Predicted (mAh)', fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle(f'Capacity LOOCV — fixed RBF l={l_cap}, joint norm  '
             f'(mean R²={mean_cap:.3f})', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'fig_cap_loocv.png', dpi=150)
plt.close()
print(f'  Saved: fig_cap_loocv.png')


# ===========================================================================
# MODEL 2 — RUL LOOCV  (6 EOL cells)  — Linear vs RBF
# ===========================================================================
print('\n' + '='*60)
print('MODEL 2 — RUL LOOCV  (7 EOL cells: linear vs RBF)')
print('='*60)

r2_lin = {}; pred_lin = {}; meas_rul = {}
r2_rbf = {}; pred_rbf = {}

for test_cell in EOL_CELLS:
    train_cells = [c for c in EOL_CELLS if c != test_cell]

    EIS_rul_tr = np.vstack([cell_eisrul[c] for c in train_cells])
    RUL_tr     = np.concatenate([cell_rul[c]    for c in train_cells])
    EIS_rul_te = cell_eisrul[test_cell]
    RUL_te     = cell_rul[test_cell]

    # Training-only normalisation for RUL (paper's approach)
    # Reference: full EIS of training cells (not just RUL cycles)
    EIS_tr_full = np.vstack([cell_eis[c] for c in train_cells])
    _, mu_x, sig_x = zscore(EIS_tr_full)
    X_tr_n = apply_norm(EIS_rul_tr, mu_x, sig_x)
    X_te_n = apply_norm(EIS_rul_te, mu_x, sig_x)

    # ---- Linear kernel (paper method) ----
    k_lin = DotProduct(sigma_0=0, sigma_0_bounds='fixed') + WhiteKernel(noise_level=1.0)
    gpr_lin = GaussianProcessRegressor(kernel=k_lin, normalize_y=False,
                                       alpha=0.4, n_restarts_optimizer=3)
    gpr_lin.fit(X_tr_n, RUL_tr)
    Y_lin = gpr_lin.predict(X_te_n)
    r2_lin[test_cell]   = r2_score(RUL_te, Y_lin)
    pred_lin[test_cell] = Y_lin
    meas_rul[test_cell] = RUL_te

    # ---- RBF kernel (extended method) ----
    k_rbf = (ConstantKernel(1.0) *
             RBF(length_scale=30.0, length_scale_bounds=(1.0, 1000.0)) +
             WhiteKernel(noise_level=1.0))
    gpr_rbf = GaussianProcessRegressor(kernel=k_rbf, normalize_y=True,
                                       alpha=0.4, n_restarts_optimizer=3)
    gpr_rbf.fit(X_tr_n, RUL_tr)
    Y_rbf = gpr_rbf.predict(X_te_n)
    r2_rbf[test_cell]   = r2_score(RUL_te, Y_rbf)
    pred_rbf[test_cell] = Y_rbf

    print(f'  {test_cell}:  Linear R²={r2_lin[test_cell]:.4f}  |  '
          f'RBF R²={r2_rbf[test_cell]:.4f}  '
          f'(EoL={len(RUL_te)-1}, RUL_max={int(RUL_te[0])})')

mean_lin = np.mean(list(r2_lin.values()))
mean_rbf = np.mean(list(r2_rbf.values()))
print(f'  Mean R²  —  Linear: {mean_lin:.4f}  |  RBF: {mean_rbf:.4f}')

# Plot — RUL scatter: linear (top row) vs RBF (bottom row)
fig, axes = plt.subplots(2, 7, figsize=(21, 7))
for col_idx, cell in enumerate(EOL_CELLS):
    rul_max = int(meas_rul[cell][0])

    # Linear
    ax = axes[0, col_idx]
    ax.scatter(meas_rul[cell], pred_lin[cell], s=8, alpha=0.7, color='tomato')
    lim = [0, rul_max * 1.05]
    ax.plot(lim, lim, 'k--', lw=1)
    ax.set_title(f'{cell}  R²={r2_lin[cell]:.3f}', fontsize=9)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Actual RUL', fontsize=8)
    if col_idx == 0:
        ax.set_ylabel('Predicted RUL\n(Linear)', fontsize=8)
    ax.grid(True, alpha=0.3)

    # RBF
    ax = axes[1, col_idx]
    ax.scatter(meas_rul[cell], pred_rbf[cell], s=8, alpha=0.7, color='seagreen')
    ax.plot(lim, lim, 'k--', lw=1)
    ax.set_title(f'{cell}  R²={r2_rbf[cell]:.3f}', fontsize=9)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Actual RUL', fontsize=8)
    if col_idx == 0:
        ax.set_ylabel('Predicted RUL\n(RBF)', fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle(f'RUL LOOCV — Linear (paper, mean R²={mean_lin:.3f})  vs  '
             f'RBF (extended, mean R²={mean_rbf:.3f})',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'fig_rul_loocv.png', dpi=150)
plt.close()
print(f'  Saved: fig_rul_loocv.png')

# Bar chart comparison
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(EOL_CELLS))
w = 0.35
bars_lin = ax.bar(x - w/2, [r2_lin[c] for c in EOL_CELLS], w,
                  label=f'Linear (paper)  mean={mean_lin:.3f}', color='tomato', alpha=0.8)
bars_rbf = ax.bar(x + w/2, [r2_rbf[c] for c in EOL_CELLS], w,
                  label=f'RBF (extended)  mean={mean_rbf:.3f}', color='seagreen', alpha=0.8)
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(EOL_CELLS)
ax.set_ylabel('R²')
ax.set_title('RUL LOOCV — Linear vs RBF kernel  (7 EOL cells, train on 6 test on 1)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUT / 'fig_rul_loocv_comparison.png', dpi=150)
plt.close()
print(f'  Saved: fig_rul_loocv_comparison.png')

# ===========================================================================
# SUMMARY
# ===========================================================================
print('\n' + '='*60)
print('LOOCV SUMMARY')
print('='*60)
print(f'\nCapacity GPR (RBF l={l_cap}, joint norm):')
for c in ALL_CELLS:
    tag = ' [DNF]' if c in DNF_CELLS else ''
    print(f'  {c}{tag}: R² = {r2_cap[c]:.4f}')
print(f'  Mean: {mean_cap:.4f}')

print(f'\nRUL GPR LOOCV (train-only norm):')
print(f'  {"Cell":<6}  {"Linear":>10}  {"RBF":>10}  {"EoL":>6}  {"RUL_max":>8}')
for c in EOL_CELLS:
    print(f'  {c:<6}  {r2_lin[c]:>10.4f}  {r2_rbf[c]:>10.4f}  '
          f'{len(meas_rul[c])-1:>6}  {int(meas_rul[c][0]):>8}')
print(f'  {"Mean":<6}  {mean_lin:>10.4f}  {mean_rbf:>10.4f}')
print(f'\n  Note: RBF kernel deviates from paper eq.(5); '
      f'linear kernel is the faithful reproduction.')
print(f'  Figures saved to: {OUT}')

# ===========================================================================
# MODEL 3 — ARD LOOCV  (fold-averaged, frequency x-axis, ±1 std bands)
# Comparison plot alongside existing single-run ARD — does NOT replace it.
# ===========================================================================
print('\n' + '='*60)
print('MODEL 3 — ARD weights across LOOCV folds (fold-averaged)')
print('='*60)

NATIVE_FREQS = np.array([
    10000.0, 7500.0, 5620.0, 4220.0, 3160.0, 2370.0, 1780.0, 1330.0,
    1000.0,  750.0,  564.0,  422.0,  316.0,  237.0,  178.0,  135.0,
    102.0,   75.0,   56.2,   42.2,   31.6,   23.7,   17.8,   13.3,
    10.0,    7.5,    5.62,   4.22,   3.16,   2.37,   1.78,   1.33,  0.999
])  # 33 freqs, high → low  (same order as feature vector)

fold_weights_re = []   # shape: (n_folds, 33)
fold_weights_im = []

for test_cell in ALL_CELLS:
    train_cells = [c for c in ALL_CELLS if c != test_cell]

    EIS_tr = np.vstack([cell_eis[c] for c in train_cells])
    Cap_tr = np.concatenate([cell_cap[c] for c in train_cells])
    EIS_te = cell_eis[test_cell]

    # Subsample every 2nd cycle to keep ARD kernel matrix tractable in memory
    EIS_tr = EIS_tr[::2]
    Cap_tr = Cap_tr[::2]

    # Joint norm (same as capacity LOOCV)
    EIS_all = np.vstack([EIS_tr, EIS_te])
    _, mu_x, sig_x = zscore(EIS_all)
    X_tr_n = apply_norm(EIS_tr, mu_x, sig_x)

    # ARD-RBF kernel (one length-scale per feature)
    k_ard = (ConstantKernel(1.0) *
             RBF(length_scale=np.ones(66)) +
             WhiteKernel(noise_level=1.0))
    gpr_ard = GaussianProcessRegressor(kernel=k_ard, normalize_y=True,
                                        alpha=0.1, n_restarts_optimizer=2)
    gpr_ard.fit(X_tr_n, Cap_tr)

    ls = gpr_ard.kernel_.k1.k2.length_scale   # 66 values
    w  = np.exp(-ls)
    w /= w.sum()

    fold_weights_re.append(w[:33])   # Re(Z) features
    fold_weights_im.append(w[33:])   # Im(Z) features
    print(f'  fold leave-{test_cell}: top Re freq={NATIVE_FREQS[np.argmax(w[:33])]:.1f} Hz  '
          f'top Im freq={NATIVE_FREQS[np.argmax(w[33:])]:.1f} Hz')

fold_weights_re = np.array(fold_weights_re)   # (8, 33)
fold_weights_im = np.array(fold_weights_im)   # (8, 33)

mean_re = fold_weights_re.mean(0)
std_re  = fold_weights_re.std(0)
mean_im = fold_weights_im.mean(0)
std_im  = fold_weights_im.std(0)

# Frequencies plotted low → high (left → right) to match reference style
freqs_plot = NATIVE_FREQS[::-1]          # 0.999 → 10000 Hz
mr = mean_re[::-1]; sr = std_re[::-1]
mi = mean_im[::-1]; si = std_im[::-1]

fig, ax = plt.subplots(figsize=(12, 4))
ax.semilogx(freqs_plot, mr, 'r-^', ms=5, lw=1.5, label='Re(Z) Ns6 (charged)')
ax.fill_between(freqs_plot, mr - sr, mr + sr, color='red', alpha=0.15)
ax.semilogx(freqs_plot, mi, color='orange', linestyle='--', marker='v',
            ms=5, lw=1.5, label='-Im(Z) Ns6 (charged)')
ax.fill_between(freqs_plot, mi - si, mi + si, color='orange', alpha=0.15)
ax.axhline(0, color='black', lw=0.6, ls='-')
ax.set_xlabel('Frequency (Hz)', fontsize=13)
ax.set_ylabel('ARD weight', fontsize=13)
ax.set_title('ARD weights across folds', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / 'fig_ARD_loocv_folds.png', dpi=150)
plt.close()
print(f'  Saved: fig_ARD_loocv_folds.png')
