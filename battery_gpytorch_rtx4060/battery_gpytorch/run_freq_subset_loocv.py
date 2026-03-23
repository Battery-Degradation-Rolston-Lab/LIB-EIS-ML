"""
Frequency-subset LOOCV — does prediction quality depend on which part of
the EIS spectrum is used?

Motivation: coupled-ARD analysis showed that ARD concentrates weight on a
narrow frequency band.  This script tests whether training on a frequency
sub-band alone matches or beats the full-spectrum model, for both:
  - Capacity GPR  (all 8 cells, fixed RBF l=30, joint norm)
  - RUL GPR       (6 EOL cells, linear/DotProduct kernel, train-only norm)

Two band-partitioning schemes are compared:
  Physics-motivated (user-specified):
    HIGH  500–10 000 Hz  : indices  0–10  (11 freqs)
    MID    10–500   Hz   : indices 11–24  (14 freqs)
    LOW     1–10    Hz   : indices 25–32  ( 8 freqs)

  Equal-count tertiles  (11 freqs each):
    HIGH  564–10 000 Hz  : indices  0–10
    MID    24–422   Hz   : indices 11–21
    LOW     1–18    Hz   : indices 22–32

Feature layout (per cell, per cycle):
  X[:, :33]  = Re(Z) at each frequency (high→low order)
  X[:, 33:]  = -Im(Z) at the same frequencies
  Subset band → take Re cols + Im cols for that index range.
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

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATA = Path(__file__).parent / "data" / "new_dataset"
OUT  = Path(__file__).parent / "output" / "new_dataset"
OUT.mkdir(parents=True, exist_ok=True)

NATIVE_FREQS = np.array([
    10000.0, 7500.0, 5620.0, 4220.0, 3160.0, 2370.0, 1780.0, 1330.0,
    1000.0,  750.0,  564.0,  422.0,  316.0,  237.0,  178.0,  135.0,
    102.0,   75.0,   56.2,   42.2,   31.6,   23.7,   17.8,   13.3,
    10.0,    7.5,    5.62,   4.22,   3.16,   2.37,   1.78,   1.33, 0.999
])  # 33 frequencies, high → low

ALL_CELLS = [f'A{i}' for i in range(1, 9)]
DNF_CELLS = {'A3', 'A6'}
EOL_CELLS = [c for c in ALL_CELLS if c not in DNF_CELLS]

L_CAP = 30.0   # calibrated RBF length-scale for capacity model

# ---------------------------------------------------------------------------
# Band definitions
#   Each entry: (label, freq_indices, description)
#   freq_indices: list/array of indices into NATIVE_FREQS (0 = 10 kHz)
# ---------------------------------------------------------------------------
idx_high_phys = np.arange(0, 11)    # 10000–564  Hz  (11 pts)
idx_mid_phys  = np.arange(11, 25)   # 422–10.0   Hz  (14 pts)
idx_low_phys  = np.arange(25, 33)   # 7.5–0.999  Hz  ( 8 pts)

idx_high_eq   = np.arange(0, 11)    # 10000–564  Hz  (11 pts)  [same as physics high]
idx_mid_eq    = np.arange(11, 22)   # 422–23.7   Hz  (11 pts)
idx_low_eq    = np.arange(22, 33)   # 17.8–0.999 Hz  (11 pts)

def _band_label(idx):
    lo, hi = NATIVE_FREQS[idx[-1]], NATIVE_FREQS[idx[0]]
    return f"{lo:.3g}–{hi:.0f} Hz  (n={len(idx)})"

BANDS = [
    ("Full",         np.arange(33),  "All 33 frequencies"),
    ("High (phys)",  idx_high_phys,  _band_label(idx_high_phys) + "  [500–10 kHz]"),
    ("Mid  (phys)",  idx_mid_phys,   _band_label(idx_mid_phys)  + "  [10–500 Hz]"),
    ("Low  (phys)",  idx_low_phys,   _band_label(idx_low_phys)  + "  [1–10 Hz]"),
    ("High (eq.)",   idx_high_eq,    _band_label(idx_high_eq)   + "  [equal-count]"),
    ("Mid  (eq.)",   idx_mid_eq,     _band_label(idx_mid_eq)    + "  [equal-count]"),
    ("Low  (eq.)",   idx_low_eq,     _band_label(idx_low_eq)    + "  [equal-count]"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def zscore(X):
    mu  = X.mean(0)
    sig = X.std(0, ddof=1)
    sig = np.where(sig == 0, 1.0, sig)
    return (X - mu) / sig, mu, sig

def apply_norm(X, mu, sig):
    return (X - mu) / sig

def select_band(X, freq_idx):
    """
    Given full 66-feature EIS matrix, return sub-matrix for freq_idx.
    Re(Z) features: X[:, freq_idx]
    Im(Z) features: X[:, 33 + freq_idx]
    """
    return np.hstack([X[:, freq_idx], X[:, 33 + freq_idx]])

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading per-cell data ...")
cell_eis    = {c: np.loadtxt(DATA / f"EIS_{c}.txt")     for c in ALL_CELLS}
cell_cap    = {c: np.loadtxt(DATA / f"cap_{c}.txt")     for c in ALL_CELLS}
cell_eisrul = {c: np.loadtxt(DATA / f"EIS_rul_{c}.txt") for c in EOL_CELLS}
cell_rul    = {c: np.loadtxt(DATA / f"rul_{c}.txt")     for c in EOL_CELLS}

for c in EOL_CELLS:
    eol = len(cell_rul[c]) - 1
    print(f"  {c}: {cell_eis[c].shape[0]} cap cycles, EoL={eol}, "
          f"RUL_max={int(cell_rul[c][0])}")
for c in DNF_CELLS:
    print(f"  {c}: {cell_eis[c].shape[0]} cap cycles, DNF (no RUL)")

# ---------------------------------------------------------------------------
# Main loop — iterate over bands
# ---------------------------------------------------------------------------
results_cap = {}   # band_name → mean R²
results_rul = {}   # band_name → mean R²
per_cell_cap = {}  # band_name → dict cell→R²
per_cell_rul = {}  # band_name → dict cell→R²

for band_name, freq_idx, band_desc in BANDS:
    print(f"\n{'='*60}")
    print(f"BAND: {band_name}   {band_desc}")
    print(f"{'='*60}")

    # ----------------------------------------------------------------
    # Capacity LOOCV  (all 8 cells, fixed RBF, joint norm)
    # ----------------------------------------------------------------
    r2_cap_fold = {}
    for test_cell in ALL_CELLS:
        train_cells = [c for c in ALL_CELLS if c != test_cell]

        X_tr_full = np.vstack([cell_eis[c] for c in train_cells])
        Cap_tr    = np.concatenate([cell_cap[c] for c in train_cells])
        X_te_full = cell_eis[test_cell]
        Cap_te    = cell_cap[test_cell]

        # Select frequency subset
        X_tr_b = select_band(X_tr_full, freq_idx)
        X_te_b = select_band(X_te_full, freq_idx)

        # Joint normalisation over train + test
        X_all  = np.vstack([X_tr_b, X_te_b])
        _, mu_x, sig_x = zscore(X_all)
        X_tr_n = apply_norm(X_tr_b, mu_x, sig_x)
        X_te_n = apply_norm(X_te_b, mu_x, sig_x)

        n_feat = X_tr_n.shape[1]
        kernel = (ConstantKernel(1.0, constant_value_bounds='fixed') *
                  RBF(length_scale=L_CAP, length_scale_bounds='fixed') +
                  WhiteKernel(noise_level=1.0))
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                       alpha=0.1, n_restarts_optimizer=0)
        gpr.fit(X_tr_n, Cap_tr)
        Y_pred = gpr.predict(X_te_n)
        r2_cap_fold[test_cell] = r2_score(Cap_te, Y_pred)

    mean_cap = np.mean(list(r2_cap_fold.values()))
    results_cap[band_name] = mean_cap
    per_cell_cap[band_name] = r2_cap_fold
    print(f"  Capacity mean R² = {mean_cap:.4f}")
    for c in ALL_CELLS:
        tag = ' [DNF]' if c in DNF_CELLS else ''
        print(f"    {c}{tag}: R²={r2_cap_fold[c]:.4f}")

    # ----------------------------------------------------------------
    # RUL LOOCV  (6 EOL cells, linear kernel, train-only norm)
    # ----------------------------------------------------------------
    r2_rul_fold = {}
    for test_cell in EOL_CELLS:
        train_cells = [c for c in EOL_CELLS if c != test_cell]

        X_rul_tr_full = np.vstack([cell_eisrul[c] for c in train_cells])
        RUL_tr        = np.concatenate([cell_rul[c] for c in train_cells])
        X_rul_te_full = cell_eisrul[test_cell]
        RUL_te        = cell_rul[test_cell]

        # Select frequency subset
        X_rul_tr_b = select_band(X_rul_tr_full, freq_idx)
        X_rul_te_b = select_band(X_rul_te_full, freq_idx)

        # Training-only normalisation using full EIS of training cells
        X_tr_full_all = np.vstack([cell_eis[c] for c in train_cells])
        X_ref_b = select_band(X_tr_full_all, freq_idx)
        _, mu_x, sig_x = zscore(X_ref_b)
        X_tr_n = apply_norm(X_rul_tr_b, mu_x, sig_x)
        X_te_n = apply_norm(X_rul_te_b, mu_x, sig_x)

        k_lin = DotProduct(sigma_0=0, sigma_0_bounds='fixed') + WhiteKernel(noise_level=1.0)
        gpr_lin = GaussianProcessRegressor(kernel=k_lin, normalize_y=False,
                                           alpha=0.4, n_restarts_optimizer=3)
        gpr_lin.fit(X_tr_n, RUL_tr)
        Y_pred = gpr_lin.predict(X_te_n)
        r2_rul_fold[test_cell] = r2_score(RUL_te, Y_pred)

    mean_rul = np.mean(list(r2_rul_fold.values()))
    results_rul[band_name] = mean_rul
    per_cell_rul[band_name] = r2_rul_fold
    print(f"  RUL    mean R² = {mean_rul:.4f}")
    for c in EOL_CELLS:
        print(f"    {c}: R²={r2_rul_fold[c]:.4f}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("SUMMARY TABLE")
print(f"{'='*60}")
hdr = f"{'Band':<16}  {'Cap R²':>8}  {'RUL R²':>8}  Freq range"
print(hdr)
print('-' * len(hdr))
for band_name, freq_idx, band_desc in BANDS:
    cap_r2 = results_cap[band_name]
    rul_r2 = results_rul[band_name]
    print(f"{band_name:<16}  {cap_r2:>8.4f}  {rul_r2:>8.4f}  {band_desc}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
band_names  = [b[0] for b in BANDS]
cap_means   = [results_cap[b] for b in band_names]
rul_means   = [results_rul[b] for b in band_names]

# ── Plot 1: Bar chart — mean R² per band, Capacity vs RUL ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x  = np.arange(len(band_names))
w  = 0.55
colors_cap = ['#2196F3' if 'Full' in n else
              '#64B5F6' if 'High' in n else
              '#1976D2' if 'Mid' in n else
              '#0D47A1' for n in band_names]
colors_rul = ['#E53935' if 'Full' in n else
              '#EF9A9A' if 'High' in n else
              '#C62828' if 'Mid' in n else
              '#B71C1C' for n in band_names]

ax = axes[0]
bars = ax.bar(x, cap_means, w, color=colors_cap, alpha=0.85, edgecolor='white')
ax.axhline(0, color='black', lw=0.7)
ax.set_xticks(x)
ax.set_xticklabels(band_names, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Mean R²', fontsize=12)
ax.set_title('Capacity LOOCV — mean R² by frequency band\n'
             '(fixed RBF l=30, joint norm, 8 cells)', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, cap_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax = axes[1]
bars = ax.bar(x, rul_means, w, color=colors_rul, alpha=0.85, edgecolor='white')
ax.axhline(0, color='black', lw=0.7)
ax.set_xticks(x)
ax.set_xticklabels(band_names, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Mean R²', fontsize=12)
ax.set_title('RUL LOOCV — mean R² by frequency band\n'
             '(linear kernel, train-only norm, 6 EOL cells)', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, rul_means):
    ypos = bar.get_height() + 0.005 if val >= 0 else bar.get_height() - 0.04
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.tight_layout()
out_path = OUT / 'fig_freq_subset_mean_r2.png'
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nSaved: {out_path}")

# ── Plot 2: Per-cell capacity R² heatmap (bands × cells) ──────────────────
fig, ax = plt.subplots(figsize=(13, 5))
mat = np.array([[per_cell_cap[bn][c] for c in ALL_CELLS] for bn in band_names])
im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
ax.set_xticks(np.arange(len(ALL_CELLS)))
ax.set_xticklabels(ALL_CELLS, fontsize=11)
ax.set_yticks(np.arange(len(band_names)))
ax.set_yticklabels(band_names, fontsize=9)
plt.colorbar(im, ax=ax, label='R²')
for i in range(len(band_names)):
    for j, c in enumerate(ALL_CELLS):
        val = mat[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=8, color='black' if 0.25 < val < 0.85 else 'white')
ax.set_title('Capacity LOOCV R² — per cell and frequency band', fontsize=12)
# Mark DNF cells
for j, c in enumerate(ALL_CELLS):
    if c in DNF_CELLS:
        ax.get_xticklabels()[j].set_color('darkorange')
ax.set_xlabel('Cell  (orange = DNF)', fontsize=10)
plt.tight_layout()
out_path = OUT / 'fig_freq_subset_cap_heatmap.png'
plt.savefig(out_path, dpi=150)
plt.close()
print(f"Saved: {out_path}")

# ── Plot 3: Per-cell RUL R² heatmap (bands × EOL cells) ───────────────────
fig, ax = plt.subplots(figsize=(12, 5))
mat_rul = np.array([[per_cell_rul[bn][c] for c in EOL_CELLS] for bn in band_names])
vmin_r = min(-0.5, mat_rul.min() - 0.05)
im = ax.imshow(mat_rul, aspect='auto', cmap='RdYlGn', vmin=vmin_r, vmax=1.0)
ax.set_xticks(np.arange(len(EOL_CELLS)))
ax.set_xticklabels(EOL_CELLS, fontsize=11)
ax.set_yticks(np.arange(len(band_names)))
ax.set_yticklabels(band_names, fontsize=9)
plt.colorbar(im, ax=ax, label='R²')
for i in range(len(band_names)):
    for j, c in enumerate(EOL_CELLS):
        val = mat_rul[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=8, color='black' if vmin_r * 0.5 < val < 0.7 else 'white')
ax.set_title('RUL LOOCV R² — per cell and frequency band\n'
             '(linear kernel; negative R² = worse than predicting mean)', fontsize=11)
ax.set_xlabel('Cell', fontsize=10)
plt.tight_layout()
out_path = OUT / 'fig_freq_subset_rul_heatmap.png'
plt.savefig(out_path, dpi=150)
plt.close()
print(f"Saved: {out_path}")

print("\nDone.")
