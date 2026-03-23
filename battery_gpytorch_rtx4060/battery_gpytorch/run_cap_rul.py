"""
Capacity-derived RUL via GPR trajectory extrapolation.

Approach
--------
Direct EIS → RUL mapping fails in LOOCV (mean R²≈-0.33) because absolute
cycle counts don't generalise across cells with different lifetimes (190–448 cycles).

Instead we use the validated capacity GPR (LOOCV mean R²=0.964):
  1. LOOCV: train capacity GPR on 7 cells (joint normalisation)
  2. Predict full capacity trajectory for the held-out cell
  3. Fit a linear degradation trend to the predicted trajectory
  4. Extrapolate to the 80% threshold → predicted EOL
  5. Derive RUL at every cycle: RUL[i] = 2*(pred_EOL - i)
  6. Compare to actual RUL for EOL cells (A1,A2,A4,A5,A7,A8)

DNF cells (A3, A6) have no EOL ground truth; predicted EOL shown as dashed
extrapolation only.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score

DATA = Path(__file__).parent / "data" / "new_dataset"
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


def predict_eol(cycles, cap_pred, eol_threshold):
    """
    Fit a linear trend to the predicted capacity trajectory and extrapolate
    to find EOL (first crossing below eol_threshold).

    Returns predicted EOL index (float) via linear interpolation, or None if
    the trend doesn't cross the threshold.
    """
    # Check if already below threshold
    below = np.where(cap_pred < eol_threshold)[0]
    if len(below) > 0:
        # Already crossed — interpolate within observed range
        idx = below[0]
        if idx == 0:
            return 0.0
        # Linear interpolation between idx-1 and idx
        c0, c1 = cap_pred[idx-1], cap_pred[idx]
        t0, t1 = cycles[idx-1], cycles[idx]
        frac = (eol_threshold - c0) / (c1 - c0)
        return float(t0 + frac * (t1 - t0))

    # Need to extrapolate — fit linear trend to last 40% of trajectory
    n = len(cycles)
    tail = max(5, int(0.4 * n))   # at least 5 points
    c_tail = cycles[-tail:]
    p_tail = cap_pred[-tail:]
    slope, intercept = np.polyfit(c_tail, p_tail, 1)

    if slope >= 0:
        return None  # capacity rising — no EOL predicted

    # Extrapolate: solve intercept + slope*t = threshold
    pred_eol = (eol_threshold - intercept) / slope
    return float(pred_eol)


# ---------------------------------------------------------------------------
# Load all cell data
# ---------------------------------------------------------------------------
ALL_CELLS = [f'A{i}' for i in range(1, 9)]
DNF_CELLS = {'A3', 'A6'}
EOL_CELLS = [c for c in ALL_CELLS if c not in DNF_CELLS]

print('Loading per-cell data ...')
cell_eis = {c: np.loadtxt(DATA / f'EIS_{c}.txt') for c in ALL_CELLS}
cell_cap = {c: np.loadtxt(DATA / f'cap_{c}.txt') for c in ALL_CELLS}
cell_rul = {c: np.loadtxt(DATA / f'rul_{c}.txt') for c in EOL_CELLS}

for c in EOL_CELLS:
    eol = len(cell_rul[c]) - 1
    print(f'  {c}: {cell_eis[c].shape[0]} cycles, '
          f'EoL index={eol}, RUL_max={int(cell_rul[c][0])} bat-cycles')
for c in DNF_CELLS:
    print(f'  {c}: {cell_eis[c].shape[0]} cycles, DNF')

l_cap = 30.0

# ===========================================================================
# LOOCV — Capacity GPR + EOL extrapolation
# ===========================================================================
print('\n' + '='*60)
print('Capacity-derived RUL  —  LOOCV  (8 cells, fixed RBF l=30)')
print('='*60)

results = {}   # per cell: pred_cap, meas_cap, pred_eol, actual_eol, rul_pred, rul_meas

for test_cell in ALL_CELLS:
    train_cells = [c for c in ALL_CELLS if c != test_cell]

    EIS_tr = np.vstack([cell_eis[c] for c in train_cells])
    Cap_tr = np.concatenate([cell_cap[c] for c in train_cells])
    EIS_te = cell_eis[test_cell]
    Cap_te = cell_cap[test_cell]

    # Joint normalisation (removes cell-to-cell impedance offset)
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
    Y_pred, Y_std = gpr.predict(X_te_n, return_std=True)

    # EOL threshold: 80% of first measured capacity
    eol_thresh = 0.80 * Cap_te[0]
    cycles = np.arange(len(Cap_te))

    pred_eol_idx = predict_eol(cycles, Y_pred, eol_thresh)

    if test_cell in EOL_CELLS:
        actual_eol_idx = len(cell_rul[test_cell]) - 1  # index where RUL=0
        rul_meas = cell_rul[test_cell]

        if pred_eol_idx is not None:
            # Derive predicted RUL at each observed EIS cycle
            # RUL[i] = 2*(EOL_index - i) — factor 2 because EIS every 2 bat-cycles
            n_obs = actual_eol_idx + 1   # same number of points as measured RUL
            idx_arr = np.arange(n_obs)
            rul_pred = 2.0 * (pred_eol_idx - idx_arr)
            rul_pred = np.clip(rul_pred, 0, None)
        else:
            rul_pred = None

        r2_cap_cell = r2_score(Cap_te, Y_pred)
        if rul_pred is not None and len(rul_pred) == len(rul_meas):
            r2_rul_cell = r2_score(rul_meas, rul_pred)
        else:
            r2_rul_cell = None

        results[test_cell] = dict(
            pred_cap=Y_pred, pred_std=Y_std, meas_cap=Cap_te,
            pred_eol=pred_eol_idx, actual_eol=actual_eol_idx,
            eol_thresh=eol_thresh, cycles=cycles,
            rul_pred=rul_pred, rul_meas=rul_meas,
            r2_cap=r2_cap_cell, r2_rul=r2_rul_cell,
        )
        r2_str = f'{r2_rul_cell:.4f}' if r2_rul_cell is not None else 'N/A'
        print(f'  {test_cell}: cap R²={r2_cap_cell:.4f}  '
              f'pred_EOL={pred_eol_idx:.1f}  actual_EOL={actual_eol_idx}  '
              f'RUL R²={r2_str}')
    else:
        # DNF — no RUL ground truth
        r2_cap_cell = r2_score(Cap_te, Y_pred)
        results[test_cell] = dict(
            pred_cap=Y_pred, pred_std=Y_std, meas_cap=Cap_te,
            pred_eol=pred_eol_idx, actual_eol=None,
            eol_thresh=eol_thresh, cycles=cycles,
            rul_pred=None, rul_meas=None,
            r2_cap=r2_cap_cell, r2_rul=None,
        )
        eol_str = f'{pred_eol_idx:.1f}' if pred_eol_idx else 'not reached'
        print(f'  {test_cell} [DNF]: cap R²={r2_cap_cell:.4f}  '
              f'pred_EOL={eol_str}  (no ground truth)')

# ===========================================================================
# FIGURE 1 — Capacity trajectories with EOL extrapolation (2×4 grid)
# ===========================================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.ravel()

for idx, cell in enumerate(ALL_CELLS):
    r = results[cell]
    ax = axes[idx]
    cyc = r['cycles']
    n = len(cyc)

    # Extend x-axis for extrapolation
    pred_eol = r['pred_eol']
    if pred_eol is not None and pred_eol > cyc[-1]:
        ext_end = int(pred_eol * 1.1) + 5
        cyc_ext = np.arange(cyc[-1] + 1, ext_end + 1)
    else:
        cyc_ext = np.array([])

    # Measured capacity
    ax.plot(cyc, r['meas_cap'], 'b-', lw=1.5, label='Measured', zorder=3)

    # Predicted capacity (observed range)
    ax.plot(cyc, r['pred_cap'], 'r-', lw=1.5, label='GPR predicted', zorder=3)
    ax.fill_between(cyc,
                    r['pred_cap'] - r['pred_std'],
                    r['pred_cap'] + r['pred_std'],
                    alpha=0.2, color='red')

    # Linear extrapolation into future
    tail = max(5, int(0.4 * n))
    slope, intercept = np.polyfit(cyc[-tail:], r['pred_cap'][-tail:], 1)

    if len(cyc_ext) > 0 and slope < 0:
        cap_ext = slope * cyc_ext + intercept
        ax.plot(cyc_ext, cap_ext, 'r--', lw=1.2, alpha=0.7, label='Extrapolation')

    # EOL threshold line
    ax.axhline(r['eol_thresh'], color='gray', lw=0.9, ls=':', label='80% threshold')

    # Mark predicted EOL
    if pred_eol is not None:
        ax.axvline(pred_eol, color='red', lw=1.2, ls='--', alpha=0.7)
        ax.text(pred_eol + 1, r['eol_thresh'] + 30,
                f'pred\nEOL={pred_eol:.0f}', fontsize=6.5, color='red')

    # Mark actual EOL (EOL cells only)
    if r['actual_eol'] is not None:
        ax.axvline(r['actual_eol'], color='navy', lw=1.2, ls='--', alpha=0.7)
        ax.text(r['actual_eol'] + 1, r['eol_thresh'] - 80,
                f'actual\nEOL={r["actual_eol"]}', fontsize=6.5, color='navy')

    dnf_tag = ' [DNF]' if cell in DNF_CELLS else ''
    r2_str  = f'R²={r["r2_cap"]:.3f}' if r['r2_cap'] is not None else ''
    ax.set_title(f'{cell}{dnf_tag}  {r2_str}', fontsize=9)
    ax.set_xlabel('EIS cycle index', fontsize=8)
    ax.set_ylabel('Capacity (mAh)', fontsize=8)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=6.5)

fig.suptitle(f'Capacity-derived RUL — GPR trajectory + linear extrapolation to 80% EOL\n'
             f'LOOCV: train on 7 cells, test on 1  (fixed RBF l={l_cap}, joint norm)',
             fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'fig_cap_rul_trajectories.png', dpi=150)
plt.close()
print(f'\n  Saved: fig_cap_rul_trajectories.png')

# ===========================================================================
# FIGURE 2 — Predicted vs Actual RUL scatter (EOL cells only)
# ===========================================================================
eol_cells_with_rul = [c for c in EOL_CELLS if results[c]['rul_pred'] is not None
                      and results[c]['r2_rul'] is not None]

fig, axes = plt.subplots(1, len(eol_cells_with_rul),
                         figsize=(4 * len(eol_cells_with_rul), 4))
if len(eol_cells_with_rul) == 1:
    axes = [axes]

for ax, cell in zip(axes, eol_cells_with_rul):
    r = results[cell]
    rul_m = r['rul_meas']
    rul_p = r['rul_pred']
    n = min(len(rul_m), len(rul_p))
    rul_m, rul_p = rul_m[:n], rul_p[:n]

    ax.scatter(rul_m, rul_p, s=12, alpha=0.7, color='darkorange')
    lim = [0, max(rul_m.max(), rul_p.max()) * 1.05]
    ax.plot(lim, lim, 'k--', lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f'{cell}  R²={r["r2_rul"]:.3f}', fontsize=9)
    ax.set_xlabel('Actual RUL (bat-cycles)', fontsize=8)
    ax.set_ylabel('Predicted RUL (bat-cycles)', fontsize=8)
    ax.grid(True, alpha=0.3)

valid_r2 = [results[c]['r2_rul'] for c in eol_cells_with_rul]
mean_r2_rul = np.mean(valid_r2)
fig.suptitle(f'Capacity-derived RUL — Predicted vs Actual  (mean R²={mean_r2_rul:.3f})',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'fig_cap_rul_scatter.png', dpi=150)
plt.close()
print(f'  Saved: fig_cap_rul_scatter.png')

# ===========================================================================
# SUMMARY
# ===========================================================================
print('\n' + '='*60)
print('CAPACITY-DERIVED RUL SUMMARY')
print('='*60)
print(f'\n{"Cell":<6}  {"Cap R²":>8}  {"RUL R²":>8}  '
      f'{"pred EOL":>10}  {"actual EOL":>10}')
print('-' * 50)
for c in ALL_CELLS:
    r = results[c]
    cap_r2 = f'{r["r2_cap"]:.4f}' if r['r2_cap'] is not None else '  N/A'
    rul_r2 = f'{r["r2_rul"]:.4f}' if r['r2_rul'] is not None else '  N/A'
    peol   = f'{r["pred_eol"]:.1f}' if r['pred_eol'] is not None else '  N/A'
    aeol   = str(r['actual_eol']) if r['actual_eol'] is not None else 'DNF'
    tag    = ' [DNF]' if c in DNF_CELLS else ''
    print(f'{c}{tag:<8}  {cap_r2:>8}  {rul_r2:>8}  {peol:>10}  {aeol:>10}')

r2_cap_all = np.mean([results[c]['r2_cap'] for c in ALL_CELLS])
print(f'\n  Mean capacity R² (all 8 cells): {r2_cap_all:.4f}')
if valid_r2:
    print(f'  Mean RUL R²     ({len(valid_r2)} EOL cells):   {mean_r2_rul:.4f}')
print(f'\n  Figures saved to: {OUT}')
