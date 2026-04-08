"""
Capacity-derived RUL via GPR trajectory extrapolation.

Approach
--------
Direct EIS -> RUL mapping fails because absolute cycle counts don't generalise
across cells with different lifetimes. Instead we use validated capacity GPR:
  1. Train capacity GPR (LOOCV for RT, fixed DOE for cold)
  2. Predict full capacity trajectory for the held-out cell
  3. Fit a linear degradation trend to the predicted trajectory
  4. Extrapolate to the 80% threshold -> predicted EOL
  5. Derive RUL at every cycle: RUL[i] = pred_EOL - i
  6. Compare to actual RUL for EOL cells

Temperature groups
------------------
  RT (~25C)  : CA1-CA8, LOOCV (8 folds), fixed RBF l=30, joint norm
  -10C       : N10_CB1-4, LOOCV (4 folds) + fixed DOE, Coupled ARD-RBF, joint norm
  -20C       : N20_CB1-4, LOOCV (4 folds) + fixed DOE, Coupled ARD-RBF, joint norm
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
    RBF, WhiteKernel, ConstantKernel, Kernel, Hyperparameter
)
from sklearn.metrics import r2_score

SCRIPT_DIR = Path(__file__).parent
CA_DATA    = SCRIPT_DIR / 'data' / 'ca_dataset'
MT_DATA    = SCRIPT_DIR / 'data' / 'multitemp_dataset'
OUT        = SCRIPT_DIR / 'output' / 'cap_rul'
OUT.mkdir(parents=True, exist_ok=True)

NATIVE_FREQS = np.array([
    10000.0, 7500.0, 5620.0, 4220.0, 3160.0, 2370.0, 1780.0, 1330.0,
    1000.0,  750.0,  564.0,  422.0,  316.0,  237.0,  178.0,  135.0,
    102.0,   75.0,   56.2,   42.2,   31.6,   23.7,   17.8,   13.3,
    10.0,    7.5,    5.62,   4.22,   3.16,   2.37,   1.78,   1.33, 0.999
])
N_FREQ = len(NATIVE_FREQS)  # 33

np.random.seed(42)
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Coupled ARD-RBF kernel (one length-scale per frequency, shared Re+Im)
# ---------------------------------------------------------------------------

class CoupledARDRBF(Kernel):
    def __init__(self, length_scale=None, length_scale_bounds=(1e-5, 1e5)):
        if length_scale is None:
            length_scale = np.ones(N_FREQ)
        self.length_scale        = np.asarray(length_scale, dtype=float)
        self.length_scale_bounds = length_scale_bounds

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter('length_scale', 'numeric',
                              self.length_scale_bounds, n_elements=N_FREQ)

    def __call__(self, X, Y=None, eval_gradient=False):
        X_re = X[:, :N_FREQ];  X_im = X[:, N_FREQ:]
        if Y is None:
            Y_re, Y_im = X_re, X_im
        else:
            Y_re, Y_im = Y[:, :N_FREQ], Y[:, N_FREQ:]
        ls = self.length_scale
        N, M = X_re.shape[0], Y_re.shape[0]
        dist2 = np.zeros((N, M))
        for i in range(N_FREQ):
            dr = X_re[:, i:i+1] - Y_re[:, i]
            di = X_im[:, i:i+1] - Y_im[:, i]
            dist2 += (dr**2 + di**2) / (2.0 * ls[i]**2)
        K = np.exp(-dist2)
        if eval_gradient:
            dK = np.empty((N, M, N_FREQ))
            for i in range(N_FREQ):
                dr = X_re[:, i:i+1] - Y_re[:, i]
                di = X_im[:, i:i+1] - Y_im[:, i]
                dK[:, :, i] = K * (dr**2 + di**2) / ls[i]**2
            return K, dK
        return K

    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return True

    def __repr__(self):
        return f'CoupledARDRBF(ls={self.length_scale.round(3)})'


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

def ard_weights(ls):
    w = np.exp(-ls);  return w / w.sum()

def predict_eol(cycles, cap_pred, eol_threshold):
    """
    Fit a linear trend to the predicted capacity trajectory and extrapolate
    to find EOL (first crossing below eol_threshold).
    """
    below = np.where(cap_pred < eol_threshold)[0]
    if len(below) > 0:
        idx = below[0]
        if idx == 0:
            return 0.0
        c0, c1 = cap_pred[idx-1], cap_pred[idx]
        t0, t1 = cycles[idx-1], cycles[idx]
        frac = (eol_threshold - c0) / (c1 - c0)
        return float(t0 + frac * (t1 - t0))

    # Extrapolate — fit linear trend to last 40% of trajectory
    n = len(cycles)
    tail = max(5, int(0.4 * n))
    c_tail = cycles[-tail:]
    p_tail = cap_pred[-tail:]
    slope, intercept = np.polyfit(c_tail, p_tail, 1)

    if slope >= 0:
        return None  # capacity rising — no EOL predicted

    pred_eol = (eol_threshold - intercept) / slope
    return float(pred_eol)


# ===========================================================================
# SECTION 1 — RT (CA1-CA8) LOOCV  (fixed RBF l=30, joint norm)
# ===========================================================================

ALL_CA   = [f'CA{i}' for i in range(1, 9)]
DNF_CA   = {'CA6'}
EOL_CA   = [c for c in ALL_CA if c not in DNF_CA]

print('Loading CA dataset (RT) ...')
ca_eis = {c: np.loadtxt(CA_DATA / f'EIS_{c}.txt') for c in ALL_CA}
ca_cap = {c: np.loadtxt(CA_DATA / f'cap_{c}.txt') for c in ALL_CA}
ca_rul = {c: np.loadtxt(CA_DATA / f'rul_{c}.txt') for c in EOL_CA}

for c in EOL_CA:
    print(f'  {c}: {ca_eis[c].shape[0]} cycles, '
          f'EoL={len(ca_rul[c])-1}, RUL_max={int(ca_rul[c][0])}')
for c in DNF_CA:
    print(f'  {c}: {ca_eis[c].shape[0]} cycles, DNF')

l_cap_rt = 30.0

print('\n' + '='*60)
print('SECTION 1 — RT Capacity-derived RUL (LOOCV, fixed RBF l=30)')
print('='*60)

rt_results = {}

for test_cell in ALL_CA:
    train_cells = [c for c in ALL_CA if c != test_cell]

    EIS_tr = np.vstack([ca_eis[c] for c in train_cells])
    Cap_tr = np.concatenate([ca_cap[c] for c in train_cells])
    EIS_te = ca_eis[test_cell]
    Cap_te = ca_cap[test_cell]

    # Joint normalisation (removes cell-to-cell impedance offset)
    EIS_all = np.vstack([EIS_tr, EIS_te])
    _, mu_x, sig_x = zscore(EIS_all)
    X_tr_n = apply_norm(EIS_tr, mu_x, sig_x)
    X_te_n = apply_norm(EIS_te, mu_x, sig_x)

    kernel = (ConstantKernel(1.0, constant_value_bounds='fixed') *
              RBF(length_scale=l_cap_rt, length_scale_bounds='fixed') +
              WhiteKernel(noise_level=1.0))
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                   alpha=0.1, n_restarts_optimizer=0)
    gpr.fit(X_tr_n, Cap_tr)
    Y_pred, Y_std = gpr.predict(X_te_n, return_std=True)

    eol_thresh = 0.80 * Cap_te[0]
    cycles = np.arange(len(Cap_te))
    pred_eol_idx = predict_eol(cycles, Y_pred, eol_thresh)

    if test_cell in EOL_CA:
        actual_eol_idx = len(ca_rul[test_cell]) - 1
        rul_meas = ca_rul[test_cell]

        if pred_eol_idx is not None:
            n_obs = actual_eol_idx + 1
            idx_arr = np.arange(n_obs)
            rul_pred = pred_eol_idx - idx_arr   # RUL factor=1 for CA
            rul_pred = np.clip(rul_pred, 0, None)
        else:
            rul_pred = None

        r2_cap_cell = r2_score(Cap_te, Y_pred)
        if rul_pred is not None and len(rul_pred) == len(rul_meas):
            r2_rul_cell = r2_score(rul_meas, rul_pred)
        else:
            r2_rul_cell = None

        rt_results[test_cell] = dict(
            pred_cap=Y_pred, pred_std=Y_std, meas_cap=Cap_te,
            pred_eol=pred_eol_idx, actual_eol=actual_eol_idx,
            eol_thresh=eol_thresh, cycles=cycles,
            rul_pred=rul_pred, rul_meas=rul_meas,
            r2_cap=r2_cap_cell, r2_rul=r2_rul_cell,
        )
        r2_str = f'{r2_rul_cell:.4f}' if r2_rul_cell is not None else 'N/A'
        print(f'  {test_cell}: cap R2={r2_cap_cell:.4f}  '
              f'pred_EOL={pred_eol_idx:.1f}  actual_EOL={actual_eol_idx}  '
              f'RUL R2={r2_str}')
    else:
        r2_cap_cell = r2_score(Cap_te, Y_pred)
        rt_results[test_cell] = dict(
            pred_cap=Y_pred, pred_std=Y_std, meas_cap=Cap_te,
            pred_eol=pred_eol_idx, actual_eol=None,
            eol_thresh=eol_thresh, cycles=cycles,
            rul_pred=None, rul_meas=None,
            r2_cap=r2_cap_cell, r2_rul=None,
        )
        eol_str = f'{pred_eol_idx:.1f}' if pred_eol_idx else 'not reached'
        print(f'  {test_cell} [DNF]: cap R2={r2_cap_cell:.4f}  '
              f'pred_EOL={eol_str}  (no ground truth)')


# ===========================================================================
# SECTION 2 — Cold temperatures: -10C and -20C (fixed DOE, Coupled ARD-RBF)
# ===========================================================================

COLD_GROUPS = {
    'N10': {
        'label'     : '-10C',
        'color'     : '#ff7f0e',
        'train'     : ['N10_CB1', 'N10_CB2', 'N10_CB3'],
        'test'      : ['N10_CB4'],
    },
    'N20': {
        'label'     : '-20C',
        'color'     : '#d62728',
        'train'     : ['N20_CB1', 'N20_CB2', 'N20_CB3'],
        'test'      : ['N20_CB4'],
    },
}

print('\nLoading multi-temp dataset (cold cells) ...')
cold_cells_all = []
for G in COLD_GROUPS.values():
    cold_cells_all.extend(G['train'] + G['test'])

mt_eis = {c: np.loadtxt(MT_DATA / f'EIS_{c}.txt') for c in cold_cells_all}
mt_cap = {c: np.loadtxt(MT_DATA / f'cap_{c}.txt') for c in cold_cells_all}
mt_rul = {c: np.loadtxt(MT_DATA / f'rul_{c}.txt') for c in cold_cells_all}

for c in cold_cells_all:
    print(f'  {c}: {mt_eis[c].shape[0]} cap cycles, '
          f'EoL={len(mt_rul[c])-1}, RUL_max={int(mt_rul[c][0])}')

cold_results = {}   # gkey -> {cell: result_dict}

for gkey, G in COLD_GROUPS.items():
    label = G['label']
    train_cells = G['train']
    test_cells  = G['test']

    print(f'\n{"="*60}')
    print(f'SECTION 2 — {label} Capacity-derived RUL (Coupled ARD-RBF)')
    print(f'  Train: {train_cells}  |  Test: {test_cells}')
    print(f'{"="*60}')

    EIS_tr = np.vstack([mt_eis[c] for c in train_cells])
    Cap_tr = np.concatenate([mt_cap[c] for c in train_cells])
    EIS_te_all = np.vstack([mt_eis[c] for c in test_cells])

    # Joint normalisation (same as RT capacity LOOCV — removes cell-to-cell offset)
    EIS_pool = np.vstack([EIS_tr, EIS_te_all])
    _, mu_x, sig_x = zscore(EIS_pool)
    X_tr_n = apply_norm(EIS_tr, mu_x, sig_x)

    # Coupled ARD-RBF capacity model
    kernel = (ConstantKernel(1.0) *
              CoupledARDRBF(length_scale=np.ones(N_FREQ)) +
              WhiteKernel(noise_level=1.0))
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                   alpha=0.1, n_restarts_optimizer=10)
    print(f'  Fitting Coupled ARD-RBF on {len(X_tr_n)} rows ...')
    gpr.fit(X_tr_n, Cap_tr)

    ls_cap = gpr.kernel_.k1.k2.length_scale
    w_cap  = ard_weights(ls_cap)
    top_idx = np.argmax(w_cap)
    print(f'  Top frequency: {NATIVE_FREQS[top_idx]:.2f} Hz  (w={w_cap[top_idx]:.4f})')

    cold_results[gkey] = {}

    for cell in test_cells:
        EIS_te = mt_eis[cell]
        Cap_te = mt_cap[cell]
        X_te_n = apply_norm(EIS_te, mu_x, sig_x)
        Y_pred = gpr.predict(X_te_n)

        eol_thresh = 0.80 * Cap_te[0]
        cycles = np.arange(len(Cap_te))
        pred_eol_idx = predict_eol(cycles, Y_pred, eol_thresh)

        actual_eol_idx = len(mt_rul[cell]) - 1
        rul_meas = mt_rul[cell]

        if pred_eol_idx is not None:
            n_obs = actual_eol_idx + 1
            idx_arr = np.arange(n_obs)
            rul_pred = pred_eol_idx - idx_arr   # RUL factor=1 for CB
            rul_pred = np.clip(rul_pred, 0, None)
        else:
            rul_pred = None

        r2_cap_cell = r2_score(Cap_te, Y_pred)
        if rul_pred is not None and len(rul_pred) == len(rul_meas):
            r2_rul_cell = r2_score(rul_meas, rul_pred)
        else:
            r2_rul_cell = None

        cold_results[gkey][cell] = dict(
            pred_cap=Y_pred, pred_std=None, meas_cap=Cap_te,
            pred_eol=pred_eol_idx, actual_eol=actual_eol_idx,
            eol_thresh=eol_thresh, cycles=cycles,
            rul_pred=rul_pred, rul_meas=rul_meas,
            r2_cap=r2_cap_cell, r2_rul=r2_rul_cell,
            ard_weights=w_cap, ard_freqs=NATIVE_FREQS,
        )

        r2_str = f'{r2_rul_cell:.4f}' if r2_rul_cell is not None else 'N/A'
        peol = f'{pred_eol_idx:.1f}' if pred_eol_idx is not None else 'N/A'
        print(f'  {cell}: cap R2={r2_cap_cell:.4f}  '
              f'pred_EOL={peol}  actual_EOL={actual_eol_idx}  '
              f'RUL R2={r2_str}')


# ===========================================================================
# SECTION 3 — Cold LOOCV (4-fold per temperature group)
# ===========================================================================

COLD_LOOCV = {
    'N10': {
        'label': '-10C',
        'cells': ['N10_CB1', 'N10_CB2', 'N10_CB3', 'N10_CB4'],
        'color': '#ff7f0e',
    },
    'N20': {
        'label': '-20C',
        'cells': ['N20_CB1', 'N20_CB2', 'N20_CB3', 'N20_CB4'],
        'color': '#d62728',
    },
}

loocv_results = {}  # gkey -> {cell: result_dict}

for gkey, G in COLD_LOOCV.items():
    label = G['label']
    all_cells = G['cells']

    print(f'\n{"="*60}')
    print(f'SECTION 3 — {label} Capacity-derived RUL (LOOCV, 4-fold)')
    print(f'{"="*60}')

    loocv_results[gkey] = {}

    for test_cell in all_cells:
        train_cells = [c for c in all_cells if c != test_cell]

        EIS_tr = np.vstack([mt_eis[c] for c in train_cells])
        Cap_tr = np.concatenate([mt_cap[c] for c in train_cells])
        EIS_te = mt_eis[test_cell]
        Cap_te = mt_cap[test_cell]

        # Joint normalisation
        EIS_pool = np.vstack([EIS_tr, EIS_te])
        _, mu_x, sig_x = zscore(EIS_pool)
        X_tr_n = apply_norm(EIS_tr, mu_x, sig_x)
        X_te_n = apply_norm(EIS_te, mu_x, sig_x)

        kernel = (ConstantKernel(1.0) *
                  CoupledARDRBF(length_scale=np.ones(N_FREQ)) +
                  WhiteKernel(noise_level=1.0))
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                       alpha=0.1, n_restarts_optimizer=10)
        gpr.fit(X_tr_n, Cap_tr)
        Y_pred = gpr.predict(X_te_n)

        eol_thresh = 0.80 * Cap_te[0]
        cycles = np.arange(len(Cap_te))
        pred_eol_idx = predict_eol(cycles, Y_pred, eol_thresh)

        actual_eol_idx = len(mt_rul[test_cell]) - 1
        rul_meas = mt_rul[test_cell]

        if pred_eol_idx is not None:
            n_obs = actual_eol_idx + 1
            idx_arr = np.arange(n_obs)
            rul_pred = pred_eol_idx - idx_arr
            rul_pred = np.clip(rul_pred, 0, None)
        else:
            rul_pred = None

        r2_cap_cell = r2_score(Cap_te, Y_pred)
        if rul_pred is not None and len(rul_pred) == len(rul_meas):
            r2_rul_cell = r2_score(rul_meas, rul_pred)
        else:
            r2_rul_cell = None

        loocv_results[gkey][test_cell] = dict(
            pred_cap=Y_pred, meas_cap=Cap_te,
            pred_eol=pred_eol_idx, actual_eol=actual_eol_idx,
            eol_thresh=eol_thresh, cycles=cycles,
            rul_pred=rul_pred, rul_meas=rul_meas,
            r2_cap=r2_cap_cell, r2_rul=r2_rul_cell,
        )

        r2_str = f'{r2_rul_cell:.4f}' if r2_rul_cell is not None else 'N/A'
        peol = f'{pred_eol_idx:.1f}' if pred_eol_idx is not None else 'N/A'
        print(f'  {test_cell}: cap R2={r2_cap_cell:.4f}  '
              f'pred_EOL={peol}  actual_EOL={actual_eol_idx}  '
              f'RUL R2={r2_str}')

    # Per-group LOOCV mean
    valid = [loocv_results[gkey][c]['r2_rul'] for c in all_cells
             if loocv_results[gkey][c]['r2_rul'] is not None]
    if valid:
        print(f'  Mean RUL R2 ({len(valid)} cells): {np.mean(valid):.4f}')
    valid_cap = [loocv_results[gkey][c]['r2_cap'] for c in all_cells]
    print(f'  Mean cap R2 ({len(valid_cap)} cells): {np.mean(valid_cap):.4f}')


# ===========================================================================
# SECTION 4 — Direct EIS→RUL (Linear kernel) for comparison
# ===========================================================================
# Reference results from run_ca_zhang.py (Experiment 6)
# These use the same cells/splits but predict RUL directly from EIS

DIRECT_RUL_RESULTS = {
    'RT':   {'r2': -4.3,   'note': 'CA1-5 train, CA7-8 test'},
    '-10C': {'r2':  0.734, 'note': 'N10_CB1-3 train, N10_CB4 test'},
    '-20C': {'r2':  0.459, 'note': 'N20_CB1-3 train, N20_CB4 test'},
}

print(f'\n{"="*60}')
print('SECTION 4 — Direct EIS->RUL reference (from run_ca_zhang.py)')
print(f'{"="*60}')
for grp, d in DIRECT_RUL_RESULTS.items():
    print(f'  {grp}: R2={d["r2"]:.3f}  ({d["note"]})')


# ===========================================================================
# FIGURES
# ===========================================================================

# ── Fig 1: RT capacity trajectories with EOL extrapolation (2x4 grid) ────

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.ravel()

for idx, cell in enumerate(ALL_CA):
    r = rt_results[cell]
    ax = axes[idx]
    cyc = r['cycles']
    n = len(cyc)

    pred_eol = r['pred_eol']
    if pred_eol is not None and pred_eol > cyc[-1]:
        ext_end = int(pred_eol * 1.1) + 5
        cyc_ext = np.arange(cyc[-1] + 1, ext_end + 1)
    else:
        cyc_ext = np.array([])

    ax.plot(cyc, r['meas_cap'], 'b-', lw=1.5, label='Measured', zorder=3)
    ax.plot(cyc, r['pred_cap'], 'r-', lw=1.5, label='GPR predicted', zorder=3)
    ax.fill_between(cyc,
                    r['pred_cap'] - r['pred_std'],
                    r['pred_cap'] + r['pred_std'],
                    alpha=0.2, color='red')

    tail = max(5, int(0.4 * n))
    slope, intercept = np.polyfit(cyc[-tail:], r['pred_cap'][-tail:], 1)

    if len(cyc_ext) > 0 and slope < 0:
        cap_ext = slope * cyc_ext + intercept
        ax.plot(cyc_ext, cap_ext, 'r--', lw=1.2, alpha=0.7, label='Extrapolation')

    ax.axhline(r['eol_thresh'], color='gray', lw=0.9, ls=':', label='80% threshold')

    if pred_eol is not None:
        ax.axvline(pred_eol, color='red', lw=1.2, ls='--', alpha=0.7)
        ax.text(pred_eol + 1, r['eol_thresh'] + 30,
                f'pred\nEOL={pred_eol:.0f}', fontsize=6.5, color='red')

    if r['actual_eol'] is not None:
        ax.axvline(r['actual_eol'], color='navy', lw=1.2, ls='--', alpha=0.7)
        ax.text(r['actual_eol'] + 1, r['eol_thresh'] - 80,
                f'actual\nEOL={r["actual_eol"]}', fontsize=6.5, color='navy')

    dnf_tag = ' [DNF]' if cell in DNF_CA else ''
    r2_str  = f'R2={r["r2_cap"]:.3f}'
    ax.set_title(f'{cell}{dnf_tag}  {r2_str}', fontsize=9)
    ax.set_xlabel('EIS cycle index', fontsize=8)
    ax.set_ylabel('Capacity (mAh)', fontsize=8)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=6.5)

fig.suptitle('RT Capacity-derived RUL — GPR trajectory + linear extrapolation to 80% EOL\n'
             f'LOOCV: train on 7 cells, test on 1  (fixed RBF l={l_cap_rt}, joint norm)',
             fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'fig_rt_cap_rul_trajectories.png', dpi=150)
plt.close()
print(f'\n  Saved: fig_rt_cap_rul_trajectories.png')

# ── Fig 2: RT predicted vs actual RUL scatter ────────────────────────────

eol_with_rul = [c for c in EOL_CA if rt_results[c]['rul_pred'] is not None
                and rt_results[c]['r2_rul'] is not None]

fig, axes = plt.subplots(1, len(eol_with_rul),
                         figsize=(4 * len(eol_with_rul), 4))
if len(eol_with_rul) == 1:
    axes = [axes]

for ax, cell in zip(axes, eol_with_rul):
    r = rt_results[cell]
    rul_m = r['rul_meas']
    rul_p = r['rul_pred']
    n = min(len(rul_m), len(rul_p))
    rul_m, rul_p = rul_m[:n], rul_p[:n]

    ax.scatter(rul_m, rul_p, s=12, alpha=0.7, color='#1f77b4')
    lim = [0, max(rul_m.max(), rul_p.max()) * 1.05]
    ax.plot(lim, lim, 'k--', lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f'{cell}  R2={r["r2_rul"]:.3f}', fontsize=9)
    ax.set_xlabel('Actual RUL (cycles)', fontsize=8)
    ax.set_ylabel('Predicted RUL (cycles)', fontsize=8)
    ax.grid(True, alpha=0.3)

valid_r2_rt = [rt_results[c]['r2_rul'] for c in eol_with_rul]
mean_r2_rt = np.mean(valid_r2_rt)
fig.suptitle(f'RT Capacity-derived RUL — Predicted vs Actual  (mean R2={mean_r2_rt:.3f})',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'fig_rt_cap_rul_scatter.png', dpi=150)
plt.close()
print(f'  Saved: fig_rt_cap_rul_scatter.png')


# ── Fig 3: Cold capacity trajectories + EOL extrapolation ────────────────

for gkey, G in COLD_GROUPS.items():
    label = G['label'];  col = G['color']
    test_cells = G['test']

    n_test = len(test_cells)
    fig, axes = plt.subplots(1, n_test, figsize=(6 * n_test, 5), squeeze=False)
    axes = axes[0]

    for idx2, cell in enumerate(test_cells):
        r = cold_results[gkey][cell]
        ax = axes[idx2]
        cyc = r['cycles']
        n = len(cyc)
        Cap_te = r['meas_cap']
        Y_pred = r['pred_cap']

        pred_eol = r['pred_eol']
        if pred_eol is not None and pred_eol > cyc[-1]:
            ext_end = int(pred_eol * 1.1) + 5
            cyc_ext = np.arange(cyc[-1] + 1, ext_end + 1)
        else:
            cyc_ext = np.array([])

        ax.plot(cyc, Cap_te, 'x', color='grey', ms=3, alpha=0.6, label='Measured')
        ax.plot(cyc, Y_pred, '-', color=col, lw=1.8, label='GPR (Coupled ARD)')

        tail = max(5, int(0.4 * n))
        slope, intercept = np.polyfit(cyc[-tail:], Y_pred[-tail:], 1)

        if len(cyc_ext) > 0 and slope < 0:
            cap_ext = slope * cyc_ext + intercept
            ax.plot(cyc_ext, cap_ext, '--', color=col, lw=1.2, alpha=0.7,
                    label='Extrapolation')

        ax.axhline(r['eol_thresh'], color='gray', lw=0.9, ls=':', label='80% threshold')

        if pred_eol is not None:
            ax.axvline(pred_eol, color='red', lw=1.2, ls='--', alpha=0.7)
            ax.text(pred_eol + 1, r['eol_thresh'] + 30,
                    f'pred EOL={pred_eol:.0f}', fontsize=7, color='red')

        if r['actual_eol'] is not None:
            ax.axvline(r['actual_eol'], color='navy', lw=1.2, ls='--', alpha=0.7)
            ax.text(r['actual_eol'] + 1, r['eol_thresh'] - 80,
                    f'actual EOL={r["actual_eol"]}', fontsize=7, color='navy')

        r2c = r['r2_cap'];  r2r = r['r2_rul']
        r2r_str = f'RUL R2={r2r:.3f}' if r2r is not None else 'RUL N/A'
        ax.set_title(f'{cell} ({label})\nCap R2={r2c:.3f}  |  {r2r_str}', fontsize=10)
        ax.set_xlabel('Cycle', fontsize=9)
        ax.set_ylabel('Capacity (mAh)', fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{label} Capacity-derived RUL — Coupled ARD-RBF + extrapolation\n'
                 f'Train: {G["train"]}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    tag = gkey.lower()
    plt.savefig(OUT / f'fig_{tag}_cap_rul_trajectories.png', dpi=150)
    plt.close()
    print(f'  Saved: fig_{tag}_cap_rul_trajectories.png')

# ── Fig 4: Cold LOOCV capacity trajectories ──────────────────────────────

for gkey, G in COLD_LOOCV.items():
    label = G['label'];  col = G['color']
    all_cells = G['cells']

    n_cells = len(all_cells)
    fig, axes = plt.subplots(1, n_cells, figsize=(5 * n_cells, 4.5), squeeze=False)
    axes = axes[0]

    for idx2, cell in enumerate(all_cells):
        r = loocv_results[gkey][cell]
        ax = axes[idx2]
        cyc = r['cycles']
        n = len(cyc)
        Cap_te = r['meas_cap']
        Y_pred = r['pred_cap']

        pred_eol = r['pred_eol']
        if pred_eol is not None and pred_eol > cyc[-1]:
            ext_end = int(pred_eol * 1.1) + 5
            cyc_ext = np.arange(cyc[-1] + 1, ext_end + 1)
        else:
            cyc_ext = np.array([])

        ax.plot(cyc, Cap_te, 'x', color='grey', ms=3, alpha=0.6, label='Measured')
        ax.plot(cyc, Y_pred, '-', color=col, lw=1.8, label='GPR (Coupled ARD)')

        tail = max(5, int(0.4 * n))
        slope, intercept = np.polyfit(cyc[-tail:], Y_pred[-tail:], 1)

        if len(cyc_ext) > 0 and slope < 0:
            cap_ext = slope * cyc_ext + intercept
            ax.plot(cyc_ext, cap_ext, '--', color=col, lw=1.2, alpha=0.7,
                    label='Extrapolation')

        ax.axhline(r['eol_thresh'], color='gray', lw=0.9, ls=':', label='80% threshold')

        if pred_eol is not None:
            ax.axvline(pred_eol, color='red', lw=1.2, ls='--', alpha=0.7)
            ax.text(pred_eol + 0.5, r['eol_thresh'] + 30,
                    f'pred EOL={pred_eol:.0f}', fontsize=6.5, color='red')

        if r['actual_eol'] is not None:
            ax.axvline(r['actual_eol'], color='navy', lw=1.2, ls='--', alpha=0.7)
            ax.text(r['actual_eol'] + 0.5, r['eol_thresh'] - 80,
                    f'actual EOL={r["actual_eol"]}', fontsize=6.5, color='navy')

        r2c = r['r2_cap'];  r2r = r['r2_rul']
        r2r_str = f'RUL R2={r2r:.3f}' if r2r is not None else 'RUL N/A'
        ax.set_title(f'{cell}\nCap R2={r2c:.3f}  |  {r2r_str}', fontsize=9)
        ax.set_xlabel('Cycle', fontsize=8)
        ax.set_ylabel('Capacity (mAh)', fontsize=8)
        ax.grid(True, alpha=0.3)
        if idx2 == 0:
            ax.legend(fontsize=6.5, frameon=False)

    fig.suptitle(f'{label} Capacity-derived RUL — LOOCV (4-fold, Coupled ARD-RBF)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    tag = gkey.lower()
    plt.savefig(OUT / f'fig_{tag}_loocv_cap_rul_trajectories.png', dpi=150)
    plt.close()
    print(f'  Saved: fig_{tag}_loocv_cap_rul_trajectories.png')

# ── Fig 5: Cold RUL scatter (fixed DOE) ─────────────────────────────────

for gkey, G in COLD_GROUPS.items():
    label = G['label'];  col = G['color']
    test_cells = G['test']

    cells_with_rul = [c for c in test_cells
                      if cold_results[gkey][c]['rul_pred'] is not None
                      and cold_results[gkey][c]['r2_rul'] is not None]
    if not cells_with_rul:
        print(f'  {label}: no valid RUL predictions — skipping scatter')
        continue

    n_test = len(cells_with_rul)
    fig, axes = plt.subplots(1, n_test, figsize=(5 * n_test, 4.5), squeeze=False)
    axes = axes[0]

    for idx2, cell in enumerate(cells_with_rul):
        r = cold_results[gkey][cell]
        ax = axes[idx2]
        rul_m = r['rul_meas']
        rul_p = r['rul_pred']
        n = min(len(rul_m), len(rul_p))
        rul_m, rul_p = rul_m[:n], rul_p[:n]

        ax.scatter(rul_m, rul_p, s=15, alpha=0.7, color=col)
        lim = [0, max(rul_m.max(), rul_p.max()) * 1.05]
        ax.plot(lim, lim, 'k--', lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(f'{cell}  R2={r["r2_rul"]:.3f}', fontsize=10)
        ax.set_xlabel('Actual RUL (cycles)', fontsize=9)
        ax.set_ylabel('Predicted RUL (cycles)', fontsize=9)
        ax.grid(True, alpha=0.3)

    tag = gkey.lower()
    fig.suptitle(f'{label} Capacity-derived RUL — Predicted vs Actual',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT / f'fig_{tag}_cap_rul_scatter.png', dpi=150)
    plt.close()
    print(f'  Saved: fig_{tag}_cap_rul_scatter.png')


# ===========================================================================
# COMBINED SUMMARY
# ===========================================================================
print('\n' + '='*60)
print('CAPACITY-DERIVED RUL — FULL SUMMARY')
print('='*60)

# RT
print(f'\n--- RT (~25C) — LOOCV (8-fold), fixed RBF l={l_cap_rt} ---')
print(f'{"Cell":<8}  {"Cap R2":>8}  {"RUL R2":>8}  {"pred EOL":>10}  {"actual EOL":>10}')
print('-' * 52)
for c in ALL_CA:
    r = rt_results[c]
    cap_r2 = f'{r["r2_cap"]:.4f}'
    rul_r2 = f'{r["r2_rul"]:.4f}' if r['r2_rul'] is not None else '  N/A'
    peol   = f'{r["pred_eol"]:.1f}' if r['pred_eol'] is not None else '  N/A'
    aeol   = str(r['actual_eol']) if r['actual_eol'] is not None else 'DNF'
    tag    = ' [DNF]' if c in DNF_CA else ''
    print(f'{c}{tag:<8}  {cap_r2:>8}  {rul_r2:>8}  {peol:>10}  {aeol:>10}')

r2_cap_rt_all = np.mean([rt_results[c]['r2_cap'] for c in ALL_CA])
print(f'\n  Mean cap R2 (all 8): {r2_cap_rt_all:.4f}')
if valid_r2_rt:
    print(f'  Mean RUL R2 ({len(valid_r2_rt)} EOL): {mean_r2_rt:.4f}')

# Cold fixed DOE
for gkey, G in COLD_GROUPS.items():
    label = G['label']
    print(f'\n--- {label} — fixed DOE (3 train, 1 test), Coupled ARD-RBF ---')
    print(f'{"Cell":<12}  {"Cap R2":>8}  {"RUL R2":>8}  '
          f'{"pred EOL":>10}  {"actual EOL":>10}')
    print('-' * 56)
    for cell in G['test']:
        r = cold_results[gkey][cell]
        cap_r2 = f'{r["r2_cap"]:.4f}'
        rul_r2 = f'{r["r2_rul"]:.4f}' if r['r2_rul'] is not None else '  N/A'
        peol   = f'{r["pred_eol"]:.1f}' if r['pred_eol'] is not None else '  N/A'
        aeol   = str(r['actual_eol'])
        print(f'{cell:<12}  {cap_r2:>8}  {rul_r2:>8}  {peol:>10}  {aeol:>10}')

# Cold LOOCV
for gkey, G in COLD_LOOCV.items():
    label = G['label']
    all_cells = G['cells']
    print(f'\n--- {label} — LOOCV (4-fold), Coupled ARD-RBF ---')
    print(f'{"Cell":<12}  {"Cap R2":>8}  {"RUL R2":>8}  '
          f'{"pred EOL":>10}  {"actual EOL":>10}')
    print('-' * 56)
    for cell in all_cells:
        r = loocv_results[gkey][cell]
        cap_r2 = f'{r["r2_cap"]:.4f}'
        rul_r2 = f'{r["r2_rul"]:.4f}' if r['r2_rul'] is not None else '  N/A'
        peol   = f'{r["pred_eol"]:.1f}' if r['pred_eol'] is not None else '  N/A'
        aeol   = str(r['actual_eol'])
        print(f'{cell:<12}  {cap_r2:>8}  {rul_r2:>8}  {peol:>10}  {aeol:>10}')
    valid_rul = [loocv_results[gkey][c]['r2_rul'] for c in all_cells
                 if loocv_results[gkey][c]['r2_rul'] is not None]
    valid_cap = [loocv_results[gkey][c]['r2_cap'] for c in all_cells]
    print(f'  Mean cap R2: {np.mean(valid_cap):.4f}')
    if valid_rul:
        print(f'  Mean RUL R2: {np.mean(valid_rul):.4f}')

# Cross-temperature + cross-method comparison
print(f'\n{"="*60}')
print('CROSS-TEMPERATURE / CROSS-METHOD COMPARISON')
print(f'{"="*60}')
print(f'{"Group":<8}  {"Method":<22}  {"Cap R2":>8}  {"RUL R2":>10}')
print('-' * 54)

# RT
print(f'{"RT":<8}  {"Cap-derived (LOOCV)":<22}  {r2_cap_rt_all:>8.4f}  {mean_r2_rt:>10.4f}')
print(f'{"RT":<8}  {"Direct EIS->RUL":<22}  {"—":>8}  {DIRECT_RUL_RESULTS["RT"]["r2"]:>10.1f}')

# Cold fixed DOE
for gkey, G in COLD_GROUPS.items():
    label = G['label']
    for cell in G['test']:
        r = cold_results[gkey][cell]
        rul_str = f'{r["r2_rul"]:.4f}' if r['r2_rul'] is not None else 'N/A'
        print(f'{label:<8}  {"Cap-derived (DOE)":<22}  {r["r2_cap"]:>8.4f}  {rul_str:>10}')
    dr = DIRECT_RUL_RESULTS[label]
    print(f'{label:<8}  {"Direct EIS->RUL":<22}  {"—":>8}  {dr["r2"]:>10.3f}')

# Cold LOOCV
for gkey, G in COLD_LOOCV.items():
    label = G['label']
    all_cells = G['cells']
    valid_cap = [loocv_results[gkey][c]['r2_cap'] for c in all_cells]
    valid_rul = [loocv_results[gkey][c]['r2_rul'] for c in all_cells
                 if loocv_results[gkey][c]['r2_rul'] is not None]
    cap_mean = np.mean(valid_cap)
    rul_mean = np.mean(valid_rul) if valid_rul else None
    rul_str = f'{rul_mean:.4f}' if rul_mean is not None else 'N/A'
    print(f'{label:<8}  {"Cap-derived (LOOCV)":<22}  {cap_mean:>8.4f}  {rul_str:>10}')

# Best per temperature
print(f'\n{"="*60}')
print('BEST APPROACH PER TEMPERATURE')
print(f'{"="*60}')
print(f'  RT   : Capacity-derived RUL (LOOCV)     R2 = {mean_r2_rt:.4f}')

# -10C: pick best
n10_doe_rul = cold_results['N10']['N10_CB4']['r2_rul']
n10_loocv_vals = [loocv_results['N10'][c]['r2_rul'] for c in COLD_LOOCV['N10']['cells']
                  if loocv_results['N10'][c]['r2_rul'] is not None]
n10_loocv_mean = np.mean(n10_loocv_vals) if n10_loocv_vals else None
n10_direct = DIRECT_RUL_RESULTS['-10C']['r2']
n10_options = {'Cap-derived (DOE)': n10_doe_rul, 'Direct EIS->RUL': n10_direct}
if n10_loocv_mean is not None:
    n10_options['Cap-derived (LOOCV)'] = n10_loocv_mean
best_n10 = max(n10_options, key=lambda k: n10_options[k] if n10_options[k] is not None else -999)
print(f'  -10C : {best_n10:<30}  R2 = {n10_options[best_n10]:.4f}')

# -20C: pick best
n20_doe_rul = cold_results['N20']['N20_CB4']['r2_rul']
n20_loocv_vals = [loocv_results['N20'][c]['r2_rul'] for c in COLD_LOOCV['N20']['cells']
                  if loocv_results['N20'][c]['r2_rul'] is not None]
n20_loocv_mean = np.mean(n20_loocv_vals) if n20_loocv_vals else None
n20_direct = DIRECT_RUL_RESULTS['-20C']['r2']
n20_options = {'Cap-derived (DOE)': n20_doe_rul, 'Direct EIS->RUL': n20_direct}
if n20_loocv_mean is not None:
    n20_options['Cap-derived (LOOCV)'] = n20_loocv_mean
best_n20 = max(n20_options, key=lambda k: n20_options[k] if n20_options[k] is not None else -999)
print(f'  -20C : {best_n20:<30}  R2 = {n20_options[best_n20]:.4f}')

print(f'\nFigures saved to: {OUT}')
