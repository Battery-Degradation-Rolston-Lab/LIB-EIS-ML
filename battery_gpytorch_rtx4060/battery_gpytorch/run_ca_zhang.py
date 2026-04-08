"""
Single-temperature GPR — Zhang Fig 1 / Fig 2 equivalent for each temperature.
Capacity model uses Coupled ARD-RBF (33 ls, Re+Im paired per frequency).

Pre-specified experimental DOE (not a post-hoc split — designated before data collection)
------------------------------------------------------------------------------------------
  RT   (~25°C) : train CA1–CA6,    test CA7, CA8
  -10°C        : train N10_CB1–3,  test N10_CB4
  -20°C        : train N20_CB1–3,  test N20_CB4

For RUL: CA6 is DNF — excluded from RT RUL training. All CB cells reached EOL.

Figures per temperature group
------------------------------
  fig_{T}_1a_capacity_trajectories.png  — GPR capacity vs cycle       (Zhang Fig 1a)
  fig_{T}_1b_capacity_scatter.png       — predicted vs actual scatter  (Zhang Fig 1b)
  fig_{T}_1c_ARD_weights.png            — Coupled ARD per frequency    (Zhang Fig 1c)
  fig_{T}_2_rul_scatter.png             — predicted vs actual RUL      (Zhang Fig 2)

Kernel
------
  Capacity : CoupledARD-RBF  (sklearn, 33 ls — one per frequency, shared Re+Im)
  RUL      : ScaleKernel(LinearKernel)  (GPyTorch, Zhang eq. 5, zero mean)

Normalisation: training-only z-score for both capacity and RUL
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import gpytorch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, WhiteKernel, Kernel, Hyperparameter
)
from sklearn.metrics import r2_score

SCRIPT_DIR = Path(__file__).parent
CA_DATA    = SCRIPT_DIR / 'data' / 'ca_dataset'
MT_DATA    = SCRIPT_DIR / 'data' / 'multitemp_dataset'
OUT        = SCRIPT_DIR / 'output' / 'ca_zhang'
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')

np.random.seed(42)
torch.manual_seed(42)
RNG = np.random.default_rng(42)

NATIVE_FREQS = np.array([
    10000.0, 7500.0, 5620.0, 4220.0, 3160.0, 2370.0, 1780.0, 1330.0,
    1000.0,  750.0,  564.0,  422.0,  316.0,  237.0,  178.0,  135.0,
    102.0,   75.0,   56.2,   42.2,   31.6,   23.7,   17.8,   13.3,
    10.0,    7.5,    5.62,   4.22,   3.16,   2.37,   1.78,   1.33, 0.999
])
N_FREQ = len(NATIVE_FREQS)  # 33

ARD_MAX_N = 600   # subsample for sklearn (K is O(N²)); RT has ~2500 rows

# ── Temperature group definitions ─────────────────────────────────────────────
GROUPS = {
    'RT': {
        'label'     : 'RT (~25°C)',
        'color'     : '#1f77b4',
        'cap_train' : ['CA1', 'CA2', 'CA3', 'CA4', 'CA5', 'CA6'],
        'rul_train' : ['CA1', 'CA2', 'CA3', 'CA4', 'CA5'],   # CA6 DNF
        'test'      : ['CA7', 'CA8'],
        'data_dir'  : None,
    },
    'N10': {
        'label'     : '-10°C',
        'color'     : '#ff7f0e',
        'cap_train' : ['N10_CB1', 'N10_CB2', 'N10_CB3'],
        'rul_train' : ['N10_CB1', 'N10_CB2', 'N10_CB3'],
        'test'      : ['N10_CB4'],
        'data_dir'  : None,
    },
    'N20': {
        'label'     : '-20°C',
        'color'     : '#d62728',
        'cap_train' : ['N20_CB1', 'N20_CB2', 'N20_CB3'],
        'rul_train' : ['N20_CB1', 'N20_CB2', 'N20_CB3'],
        'test'      : ['N20_CB4'],
        'data_dir'  : None,
    },
}
GROUPS['RT']['data_dir']  = CA_DATA
GROUPS['N10']['data_dir'] = MT_DATA
GROUPS['N20']['data_dir'] = MT_DATA


# ── Coupled ARD-RBF kernel (sklearn) ─────────────────────────────────────────
# One length-scale per frequency, shared between Re(Z) and -Im(Z).
# k(x,x') = exp(−Σᵢ [(Re_i−Re_i')² + (Im_i−Im_i')²] / (2·lᵢ²))

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_cap(cell, data_dir):
    return (np.loadtxt(data_dir / f'EIS_{cell}.txt'),
            np.loadtxt(data_dir / f'cap_{cell}.txt'))

def load_rul(cell, data_dir):
    return (np.loadtxt(data_dir / f'EIS_rul_{cell}.txt'),
            np.loadtxt(data_dir / f'rul_{cell}.txt'))

def zscore(X):
    mu = X.mean(0);  sig = X.std(0, ddof=1)
    return (X - mu) / np.where(sig == 0, 1.0, sig), mu, sig

def apply_norm(X, mu, sig):
    return (X - mu) / np.where(sig == 0, 1.0, sig)

def ard_weights(ls):
    w = np.exp(-ls);  return w / w.sum()

def to_t(a): return torch.tensor(a, dtype=torch.float32, device=DEVICE)


# ── GPyTorch Linear model (RUL) ───────────────────────────────────────────────

class LinearModel(gpytorch.models.ExactGP):
    def __init__(self, tx, ty, lh):
        super().__init__(tx, ty, lh)
        self.mean_module  = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel())
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))

def train_gp(model, lh, tx, ty, n_iter=800, lr=0.05):
    model.train(); lh.train()
    opt = torch.optim.Adam(list(model.parameters()) + list(lh.parameters()), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lh, model)
    best, best_sd = float('inf'), None
    for i in range(n_iter):
        opt.zero_grad()
        loss = -mll(model(tx), ty); loss.backward()
        opt.step(); sch.step()
        if loss.item() < best:
            best = loss.item()
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        if (i + 1) % 200 == 0:
            print(f'    iter {i+1}/{n_iter}  loss={loss.item():.4f}')
    if best_sd: model.load_state_dict(best_sd)
    return model, lh

def predict_gp(model, lh, tx):
    model.eval(); lh.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        p = lh(model(tx))
    return p.mean.cpu().numpy(), p.stddev.cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# Run for each temperature group
# ═══════════════════════════════════════════════════════════════════════════════

summary = {}

for gkey, G in GROUPS.items():
    label = G['label'];  col = G['color'];  ddir = G['data_dir']
    test  = G['test'];   tag = gkey.lower()

    print('\n' + '═'*60)
    print(f'TEMPERATURE GROUP: {label}')
    print(f'  Cap train : {G["cap_train"]}')
    print(f'  RUL train : {G["rul_train"]}')
    print(f'  Test      : {test}')
    print('═'*60)

    # ── Load ─────────────────────────────────────────────────────────────────
    EIS_cap_tr = np.vstack([load_cap(c, ddir)[0] for c in G['cap_train']])
    Cap_tr     = np.concatenate([load_cap(c, ddir)[1] for c in G['cap_train']])
    EIS_rul_tr = np.vstack([load_rul(c, ddir)[0] for c in G['rul_train']])
    RUL_tr     = np.concatenate([load_rul(c, ddir)[1] for c in G['rul_train']])

    print(f'  Capacity train : {EIS_cap_tr.shape[0]} rows')
    print(f'  RUL train      : {EIS_rul_tr.shape[0]} rows  '
          f'(range {RUL_tr.min():.0f}–{RUL_tr.max():.0f})')

    _, mu_cap, sig_cap = zscore(EIS_cap_tr)
    _, mu_rul, sig_rul = zscore(EIS_rul_tr)
    X_cap_tr_n = apply_norm(EIS_cap_tr, mu_cap, sig_cap)
    X_rul_tr_n = apply_norm(EIS_rul_tr, mu_rul, sig_rul)

    # ── Capacity: Coupled ARD-RBF (sklearn) ───────────────────────────────────
    print(f'\n  [Capacity — Coupled ARD-RBF, 33 ls]')
    n_sub = min(ARD_MAX_N, len(X_cap_tr_n))
    idx   = RNG.choice(len(X_cap_tr_n), n_sub, replace=False)
    X_sub = X_cap_tr_n[idx];  y_sub = Cap_tr[idx]

    kernel = (ConstantKernel(1.0) *
              CoupledARDRBF(length_scale=np.ones(N_FREQ)) +
              WhiteKernel(noise_level=1.0))
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                   alpha=0.1, n_restarts_optimizer=10)
    print(f'  Fitting on {n_sub} subsampled rows ...')
    gpr.fit(X_sub, y_sub)

    ls_cap = gpr.kernel_.k1.k2.length_scale   # (33,)
    w_cap  = ard_weights(ls_cap)
    top_idx = np.argmax(w_cap)
    print(f'  Top frequency: {NATIVE_FREQS[top_idx]:.2f} Hz  (w={w_cap[top_idx]:.4f})')

    r2_cap = {};  pred_cap = {};  meas_cap = {}
    all_pred = [];  all_meas = []

    for cell in test:
        eis_te, cap_te = load_cap(cell, ddir)
        X_te_n = apply_norm(eis_te, mu_cap, sig_cap)
        mn = gpr.predict(X_te_n)
        r2 = r2_score(cap_te, mn)
        r2_cap[cell] = r2;  pred_cap[cell] = mn;  meas_cap[cell] = cap_te
        all_pred.extend(mn.tolist());  all_meas.extend(cap_te.tolist())
        print(f'  {cell}: Cap R² = {r2:.4f}')

    r2_scatter = r2_score(np.array(all_meas), np.array(all_pred))

    # ── RUL: Linear GPyTorch ──────────────────────────────────────────────────
    print(f'\n  [RUL — Linear kernel]')
    tx_r = to_t(X_rul_tr_n);  ty_r = to_t(RUL_tr.astype(np.float32))
    lh_r = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
    m_r  = LinearModel(tx_r, ty_r, lh_r).to(DEVICE)
    m_r, lh_r = train_gp(m_r, lh_r, tx_r, ty_r, n_iter=800)

    r2_rul = {};  pred_rul = {};  meas_rul = {}
    for cell in test:
        eis_te, rul_te = load_rul(cell, ddir)
        mn_r, sd_r = predict_gp(m_r, lh_r,
                                 to_t(apply_norm(eis_te, mu_rul, sig_rul)))
        r2 = r2_score(rul_te, mn_r)
        r2_rul[cell] = r2;  pred_rul[cell] = (mn_r, sd_r);  meas_rul[cell] = rul_te
        print(f'  {cell}: RUL R² = {r2:.4f}  '
              f'(max={int(rul_te[0])}, pred_mean={mn_r.mean():.1f})')

    summary[gkey] = {'r2_cap': r2_cap, 'r2_rul': r2_rul, 'r2_scatter': r2_scatter}

    test_colors = [col] * len(test)

    # ── Fig 1a — Capacity trajectories ───────────────────────────────────────
    n_test = len(test)
    fig, axes = plt.subplots(1, n_test, figsize=(6 * n_test, 5), squeeze=False)
    axes = axes[0]
    for idx2, cell in enumerate(test):
        ax = axes[idx2]
        cap_te = meas_cap[cell];  mn = pred_cap[cell]
        cap0 = cap_te[0];  cyc = np.arange(len(cap_te))
        ax.plot(cyc, cap_te / cap0, 'x', color='grey', ms=3, alpha=0.6,
                label='Measured')
        ax.plot(cyc, mn / cap0, '-', color=test_colors[idx2], lw=1.8,
                label='GPR (Coupled ARD)')
        ax.axhline(0.8, color='black', lw=0.8, ls='--', alpha=0.5)
        ax.set_title(f'{cell} ({label})\nR²={r2_cap[cell]:.3f}', fontsize=10)
        ax.set_xlabel('Cycle', fontsize=9);  ax.set_ylabel('Norm. Capacity', fontsize=9)
        ax.legend(fontsize=8, frameon=False);  ax.grid(True, alpha=0.3)
    fig.suptitle(f'Capacity GPR — Single-T {label}\n'
                 f'Train: {G["cap_train"]}  |  Coupled ARD-RBF (33 ls)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT / f'fig_{tag}_1a_capacity_trajectories.png', dpi=150)
    plt.close()

    # ── Fig 1b — Capacity scatter ─────────────────────────────────────────────
    all_pred_a = np.array(all_pred);  all_meas_a = np.array(all_meas)
    fig, ax = plt.subplots(figsize=(5, 5))
    for idx2, cell in enumerate(test):
        ax.scatter(meas_cap[cell], pred_cap[cell],
                   s=8, alpha=0.6, color=test_colors[idx2], label=cell)
    lo = min(all_meas_a.min(), all_pred_a.min()) * 0.98
    hi = max(all_meas_a.max(), all_pred_a.max()) * 1.02
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.6)
    ax.set_xlim(lo, hi);  ax.set_ylim(lo, hi)
    ax.set_xlabel('Measured Capacity (mAh)', fontsize=11)
    ax.set_ylabel('Predicted Capacity (mAh)', fontsize=11)
    ax.set_title(f'Capacity Scatter — Single-T {label}\nR²={r2_scatter:.3f}',
                 fontsize=11)
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / f'fig_{tag}_1b_capacity_scatter.png', dpi=150)
    plt.close()

    # ── Fig 1c — Coupled ARD weights (one per frequency) ─────────────────────
    freqs_plot = NATIVE_FREQS[::-1]   # low → high
    w_plot     = w_cap[::-1]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.semilogx(freqs_plot, w_plot, 'b-o', ms=5, lw=1.8,
                label='|Z(ω)| — Re & Im paired')
    ax.fill_between(freqs_plot, 0, w_plot, alpha=0.15, color='blue')
    ax.axhline(0, color='black', lw=0.6)

    top_plot_idx = N_FREQ - 1 - top_idx
    ax.annotate(f'{NATIVE_FREQS[top_idx]:.2f} Hz',
                xy=(freqs_plot[top_plot_idx], w_plot[top_plot_idx]),
                xytext=(freqs_plot[top_plot_idx] * 3,
                        w_plot[top_plot_idx] * 1.1),
                fontsize=10, color='blue',
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.2))

    ax.set_xlabel('Frequency (Hz)', fontsize=13)
    ax.set_ylabel('ARD weight', fontsize=13)
    ax.set_title(f'Coupled ARD — Single-T {label}\n'
                 f'Train: {G["cap_train"]}  |  33 ls, Re+Im paired per frequency',
                 fontsize=11)
    ax.legend(fontsize=10);  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / f'fig_{tag}_1c_ARD_weights.png', dpi=150)
    plt.close()

    # ── Fig 2 — RUL scatter ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_test, figsize=(5 * n_test, 5), squeeze=False)
    axes = axes[0]
    for idx2, cell in enumerate(test):
        ax = axes[idx2]
        rul_te = meas_rul[cell];  mn_r, sd_r = pred_rul[cell]
        lim = max(rul_te.max(), mn_r.max()) * 1.1
        ax.fill_between(rul_te, mn_r - sd_r, mn_r + sd_r,
                        alpha=0.3, color=test_colors[idx2])
        ax.scatter(rul_te, mn_r, s=20, color=test_colors[idx2],
                   alpha=0.85, zorder=3)
        ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5)
        ax.set_xlim(0, lim);  ax.set_ylim(0, lim)
        ax.set_xlabel('Actual RUL (bat-cycles)', fontsize=10)
        ax.set_ylabel('Predicted RUL', fontsize=10)
        ax.set_title(f'{cell} ({label})\nR²={r2_rul[cell]:.3f}', fontsize=10)
        ax.grid(True, alpha=0.3)
    mean_r2_rul = np.mean(list(r2_rul.values()))
    fig.suptitle(f'RUL GPR — Single-T {label}  (mean R²={mean_r2_rul:.3f})\n'
                 f'Train: {G["rul_train"]}  |  Linear kernel',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT / f'fig_{tag}_2_rul_scatter.png', dpi=150)
    plt.close()

    print(f'\n  Figures saved: fig_{tag}_1a/1b/1c/2')


# ═══════════════════════════════════════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*60)
print('FINAL SUMMARY — Single-T Coupled ARD results per temperature')
print('═'*60)
print(f'\n{"Group":<6}  {"Temp":>7}  {"Cap scatter R²":>15}  {"RUL R² (mean)":>14}')
print('-'*50)
for gkey, G in GROUPS.items():
    s = summary[gkey]
    mean_rul = np.mean(list(s['r2_rul'].values()))
    print(f'{gkey:<6}  {G["label"]:>7}  {s["r2_scatter"]:>15.4f}  {mean_rul:>14.4f}')

print(f'\nAll figures saved to: {OUT}')
