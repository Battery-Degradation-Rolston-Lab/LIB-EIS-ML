"""
Multi-temperature GPR — Zhang-faithful DOE with Coupled ARD.

Zhang (2020) approach:
  Train on cells from ALL temperature groups, hold out one cell per non-RT temp.

Split
-----
  Capacity train : CA1-CA8 (RT)  +  N10_CB1-CB3 (-10°C)  +  N20_CB1-CB3 (-20°C)
  Capacity test  : N10_CB4  (-10°C held-out)
                   N20_CB4  (-20°C held-out)
  RUL train      : CA1-CA5 EOL  +  N10_CB1-CB3  +  N20_CB1-CB3
  RUL test       : N10_CB4  and  N20_CB4

Kernel
------
  Capacity / ARD : CoupledARD-RBF (sklearn) — 33 length-scales, one per frequency,
                   shared between Re(Z) and -Im(Z)  [K-K physically correct]
  RUL            : Linear (GPyTorch)  —  Zhang eq. 5, zero mean

Normalisation: training-only z-score  (Zhang multi-T convention)
               Temperature-shifted EIS lands away from training mean — that IS
               the signal that encodes temperature and health state jointly.
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
    Kernel, ConstantKernel, WhiteKernel, Hyperparameter
)
from sklearn.metrics import r2_score

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
CA_DATA    = SCRIPT_DIR / 'data' / 'ca_dataset'
MT_DATA    = SCRIPT_DIR / 'data' / 'multitemp_dataset'
OUT        = SCRIPT_DIR / 'output' / 'multitemp_zhang'
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
])  # 33 freqs high → low (same order as feature vector Re then Im)
N_FREQ = len(NATIVE_FREQS)  # 33

# Zhang DOE — include representative cells from every temperature in training
CA_TRAIN       = ['CA1', 'CA2', 'CA3', 'CA4', 'CA5', 'CA6', 'CA7', 'CA8']  # all RT
CA_EOL_TRAIN   = ['CA1', 'CA2', 'CA3', 'CA4', 'CA5']                        # CA6 DNF
N10_TRAIN      = ['N10_CB1', 'N10_CB2', 'N10_CB3']
N20_TRAIN      = ['N20_CB1', 'N20_CB2', 'N20_CB3']
N10_TEST       = ['N10_CB4']   # -10°C held-out
N20_TEST       = ['N20_CB4']   # -20°C held-out

ARD_MAX_N = 400   # subsample for sklearn ARD (K is O(N²))

print(f'\nZhang DOE:')
print(f'  Capacity train : {CA_TRAIN}')
print(f'                   {N10_TRAIN}')
print(f'                   {N20_TRAIN}')
print(f'  Test           : {N10_TEST}  {N20_TEST}')


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_ca(cell):
    return (np.loadtxt(CA_DATA / f'EIS_{cell}.txt'),
            np.loadtxt(CA_DATA / f'cap_{cell}.txt'))

def load_ca_rul(cell):
    return (np.loadtxt(CA_DATA / f'EIS_rul_{cell}.txt'),
            np.loadtxt(CA_DATA / f'rul_{cell}.txt'))

def load_mt(cell):
    return (np.loadtxt(MT_DATA / f'EIS_{cell}.txt'),
            np.loadtxt(MT_DATA / f'cap_{cell}.txt'))

def load_mt_rul(cell):
    return (np.loadtxt(MT_DATA / f'EIS_rul_{cell}.txt'),
            np.loadtxt(MT_DATA / f'rul_{cell}.txt'))

def zscore(X):
    mu  = X.mean(0)
    sig = X.std(0, ddof=1)
    return (X - mu) / np.where(sig == 0, 1.0, sig), mu, sig

def apply_norm(X, mu, sig):
    return (X - mu) / np.where(sig == 0, 1.0, sig)

def ard_weights(ls):
    """w = exp(-ls), normalised (Zhang eq. 4)."""
    w = np.exp(-ls)
    return w / w.sum()

def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32, device=DEVICE)


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


# ── GPyTorch Linear model (RUL) ───────────────────────────────────────────────

class LinearModel(gpytorch.models.ExactGP):
    def __init__(self, tx, ty, lh):
        super().__init__(tx, ty, lh)
        self.mean_module  = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel()
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def train_gp(model, likelihood, tx, ty, n_iter=600, lr=0.05):
    model.train(); likelihood.train()
    opt = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()), lr=lr
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    best, best_sd = float('inf'), None
    for i in range(n_iter):
        opt.zero_grad()
        loss = -mll(model(tx), ty)
        loss.backward(); opt.step(); sch.step()
        if loss.item() < best:
            best = loss.item()
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        if (i + 1) % 200 == 0:
            print(f'    iter {i+1}/{n_iter}  loss={loss.item():.4f}')
    if best_sd:
        model.load_state_dict(best_sd)
    return model, likelihood


def predict_gp(model, likelihood, tx):
    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        p = likelihood(model(tx))
    return p.mean.cpu().numpy(), p.stddev.cpu().numpy()


# ── Load all data ─────────────────────────────────────────────────────────────
print('\nLoading data ...')

# Capacity
EIS_cap_tr = np.vstack(
    [load_ca(c)[0] for c in CA_TRAIN] +
    [load_mt(c)[0] for c in N10_TRAIN + N20_TRAIN]
)
Cap_tr = np.concatenate(
    [load_ca(c)[1] for c in CA_TRAIN] +
    [load_mt(c)[1] for c in N10_TRAIN + N20_TRAIN]
)

# RUL
EIS_rul_tr = np.vstack(
    [load_ca_rul(c)[0] for c in CA_EOL_TRAIN] +
    [load_mt_rul(c)[0] for c in N10_TRAIN + N20_TRAIN]
)
RUL_tr = np.concatenate(
    [load_ca_rul(c)[1] for c in CA_EOL_TRAIN] +
    [load_mt_rul(c)[1] for c in N10_TRAIN + N20_TRAIN]
)

print(f'  Capacity train : {EIS_cap_tr.shape[0]} rows')
print(f'  RUL train      : {EIS_rul_tr.shape[0]} rows  '
      f'(RUL range {RUL_tr.min():.0f}–{RUL_tr.max():.0f})')

# Training-only z-score (Zhang multi-T convention)
_, mu_cap, sig_cap = zscore(EIS_cap_tr)
_, mu_rul, sig_rul = zscore(EIS_rul_tr)

X_cap_tr_n = apply_norm(EIS_cap_tr, mu_cap, sig_cap)
X_rul_tr_n = apply_norm(EIS_rul_tr, mu_rul, sig_rul)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — Capacity: Coupled ARD-RBF  (sklearn)
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*60)
print('MODEL 1 — Capacity  (Coupled ARD-RBF, 33 length-scales)')
print(f'  Train: {len(CA_TRAIN)} RT + {len(N10_TRAIN)} × -10°C + {len(N20_TRAIN)} × -20°C')
print('═'*60)

# Subsample for sklearn (K is O(N²))
idx_sub = RNG.choice(len(X_cap_tr_n), min(ARD_MAX_N, len(X_cap_tr_n)), replace=False)
X_sub   = X_cap_tr_n[idx_sub]
y_sub   = Cap_tr[idx_sub]

kernel_cap = (ConstantKernel(1.0) *
              CoupledARDRBF(length_scale=np.ones(N_FREQ)) +
              WhiteKernel(noise_level=1.0))
gpr_cap = GaussianProcessRegressor(kernel=kernel_cap, normalize_y=True,
                                   alpha=0.1, n_restarts_optimizer=3)
print('  Fitting Coupled ARD-RBF ...')
gpr_cap.fit(X_sub, y_sub)

ls_cap = gpr_cap.kernel_.k1.k2.length_scale   # (33,)
w_cap  = ard_weights(ls_cap)

top_idx_cap  = np.argmax(w_cap)
top_freq_cap = NATIVE_FREQS[top_idx_cap]
print(f'  Top frequency: {top_freq_cap:.2f} Hz  (weight={w_cap[top_idx_cap]:.4f})')

# Predict test cells
test_cells_cap = N10_TEST + N20_TEST
r2_cap = {}
pred_cap_all = {}
meas_cap_all = {}

for cell in test_cells_cap:
    eis_te, cap_te = load_mt(cell)
    X_te_n = apply_norm(eis_te, mu_cap, sig_cap)
    mean_pred = gpr_cap.predict(X_te_n)
    r2 = r2_score(cap_te, mean_pred)
    r2_cap[cell]       = r2
    pred_cap_all[cell] = mean_pred
    meas_cap_all[cell] = cap_te
    temp = '-10°C' if cell.startswith('N10') else '-20°C'
    print(f'  {cell} ({temp}): R² = {r2:.4f}')


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — RUL: Linear kernel  (GPyTorch, Zhang eq. 5)
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*60)
print('MODEL 2 — RUL  (Linear kernel, GPyTorch, zero mean)')
print(f'  Train: {len(CA_EOL_TRAIN)} RT EOL + {len(N10_TRAIN)} × -10°C + {len(N20_TRAIN)} × -20°C')
print('═'*60)

tx_rul = to_tensor(X_rul_tr_n)
ty_rul = to_tensor(RUL_tr.astype(np.float32))

lh_rul  = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
m_rul   = LinearModel(tx_rul, ty_rul, lh_rul).to(DEVICE)

print('  Training ...')
m_rul, lh_rul = train_gp(m_rul, lh_rul, tx_rul, ty_rul, n_iter=600, lr=0.05)

test_cells_rul = N10_TEST + N20_TEST
r2_rul = {}
pred_rul_all = {}
meas_rul_all = {}

for cell in test_cells_rul:
    eis_te, rul_te = load_mt_rul(cell)
    X_te_n  = apply_norm(eis_te, mu_rul, sig_rul)
    mean_r, std_r = predict_gp(m_rul, lh_rul, to_tensor(X_te_n))
    r2 = r2_score(rul_te, mean_r)
    r2_rul[cell]       = r2
    pred_rul_all[cell] = (mean_r, std_r)
    meas_rul_all[cell] = rul_te
    temp = '-10°C' if cell.startswith('N10') else '-20°C'
    print(f'  {cell} ({temp}): R² = {r2:.4f}  '
          f'(RUL_max={int(rul_te[0])}, n={len(rul_te)}, '
          f'pred_mean={mean_r.mean():.1f}, actual_mean={rul_te.mean():.1f})')


# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════

freqs_plot = NATIVE_FREQS[::-1]   # low → high for x-axis
w_cap_plot = w_cap[::-1]

# ── Figure 1: ARD weights ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4))
ax.semilogx(freqs_plot, w_cap_plot, 'b-o', ms=5, lw=1.8,
            label='|Z(ω)| — Re & Im paired')
ax.fill_between(freqs_plot, 0, w_cap_plot, alpha=0.15, color='blue')
ax.axhline(0, color='black', lw=0.6)

# Annotate top frequency
top_plot_idx = len(freqs_plot) - 1 - top_idx_cap
ax.annotate(f'{top_freq_cap:.1f} Hz',
            xy=(freqs_plot[top_plot_idx], w_cap_plot[top_plot_idx]),
            xytext=(freqs_plot[top_plot_idx] * 3, w_cap_plot[top_plot_idx] * 1.1),
            fontsize=10, color='blue',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.2))

ax.set_xlabel('Frequency (Hz)', fontsize=13)
ax.set_ylabel('ARD weight', fontsize=13)
ax.set_title('Coupled ARD — Multi-T training (RT + -10°C + -20°C)\n'
             '33 length-scales, Re(Z) and -Im(Z) paired per frequency', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / 'fig_zhang_ard.png', dpi=150)
plt.close()
print(f'\n  Saved: fig_zhang_ard.png')

# ── Figure 2: Capacity trajectories ──────────────────────────────────────────
temp_colors = {'-10°C': '#ff7f0e', '-20°C': '#d62728'}

fig, axes = plt.subplots(1, len(test_cells_cap), figsize=(5 * len(test_cells_cap), 5))
if len(test_cells_cap) == 1:
    axes = [axes]

for idx, cell in enumerate(test_cells_cap):
    ax   = axes[idx]
    cap_te     = meas_cap_all[cell]
    mean_pred  = pred_cap_all[cell]
    cap0       = cap_te[0]
    cyc        = np.arange(len(cap_te))
    temp       = '-10°C' if cell.startswith('N10') else '-20°C'
    col        = temp_colors[temp]

    ax.plot(cyc, cap_te / cap0, 'x', color='grey', ms=4, alpha=0.6, label='Measured')
    ax.plot(cyc, mean_pred / cap0, '-', color=col, lw=1.8, label='GPR (Coupled ARD)')
    ax.axhline(0.8, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.set_title(f'{cell} ({temp})\nR²={r2_cap[cell]:.3f}', fontsize=10)
    ax.set_xlabel('Cycle', fontsize=9)
    ax.set_ylabel('Norm. Capacity', fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)

fig.suptitle('Capacity GPR — Zhang DOE  (Train: RT + -10°C + -20°C, Test: held-out)\n'
             'Coupled ARD-RBF, training-only z-score', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'fig_zhang_capacity.png', dpi=150)
plt.close()
print(f'  Saved: fig_zhang_capacity.png')

# ── Figure 3: RUL scatter ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(test_cells_rul), figsize=(5 * len(test_cells_rul), 5))
if len(test_cells_rul) == 1:
    axes = [axes]

for idx, cell in enumerate(test_cells_rul):
    ax         = axes[idx]
    rul_te     = meas_rul_all[cell]
    mean_r, std_r = pred_rul_all[cell]
    temp       = '-10°C' if cell.startswith('N10') else '-20°C'
    col        = temp_colors[temp]
    lim        = max(rul_te.max(), mean_r.max()) * 1.1

    ax.fill_between(rul_te, mean_r - std_r, mean_r + std_r, alpha=0.3, color=col)
    ax.scatter(rul_te, mean_r, s=25, color=col, alpha=0.85, zorder=3)
    ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('Actual RUL (bat-cycles)', fontsize=10)
    ax.set_ylabel('Predicted RUL', fontsize=10)
    ax.set_title(f'{cell} ({temp})\nR²={r2_rul[cell]:.3f}', fontsize=10)
    ax.grid(True, alpha=0.3)

mean_r2_rul = np.mean(list(r2_rul.values()))
fig.suptitle(f'RUL GPR — Zhang DOE  (mean R²={mean_r2_rul:.3f})\n'
             f'Train: RT + -10°C + -20°C  |  Linear kernel  |  train-only z-score',
             fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'fig_zhang_rul_scatter.png', dpi=150)
plt.close()
print(f'  Saved: fig_zhang_rul_scatter.png')


# ── Summary ───────────────────────────────────────────────────────────────────
print('\n' + '═'*60)
print('SUMMARY — Zhang-faithful DOE + Coupled ARD')
print('═'*60)
print(f'\n{"Cell":<12} {"Temp":>6}  {"Cap R²":>8}  {"RUL R²":>8}')
print('-'*40)
for cell in test_cells_cap:
    temp = '-10°C' if cell.startswith('N10') else '-20°C'
    cr   = r2_cap.get(cell, float('nan'))
    rr   = r2_rul.get(cell, float('nan'))
    print(f'{cell:<12} {temp:>6}  {cr:>8.4f}  {rr:>8.4f}')

print(f'\nCoupled ARD top frequency : {top_freq_cap:.2f} Hz  (w={w_cap[top_idx_cap]:.4f})')
print(f'Mean RUL R²               : {mean_r2_rul:.4f}')
print(f'\nFigures saved to: {OUT}')
