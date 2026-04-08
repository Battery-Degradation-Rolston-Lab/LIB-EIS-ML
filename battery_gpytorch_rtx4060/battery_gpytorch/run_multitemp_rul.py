"""
Multi-temperature GPR: RT + -10°C → -20°C prediction.
Follows Zhang et al. (2020) multi-T DOE approach.
Uses GPyTorch (GPU-accelerated) — runs on RTX 4060 Laptop and DGX Spark unchanged.

Training temperatures : RT  (CA1-CA6)    +  -10°C (N10_CB1-N10_CB3)
Test cells            :
  CA7, CA8            — RT held-out         (in-distribution sanity check)
  N10_CB4             — -10°C held-out      (within-temperature generalisation)
  N20_CB1 - N20_CB4   — -20°C entirely new  (cross-temperature: key result)

Normalisation: training-only z-score (paper's multi-T convention).
               Temperature-shifted EIS lands far from training mean → this IS the
               signal that lets the model discriminate RUL by temperature.

Models
------
  Model 1 — Capacity GPR  : ScaleKernel(RBF) + noise, y-normalised
  Model 2 — RUL GPR       : ScaleKernel(Linear) + noise, raw y (paper eq. 5)
  Model 3 — ARD GPR       : ScaleKernel(RBF-ARD, 66 ls) — feature importance
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
from sklearn.metrics import r2_score

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
CA_DATA    = SCRIPT_DIR / 'data' / 'ca_dataset'
MT_DATA    = SCRIPT_DIR / 'data' / 'multitemp_dataset'
OUT        = SCRIPT_DIR / 'output' / 'multitemp'
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')

np.random.seed(42)
torch.manual_seed(42)

NATIVE_FREQS = np.array([
    10000.0, 7500.0, 5620.0, 4220.0, 3160.0, 2370.0, 1780.0, 1330.0,
    1000.0,  750.0,  564.0,  422.0,  316.0,  237.0,  178.0,  135.0,
    102.0,   75.0,   56.2,   42.2,   31.6,   23.7,   17.8,   13.3,
    10.0,    7.5,    5.62,   4.22,   3.16,   2.37,   1.78,   1.33, 0.999
])  # 33 freqs, high → low

# ── Helpers ───────────────────────────────────────────────────────────────────

def zscore(X):
    mu  = X.mean(0)
    sig = X.std(0, ddof=1)
    sig = np.where(sig == 0, 1.0, sig)
    return (X - mu) / sig, mu, sig


def apply_norm(X, mu, sig):
    return (X - mu) / sig


def to_tensor(arr, device=DEVICE):
    return torch.tensor(arr, dtype=torch.float32, device=device)


# ── GPyTorch model classes ────────────────────────────────────────────────────

class RBFModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class LinearModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ZeroMean()   # paper: zero mean for RUL
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel()
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class ARDModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_feat=66):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=n_feat)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def train_gp(model, likelihood, train_x, train_y, n_iter=600, lr=0.05):
    """Train GP via Adam + marginal log-likelihood maximisation."""
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    best_loss = float('inf')
    best_state = None
    for i in range(n_iter):
        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (i + 1) % 200 == 0:
            print(f'    iter {i+1}/{n_iter}  loss={loss.item():.4f}')

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, likelihood


def predict_gp(model, likelihood, test_x):
    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(test_x))
    mean = preds.mean.cpu().numpy()
    std  = preds.stddev.cpu().numpy()
    return mean, std


def ard_weights(model):
    """w_m = exp(-σ_m), normalised (paper eq. 4)."""
    ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
    w  = np.exp(-ls)
    w /= w.sum()
    return w


# ── Load data ─────────────────────────────────────────────────────────────────
print('\nLoading data ...')

CA_TRAIN_CELLS  = ['CA1', 'CA2', 'CA3', 'CA4', 'CA5', 'CA6']   # RT, DNF CA6 included
CA_TEST_CELLS   = ['CA7', 'CA8']                                 # RT held-out
N10_TRAIN_CELLS = ['N10_CB1', 'N10_CB2', 'N10_CB3']             # -10°C train
N10_TEST_CELLS  = ['N10_CB4']                                    # -10°C held-out
N20_TEST_CELLS  = ['N20_CB1', 'N20_CB2', 'N20_CB3', 'N20_CB4'] # -20°C (new temp)

# CA cells with EOL (for RUL training) — CA6 is DNF
CA_EOL_TRAIN = ['CA1', 'CA2', 'CA3', 'CA4', 'CA5']  # CA6 DNF — no RUL file

def load_ca(cell):
    eis = np.loadtxt(CA_DATA / f'EIS_{cell}.txt')
    cap = np.loadtxt(CA_DATA / f'cap_{cell}.txt')
    return eis, cap

def load_ca_rul(cell):
    eis = np.loadtxt(CA_DATA / f'EIS_rul_{cell}.txt')
    rul = np.loadtxt(CA_DATA / f'rul_{cell}.txt')
    return eis, rul

def load_mt(cell):
    eis = np.loadtxt(MT_DATA / f'EIS_{cell}.txt')
    cap = np.loadtxt(MT_DATA / f'cap_{cell}.txt')
    return eis, cap

def load_mt_rul(cell):
    eis = np.loadtxt(MT_DATA / f'EIS_rul_{cell}.txt')
    rul = np.loadtxt(MT_DATA / f'rul_{cell}.txt')
    return eis, rul

# Capacity training set
EIS_cap_tr_parts = ([load_ca(c)[0] for c in CA_TRAIN_CELLS] +
                    [load_mt(c)[0] for c in N10_TRAIN_CELLS])
Cap_tr_parts     = ([load_ca(c)[1] for c in CA_TRAIN_CELLS] +
                    [load_mt(c)[1] for c in N10_TRAIN_CELLS])
EIS_cap_tr = np.vstack(EIS_cap_tr_parts)
Cap_tr     = np.concatenate(Cap_tr_parts)

# RUL training set (pre-EOL only)
EIS_rul_tr_parts = ([load_ca_rul(c)[0] for c in CA_EOL_TRAIN] +
                    [load_mt_rul(c)[0] for c in N10_TRAIN_CELLS])
RUL_tr_parts     = ([load_ca_rul(c)[1] for c in CA_EOL_TRAIN] +
                    [load_mt_rul(c)[1] for c in N10_TRAIN_CELLS])
EIS_rul_tr = np.vstack(EIS_rul_tr_parts)
RUL_tr     = np.concatenate(RUL_tr_parts)

print(f'  Capacity train : {EIS_cap_tr.shape[0]} rows  '
      f'({" + ".join(CA_TRAIN_CELLS + N10_TRAIN_CELLS)})')
print(f'  RUL train      : {EIS_rul_tr.shape[0]} rows  '
      f'({" + ".join(CA_EOL_TRAIN + N10_TRAIN_CELLS)})')
print(f'  RUL train range: {RUL_tr.min():.0f} – {RUL_tr.max():.0f} bat-cycles')

# Training-only z-score (paper multi-T convention)
_, mu_eis, sig_eis = zscore(EIS_cap_tr)
_, mu_rul, sig_rul = zscore(EIS_rul_tr)   # separate norm ref for RUL training

X_cap_tr_n = apply_norm(EIS_cap_tr, mu_eis, sig_eis)
X_rul_tr_n = apply_norm(EIS_rul_tr, mu_rul, sig_rul)

# y normalisation for capacity (stability)
y_cap_mean = Cap_tr.mean();  y_cap_std = Cap_tr.std()
y_cap_tr_n = (Cap_tr - y_cap_mean) / y_cap_std

# ── Model 1: Capacity GPR (RBF) ───────────────────────────────────────────────
print('\n' + '=' * 60)
print('MODEL 1 — Multi-T Capacity GPR  (ScaleKernel + RBF)')
print(f'  Train: {len(CA_TRAIN_CELLS)} RT + {len(N10_TRAIN_CELLS)} × -10°C cells')
print('=' * 60)

X_cap_tr_t = to_tensor(X_cap_tr_n)
y_cap_tr_t = to_tensor(y_cap_tr_n)

likelihood_cap = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
model_cap      = RBFModel(X_cap_tr_t, y_cap_tr_t, likelihood_cap).to(DEVICE)

print('  Training ...')
model_cap, likelihood_cap = train_gp(
    model_cap, likelihood_cap, X_cap_tr_t, y_cap_tr_t, n_iter=600, lr=0.05
)
ls_cap = model_cap.covar_module.base_kernel.lengthscale.item()
print(f'  Optimised RBF length-scale: {ls_cap:.3f}')

# Predict per test cell
all_cap_cells = CA_TEST_CELLS + N10_TEST_CELLS + N20_TEST_CELLS
r2_cap = {}
pred_cap_all = {}
meas_cap_all = {}

for cell in all_cap_cells:
    eis_te, cap_te = (load_ca(cell) if cell.startswith('CA')
                      else (load_mt(cell) if cell.startswith('N')
                            else (None, None)))
    X_te_n = apply_norm(eis_te, mu_eis, sig_eis)
    X_te_t = to_tensor(X_te_n)

    mean_n, std_n = predict_gp(model_cap, likelihood_cap, X_te_t)
    mean_mah = mean_n * y_cap_std + y_cap_mean
    std_mah  = std_n  * y_cap_std

    r2 = r2_score(cap_te, mean_mah)
    r2_cap[cell]        = r2
    pred_cap_all[cell]  = (mean_mah, std_mah)
    meas_cap_all[cell]  = cap_te

    temp_tag = 'RT' if cell.startswith('CA') else ('-10°C' if cell.startswith('N10') else '-20°C')
    print(f'  {cell} ({temp_tag}): R² = {r2:.4f}')

# Plot capacity trajectories
n_cells = len(all_cap_cells)
fig, axes = plt.subplots(2, (n_cells + 1) // 2, figsize=(4 * ((n_cells + 1) // 2), 8))
axes = axes.ravel()
temp_colors = {'CA': '#1f77b4', 'N10': '#ff7f0e', 'N20': '#d62728'}

for idx, cell in enumerate(all_cap_cells):
    ax = axes[idx]
    cap_te = meas_cap_all[cell]
    mean_mah, std_mah = pred_cap_all[cell]
    cyc = np.arange(len(cap_te))
    cap0 = cap_te[0]

    col = temp_colors['CA'] if cell.startswith('CA') else (
          temp_colors['N10'] if cell.startswith('N10') else temp_colors['N20'])

    ax.fill_between(cyc,
                    (mean_mah - std_mah) / cap0, (mean_mah + std_mah) / cap0,
                    alpha=0.25, color=col)
    ax.plot(cyc, cap_te / cap0, 'x', color='grey', ms=3, alpha=0.6, label='Measured')
    ax.plot(cyc, mean_mah / cap0, '-', color=col, lw=1.5, label='GPR')
    ax.axhline(0.8, color='black', lw=0.8, ls='--', alpha=0.5)

    temp_tag = 'RT' if cell.startswith('CA') else ('-10°C' if cell.startswith('N10') else '-20°C')
    ax.set_title(f'{cell} ({temp_tag})\nR²={r2_cap[cell]:.3f}', fontsize=9)
    ax.set_xlabel('Cycle', fontsize=8)
    ax.set_ylabel('Norm. Capacity', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, frameon=False)

for idx in range(len(all_cap_cells), len(axes)):
    axes[idx].set_visible(False)

fig.suptitle('Multi-T Capacity GPR — Train: RT+(-10°C), Test per temperature',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'fig_multitemp_capacity.png', dpi=150)
plt.close()
print(f'  Saved: fig_multitemp_capacity.png')


# ── Model 2: RUL GPR (Linear) ─────────────────────────────────────────────────
print('\n' + '=' * 60)
print('MODEL 2 — Multi-T RUL GPR  (ScaleKernel + Linear, zero mean)')
print(f'  Train: {len(CA_EOL_TRAIN)} RT EOL + {len(N10_TRAIN_CELLS)} × -10°C cells')
print('=' * 60)

X_rul_tr_t = to_tensor(X_rul_tr_n)
y_rul_tr_t = to_tensor(RUL_tr.astype(np.float32))

likelihood_rul = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
model_rul      = LinearModel(X_rul_tr_t, y_rul_tr_t, likelihood_rul).to(DEVICE)

print('  Training ...')
model_rul, likelihood_rul = train_gp(
    model_rul, likelihood_rul, X_rul_tr_t, y_rul_tr_t, n_iter=600, lr=0.05
)

# Predict per N20 test cell
rul_test_cells = N20_TEST_CELLS
r2_rul  = {}
pred_rul_all = {}
meas_rul_all = {}

for cell in rul_test_cells:
    eis_te, rul_te = load_mt_rul(cell)
    X_te_n = apply_norm(eis_te, mu_rul, sig_rul)
    X_te_t = to_tensor(X_te_n)

    mean_rul, std_rul = predict_gp(model_rul, likelihood_rul, X_te_t)
    r2 = r2_score(rul_te, mean_rul)
    r2_rul[cell]       = r2
    pred_rul_all[cell] = (mean_rul, std_rul)
    meas_rul_all[cell] = rul_te
    print(f'  {cell} (-20°C): R² = {r2:.4f}  '
          f'(RUL_max={int(rul_te[0])}, n={len(rul_te)}, '
          f'pred_mean={mean_rul.mean():.1f}, actual_mean={rul_te.mean():.1f})')

# Plot RUL scatter
fig, axes = plt.subplots(1, len(rul_test_cells), figsize=(4 * len(rul_test_cells), 5))
if len(rul_test_cells) == 1:
    axes = [axes]

for idx, cell in enumerate(rul_test_cells):
    ax = axes[idx]
    rul_te  = meas_rul_all[cell]
    mean_r, std_r = pred_rul_all[cell]
    rul_max = max(rul_te.max(), mean_r.max()) * 1.1

    ax.fill_between(rul_te, mean_r - std_r, mean_r + std_r, alpha=0.3, color='#d62728')
    ax.scatter(rul_te, mean_r, s=20, color='#d62728', alpha=0.8, zorder=3)
    ax.plot([0, rul_max], [0, rul_max], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('Actual RUL (bat-cycles)', fontsize=10)
    ax.set_ylabel('Predicted RUL', fontsize=10)
    ax.set_title(f'{cell} (-20°C)\nR²={r2_rul[cell]:.3f}', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, rul_max); ax.set_ylim(0, rul_max)

mean_r2_rul = np.mean(list(r2_rul.values()))
fig.suptitle(f'Multi-T RUL GPR  (-20°C test, mean R²={mean_r2_rul:.3f})\n'
             f'Train: RT+(-10°C)  |  Linear kernel  |  train-only norm',
             fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'fig_multitemp_rul_scatter.png', dpi=150)
plt.close()
print(f'  Saved: fig_multitemp_rul_scatter.png')


# ── Model 3: ARD feature importance ───────────────────────────────────────────
print('\n' + '=' * 60)
print('MODEL 3 — ARD  (RBF-ARD, 66 features, multi-T training set)')
print('=' * 60)

# Subsample every 2nd row to keep kernel matrix tractable
EIS_ard_tr = EIS_cap_tr[::2]
Cap_ard_tr = Cap_tr[::2]
X_ard_tr_n = apply_norm(EIS_ard_tr, mu_eis, sig_eis)
y_ard_n    = (Cap_ard_tr - y_cap_mean) / y_cap_std

print(f'  ARD train size (every-2nd): {EIS_ard_tr.shape[0]} rows')

X_ard_t = to_tensor(X_ard_tr_n)
y_ard_t = to_tensor(y_ard_n)

likelihood_ard = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
model_ard      = ARDModel(X_ard_t, y_ard_t, likelihood_ard, n_feat=66).to(DEVICE)

print('  Training ARD ...')
model_ard, likelihood_ard = train_gp(
    model_ard, likelihood_ard, X_ard_t, y_ard_t, n_iter=500, lr=0.05
)

w = ard_weights(model_ard)
w_re = w[:33];  w_im = w[33:]

top_feat_re = int(np.argmax(w_re)) + 1       # 1-indexed in Re block
top_feat_im = int(np.argmax(w_im)) + 33 + 1  # 1-indexed in Im block (1-66)
top_freq_re = NATIVE_FREQS[np.argmax(w_re)]
top_freq_im = NATIVE_FREQS[np.argmax(w_im)]

print(f'  Top Re(Z) feature: #{top_feat_re}  ({top_freq_re:.2f} Hz)')
print(f'  Top Im(Z) feature: #{top_feat_im}  ({top_freq_im:.2f} Hz)')

# Frequencies low → high for x-axis (same as existing LOOCV plots)
freqs_plot = NATIVE_FREQS[::-1]
wr_plot = w_re[::-1]
wi_plot = w_im[::-1]

fig, ax = plt.subplots(figsize=(12, 4))
ax.semilogx(freqs_plot, wr_plot, 'r-^', ms=5, lw=1.5, label='Re(Z)')
ax.semilogx(freqs_plot, wi_plot, color='orange', linestyle='--',
            marker='v', ms=5, lw=1.5, label='-Im(Z)')

# Annotate top features
for freq, w_val, label, col in [
    (top_freq_re, w_re[np.argmax(w_re)], f'{top_freq_re:.1f} Hz', 'red'),
    (top_freq_im, w_im[np.argmax(w_im)], f'{top_freq_im:.1f} Hz', 'darkorange'),
]:
    ax.annotate(label, xy=(freq, w_val),
                xytext=(freq * 2, w_val * 1.15), fontsize=10, color=col,
                arrowprops=dict(arrowstyle='->', color=col))

ax.axhline(0, color='black', lw=0.6)
ax.set_xlabel('Frequency (Hz)', fontsize=13)
ax.set_ylabel('ARD weight', fontsize=13)
ax.set_title('ARD feature importance — Multi-T training (RT + -10°C)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / 'fig_multitemp_ARD.png', dpi=150)
plt.close()
print(f'  Saved: fig_multitemp_ARD.png')


# ── Summary ───────────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('RESULTS SUMMARY')
print('=' * 60)

print('\nCapacity GPR R² per test cell:')
for cell in all_cap_cells:
    temp = 'RT' if cell.startswith('CA') else ('-10°C' if cell.startswith('N10') else '-20°C')
    tag  = '← in-distrib'  if temp != '-20°C' else '← NEW TEMP'
    print(f'  {cell:<10} ({temp:>5})  R² = {r2_cap[cell]:.4f}  {tag}')

print(f'\nRUL GPR R² per test cell (-20°C only):')
for cell in rul_test_cells:
    print(f'  {cell:<10} (-20°C)  R² = {r2_rul[cell]:.4f}  '
          f'RUL_max={int(meas_rul_all[cell][0])}')
print(f'  Mean RUL R²: {mean_r2_rul:.4f}')

print(f'\nARD top features:')
print(f'  Re(Z): #{top_feat_re} @ {top_freq_re:.2f} Hz')
print(f'  Im(Z): #{top_feat_im} @ {top_freq_im:.2f} Hz')
print(f'\nFigures saved to: {OUT}')
