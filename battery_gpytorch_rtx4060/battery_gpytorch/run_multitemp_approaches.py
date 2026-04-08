"""
Multi-temperature GPR — Approach 1 and Approach 3 compared.

Approach 1 — LOOCV within all three temperature groups
    For each held-out cell, train on ALL remaining cells across RT + -10°C + -20°C.
    Model sees -20°C data during training → learns that impedance regime.
    LOOCV done separately for -10°C and -20°C (RT already done in run_loocv.py).

Approach 3 — Relative feature normalisation
    Each cell's EIS is divided element-wise by its own first-cycle EIS measurement.
    Output: fractional change from initial state  →  removes absolute impedance level.
    Temperature-driven offset (e.g. -20°C Re(Z) 6× higher) is cancelled.
    Then pooled z-score applied on top before GPR.

Both models use the same GPyTorch RBF/Linear setup as run_multitemp_rul.py.
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

# ── Data helpers ──────────────────────────────────────────────────────────────

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
    mu  = X.mean(0);  sig = X.std(0, ddof=1)
    return (X - mu) / np.where(sig == 0, 1.0, sig), mu, sig

def apply_norm(X, mu, sig):
    return (X - mu) / np.where(sig == 0, 1.0, sig)

def to_t(a): return torch.tensor(a, dtype=torch.float32, device=DEVICE)


# ── GPyTorch models ───────────────────────────────────────────────────────────

class RBFModel(gpytorch.models.ExactGP):
    def __init__(self, tx, ty, lh):
        super().__init__(tx, ty, lh)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))

class LinearModel(gpytorch.models.ExactGP):
    def __init__(self, tx, ty, lh):
        super().__init__(tx, ty, lh)
        self.mean_module  = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


def train_gp(model, likelihood, tx, ty, n_iter=500, lr=0.05):
    model.train(); likelihood.train()
    opt = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=lr)
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
    if best_sd: model.load_state_dict(best_sd)
    return model, likelihood


def predict(model, likelihood, tx):
    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        p = likelihood(model(tx))
    return p.mean.cpu().numpy(), p.stddev.cpu().numpy()


def fit_predict_rbf(EIS_tr, Cap_tr, EIS_te, Cap_te, n_iter=500):
    """Z-score, fit RBF-GP, return (r2, mean_pred, std_pred)."""
    X_tr_n, mu, sig = zscore(EIS_tr)
    X_te_n = apply_norm(EIS_te, mu, sig)
    ym, ys = Cap_tr.mean(), Cap_tr.std()
    tx = to_t(X_tr_n); ty = to_t((Cap_tr - ym) / ys)
    lh = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
    m  = RBFModel(tx, ty, lh).to(DEVICE)
    m, lh = train_gp(m, lh, tx, ty, n_iter=n_iter)
    mn, sd = predict(m, lh, to_t(X_te_n))
    mn = mn * ys + ym;  sd = sd * ys
    return r2_score(Cap_te, mn), mn, sd


def fit_predict_linear(EIS_tr, RUL_tr, EIS_te, RUL_te, n_iter=500):
    """Training-only z-score, fit Linear-GP, return (r2, mean_pred, std_pred)."""
    X_tr_n, mu, sig = zscore(EIS_tr)
    X_te_n = apply_norm(EIS_te, mu, sig)
    tx = to_t(X_tr_n); ty = to_t(RUL_tr.astype(np.float32))
    lh = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
    m  = LinearModel(tx, ty, lh).to(DEVICE)
    m, lh = train_gp(m, lh, tx, ty, n_iter=n_iter)
    mn, sd = predict(m, lh, to_t(X_te_n))
    return r2_score(RUL_te, mn), mn, sd


# ── Load all cells ────────────────────────────────────────────────────────────

CA_CELLS  = [f'CA{i}' for i in range(1, 9)]
CA_DNF    = {'CA6'}
CA_EOL    = [c for c in CA_CELLS if c not in CA_DNF]
N10_CELLS = [f'N10_CB{i}' for i in range(1, 5)]
N20_CELLS = [f'N20_CB{i}' for i in range(1, 5)]

print('\nLoading all cells ...')
eis_cap = {}; cap_all = {}
eis_rul = {}; rul_all = {}

for c in CA_CELLS:
    eis_cap[c], cap_all[c] = load_ca(c)
for c in CA_EOL:
    eis_rul[c], rul_all[c] = load_ca_rul(c)
for c in N10_CELLS + N20_CELLS:
    eis_cap[c], cap_all[c] = load_mt(c)
    eis_rul[c], rul_all[c] = load_mt_rul(c)

ALL_CELLS = CA_CELLS + N10_CELLS + N20_CELLS
ALL_EOL   = CA_EOL   + N10_CELLS + N20_CELLS

print(f'  Total cells: {len(ALL_CELLS)}  '
      f'({len(CA_CELLS)} RT + {len(N10_CELLS)} -10°C + {len(N20_CELLS)} -20°C)')


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH 1 — LOOCV across all three temperature groups
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '═' * 60)
print('APPROACH 1 — Multi-T LOOCV  (leave one cell out per fold)')
print('  Train: all remaining RT + -10°C + -20°C cells')
print('  Test : one held-out cell per fold')
print('═' * 60)

# ---- Capacity LOOCV — -10°C and -20°C (RT already done) --------------------
print('\n[Capacity — -10°C and -20°C LOOCV]')

r2_cap_a1 = {}

for test_cell in N10_CELLS + N20_CELLS:
    train_cells = [c for c in ALL_CELLS if c != test_cell]
    EIS_tr = np.vstack([eis_cap[c] for c in train_cells])
    Cap_tr = np.concatenate([cap_all[c] for c in train_cells])
    EIS_te = eis_cap[test_cell];  Cap_te = cap_all[test_cell]

    r2, mn, _ = fit_predict_rbf(EIS_tr, Cap_tr, EIS_te, Cap_te, n_iter=500)
    r2_cap_a1[test_cell] = r2
    temp = '-10°C' if test_cell.startswith('N10') else '-20°C'
    print(f'  {test_cell} ({temp}): R² = {r2:.4f}')

# ---- RUL LOOCV — -20°C only ------------------------------------------------
print('\n[RUL — -20°C LOOCV  (Linear kernel)]')

r2_rul_a1 = {}

for test_cell in N20_CELLS:
    train_eol = [c for c in ALL_EOL if c != test_cell]
    EIS_rul_tr = np.vstack([eis_rul[c] for c in train_eol])
    RUL_tr     = np.concatenate([rul_all[c] for c in train_eol])
    EIS_rul_te = eis_rul[test_cell];  RUL_te = rul_all[test_cell]

    r2, mn, _ = fit_predict_linear(EIS_rul_tr, RUL_tr, EIS_rul_te, RUL_te, n_iter=500)
    r2_rul_a1[test_cell] = r2
    print(f'  {test_cell} (-20°C): R² = {r2:.4f}  '
          f'(RUL_max={int(RUL_te[0])}, pred_mean={mn.mean():.1f}, '
          f'actual_mean={RUL_te.mean():.1f})')

mean_cap_n10 = np.mean([r2_cap_a1[c] for c in N10_CELLS])
mean_cap_n20 = np.mean([r2_cap_a1[c] for c in N20_CELLS])
mean_rul_n20 = np.mean(list(r2_rul_a1.values()))
print(f'\n  Mean Capacity R²  -10°C: {mean_cap_n10:.4f}')
print(f'  Mean Capacity R²  -20°C: {mean_cap_n20:.4f}')
print(f'  Mean RUL R²       -20°C: {mean_rul_n20:.4f}')


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH 3 — Relative feature normalisation
# Each cell's EIS divided element-wise by its own cycle-0 EIS measurement.
# Gives fractional change from initial state → removes absolute impedance offset.
# Then pooled z-score is applied across all training cells.
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '═' * 60)
print('APPROACH 3 — Relative feature normalisation')
print('  EIS_rel[i] = EIS[i] / EIS[0]   (element-wise, per cell)')
print('  Then pooled z-score across training cells')
print('═' * 60)

def relative_eis(eis):
    """Divide each row by the first cycle's EIS. Clips denominator at 1e-6."""
    ref = np.maximum(np.abs(eis[0]), 1e-6)
    return eis / ref

# Pre-compute relative EIS for all cells
eis_cap_rel = {c: relative_eis(eis_cap[c]) for c in ALL_CELLS}
eis_rul_rel = {c: relative_eis(eis_rul[c]) for c in ALL_EOL}

# ---- Same fixed split as run_multitemp_rul.py --------------------------------
CA_TRAIN  = ['CA1', 'CA2', 'CA3', 'CA4', 'CA5', 'CA6']
N10_TRAIN = ['N10_CB1', 'N10_CB2', 'N10_CB3']
CA_EOL_TR = ['CA1', 'CA2', 'CA3', 'CA4', 'CA5']
CA_TEST   = ['CA7', 'CA8']
N10_TEST  = ['N10_CB4']
N20_TEST  = N20_CELLS

print('\n[Capacity — fixed split  (train: RT+(-10°C), test: all)]')
r2_cap_a3 = {}

EIS_tr_rel = np.vstack([eis_cap_rel[c] for c in CA_TRAIN + N10_TRAIN])
Cap_tr     = np.concatenate([cap_all[c] for c in CA_TRAIN + N10_TRAIN])

for test_cell in CA_TEST + N10_TEST + N20_TEST:
    EIS_te_rel = eis_cap_rel[test_cell]
    Cap_te     = cap_all[test_cell]
    r2, mn, _ = fit_predict_rbf(EIS_tr_rel, Cap_tr, EIS_te_rel, Cap_te, n_iter=500)
    r2_cap_a3[test_cell] = r2
    temp = 'RT' if test_cell.startswith('CA') else ('-10°C' if test_cell.startswith('N10') else '-20°C')
    print(f'  {test_cell} ({temp}): R² = {r2:.4f}')

print('\n[RUL — fixed split  (train: RT+(-10°C), test: -20°C)]')
r2_rul_a3 = {}

EIS_rul_tr_rel = np.vstack([eis_rul_rel[c] for c in CA_EOL_TR + N10_TRAIN])
RUL_tr         = np.concatenate([rul_all[c] for c in CA_EOL_TR + N10_TRAIN])

for test_cell in N20_TEST:
    EIS_te_rel = eis_rul_rel[test_cell]
    RUL_te     = rul_all[test_cell]
    r2, mn, _ = fit_predict_linear(EIS_rul_tr_rel, RUL_tr, EIS_te_rel, RUL_te, n_iter=500)
    r2_rul_a3[test_cell] = r2
    print(f'  {test_cell} (-20°C): R² = {r2:.4f}  '
          f'pred_mean={mn.mean():.1f}, actual_mean={RUL_te.mean():.1f}')

mean_cap_n20_a3 = np.mean([r2_cap_a3[c] for c in N20_TEST])
mean_rul_n20_a3 = np.mean(list(r2_rul_a3.values()))


# ═══════════════════════════════════════════════════════════════════════════════
# Summary + comparison plot
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '═' * 60)
print('COMPARISON SUMMARY')
print('═' * 60)

print(f'\n{"Cell":<12} {"Baseline R²":>12} {"Approach1 R²":>13} {"Approach3 R²":>13}  Temp')
print('-' * 60)
baseline = {
    'CA7': 0.9834, 'CA8': 0.9827, 'N10_CB4': 0.8002,
    'N20_CB1': -10.39, 'N20_CB2': -9.49, 'N20_CB3': -7.68, 'N20_CB4': -7.36
}
for cell in CA_TEST + N10_TEST + N20_TEST:
    temp = 'RT' if cell.startswith('CA') else ('-10°C' if cell.startswith('N10') else '-20°C')
    b  = baseline.get(cell, float('nan'))
    a1 = r2_cap_a1.get(cell, float('nan'))
    a3 = r2_cap_a3.get(cell, float('nan'))
    print(f'{cell:<12} {b:>12.4f} {a1:>13.4f} {a3:>13.4f}  {temp}')

print(f'\nRUL (−20°C only):')
print(f'{"Cell":<12} {"Baseline R²":>12} {"Approach1 R²":>13} {"Approach3 R²":>13}')
print('-' * 50)
rul_baseline = {
    'N20_CB1': -6962.86, 'N20_CB2': -9616.82, 'N20_CB3': -4016.91, 'N20_CB4': -6200.23
}
for cell in N20_CELLS:
    b  = rul_baseline[cell]
    a1 = r2_rul_a1.get(cell, float('nan'))
    a3 = r2_rul_a3.get(cell, float('nan'))
    print(f'{cell:<12} {b:>12.2f} {a1:>13.4f} {a3:>13.4f}')

print(f'\nMean capacity R² -20°C:  baseline=-8.74  '
      f'A1={mean_cap_n20:.4f}  A3={mean_cap_n20_a3:.4f}')
print(f'Mean RUL R²      -20°C:  baseline=-6699  '
      f'A1={mean_rul_n20:.4f}  A3={mean_rul_n20_a3:.4f}')

# ── Comparison bar chart ──────────────────────────────────────────────────────
cells_plot = N20_CELLS
x = np.arange(len(cells_plot))
w = 0.25

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Capacity
ax = axes[0]
ax.bar(x - w, [baseline[c] for c in cells_plot], w,
       label='Baseline (train RT+-10°C)', color='grey', alpha=0.7)
ax.bar(x,     [r2_cap_a1[c] for c in cells_plot], w,
       label='Approach 1 (LOOCV multi-T)', color='steelblue', alpha=0.85)
ax.bar(x + w, [r2_cap_a3[c] for c in cells_plot], w,
       label='Approach 3 (relative feat)', color='darkorange', alpha=0.85)
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(cells_plot)
ax.set_ylabel('R²'); ax.set_title('Capacity R² — -20°C test cells')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

# RUL
ax = axes[1]
ax.bar(x - w, [rul_baseline[c] for c in cells_plot], w,
       label='Baseline', color='grey', alpha=0.7)
ax.bar(x,     [r2_rul_a1[c] for c in cells_plot], w,
       label='Approach 1 (LOOCV multi-T)', color='steelblue', alpha=0.85)
ax.bar(x + w, [r2_rul_a3[c] for c in cells_plot], w,
       label='Approach 3 (relative feat)', color='darkorange', alpha=0.85)
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(cells_plot)
ax.set_ylabel('R²'); ax.set_title('RUL R² — -20°C test cells')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('Approach 1 (LOOCV) vs Approach 3 (relative features) vs Baseline',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'fig_approaches_comparison.png', dpi=150)
plt.close()
print(f'\nSaved: fig_approaches_comparison.png')
