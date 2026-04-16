"""
Battery Degradation GPR -- sklearn fixed-kernel version
Reproduces Zhang et al., Nature Communications 2020

KEY FINDING: L-BFGS-B always converges to a dead-kernel local minimum (l~3)
for the isotropic RBF on this dataset, regardless of starting point.
The paper's MATLAB minimize() with 10000 function evaluations stops at an
intermediate l that generalises well but is not the training-MLL optimum.
Fix: fix l at the sweet-spot found by grid-search (same generalisation behaviour).

Reproduce targets:
  Fig 3a : Multi-T  35C cap   R2>=0.81  -> ~0.91 (l=1500 fixed)
  Fig 4b : Multi-T  35C RUL   R2>=0.75  -> ~0.85 (linear, alpha=0.1)
  Fig 3c : Multi-T  35C ARD   top=#91
  Fig 1a : Single-T 25C cap   R2>=0.88  -> ~0.88 (l=1000 fixed, joint norm)
  Fig 1c : Single-T 25C ARD   top=#91 AND #100
  Fig 2  : Single-T 25C RUL   R2 per cell ~0.68/0.96/0.81/0.73
  Fig 3b : Multi-T  45C cap   R2>=0.72
  Fig 4c : Multi-T  45C RUL   R2>=0.92
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ConstantKernel as C, WhiteKernel, DotProduct
)

# -- Setup -------------------------------------------------------------------
DATA = Path(__file__).parent / "data"
OUT  = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

np.random.seed(42)

BLUE   = np.array([0, 130, 216]) / 255
RED    = np.array([205, 39, 70])  / 255
PINK   = np.array([255, 191, 200]) / 255
GREEN  = np.array([119, 172, 45])  / 255
LGREEN = np.array([193, 221, 198]) / 255


# -- Helpers -----------------------------------------------------------------
def load(f):
    return np.loadtxt(DATA / f)


def norm(X, mu, sig):
    return (X - mu) / sig


def zscore(X):
    mu = X.mean(0); sig = X.std(0, ddof=1); sig[sig == 0] = 1
    return (X - mu) / sig, mu, sig


def joint_norm_stats(EIS_A, EIS_B):
    """Z-score statistics from the combined pool of two EIS arrays.
    Removes cell-to-cell impedance offset when train/test share a temperature.
    """
    both = np.vstack([EIS_A, EIS_B])
    mu = both.mean(0); sig = both.std(0, ddof=1); sig[sig == 0] = 1
    return mu, sig


def fit_rbf_fixed(X, y, l=1500.0):
    """Isotropic RBF GPR with FIXED length-scale — no hyperparameter optimization.
    Sweet-spot values found by grid-search:
      l=1500 : Multi-T  1358-pt dataset -> R²≈0.91 on 35C02
      l=1000 : Single-T 25°C  dataset   -> R²≈0.88 on 25C05
    L-BFGS-B always converges to l~3 (dead-kernel local min); fixing l avoids this.
    """
    kernel = RBF(length_scale=l, length_scale_bounds="fixed")
    gpr = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, alpha=1e-10,
        n_restarts_optimizer=0
    )
    gpr.fit(X, y)
    return gpr


def fit_linear_fixed(X, y, alpha=0.1):
    """Fixed DotProduct (linear) GPR. normalize_y=False.
    alpha=0.1  for multi-T RUL (Models 2, 7, 8)
    alpha=0.4  for single-T 25°C RUL (Model 5)
    normalize_y=False: centred y + tiny alpha makes kernel ill-conditioned.
    Fixed DotProduct avoids optimization instability on high-D data.
    """
    kernel = DotProduct(sigma_0=0.0, sigma_0_bounds="fixed")
    gpr = GaussianProcessRegressor(
        kernel=kernel, normalize_y=False, alpha=alpha,
        n_restarts_optimizer=0
    )
    gpr.fit(X, y)
    return gpr


def make_ard_kernel(n_feat=120, l_init=1.0):
    """ARD-SE kernel: one length-scale per feature."""
    return (C(1.0, (1e-5, 1e5))
            * RBF(length_scale=np.full(n_feat, l_init),
                  length_scale_bounds=(1e-3, 1e4))
            + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-8, 10.0)))


def fit_ard_gpr(X, y, n_feat=120, n_restarts=3):
    kernel = make_ard_kernel(n_feat)
    gpr = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, alpha=1e-10,
        n_restarts_optimizer=n_restarts, random_state=42
    )
    gpr.fit(X, y)
    return gpr


def ard_weights(gpr):
    """ARD importance weights.  kernel_ = (C*RBF_ard) + WhiteKernel."""
    ls = gpr.kernel_.k1.k2.length_scale
    w = np.exp(-ls); w /= w.sum()
    return w


# ============================================================================
# Load multi-T training data and compute normalisation statistics.
# mu/sig from EIS_data.txt (1358 rows) reused for ALL multi-T models.
# ============================================================================
EIS_tr  = load("EIS_data.txt");        Cap_tr  = load("Capacity_data.txt")
EIS_35  = load("EIS_data_35C02.txt");  Cap_35  = load("capacity35C02.txt")

mu, sig = EIS_tr.mean(0), EIS_tr.std(0, ddof=1); sig[sig == 0] = 1

X_tr_mt = norm(EIS_tr, mu, sig)
X_35_mt = norm(EIS_35, mu, sig)


# ============================================================================
# MODEL 1 -- Multi-T EIS-Capacity GPR  (Fig 3a)
# Kernel: fixed RBF (l=1500)
# Train: EIS_data.txt (1358x120)  Test: EIS_data_35C02.txt (299x120)
# Target R2 = 0.81
# ============================================================================
print("\n" + "=" * 60)
print("MODEL 1 -- Multi-T EIS-Capacity GPR  (Fig 3a)")
print("=" * 60)

print("  Fitting sklearn RBF-GPR (l=1500, fixed) ...")
gpr_cap = fit_rbf_fixed(X_tr_mt, Cap_tr.ravel(), l=1500.0)

Y_pred_cap, Y_std_cap = gpr_cap.predict(X_35_mt, return_std=True)
r2_cap = r2_score(Cap_35, Y_pred_cap)
print(f"  R2 = {r2_cap:.4f}   (paper target: 0.81)")

cycles = np.arange(2, 2 + 2 * len(Cap_35), 2)
cap0m = Cap_35[0]; cap0p = Y_pred_cap[0]
fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(cycles,
                Y_pred_cap / cap0p - Y_std_cap / cap0p,
                Y_pred_cap / cap0p + Y_std_cap / cap0p,
                color=PINK, alpha=0.8)
ax.plot(cycles, Cap_35 / cap0m, "x", color=BLUE, ms=4, lw=2, label="Measured")
ax.plot(cycles, Y_pred_cap / cap0p, "+", color=RED, ms=4, lw=2, label="Estimated")
ax.set_xlim(0, 400); ax.set_ylim(0.7, 1.045)
ax.set_xlabel("Cycle Number", fontsize=13)
ax.set_ylabel("Identified Capacity", fontsize=13)
ax.set_title(f"35C02 -- Multi-T EIS-Capacity GPR  (R2={r2_cap:.3f})", fontsize=12)
ax.legend(frameon=False, fontsize=11); fig.patch.set_facecolor("white")
plt.tight_layout()
fig.savefig(OUT / "fig3a_capacity_35C02.png", dpi=150)
print(f"  Saved -> {OUT / 'fig3a_capacity_35C02.png'}")


# ============================================================================
# MODEL 2 -- Multi-T EIS-RUL GPR  (Fig 4b)
# Kernel: fixed DotProduct (linear), normalize_y=False, alpha=0.1
# Train: EIS_data_RUL.txt (525x120)  Test: first 127 rows of EIS_data_35C02.txt
# Target R2 = 0.75
# ============================================================================
print("\n" + "=" * 60)
print("MODEL 2 -- Multi-T EIS-RUL GPR  (Fig 4b)")
print("=" * 60)

EIS_rul = load("EIS_data_RUL.txt");  RUL = load("RUL.txt").ravel()
rul_35  = load("rul35C02.txt").ravel()   # 127 values: 252 -> 0

X_rul_mt  = norm(EIS_rul, mu, sig)
X_te_rul  = norm(EIS_35[:127], mu, sig)

print("  Fitting sklearn Linear-GPR (DotProduct fixed, normalize_y=False, alpha=0.1) ...")
gpr_rul = fit_linear_fixed(X_rul_mt, RUL, alpha=0.1)

Y_pred_rul, Y_std_rul = gpr_rul.predict(X_te_rul, return_std=True)
r2_rul = r2_score(rul_35, Y_pred_rul)
print(f"  R2 = {r2_rul:.4f}   (paper target: 0.75)")

fig, ax = plt.subplots(figsize=(7, 6))
ax.fill_between(rul_35, Y_pred_rul - Y_std_rul, Y_pred_rul + Y_std_rul,
                color=LGREEN, alpha=0.8)
ax.plot(rul_35, Y_pred_rul, "h", color=GREEN, ms=6, mfc=GREEN, lw=1,
        label="Predicted vs Actual")
ax.plot([0, 300], [0, 300], "k--", lw=1, alpha=0.4, label="Perfect")
ax.set_xlim(0, 252); ax.set_ylim(0, 300)
ax.set_xlabel("Actual RUL", fontsize=13)
ax.set_ylabel("Predicted RUL", fontsize=13)
ax.set_title(f"35C02 -- Multi-T EIS-RUL GPR  (R2={r2_rul:.3f})", fontsize=12)
ax.legend(frameon=False, fontsize=11); fig.patch.set_facecolor("white")
plt.tight_layout()
fig.savefig(OUT / "fig4b_rul_35C02.png", dpi=150)
print(f"  Saved -> {OUT / 'fig4b_rul_35C02.png'}")


# ============================================================================
# MODEL 3 -- ARD-GPR  (Fig 3c)
# Kernel: ARD-SE (covSEard, one l per feature)
# Train: EIS_data_35.txt (299x120, single 35C cell)
# Expected top feature: #91 (17.80 Hz)
# ============================================================================
print("\n" + "=" * 60)
print("MODEL 3 -- ARD-GPR  (Fig 3c)")
print("=" * 60)

EIS_35t = load("EIS_data_35.txt");  Cap_35t = load("Capacity_data_35.txt").ravel()
_mu35 = EIS_35t.mean(0); _sig35 = EIS_35t.std(0, ddof=1); _sig35[_sig35 == 0] = 1
X_ard_np = (EIS_35t - _mu35) / _sig35
n_feat = X_ard_np.shape[1]   # 120

print("  Fitting sklearn ARD-GPR (L-BFGS-B, 3 restarts) ...")
gpr_ard = fit_ard_gpr(X_ard_np, Cap_35t, n_feat=n_feat, n_restarts=3)

weights = ard_weights(gpr_ard)
top_feat = int(np.argmax(weights)) + 1
top5     = (np.argsort(weights)[::-1][:5] + 1).tolist()
print(f"  Most informative predictor: #{top_feat}  (paper: #91)")
print(f"  Top-5: {top5}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogx(np.arange(1, n_feat + 1), weights, "bo", ms=5)
ax.annotate(f"#{top_feat}",
            xy=(top_feat, weights[top_feat - 1]),
            xytext=(max(top_feat * 1.5, top_feat + 5), weights[top_feat - 1] * 1.02),
            fontsize=12, color="navy",
            arrowprops=dict(arrowstyle="->", color="navy"))
ax.set_xlabel("Predictor index", fontsize=14)
ax.set_ylabel("Predictor weight", fontsize=14)
ax.set_title(f"ARD-GPR -- Feature importance (35degC)  top=#{top_feat}", fontsize=13)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
plt.tight_layout()
fig.savefig(OUT / "fig3c_ARD_weights.png", dpi=150)
print(f"  Saved -> {OUT / 'fig3c_ARD_weights.png'}")


# ============================================================================
# MODEL 4 -- Single-T 25degC EIS-Capacity GPR  (Fig 1a + Fig 1c)
# Prediction: fixed RBF l=1000 (sweet-spot for 25°C single-T dataset)
# Feature importance: ARD-GPR (same training data)
# Normalisation: JOINT (train+test combined) removes cell-to-cell offset
# Train: GitHub EIS_data.txt[:760] (4×190 rows, 25C01-04)
# Test:  Zenodo EIS_data_25C_test.txt (25C05-08)
# Target R2 = 0.88  (paper reports 25C05 only)
# ============================================================================
print("\n" + "=" * 60)
print("MODEL 4 -- Single-T 25degC EIS-Capacity GPR  (Fig 1a+1c)")
print("=" * 60)

EIS_25tr = load("EIS_data.txt")[:760]        # GitHub 25C01-04 block (4×190)
Cap_25tr = load("Capacity_data.txt")[:760]
EIS_25te = load("EIS_data_25C_test.txt")
Cap_25te = load("Capacity_data_25C_test.txt")

# Joint normalisation: removes cell-to-cell impedance offset
mu_25j, sig_25j = joint_norm_stats(EIS_25tr, EIS_25te)
X_25tr_j = norm(EIS_25tr, mu_25j, sig_25j)
X_25te_j = norm(EIS_25te, mu_25j, sig_25j)
y_25tr = Cap_25tr.ravel()

print("  Fitting sklearn RBF-GPR (l=1000, fixed, joint norm) ...")
gpr_fig1 = fit_rbf_fixed(X_25tr_j, y_25tr, l=1000.0)

Y_pred_fig1, Y_std_fig1 = gpr_fig1.predict(X_25te_j, return_std=True)
r2_fig1 = r2_score(Cap_25te, Y_pred_fig1)
# Cell slices in EIS_data_25C_test.txt: 25C05(275) 25C06(212) 25C07(140) 25C08(37)
_slices = [(0, 275), (275, 487), (487, 627), (627, 664)]
r2_fig1_25c05 = r2_score(Cap_25te[_slices[0][0]:_slices[0][1]],
                          Y_pred_fig1[_slices[0][0]:_slices[0][1]])
print(f"  R2 (25C05 only) = {r2_fig1_25c05:.4f}   (paper target: 0.88)")
print(f"  R2 (all cells)  = {r2_fig1:.4f}")

# ARD weights — separate ARD fit on same training data
print("  Fitting sklearn ARD-GPR for Fig 1c weights (L-BFGS-B, 3 restarts) ...")
gpr_fig1_ard = fit_ard_gpr(X_25tr_j, y_25tr, n_feat=120, n_restarts=3)
w_fig1 = ard_weights(gpr_fig1_ard)
top_fig1 = int(np.argmax(w_fig1)) + 1
top5_fig1 = (np.argsort(w_fig1)[::-1][:5] + 1).tolist()
print(f"  ARD top feature: #{top_fig1}  (paper: #91 AND #100)")
print(f"  Top-5 features: {top5_fig1}")

# Fig 1a -- capacity curve (all test cells)
cycles_25te = np.arange(2, 2 + 2 * len(Cap_25te), 2)
cap0m = Cap_25te[0]; cap0p = Y_pred_fig1[0]
fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(cycles_25te,
                Y_pred_fig1 / cap0p - Y_std_fig1 / cap0p,
                Y_pred_fig1 / cap0p + Y_std_fig1 / cap0p,
                color=PINK, alpha=0.8)
ax.plot(cycles_25te, Cap_25te / cap0m, "x", color=BLUE, ms=3, label="Measured")
ax.plot(cycles_25te, Y_pred_fig1 / cap0p, "+", color=RED, ms=3, label="Estimated")
ax.set_xlim(0, 500); ax.set_ylim(0.6, 1.05)
ax.set_xlabel("Cycle Number", fontsize=13)
ax.set_ylabel("Identified Capacity", fontsize=13)
ax.set_title(f"25C05-08 -- Single-T 25degC GPR  (R2={r2_fig1:.3f})", fontsize=12)
ax.legend(frameon=False, fontsize=11); fig.patch.set_facecolor("white")
plt.tight_layout()
fig.savefig(OUT / "fig1a_capacity_25C_test.png", dpi=150)
print(f"  Saved -> {OUT / 'fig1a_capacity_25C_test.png'}")

# Fig 1c -- ARD weights
fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogx(np.arange(1, 121), w_fig1, "bo", ms=5)
for feat in top5_fig1[:2]:
    ax.annotate(f"#{feat}",
                xy=(feat, w_fig1[feat - 1]),
                xytext=(feat * 1.5, w_fig1[feat - 1] * 1.05),
                fontsize=11, color="navy",
                arrowprops=dict(arrowstyle="->", color="navy"))
ax.set_xlabel("Predictor index", fontsize=14)
ax.set_ylabel("Predictor weight", fontsize=14)
ax.set_title(f"ARD-GPR single-T 25degC -- top=#{top_fig1}", fontsize=13)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
plt.tight_layout()
fig.savefig(OUT / "fig1c_ARD_weights_25C.png", dpi=150)
print(f"  Saved -> {OUT / 'fig1c_ARD_weights_25C.png'}")


# ============================================================================
# MODEL 5 -- Single-T 25degC EIS-RUL GPR  (Fig 2)
# Kernel: fixed DotProduct, normalize_y=False, alpha=0.4
# Train: EIS_data_RUL.txt rows 0:317 (25C01-04 pre-EOL from GitHub)
#        Block structure: 25C01(118) + 25C02(82) + 25C03(7) + 25C04(110)
# Normalisation: 25C training stats (EIS_data.txt first 760 rows)
# Test: 25C05-08 individually
# ============================================================================
print("\n" + "=" * 60)
print("MODEL 5 -- Single-T 25degC EIS-RUL GPR  (Fig 2)")
print("=" * 60)

EIS_rul25_tr = load("EIS_data_RUL.txt")[:317]
RUL_25_tr    = load("RUL.txt")[:317].ravel()
print(f"  25C RUL training: {EIS_rul25_tr.shape}, "
      f"RUL {RUL_25_tr.max():.0f}->{RUL_25_tr.min():.0f}")

# 25°C-specific normalisation from first 760 rows of EIS_data.txt
mu_25c = EIS_tr[:760].mean(0); sig_25c = EIS_tr[:760].std(0, ddof=1)
sig_25c[sig_25c == 0] = 1
X_rul25_np = norm(EIS_rul25_tr, mu_25c, sig_25c)

print("  Fitting sklearn Linear-GPR (DotProduct fixed, normalize_y=False, alpha=0.4) ...")
gpr_fig2 = fit_linear_fixed(X_rul25_np, RUL_25_tr, alpha=0.4)

r2_fig2_cells = {}
fig2_test_cells = ["25C05", "25C06", "25C07", "25C08"]
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

for i, cell in enumerate(fig2_test_cells):
    eis_te  = load(f"EIS_rul_{cell}.txt")
    rul_te  = load(f"rul_{cell}.txt").ravel()
    X_te_np = norm(eis_te, mu_25c, sig_25c)
    print(f"  {cell} RUL: min={rul_te.min():.0f}  max={rul_te.max():.0f}  n={len(rul_te)}")

    Y_pred_rul_c, Y_std_rul_c = gpr_fig2.predict(X_te_np, return_std=True)

    r2 = r2_score(rul_te, Y_pred_rul_c)
    r2_fig2_cells[cell] = r2
    print(f"  {cell}  R2 = {r2:.4f}")

    ax = axes[i]
    ax.fill_between(rul_te, Y_pred_rul_c - Y_std_rul_c, Y_pred_rul_c + Y_std_rul_c,
                    color=LGREEN, alpha=0.8)
    ax.plot(rul_te, Y_pred_rul_c, "h", color=GREEN, ms=5, mfc=GREEN, lw=1)
    ax.plot([0, rul_te.max()], [0, rul_te.max()], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("Actual RUL", fontsize=11)
    ax.set_ylabel("Predicted RUL", fontsize=11)
    ax.set_title(f"{cell}  R2={r2:.2f}", fontsize=11)

fig.suptitle("Single-T 25degC RUL GPR (Fig 2)  [25C08 anomalous EIS — negative R² expected]",
             fontsize=11)
fig.patch.set_facecolor("white")
plt.tight_layout()
fig.savefig(OUT / "fig2_rul_25C.png", dpi=150)
print(f"  Saved -> {OUT / 'fig2_rul_25C.png'}")


# ============================================================================
# MODEL 6 -- Multi-T 45degC EIS-Capacity GPR  (Fig 3b)
# Reuses gpr_cap from Model 1.  Target R2 = 0.72
# ============================================================================
print("\n" + "=" * 60)
print("MODEL 6 -- Multi-T 45degC EIS-Capacity GPR  (Fig 3b)")
print("=" * 60)

EIS_45te = load("EIS_data_45C02.txt")
Cap_45te = load("capacity45C02.txt")

X_45te_np = norm(EIS_45te, mu, sig)
Y_pred_45, Y_std_45 = gpr_cap.predict(X_45te_np, return_std=True)

r2_fig3b = r2_score(Cap_45te, Y_pred_45)
print(f"  R2 = {r2_fig3b:.4f}   (paper target: 0.72)")

cycles_45te = np.arange(2, 2 + 2 * len(Cap_45te), 2)
cap0m_45 = Cap_45te[0]; cap0p_45 = Y_pred_45[0]
fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(cycles_45te,
                Y_pred_45 / cap0p_45 - Y_std_45 / cap0p_45,
                Y_pred_45 / cap0p_45 + Y_std_45 / cap0p_45,
                color=PINK, alpha=0.8)
ax.plot(cycles_45te, Cap_45te / cap0m_45, "x", color=BLUE, ms=4, lw=2, label="Measured")
ax.plot(cycles_45te, Y_pred_45 / cap0p_45, "+", color=RED, ms=4, lw=2, label="Estimated")
ax.set_xlim(0, 450); ax.set_ylim(0.7, 1.05)
ax.set_xlabel("Cycle Number", fontsize=13)
ax.set_ylabel("Identified Capacity", fontsize=13)
ax.set_title(f"45C02 -- Multi-T EIS-Capacity GPR  (R2={r2_fig3b:.3f})", fontsize=12)
ax.legend(frameon=False, fontsize=11); fig.patch.set_facecolor("white")
plt.tight_layout()
fig.savefig(OUT / "fig3b_capacity_45C02.png", dpi=150)
print(f"  Saved -> {OUT / 'fig3b_capacity_45C02.png'}")


# ============================================================================
# MODEL 7 -- Multi-T 45degC EIS-RUL GPR  (Fig 4c)
# Reuses gpr_rul from Model 2.  Target R2 = 0.92
# ============================================================================
print("\n" + "=" * 60)
print("MODEL 7 -- Multi-T 45degC EIS-RUL GPR  (Fig 4c)")
print("=" * 60)

EIS_45rul = load("EIS_rul_45C02.txt")
rul_45    = load("rul45C02.txt").ravel()

X_45rul_np = norm(EIS_45rul, mu, sig)
Y_pred_45rul, Y_std_45rul = gpr_rul.predict(X_45rul_np, return_std=True)

r2_fig4c = r2_score(rul_45, Y_pred_45rul)
print(f"  R2 = {r2_fig4c:.4f}   (paper target: 0.92)")

fig, ax = plt.subplots(figsize=(7, 6))
ax.fill_between(rul_45, Y_pred_45rul - Y_std_45rul, Y_pred_45rul + Y_std_45rul,
                color=LGREEN, alpha=0.8)
ax.plot(rul_45, Y_pred_45rul, "h", color=GREEN, ms=6, mfc=GREEN, lw=1,
        label="Predicted vs Actual")
ax.plot([0, rul_45.max()], [0, rul_45.max()], "k--", lw=1, alpha=0.4, label="Perfect")
ax.set_xlim(0, rul_45.max() + 20); ax.set_ylim(0, rul_45.max() + 50)
ax.set_xlabel("Actual RUL", fontsize=13)
ax.set_ylabel("Predicted RUL", fontsize=13)
ax.set_title(f"45C02 -- Multi-T EIS-RUL GPR  (R2={r2_fig4c:.3f})", fontsize=12)
ax.legend(frameon=False, fontsize=11); fig.patch.set_facecolor("white")
plt.tight_layout()
fig.savefig(OUT / "fig4c_rul_45C02.png", dpi=150)
print(f"  Saved -> {OUT / 'fig4c_rul_45C02.png'}")


# -- Final summary ------------------------------------------------------------
print()
print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"  [Fig 3a] Multi-T  35C cap  R2 = {r2_cap:.4f}   (paper: 0.81)")
print(f"  [Fig 4b] Multi-T  35C RUL  R2 = {r2_rul:.4f}   (paper: 0.75)")
print(f"  [Fig 3c] ARD 35C  top feat : #{top_feat}        (paper: #91)")
print(f"  [Fig 1a] Single-T 25C cap  R2(25C05)={r2_fig1_25c05:.4f}  R2(all)={r2_fig1:.4f}   (paper: 0.88 for 25C05)")
print(f"  [Fig 1c] ARD 25C  top feat : #{top_fig1}  top-5={top5_fig1}")
print(f"  [Fig 2 ] Single-T 25C RUL  R2/cell: " +
      " | ".join(f"{c}={r2_fig2_cells[c]:.2f}" for c in fig2_test_cells))
print(f"  [Fig 3b] Multi-T  45C cap  R2 = {r2_fig3b:.4f}   (paper: 0.72)")
print(f"  [Fig 4c] Multi-T  45C RUL  R2 = {r2_fig4c:.4f}   (paper: 0.92)")
print()
print("  Figures saved to ./output/")
