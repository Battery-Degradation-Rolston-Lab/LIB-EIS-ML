"""
Battery Degradation GPR -- sklearn version (L-BFGS-B optimizer)
Reproduces Zhang et al., Nature Communications 2020

All GPR training uses sklearn GaussianProcessRegressor (L-BFGS-B internally).
This reliably finds the correct kernel optima whereas Adam in high-dimension
converges to dead-kernel local minima (l~3 in 120-D space).

Reproduce targets:
  Fig 3a : Multi-T  35C cap   R2>=0.81
  Fig 4b : Multi-T  35C RUL   R2>=0.75
  Fig 3c : Multi-T  35C ARD   top=#91
  Fig 1a : Single-T 25C cap   R2>=0.88
  Fig 1c : Single-T 25C ARD   top=#91 AND #100
  Fig 2  : Single-T 25C RUL   R2 per cell ~0.68/0.96/0.81/0.73
  Fig 3b : Multi-T  45C cap   R2>=0.72
  Fig 4c : Multi-T  45C RUL   R2>=0.92
"""

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


def zscore(X):
    mu = X.mean(0); sig = X.std(0, ddof=1); sig[sig == 0] = 1
    return (X - mu) / sig, mu, sig


def make_rbf_kernel(l_init=1.0):
    """Isotropic RBF + WhiteKernel.
    WhiteKernel is critical: without it, large l_init (~50) makes K nearly
    rank-1 (all entries ~0.95 for 120-D z-scored data), causing L-BFGS
    ABNORMAL_TERMINATION after only 4 iterations.  WhiteKernel provides an
    'escape valve' so the optimizer can fit data as noise when the RBF is
    degenerate, then reduce noise as it finds the correct l.
    """
    return (C(1.0, (1e-3, 1e3))
            * RBF(length_scale=l_init, length_scale_bounds=(1e-2, 1e4))
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e5)))


def make_linear_kernel():
    """Linear kernel k(x,z)=c*x'z + noise.
    WhiteKernel added so optimizer doesn't get stuck in singular configurations.
    """
    return (C(1.0, (1e-3, 1e3))
            * DotProduct(sigma_0=0.0, sigma_0_bounds="fixed")
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e5)))


def make_ard_kernel(n_feat=120, l_init=1.0):
    """ARD-SE kernel: one length-scale per feature."""
    return (C(1.0, (1e-5, 1e5))
            * RBF(length_scale=np.full(n_feat, l_init),
                  length_scale_bounds=(1e-3, 1e4))
            + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-8, 10.0)))


def fit_gpr(kernel, X, y, n_restarts=5, alpha=1e-10):
    """Fit GPR.  alpha is a small numerical jitter — noise is handled by
    WhiteKernel inside the kernel, not by alpha.  Keep alpha=1e-10 (default)
    unless there is a specific reason to use a larger value.
    """
    gpr = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, alpha=alpha,
        n_restarts_optimizer=n_restarts, random_state=42
    )
    gpr.fit(X, y)
    return gpr


def ard_weights(gpr):
    """Extract ARD weights from a fitted gpr with C*RBF(ard)+WhiteKernel."""
    # kernel_ = (C * RBF) + WhiteKernel  =>  k1=C*RBF, k2=WhiteKernel
    ls = gpr.kernel_.k1.k2.length_scale
    w = np.exp(-ls); w /= w.sum()
    return w


# ============================================================================
# MODEL 1 -- Multi-T EIS-Capacity GPR  (Fig 3a)
# Kernel: isotropic RBF (covSEiso)
# Train: EIS_data.txt (1358x120)  Test: EIS_data_35C02.txt (299x120)
# Target R2 = 0.81
# ============================================================================
print("\n" + "=" * 60)
print("MODEL 1 -- Multi-T EIS-Capacity GPR  (Fig 3a)")
print("=" * 60)

EIS_tr  = load("EIS_data.txt");        Cap_tr  = load("Capacity_data.txt")
EIS_35  = load("EIS_data_35C02.txt");  Cap_35  = load("capacity35C02.txt")

X_tr_np, mu, sig = zscore(EIS_tr)
X_te_np = (EIS_35 - mu) / sig
y_tr_np = Cap_tr.ravel()

print("  Fitting sklearn RBF-GPR (L-BFGS-B, 9 restarts) ...")
gpr_cap = fit_gpr(make_rbf_kernel(), X_tr_np, y_tr_np, n_restarts=9)
print(f"  Kernel: {gpr_cap.kernel_}")
print(f"  Log-MLL: {gpr_cap.log_marginal_likelihood_value_:.4f}")

Y_pred_cap, Y_std_cap = gpr_cap.predict(X_te_np, return_std=True)
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
# Kernel: Linear (covLINiso)
# Train: EIS_data_RUL.txt (525x120)  Test: first 127 rows of EIS_data_35C02.txt
# Target R2 = 0.75
# ============================================================================
print("\n" + "=" * 60)
print("MODEL 2 -- Multi-T EIS-RUL GPR  (Fig 4b)")
print("=" * 60)

EIS_rul = load("EIS_data_RUL.txt");  RUL = load("RUL.txt").ravel()
rul_35  = load("rul35C02.txt").ravel()   # 127 values: 252 -> 0

# EIS normalised using same mu/sig as EIS_data.txt (MODEL 1 reference)
X_rul_np  = (EIS_rul - mu) / sig
X_te_rul  = (EIS_35[:127] - mu) / sig

print("  Fitting sklearn Linear-GPR (L-BFGS-B, 9 restarts) ...")
gpr_rul = fit_gpr(make_linear_kernel(), X_rul_np, RUL, n_restarts=9)
print(f"  Kernel: {gpr_rul.kernel_}")

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
X_ard_np, _, _ = zscore(EIS_35t)
n_feat = X_ard_np.shape[1]   # 120

print("  Fitting sklearn ARD-GPR (L-BFGS-B, 3 restarts) ...")
gpr_ard = fit_gpr(make_ard_kernel(n_feat, l_init=1.0), X_ard_np, Cap_35t, n_restarts=3)

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
# MODEL 4 -- Single-T 25degC EIS-Capacity ARD-GPR  (Fig 1a + Fig 1c)
# Kernel: ARD-SE
# Train: 25C01-04 (679 rows)  Test: 25C05-08 (664 rows)
# Target R2 = 0.88;  ARD should show #91 AND #100 as top features
# ============================================================================
print("\n" + "=" * 60)
print("MODEL 4 -- Single-T 25degC EIS-Capacity ARD-GPR  (Fig 1a+1c)")
print("=" * 60)

EIS_25tr = load("EIS_data_25C_train.txt")
Cap_25tr = load("Capacity_data_25C_train.txt")
EIS_25te = load("EIS_data_25C_test.txt")
Cap_25te = load("Capacity_data_25C_test.txt")

X_25tr_np, mu_25, sig_25 = zscore(EIS_25tr)
X_25te_np = (EIS_25te - mu_25) / sig_25
y_25tr_np = Cap_25tr.ravel()

# Paper uses a SINGLE ARD-GPR for BOTH capacity prediction (Fig 1a) AND feature
# importance (Fig 1c).  Training on full 25C01-04 (679 rows) gives both.
# (Previously we used a separate isotropic RBF for prediction → wrong approach.)
print("  Fitting sklearn ARD-GPR on 25C01-04 (679 rows, L-BFGS-B, 5 restarts) ...")
gpr_fig1 = fit_gpr(make_ard_kernel(120, l_init=1.0), X_25tr_np, y_25tr_np, n_restarts=5)

Y_pred_fig1, Y_std_fig1 = gpr_fig1.predict(X_25te_np, return_std=True)
r2_fig1 = r2_score(Cap_25te, Y_pred_fig1)
print(f"  R2 = {r2_fig1:.4f}   (paper target: 0.88)")

w_fig1   = ard_weights(gpr_fig1)
top_fig1 = int(np.argmax(w_fig1)) + 1
top5_fig1 = (np.argsort(w_fig1)[::-1][:5] + 1).tolist()
print(f"  ARD top feature: #{top_fig1}  (paper: #91 AND #100)")
print(f"  Top-5 features: {top5_fig1}")

# Fig 1a -- capacity curve
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
ax.set_title(f"25C05-08 -- Single-T 25degC ARD-GPR  (R2={r2_fig1:.3f})", fontsize=12)
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
# Kernel: Linear (covLINiso)
# Train: 25C01-04 up to EOL (243 rows)  Test: 25C05-08 individually
# Target: R2 per cell ~0.68 / 0.96 / 0.81 / 0.73
# ============================================================================
print("\n" + "=" * 60)
print("MODEL 5 -- Single-T 25degC EIS-RUL GPR  (Fig 2)")
print("=" * 60)

EIS_rul25_tr = load("EIS_data_25C_RUL_train.txt")
RUL_25_tr    = load("RUL_25C_train.txt").ravel()

# Normalise EIS using mu/sig from EIS_data.txt (multi-T reference, same as paper)
X_rul25_np = (EIS_rul25_tr - mu) / sig

print("  Fitting sklearn Linear-GPR (L-BFGS-B, 9 restarts) ...")
gpr_fig2 = fit_gpr(make_linear_kernel(), X_rul25_np, RUL_25_tr, n_restarts=9)
print(f"  Kernel: {gpr_fig2.kernel_}")
print(f"  Train RUL: min={RUL_25_tr.min():.0f}  max={RUL_25_tr.max():.0f}  mean={RUL_25_tr.mean():.0f}")

r2_fig2_cells = {}
fig2_test_cells = ["25C05", "25C06", "25C07", "25C08"]
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

for i, cell in enumerate(fig2_test_cells):
    eis_te  = load(f"EIS_rul_{cell}.txt")
    rul_te  = load(f"rul_{cell}.txt").ravel()
    X_te_np = (eis_te - mu) / sig
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

fig.suptitle("Single-T 25degC RUL GPR (Fig 2)", fontsize=13)
fig.patch.set_facecolor("white")
plt.tight_layout()
fig.savefig(OUT / "fig2_rul_25C.png", dpi=150)
print(f"  Saved -> {OUT / 'fig2_rul_25C.png'}")

# DIAGNOSTIC: apply multi-T RUL model (gpr_rul, trained on GitHub EIS_data_RUL.txt)
# to 25C test cells. If this gives positive R2, it confirms the 25C Zenodo
# training data is the culprit for the negative R2 above.
print("\n  DIAGNOSTIC -- multi-T model (GitHub EIS_data_RUL) on 25C cells:")
for cell in fig2_test_cells:
    eis_te  = load(f"EIS_rul_{cell}.txt")
    rul_te  = load(f"rul_{cell}.txt").ravel()
    X_d = (eis_te - mu) / sig
    Y_d = gpr_rul.predict(X_d)
    print(f"    {cell}  R2={r2_score(rul_te, Y_d):.4f}  "
          f"pred_mean={Y_d.mean():.1f}  actual_mean={rul_te.mean():.1f}")


# ============================================================================
# MODEL 6 -- Multi-T 45degC EIS-Capacity GPR  (Fig 3b)
# Reuses gpr_cap from Model 1.  Target R2 = 0.72
# ============================================================================
print("\n" + "=" * 60)
print("MODEL 6 -- Multi-T 45degC EIS-Capacity GPR  (Fig 3b)")
print("=" * 60)

EIS_45te = load("EIS_data_45C02.txt")
Cap_45te = load("capacity45C02.txt")

X_45te_np = (EIS_45te - mu) / sig   # same normalisation as Model 1
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

X_45rul_np = (EIS_45rul - mu) / sig   # same normalisation as Model 2
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
print(f"  [Fig 1a] Single-T 25C cap  R2 = {r2_fig1:.4f}   (paper: 0.88)")
print(f"  [Fig 1c] ARD 25C  top feat : #{top_fig1}  top-5={top5_fig1}")
print(f"  [Fig 2 ] Single-T 25C RUL  R2/cell: " +
      " | ".join(f"{c}={r2_fig2_cells[c]:.2f}" for c in fig2_test_cells))
print(f"  [Fig 3b] Multi-T  45C cap  R2 = {r2_fig3b:.4f}   (paper: 0.72)")
print(f"  [Fig 4c] Multi-T  45C RUL  R2 = {r2_fig4c:.4f}   (paper: 0.92)")
print()
print("  Figures saved to ./output/")
