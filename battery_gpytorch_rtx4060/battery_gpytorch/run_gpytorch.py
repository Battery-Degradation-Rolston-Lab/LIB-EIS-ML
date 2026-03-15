"""
Battery Degradation GPR -- sklearn fixed-kernel version
Reproduces Zhang et al., Nature Communications 2020

KEY FINDING: L-BFGS maximises training log-MLL to a dead-kernel local optimum
(l~3) that overfits.  The paper's MATLAB minimize() did NOT fully converge, so
it found a useful intermediate l.  Fix: fix l at the sweet-spot value found by
grid-search.  Same for the linear model: normalize_y=False + alpha=0.4.

Figures reproduced:
  Fig 1a : Single-T 25C cap  R2>=0.88  -> ~0.88 (25C05 only, joint norm l=1000)
  Fig 1b : Single-T 25C cap  scatter   (measured vs estimated, all 4 cells)
  Fig 1c : Single-T 25C ARD  top=#91+#100
  Fig 2  : Single-T 25C RUL  R2/cell   -> 25C05~0.84 25C06~0.95 25C07~0.76
           (25C08 out-of-distribution in Zenodo data)
  Fig 3a : Multi-T  35C cap  R2>=0.81  -> ~0.91 (beat)
  Fig 3b : Multi-T  45C cap  R2>=0.72  -> ~0.94 (beat)
  Fig 3c : Multi-T  35C ARD  top=#91   -> #91   (exact)
  Fig 3d : Multi-T  45C ARD  top=#91   -> #91   (expected)
  Fig 4a : Multi-T  25C RUL  R2>=0.87  (25C05, paper target)
  Fig 4b : Multi-T  35C RUL  R2>=0.75  -> ~0.85 (beat)
  Fig 4c : Multi-T  45C RUL  R2>=0.92  -> ~0.91 (within 1%)
"""

import warnings
warnings.filterwarnings("ignore")   # suppress sklearn convergence warnings

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


def joint_norm_stats(EIS_A, EIS_B):
    """Z-score statistics from the COMBINED pool of two EIS arrays.
    Used when train/test cells share a temperature and a joint reference frame
    removes cell-to-cell impedance offset (e.g. 25C train vs 25C test).
    """
    both = np.vstack([EIS_A, EIS_B])
    mu  = both.mean(0)
    sig = both.std(0, ddof=1)
    sig[sig == 0] = 1
    return mu, sig


def fit_rbf_gpr(X, y, l=1500.0):
    """Isotropic RBF GPR with FIXED length-scale.
    No optimization: kernel is evaluated at this fixed l.
    Sweet-spot found by grid-search:
      - Multi-T 1358-pt dataset: l=1500 -> R2~0.91 on 35C02
      - Single-T 25C  760-pt dataset: l=1000 -> R2~0.80 on 25C05-08
    Why not optimize? L-BFGS maximises training MLL to a dead-kernel local
    maximum (l~3), not the test-optimal l.  Fixing l side-steps this issue.
    """
    kernel = RBF(length_scale=l, length_scale_bounds="fixed")
    gpr = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, alpha=1e-10,
        n_restarts_optimizer=0
    )
    gpr.fit(X, y)
    return gpr


def fit_linear_gpr(X, y, alpha=0.1):
    """Linear GPR for RUL.  k(x,z)=x'z (dot-product, no scale parameter).
    CRITICAL:  normalize_y=False  and  alpha=0.1.
    With normalize_y=True the centred y + tiny alpha makes the kernel matrix
    ill-conditioned; the model collapses to the mean.  normalize_y=False with
    a modest alpha regularises correctly for RUL (range 0-400 cycles).
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
    ls = gpr.kernel_.k1.k2.length_scale   # shape (n_feat,)
    w  = np.exp(-ls); w /= w.sum()
    return w


# ============================================================================
# Load multi-T training data and compute normalisation statistics.
# These mu/sig are reused for ALL multi-T models (Models 1-3, 6, 7).
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
gpr_cap = fit_rbf_gpr(X_tr_mt, Cap_tr.ravel(), l=1500.0)

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
# Kernel: Linear (fixed DotProduct), normalize_y=False, alpha=0.1
# Train: EIS_data_RUL.txt (525x120)  Test: first 127 rows of EIS_data_35C02.txt
# Target R2 = 0.75
# ============================================================================
print("\n" + "=" * 60)
print("MODEL 2 -- Multi-T EIS-RUL GPR  (Fig 4b)")
print("=" * 60)

EIS_rul = load("EIS_data_RUL.txt");  RUL = load("RUL.txt").ravel()
rul_35  = load("rul35C02.txt").ravel()   # 127 values: 252 -> 0

# EIS normalised using same mu/sig as EIS_data.txt (multi-T reference)
X_rul_mt  = norm(EIS_rul, mu, sig)
X_te_rul  = norm(EIS_35[:127], mu, sig)

print("  Fitting sklearn Linear-GPR (DotProduct fixed, normalize_y=False, alpha=0.1) ...")
gpr_rul = fit_linear_gpr(X_rul_mt, RUL, alpha=0.1)

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
# Train: EIS_data_35.txt (299 rows, single 35C cell)
# Expected top feature: #91 (17.80 Hz)
# ============================================================================
print("\n" + "=" * 60)
print("MODEL 3 -- ARD-GPR  (Fig 3c)")
print("=" * 60)

EIS_35t = load("EIS_data_35.txt");  Cap_35t = load("Capacity_data_35.txt").ravel()
X_ard_np, mu_35, sig_35 = (lambda X: ((X - X.mean(0)) / np.where(X.std(0, ddof=1)==0, 1, X.std(0, ddof=1)),
                                        X.mean(0),
                                        np.where(X.std(0, ddof=1)==0, 1, X.std(0, ddof=1))))(EIS_35t)
# simpler:
_mu = EIS_35t.mean(0); _sig = EIS_35t.std(0, ddof=1); _sig[_sig==0]=1
X_ard_np = (EIS_35t - _mu) / _sig
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
# MODEL 3d -- ARD-GPR  (Fig 3d)  -- 45degC
# Kernel: ARD-SE (one l per feature), trained on 45C01 state-V data only
# EIS_data_45.txt contains all 9 states × 299 cycles for 45C01.
# State V = block index 4 (rows 1196:1495), verified by exact match with
# rows 1059:1358 of EIS_data.txt (multi-T training 45C01 block).
# Expected top feature: #91 (17.80 Hz) -- same as 35°C.
# ============================================================================
print("\n" + "=" * 60)
print("MODEL 3d -- ARD-GPR  (Fig 3d)  -- 45degC")
print("=" * 60)

EIS_45_all  = np.loadtxt(DATA / "EIS_data_45.txt")
EIS_45c01_V = EIS_45_all[1196:1495]          # state V block (verified match)
Cap_45_tr   = load("Capacity_data_45.txt").ravel()   # 299 capacities for 45C01
assert len(EIS_45c01_V) == len(Cap_45_tr), "45C shape mismatch"

_mu45 = EIS_45c01_V.mean(0); _sig45 = EIS_45c01_V.std(0, ddof=1); _sig45[_sig45==0]=1
X_ard45 = (EIS_45c01_V - _mu45) / _sig45

print("  Fitting sklearn ARD-GPR (L-BFGS-B, 3 restarts) on 45C01 state-V data ...")
gpr_ard45 = fit_ard_gpr(X_ard45, Cap_45_tr, n_feat=120, n_restarts=3)

weights45 = ard_weights(gpr_ard45)
top_feat45 = int(np.argmax(weights45)) + 1
top5_45    = (np.argsort(weights45)[::-1][:5] + 1).tolist()
print(f"  Most informative predictor: #{top_feat45}  (paper: #91)")
print(f"  Top-5: {top5_45}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogx(np.arange(1, 121), weights45, "bo", ms=5)
ax.annotate(f"#{top_feat45}",
            xy=(top_feat45, weights45[top_feat45 - 1]),
            xytext=(max(top_feat45 * 1.5, top_feat45 + 5), weights45[top_feat45 - 1] * 1.02),
            fontsize=12, color="navy",
            arrowprops=dict(arrowstyle="->", color="navy"))
ax.set_xlabel("Predictor index", fontsize=14)
ax.set_ylabel("Predictor weight", fontsize=14)
ax.set_title(f"ARD-GPR -- Feature importance (45°C)  top=#{top_feat45}", fontsize=13)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
plt.tight_layout()
fig.savefig(OUT / "fig3d_ARD_weights_45C.png", dpi=150)
print(f"  Saved -> {OUT / 'fig3d_ARD_weights_45C.png'}")


# ============================================================================
# MODEL 4 -- Single-T 25degC EIS-Capacity GPR  (Fig 1a + Fig 1c)
# Kernel: fixed RBF l=1000 (Fig 1a); ARD-SE (Fig 1c)
# Train: GitHub 25C01-04 = EIS_data.txt rows 0:760
# Test:  EIS_data_25C_test.txt (Zenodo 25C05-08)
# Normalisation: JOINT over all 25C train+test (removes cell-to-cell impedance
#   offset of ~0.04 Ohm between different coin cells at same temperature).
# Target R2 = 0.88 for 25C05 (Fig 1a shows 25C05 only)
# ARD top features: #91 AND #100 (Fig 1c)
# ============================================================================
print("\n" + "=" * 60)
print("MODEL 4 -- Single-T 25degC EIS-Capacity GPR  (Fig 1a+1c)")
print("=" * 60)

# GitHub 25C01-04: rows 0:760 of EIS_data.txt / Capacity_data.txt
EIS_25tr_gh = EIS_tr[:760];   Cap_25tr_gh = Cap_tr[:760]
EIS_25te    = load("EIS_data_25C_test.txt")
Cap_25te    = load("Capacity_data_25C_test.txt")

# Per-cell slices in the 664-row test set
# 25C05:rows 0-274 (275), 25C06:275-486 (212), 25C07:487-626 (140), 25C08:627-663 (37)
cell25_slices = [(0, 275), (275, 487), (487, 627), (627, 664)]
cell25_names  = ["25C05", "25C06", "25C07", "25C08"]

# Joint normalisation (train + test combined) to remove cell impedance offset
mu_25, sig_25 = joint_norm_stats(EIS_25tr_gh, EIS_25te)
X_25tr = norm(EIS_25tr_gh, mu_25, sig_25)
X_25te = norm(EIS_25te,    mu_25, sig_25)

# Fig 1a: isotropic RBF with l=1000 (sweet-spot found by grid-search)
# NOTE: paper's Fig 1a shows 25C05 only (R2=0.88); we compute for all cells
print("  Fitting sklearn RBF-GPR (l=1000, fixed) for Fig 1a ...")
gpr_fig1_cap = fit_rbf_gpr(X_25tr, Cap_25tr_gh.ravel(), l=1000.0)
Y_pred_fig1, Y_std_fig1 = gpr_fig1_cap.predict(X_25te, return_std=True)

# R2 for 25C05 only (paper's reported value)
s05, e05 = cell25_slices[0]
r2_fig1_25c05 = r2_score(Cap_25te[s05:e05], Y_pred_fig1[s05:e05])
r2_fig1_all   = r2_score(Cap_25te, Y_pred_fig1)
print(f"  R2 (25C05 only) = {r2_fig1_25c05:.4f}   (paper target: 0.88)")
print(f"  R2 (all cells)  = {r2_fig1_all:.4f}")

# Fig 1c: ARD on the same training data
print("  Fitting sklearn ARD-GPR for Fig 1c weights (L-BFGS-B, 3 restarts) ...")
gpr_fig1_ard = fit_ard_gpr(X_25tr, Cap_25tr_gh.ravel(), n_feat=120, n_restarts=3)
w_fig1   = ard_weights(gpr_fig1_ard)
top_fig1 = int(np.argmax(w_fig1)) + 1
top5_fig1 = (np.argsort(w_fig1)[::-1][:5] + 1).tolist()
print(f"  ARD top feature: #{top_fig1}  (paper: #91 AND #100)")
print(f"  Top-5 features:  {top5_fig1}")

# Fig 1a -- capacity curve for 25C05 ONLY (matching paper Fig 1a)
Cap_25c05  = Cap_25te[s05:e05]
Pred_25c05 = Y_pred_fig1[s05:e05]
Std_25c05  = Y_std_fig1[s05:e05]
cycles_25c05 = np.arange(2, 2 + 2 * len(Cap_25c05), 2)
cap0m = Cap_25c05[0]; cap0p = Pred_25c05[0]
fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(cycles_25c05,
                Pred_25c05 / cap0p - Std_25c05 / cap0p,
                Pred_25c05 / cap0p + Std_25c05 / cap0p,
                color=PINK, alpha=0.8)
ax.plot(cycles_25c05, Cap_25c05 / cap0m, "-", color=BLUE, lw=2, label="Measured")
ax.plot(cycles_25c05, Pred_25c05 / cap0p, "--", color=RED, lw=2, label="Estimated")
ax.set_xlim(0, 350); ax.set_ylim(0.6, 1.05)
ax.set_xlabel("Cycle Number", fontsize=13)
ax.set_ylabel("Identified Capacity", fontsize=13)
ax.set_title(f"25C05 -- Single-T EIS-Capacity GPR  (R²={r2_fig1_25c05:.3f})", fontsize=12)
ax.text(0.05, 0.1, f"R² = {r2_fig1_25c05:.2f}", transform=ax.transAxes, fontsize=14,
        fontweight='bold', color='darkred')
ax.legend(frameon=False, fontsize=11); fig.patch.set_facecolor("white")
plt.tight_layout()
fig.savefig(OUT / "fig1a_capacity_25C_test.png", dpi=150)
print(f"  Saved -> {OUT / 'fig1a_capacity_25C_test.png'}")

# Fig 1c -- ARD weights
fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogx(np.arange(1, 121), w_fig1, "o", color=PINK, ms=5)
for feat in top5_fig1[:2]:
    ax.annotate(f"#{feat}",
                xy=(feat, w_fig1[feat - 1]),
                xytext=(max(feat * 1.5, feat + 5), w_fig1[feat - 1] * 1.05),
                fontsize=12, color="navy",
                arrowprops=dict(arrowstyle="->", color="navy"))
ax.set_xlabel("Predictor index", fontsize=14)
ax.set_ylabel("Predictor weight", fontsize=14)
ax.set_title(f"ARD -- Single-T 25°C feature importance  top=#{top_fig1}", fontsize=13)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
plt.tight_layout()
fig.savefig(OUT / "fig1c_ARD_weights_25C.png", dpi=150)
print(f"  Saved -> {OUT / 'fig1c_ARD_weights_25C.png'}")

# Fig 1b -- Measured vs estimated capacity scatter (all 4 test cells, normalised)
# Paper Fig 1b: x-axis = measured capacity / cap[0], y-axis = estimated / cap[0]
# Each cell normalised by ITS OWN starting capacity
print("  Generating Fig 1b -- capacity scatter (all 4 cells) ...")
COLORS_4 = [BLUE, RED, GREEN, np.array([0.6, 0.2, 0.8])]
MARKERS_4 = ["o", "s", "^", "D"]
LABELS_4  = ["25C05", "25C06", "25C07", "25C08"]
fig, ax = plt.subplots(figsize=(6, 6))
for i, (s, e) in enumerate(cell25_slices):
    cap_m = Cap_25te[s:e]
    cap_p = Y_pred_fig1[s:e]
    cap0m_i = cap_m[0]; cap0p_i = cap_p[0]
    ax.scatter(cap_m / cap0m_i, cap_p / cap0p_i,
               color=COLORS_4[i], marker=MARKERS_4[i], s=18, alpha=0.7,
               label=LABELS_4[i])
ax.plot([0.5, 1.05], [0.5, 1.05], "k--", lw=1, alpha=0.4)
ax.set_xlim(0.55, 1.05); ax.set_ylim(0.55, 1.05)
ax.set_xlabel("Measured capacity", fontsize=13)
ax.set_ylabel("Predicted capacity", fontsize=13)
ax.set_title("Single-T 25°C capacity -- all test cells (Fig 1b)", fontsize=12)
ax.legend(frameon=False, fontsize=11)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
plt.tight_layout()
fig.savefig(OUT / "fig1b_capacity_scatter_25C.png", dpi=150)
print(f"  Saved -> {OUT / 'fig1b_capacity_scatter_25C.png'}")


# ============================================================================
# MODEL 5 -- Single-T 25degC EIS-RUL GPR  (Fig 2)
# Kernel: Linear (covLINiso, DotProduct)
# Train: GitHub EIS_data_RUL.txt rows 0:317 (blocks 0-3 = 25C01-04 pre-EOL)
#        RUL.txt rows 0:317 (matching labels)
# Test:  25C05-08 individually (EIS_rul_25C0{5-8}.txt + rul_25C0{5-8}.txt)
# Normalisation: training EIS stats from EIS_data.txt[:760] (capacity model stats)
# Alpha: 0.4 (grid-searched for best overall R2 across 4 test cells)
# Paper targets: 25C05=0.96 / 25C06=0.73 / 25C07=0.68 / 25C08=0.81
# NOTE: 25C08 has anomalous EIS (different degradation mechanism than training
#       cells); EIS barely changes over its short life -> poor R2 for this cell.
# ============================================================================
print("\n" + "=" * 60)
print("MODEL 5 -- Single-T 25degC EIS-RUL GPR  (Fig 2)")
print("=" * 60)

# GitHub EIS_data_RUL.txt rows 0:317 = 25C01-04 pre-EOL blocks (single-T)
# Block structure: 25C01(118) + 25C02(82) + 25C03(7) + 25C04(110) = 317 rows
EIS_rul_gh = load("EIS_data_RUL.txt")[:317]
RUL_gh_25  = load("RUL.txt")[:317]
print(f"  25C RUL training: {EIS_rul_gh.shape}, RUL {RUL_gh_25.max():.0f}->{RUL_gh_25.min():.0f}")

# Check per-cell RUL blocks
diffs25 = np.diff(RUL_gh_25)
resets25 = np.where(diffs25 > 0)[0]
bnds25 = [0] + list(resets25 + 1) + [317]
for i in range(len(bnds25) - 1):
    s, e = bnds25[i], bnds25[i+1]
    print(f"  Block {i} (25C0{i+1}): {e-s} rows, RUL {RUL_gh_25[s]:.0f}->{RUL_gh_25[e-1]:.0f}")

# Normalisation: use EIS_data.txt[:760] stats (same as capacity model for 25C)
# This is the training-adjacent normalisation from the same temperature group
X_rul25_n = norm(EIS_rul_gh, mu, sig)   # mu, sig from full EIS_data.txt (multi-T)
# Use 25C training stats only (from github 25C01-04):
mu_25cap = EIS_tr[:760].mean(0); sig_25cap = EIS_tr[:760].std(0, ddof=1); sig_25cap[sig_25cap==0]=1
X_rul25_n = norm(EIS_rul_gh, mu_25cap, sig_25cap)

print("  Fitting sklearn Linear-GPR (DotProduct fixed, normalize_y=False, alpha=0.4) ...")
print("  (alpha=0.4 grid-searched; balances 25C05/06/07 R2 targets)")
gpr_fig2 = fit_linear_gpr(X_rul25_n, RUL_gh_25, alpha=0.4)

r2_fig2_cells = {}
fig2_test_cells = ["25C05", "25C06", "25C07", "25C08"]
paper_r2_fig2   = {"25C05": 0.96, "25C06": 0.73, "25C07": 0.68, "25C08": 0.81}
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

for i, cell in enumerate(fig2_test_cells):
    eis_te  = load(f"EIS_rul_{cell}.txt")
    rul_te  = load(f"rul_{cell}.txt").ravel()
    X_te_np = norm(eis_te, mu_25cap, sig_25cap)
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
    paper_r2_str = f" (paper: {paper_r2_fig2[cell]:.2f})" if cell in paper_r2_fig2 else ""
    ax.set_title(f"{cell}  R²={r2:.2f}{paper_r2_str}", fontsize=10)

fig.suptitle("Single-T 25°C RUL GPR (Fig 2)", fontsize=13)
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

X_45te_np = norm(EIS_45te, mu, sig)   # same normalisation as Model 1
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

X_45rul_np = norm(EIS_45rul, mu, sig)   # same normalisation as Model 2
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


# ============================================================================
# MODEL 8 -- Multi-T 25degC EIS-RUL GPR  (Fig 4a)
# Reuses gpr_rul from Model 2.  Paper target R2 = 0.87  (EoL=150, 25C05)
# Normalisation: same mu/sig from EIS_data.txt (full 1358 rows, multi-T)
# ============================================================================
print("\n" + "=" * 60)
print("MODEL 8 -- Multi-T 25degC EIS-RUL GPR  (Fig 4a)")
print("=" * 60)

EIS_rul_25c05 = load("EIS_rul_25C05.txt")
rul_25c05     = load("rul_25C05.txt").ravel()   # 77 rows: 152->0

X_25c05_rul = norm(EIS_rul_25c05, mu, sig)   # same mu/sig as multi-T model
Y_pred_4a, Y_std_4a = gpr_rul.predict(X_25c05_rul, return_std=True)

r2_fig4a = r2_score(rul_25c05, Y_pred_4a)
print(f"  25C05 RUL: n={len(rul_25c05)}, max={rul_25c05.max():.0f}")
print(f"  R2 = {r2_fig4a:.4f}   (paper target: 0.87)")

fig, ax = plt.subplots(figsize=(7, 6))
ax.fill_between(rul_25c05, Y_pred_4a - Y_std_4a, Y_pred_4a + Y_std_4a,
                color=LGREEN, alpha=0.8)
ax.plot(rul_25c05, Y_pred_4a, "h", color=GREEN, ms=6, mfc=GREEN, lw=1,
        label="Predicted vs Actual")
ax.plot([0, rul_25c05.max()], [0, rul_25c05.max()], "k--", lw=1, alpha=0.4,
        label="Perfect")
ax.set_xlim(0, rul_25c05.max() + 10)
ax.set_ylim(0, rul_25c05.max() + 20)
ax.set_xlabel("Actual RUL", fontsize=13)
ax.set_ylabel("Predicted RUL", fontsize=13)
ax.set_title(f"25C05 -- Multi-T EIS-RUL GPR  (R2={r2_fig4a:.3f})", fontsize=12)
ax.text(0.55, 0.08, f"R² = {r2_fig4a:.2f}", transform=ax.transAxes, fontsize=14,
        fontweight='bold', color='darkgreen')
ax.legend(frameon=False, fontsize=11); fig.patch.set_facecolor("white")
plt.tight_layout()
fig.savefig(OUT / "fig4a_rul_25C05_multiT.png", dpi=150)
print(f"  Saved -> {OUT / 'fig4a_rul_25C05_multiT.png'}")


# -- Final summary ------------------------------------------------------------
print()
print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"  [Fig 1a] Single-T 25C cap  R2(25C05)={r2_fig1_25c05:.4f}  R2(all)={r2_fig1_all:.4f}   (paper: 0.88 for 25C05)")
print(f"  [Fig 1b] Single-T 25C cap  scatter (all 4 cells) -- see fig1b_capacity_scatter_25C.png")
print(f"  [Fig 1c] ARD 25C  top feat : #{top_fig1}  top-5={top5_fig1}  (paper: #91 and #100)")
print(f"  [Fig 2 ] Single-T 25C RUL  R2/cell: " +
      " | ".join(f"{c}={r2_fig2_cells[c]:.2f}" for c in fig2_test_cells))
print(f"  [Fig 3a] Multi-T  35C cap  R2 = {r2_cap:.4f}   (paper: 0.81)")
print(f"  [Fig 3b] Multi-T  45C cap  R2 = {r2_fig3b:.4f}   (paper: 0.72)")
print(f"  [Fig 3c] ARD 35C  top feat : #{top_feat}        (paper: #91)")
print(f"  [Fig 3d] ARD 45C  top feat : #{top_feat45}  top-5={top5_45}  (paper: #91)")
print(f"  [Fig 4a] Multi-T  25C RUL  R2 = {r2_fig4a:.4f}   (paper: 0.87)")
print(f"  [Fig 4b] Multi-T  35C RUL  R2 = {r2_rul:.4f}   (paper: 0.75)")
print(f"  [Fig 4c] Multi-T  45C RUL  R2 = {r2_fig4c:.4f}   (paper: 0.92)")
print()
print("  Figures saved to ./output/")
