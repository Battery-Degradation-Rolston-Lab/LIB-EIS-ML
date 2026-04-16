"""
Supplementary Information Reproduction -- Zhang et al., Nature Communications 2020

Reproduces:
  Supp Fig 3a : 3D EIS spectra evolution for 25C01 (State V) with ARD highlights
  Supp Fig 3b : -Im[Z] at 17.80 Hz (#91) and 2.16 Hz (#100) vs cycle number
  Supp Fig 4  : Capacity retention curves for all 12 cells (train + test)

Not reproducible:
  Supp Fig 2  : Multi-state GPR (States VII/VIII not released on Zenodo)
  Supp Table 1: Requires capacity+voltage baseline model (not implemented)

All inputs are loaded from data/ (committed to git).  raw_data/ is not required.

Frequency grid (60 pts, descending):
  Index 30 (0-based) = 17.796 Hz  -> feature #91  (1-indexed)
  Index 39 (0-based) =  2.161 Hz  -> feature #100 (1-indexed)
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from pathlib import Path

DATA = Path(__file__).parent / "data"
OUT  = Path(__file__).parent / "output" / "supplement"
OUT.mkdir(parents=True, exist_ok=True)

BLUE  = np.array([0,   114, 189]) / 255
GREEN = np.array([77,  190,  82]) / 255
RED   = np.array([217,  83,  25]) / 255

# Known frequency grid (60 pts, descending) — confirmed from Zenodo raw files.
# Index 30 = 17.796 Hz (feature #91), index 39 = 2.161 Hz (feature #100).
FREQS = np.array([
    20004.453, 15829.126, 12516.703,  9909.442,  7835.480,  6217.246,
     4905.291,  3881.274,  3070.983,  2430.778,  1923.154,  1522.436,
     1203.845,   952.866,   754.276,   596.719,   471.963,   373.209,
      295.473,   233.877,   185.059,   146.358,   115.778,    91.672,
       72.517,    57.368,    45.363,    35.931,    28.409,    22.482,
       17.796,    14.068,    11.145,     8.818,     6.975,     5.517,
        4.369,     3.457,     2.735,     2.161,     1.710,     1.354,
        1.071,     0.847,     0.671,     0.531,     0.420,     0.332,
        0.263,     0.208,     0.165,     0.130,     0.103,     0.082,
        0.064,     0.051,     0.040,     0.032,     0.025,     0.020,
])


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_eis(fname: str) -> np.ndarray:
    """Load a preprocessed (N, 120) EIS matrix from data/."""
    return np.loadtxt(DATA / fname)


def load_cap(fname: str) -> np.ndarray:
    """Load a preprocessed capacity vector from data/."""
    return np.loadtxt(DATA / fname)


# ── SUPP FIG 3 ── EIS spectra 3D + -Im[Z] trend for 25C01 ───────────────────

print("\n" + "=" * 60)
print("SUPP FIG 3 — 25C01 EIS spectra 3D + ARD feature trends")
print("=" * 60)

eis    = load_eis("EIS_data_25C01.txt")    # (261, 120)
freqs  = FREQS                              # 60-pt grid, descending
cycles = np.arange(1, len(eis) + 1, dtype=float)   # cycle index 1, 2, ...
print(f"  Loaded EIS_data_25C01.txt: {eis.shape[0]} cycles, {eis.shape[1]} features")

# Feature #91 (1-indexed) = index 90 -> -Im(Z) at freq index 30 = 17.796 Hz
# Feature #100 (1-indexed) = index 99 -> -Im(Z) at freq index 39 = 2.161 Hz
idx91  = np.argmin(np.abs(freqs - 17.80))   # 30
idx100 = np.argmin(np.abs(freqs -  2.16))   # 39
f91    = freqs[idx91]
f100   = freqs[idx100]
print(f"  Feature #91 : freq index {idx91}, actual = {f91:.3f} Hz (paper: 17.80 Hz)")
print(f"  Feature #100: freq index {idx100}, actual = {f100:.3f} Hz (paper: 2.16 Hz)")

re_all = eis[:,  :60]   # (N, 60) Re(Z)
im_all = eis[:, 60:]    # (N, 60) -Im(Z)

# ── Supp Fig 3a — 3D Nyquist stack ──────────────────────────────────────────
fig = plt.figure(figsize=(11, 7))
ax  = fig.add_subplot(111, projection="3d")

# Plot each cycle's spectrum as red dots
step = max(1, len(cycles) // 80)   # thin out cycles for clarity (~80 shown)
for i in range(0, len(cycles), step):
    ax.plot(re_all[i], np.full(60, cycles[i]), im_all[i],
            ".", color=(*RED, 0.25), ms=1.8)

# Highlight salient frequencies across ALL cycles
ax.scatter(re_all[:, idx91],  cycles, im_all[:, idx91],
           c=[BLUE.tolist()]  * len(cycles), s=12, zorder=5,
           label=f"17.80 Hz (91st)")
ax.scatter(re_all[:, idx100], cycles, im_all[:, idx100],
           c=[GREEN.tolist()] * len(cycles), s=12, zorder=5,
           label=f"2.16 Hz (100th)")

ax.set_xlabel("Re[Z] (Ohm)", fontsize=10, labelpad=8)
ax.set_ylabel("Cycle number",  fontsize=10, labelpad=8)
ax.set_zlabel("-Im[Z] (Ohm)", fontsize=10, labelpad=8)
ax.set_title("25C01 — EIS spectra at State V  (Supp Fig 3a)", fontsize=12)
ax.legend(fontsize=10, loc="upper left")
# Match paper: Re[Z] runs right-to-left (high→low), cycle 0→300 front-to-back
ax.set_xlim(re_all.max() * 1.05, 0)
ax.set_ylim(0, max(cycles[-1], 300))
ax.set_zlim(-0.1, 0.5)
ax.view_init(elev=20, azim=-135)
fig.patch.set_facecolor("white")
plt.tight_layout()
fig.savefig(OUT / "suppfig3a_EIS_spectra_3D_25C01.png", dpi=150)
print(f"  Saved -> suppfig3a_EIS_spectra_3D_25C01.png")
plt.close(fig)

# ── Supp Fig 3b — -Im[Z] vs cycle for the two salient frequencies ────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cycles, im_all[:, idx91],  color=BLUE,  lw=1.5,
        label=f"91st  ({f91:.2f} Hz)")
ax.plot(cycles, im_all[:, idx100], color=GREEN, lw=1.5,
        label=f"100th ({f100:.2f} Hz)")
ax.set_xlabel("Cycle Number",  fontsize=13)
ax.set_ylabel("-Im[Z] (Ohm)", fontsize=13)
ax.set_title("25C01 — -Im[Z] at salient frequencies vs cycle  (Supp Fig 3b)",
             fontsize=11)
ax.legend(frameon=False, fontsize=11)
ax.set_xlim(0, max(cycles[-1], 300))
ax.set_ylim(0.04, 0.30)
fig.patch.set_facecolor("white")
plt.tight_layout()
fig.savefig(OUT / "suppfig3b_ImZ_trend_25C01.png", dpi=150)
print(f"  Saved -> suppfig3b_ImZ_trend_25C01.png")
plt.close(fig)


# ── SUPP FIG 4 — Capacity retention curves, all 12 cells ────────────────────

print("\n" + "=" * 60)
print("SUPP FIG 4 — Capacity retention curves (all 12 cells)")
print("=" * 60)

TRAIN = ["25C01", "25C02", "25C03", "25C04", "35C01", "45C01"]
TEST  = ["25C05", "25C06", "25C07", "25C08", "35C02", "45C02"]

# Colour palette matching MATLAB tab colours used in the paper figure
TRAIN_COLORS = [
    "#FF69B4",  # pink        25C01
    "#CC0000",  # dark red    25C02
    "#00CCCC",  # cyan        25C03
    "#00AA00",  # green       25C04
    "#808080",  # dark grey   35C01
    "#FFD700",  # gold        45C01
]
TEST_COLORS = [
    "#ADD8E6",  # light blue  25C05
    "#00008B",  # dark blue   25C06
    "#FF8C00",  # orange      25C07
    "#800080",  # purple      25C08
    "#A9A9A9",  # grey        35C02
    "#9400D3",  # violet      45C02
]

fig, ax = plt.subplots(figsize=(10, 6))

# Capacity file mapping — all from data/ (committed to git)
# 25C cells: Capacity_data_25C0{n}.txt (generated by preprocess_zenodo.py)
# 35C/45C:   existing files already committed for run_gpytorch.py
CAP_FILES = {
    "25C01": "Capacity_data_25C01.txt", "25C02": "Capacity_data_25C02.txt",
    "25C03": "Capacity_data_25C03.txt", "25C04": "Capacity_data_25C04.txt",
    # Note: Zenodo preprocessed 35C data contains 35C02 for both train/test cells;
    # 35C01 raw individual file is available but was not part of the original release.
    "35C01": "Capacity_data_35.txt",    "45C01": "Capacity_data_45C01.txt",
    "25C05": "Capacity_data_25C05.txt", "25C06": "Capacity_data_25C06.txt",
    "25C07": "Capacity_data_25C07.txt", "25C08": "Capacity_data_25C08.txt",
    "35C02": "Capacity_data_35C02.txt", "45C02": "Capacity_data_45C02.txt",
}

for cell, col in zip(TRAIN, TRAIN_COLORS):
    caps = load_cap(CAP_FILES[cell])
    cyc  = np.arange(len(caps))
    ax.plot(cyc, caps, color=col, lw=1.2, label=f"{cell}-train")
    print(f"  {cell} (train): {len(caps)} cycles, "
          f"init={caps[0]:.1f}, final={caps[-1]:.1f} mAh")

for cell, col in zip(TEST, TEST_COLORS):
    caps = load_cap(CAP_FILES[cell])
    cyc  = np.arange(len(caps))
    ax.plot(cyc, caps, color=col, lw=1.2, label=f"{cell}-test")
    print(f"  {cell} (test): {len(caps)} cycles, "
          f"init={caps[0]:.1f}, final={caps[-1]:.1f} mAh")

ax.set_xlabel("Cycle number",  fontsize=13)
ax.set_ylabel("Capacity (mAh)", fontsize=13)
ax.set_title("Capacity retention curves — all 12 cells  (Supp Fig 4)", fontsize=12)
ax.legend(fontsize=8, ncol=2, frameon=False, loc="lower left")
ax.set_xlim(0, 500)
ax.set_ylim(15, 45)
fig.patch.set_facecolor("white")
plt.tight_layout()
fig.savefig(OUT / "suppfig4_capacity_retention_all_cells.png", dpi=150)
print(f"  Saved -> suppfig4_capacity_retention_all_cells.png")
plt.close(fig)


# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("DONE — figures saved to:", OUT)
print("=" * 60)
print("  suppfig3a_EIS_spectra_3D_25C01.png")
print("  suppfig3b_ImZ_trend_25C01.png")
print("  suppfig4_capacity_retention_all_cells.png")
print()
print("NOT reproduced:")
print("  Supp Fig 2  — States VII/VIII EIS not released on Zenodo")
print("  Supp Table 1— Requires capacity+voltage baseline (not implemented)")
