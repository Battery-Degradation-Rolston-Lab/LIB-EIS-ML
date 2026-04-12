"""
Supplementary Information Reproduction -- Zhang et al., Nature Communications 2020

Reproduces:
  Supp Fig 3a : 3D EIS spectra evolution for 25C01 (State V) with ARD highlights
  Supp Fig 3b : -Im[Z] at 17.80 Hz (#91) and 2.16 Hz (#100) vs cycle number
  Supp Fig 4  : Capacity retention curves for all 12 cells (train + test)

Not reproducible:
  Supp Fig 2  : Multi-state GPR (States VII/VIII not released on Zenodo)
  Supp Table 1: Requires capacity+voltage baseline model (not implemented)

Frequency grid (60 pts, descending):
  Index 30 (0-based) = 17.796 Hz  → feature #91  (1-indexed)
  Index 39 (0-based) =  2.161 Hz  → feature #100 (1-indexed)
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from pathlib import Path

EIS_DIR = Path(__file__).parents[2] / "raw_data" / "zenodo_eis"
CAP_DIR = Path(__file__).parents[2] / "raw_data" / "zenodo_capacity"
OUT     = Path(__file__).parent / "output" / "supplement"
OUT.mkdir(parents=True, exist_ok=True)

BLUE  = np.array([0,   114, 189]) / 255
GREEN = np.array([77,  190,  82]) / 255
RED   = np.array([217,  83,  25]) / 255


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_eis_stateV(cell: str):
    """
    Load EIS_state_V_{cell}.txt.
    Returns:
      eis     : (N, 120) array  — cols 0-59 Re(Z), cols 60-119 -Im(Z), freqs descending
      freqs   : (60,) array    — frequency grid in Hz, descending
      cycles  : (N,) array     — battery cycle numbers (one entry per complete EIS)
    Skips incomplete EIS sweeps (not exactly 60 frequency points).
    """
    path = EIS_DIR / f"EIS_state_V_{cell}.txt"
    raw  = np.loadtxt(path, skiprows=1)
    # cols: time(0) cycle(1) freq(2) Re(Z)(3) -Im(Z)(4) |Z|(5) Phase(6)
    unique_cycles = np.unique(raw[:, 1])
    rows, cyc_out = [], []
    freq_grid = None
    for c in unique_cycles:
        mask = raw[:, 1] == c
        pts  = raw[mask]
        if len(pts) != 60:
            continue
        pts = pts[np.argsort(pts[:, 2])[::-1]]   # sort freq descending
        if freq_grid is None:
            freq_grid = pts[:, 2]
        rows.append(np.concatenate([pts[:, 3], pts[:, 4]]))
        cyc_out.append(c)
    return np.array(rows), freq_grid, np.array(cyc_out)


def load_capacity(cell: str):
    """
    Load Data_Capacity_{cell}.txt.
    Returns (caps, cycles): discharge capacity per battery cycle (mAh).
    """
    path = CAP_DIR / f"Data_Capacity_{cell}.txt"
    raw  = np.loadtxt(path, skiprows=1)
    # cols: time(0) cycle(1) ox/red(2) ... Capacity/mAh(-1)
    caps, cyc_out = {}, []
    for c in np.unique(raw[:, 1]):
        d   = raw[raw[:, 1] == c]
        dis = d[d[:, 2] == 0]          # ox/red=0 → discharge
        if len(dis):
            caps[c] = dis[:, -1].max() # cumulative max = total discharge capacity
    sorted_c = sorted(caps)
    return np.array([caps[c] for c in sorted_c]), np.array(sorted_c)


# ── SUPP FIG 3 ── EIS spectra 3D + -Im[Z] trend for 25C01 ───────────────────

print("\n" + "=" * 60)
print("SUPP FIG 3 — 25C01 EIS spectra 3D + ARD feature trends")
print("=" * 60)

eis, freqs, cycles = load_eis_stateV("25C01")
print(f"  Loaded: {eis.shape[0]} cycles, {eis.shape[1]} features")
print(f"  Freq range: {freqs[-1]:.4f} – {freqs[0]:.2f} Hz")

# Identify ARD feature frequencies
idx91  = np.argmin(np.abs(freqs - 17.80))   # should be index 30
idx100 = np.argmin(np.abs(freqs -  2.16))   # should be index 39
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
ax.view_init(elev=22, azim=-55)
fig.patch.set_facecolor("white")
plt.tight_layout()
fig.savefig(OUT / "suppfig3a_EIS_spectra_3D_25C01.png", dpi=150)
print(f"  Saved → suppfig3a_EIS_spectra_3D_25C01.png")
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
ax.set_xlim(left=0)
fig.patch.set_facecolor("white")
plt.tight_layout()
fig.savefig(OUT / "suppfig3b_ImZ_trend_25C01.png", dpi=150)
print(f"  Saved → suppfig3b_ImZ_trend_25C01.png")
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

for cell, col in zip(TRAIN, TRAIN_COLORS):
    caps, cyc = load_capacity(cell)
    ax.plot(cyc, caps, color=col, lw=1.2, label=f"{cell}-train")
    print(f"  {cell} (train): {len(caps)} cycles, "
          f"init={caps[0]:.1f}, final={caps[-1]:.1f} mAh")

for cell, col in zip(TEST, TEST_COLORS):
    caps, cyc = load_capacity(cell)
    ax.plot(cyc, caps, color=col, lw=1.2, label=f"{cell}-test")
    print(f"  {cell} (test): {len(caps)} cycles, "
          f"init={caps[0]:.1f}, final={caps[-1]:.1f} mAh")

ax.set_xlabel("Cycle number",  fontsize=13)
ax.set_ylabel("Capacity (mAh)", fontsize=13)
ax.set_title("Capacity retention curves — all 12 cells  (Supp Fig 4)", fontsize=12)
ax.legend(fontsize=8, ncol=2, frameon=False, loc="lower left")
ax.set_xlim(left=0)
fig.patch.set_facecolor("white")
plt.tight_layout()
fig.savefig(OUT / "suppfig4_capacity_retention_all_cells.png", dpi=150)
print(f"  Saved → suppfig4_capacity_retention_all_cells.png")
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
