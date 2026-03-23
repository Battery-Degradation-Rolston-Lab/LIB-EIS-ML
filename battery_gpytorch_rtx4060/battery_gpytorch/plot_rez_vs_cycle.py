"""
Plot Re(Z) at 7500 Hz (feature #2, index 1) vs cycle for all A1-A8 cells,
alongside capacity, to show the bulk resistance increase with degradation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "new_dataset")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "output", "new_dataset")
os.makedirs(OUT_DIR, exist_ok=True)

CELLS = [f"A{i}" for i in range(1, 9)]
COLOURS = plt.cm.tab10(np.linspace(0, 0.8, 8))

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
ax_rez, ax_cap = axes

for cell, colour in zip(CELLS, COLOURS):
    eis  = np.loadtxt(os.path.join(DATA_DIR, f"EIS_{cell}.txt"))   # (N, 66)
    cap  = np.loadtxt(os.path.join(DATA_DIR, f"cap_{cell}.txt"))   # (N,)
    cyc  = np.loadtxt(os.path.join(DATA_DIR, f"cyc_{cell}.txt"))   # (N,)

    rez = eis[:, 1]   # feature #2 = Re(Z) at 7500 Hz (index 1, 0-based)

    ax_rez.plot(cyc, rez * 1000, color=colour, label=cell, linewidth=1.5)
    ax_cap.plot(cyc, cap,        color=colour, label=cell, linewidth=1.5)

# ── Re(Z) panel ──────────────────────────────────────────────────────────────
ax_rez.set_ylabel("Re(Z) at 7500 Hz  [mΩ]", fontsize=12)
ax_rez.set_title("Bulk ohmic resistance (Re(Z) at 7500 Hz) vs cycle — A1–A8", fontsize=13)
ax_rez.legend(fontsize=9, ncol=4, loc="upper left")
ax_rez.grid(True, alpha=0.3)
ax_rez.set_xlabel("Cycle number", fontsize=11)

# ── Capacity panel ────────────────────────────────────────────────────────────
ax_cap.set_ylabel("Capacity  [mAh]", fontsize=12)
ax_cap.set_title("Capacity vs cycle — A1–A8", fontsize=13)
ax_cap.legend(fontsize=9, ncol=4, loc="lower left")
ax_cap.grid(True, alpha=0.3)
ax_cap.set_xlabel("Cycle number", fontsize=11)

plt.tight_layout()
out = os.path.join(OUT_DIR, "fig_ReZ_vs_cycle.png")
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.show()
