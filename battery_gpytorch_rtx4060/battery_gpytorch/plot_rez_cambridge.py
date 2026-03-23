"""
Plot Re(Z) at highest frequency vs cycle for Cambridge (Zenodo) cells,
mirroring the A1-A8 plot for direct comparison.

Cambridge: 120 features = Re(Z)[0..59] || Im(Z)[60..119], sorted freq descending.
Feature index 0 = highest-frequency Re(Z) = bulk ohmic resistance.
EIS measured every 2 battery cycles → cycle = row_index * 2.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

DATA = r"C:\Users\hithe\Downloads\paper reproduce\battery_gpytorch_rtx4060\battery_gpytorch\data"
OUT  = r"C:\Users\hithe\Downloads\paper reproduce\battery_gpytorch_rtx4060\battery_gpytorch\output\new_dataset"
os.makedirs(OUT, exist_ok=True)

# ── Load Cambridge cells ──────────────────────────────────────────────────────
# 25C train = 25C01 (200) + 25C02 (250) + 25C03 (229) = 679 rows
eis_25_train = np.loadtxt(os.path.join(DATA, "EIS_data_25C_train.txt"))
cap_25_train = np.loadtxt(os.path.join(DATA, "Capacity_data_25C_train.txt"))

splits = {"25C01": (0, 200), "25C02": (200, 450), "25C03": (450, 679)}
cells_25 = {
    name: {
        "eis": eis_25_train[a:b],
        "cap": cap_25_train[a:b],
        "cyc": np.arange(b - a) * 2,
    }
    for name, (a, b) in splits.items()
}

# 35C02 and 45C02 — isolated files
cells_other = {}
for name, eis_f, cap_f in [
    ("35C02", "EIS_data_35C02.txt",  "capacity35C02.txt"),
    ("45C02", "EIS_data_45C02.txt",  "capacity45C02.txt"),
]:
    eis = np.loadtxt(os.path.join(DATA, eis_f))
    cap = np.loadtxt(os.path.join(DATA, cap_f))
    cells_other[name] = {"eis": eis, "cap": cap, "cyc": np.arange(len(cap)) * 2}

all_cells = {**cells_25, **cells_other}

# Colour map: warm = high temp, cool = low temp
colour_map = {
    "25C01": "#1f77b4", "25C02": "#4a90d9", "25C03": "#7bb3e8",
    "35C02": "#e87c1f",
    "45C02": "#c0392b",
}

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
ax_rez, ax_cap = axes

for name, d in all_cells.items():
    rez = d["eis"][:, 0] * 1000   # Ω → mΩ, index 0 = highest-freq Re(Z)
    cap = d["cap"]
    cyc = d["cyc"]
    c   = colour_map[name]
    ax_rez.plot(cyc, rez, color=c, label=name, linewidth=1.8)
    ax_cap.plot(cyc, cap, color=c, label=name, linewidth=1.8)

ax_rez.set_ylabel("Re(Z) at highest freq  [mΩ]", fontsize=12)
ax_rez.set_title("Cambridge (Zenodo) — Bulk ohmic resistance vs cycle", fontsize=13)
ax_rez.legend(fontsize=10, ncol=5)
ax_rez.grid(True, alpha=0.3)
ax_rez.set_xlabel("Cycle number", fontsize=11)

ax_cap.set_ylabel("Capacity  [mAh]", fontsize=12)
ax_cap.set_title("Cambridge (Zenodo) — Capacity vs cycle", fontsize=13)
ax_cap.legend(fontsize=10, ncol=5)
ax_cap.grid(True, alpha=0.3)
ax_cap.set_xlabel("Cycle number", fontsize=11)

plt.tight_layout()
out_path = os.path.join(OUT, "fig_ReZ_cambridge.png")
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
plt.show()
