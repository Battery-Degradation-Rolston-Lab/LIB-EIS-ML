"""Generic plotter for capacity-retention (%) or raw-capacity (mAh) curves.

Expected file layout inside --data-dir:
  cap_<CELL>.txt   capacity vector
  cyc_<CELL>.txt   cycle numbers (optional; falls back to 0..N-1)

Examples
--------
# A1-A8 retention plot
python plot_capacity_retention_generic.py \
  --data-dir data/new_dataset \
  --title "In-house cells (A1-A8) — capacity retention curves" \
  --output output/new_dataset/fig_capacity_retention_A1_A8.png

# CA1-CA8 raw capacity plot in mAh
python plot_capacity_retention_generic.py \
  --data-dir data/ca_dataset \
  --prefix CA \
  --mode capacity \
  --title "RT complete-lifecycle cells (CA1-CA8) — capacity vs cycle" \
  --output output/ca_zhang/fig_capacity_CA1_CA8_mAh.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def natural_key(text: str):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


CAMBRIDGE_25_TRAIN = ["25C01", "25C02", "25C03", "25C04"]
CAMBRIDGE_25_TEST = ["25C05", "25C06", "25C07", "25C08"]
CAMBRIDGE_SINGLE_FILES = {
    "35C01": "Capacity_data_35.txt",
    "35C02": "capacity35C02.txt",
    "45C01": "Capacity_data_45.txt",
    "45C02": "capacity45C02.txt",
}
CAMBRIDGE_CACHE: dict[str, dict[str, np.ndarray]] = {}


def split_concatenated_capacity(caps: np.ndarray, cell_names: list[str]) -> dict[str, np.ndarray]:
    thresholds = [max(0.5, 0.03 * float(np.max(caps))), max(0.3, 0.02 * float(np.max(caps)))]

    for jump_threshold in thresholds:
        reset_points = np.where(np.diff(caps) > jump_threshold)[0] + 1
        bounds = [0, *reset_points.tolist(), len(caps)]
        segments = [caps[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1) if bounds[i] < bounds[i + 1]]
        if len(segments) == len(cell_names):
            return {name: seg for name, seg in zip(cell_names, segments)}

    raise ValueError(
        f"Could not split concatenated capacity file into {len(cell_names)} cells. "
        f"Detected reset points at {reset_points.tolist()}"
    )


def discover_cells(data_dir: Path) -> list[str]:
    cells = [path.stem.replace("cap_", "", 1) for path in data_dir.glob("cap_*.txt")]

    if (data_dir / "Capacity_data_25C_train.txt").exists():
        cells.extend(CAMBRIDGE_25_TRAIN)
    if (data_dir / "Capacity_data_25C_test.txt").exists():
        cells.extend(CAMBRIDGE_25_TEST)
    for cell, filename in CAMBRIDGE_SINGLE_FILES.items():
        if (data_dir / filename).exists():
            cells.append(cell)

    return sorted(set(cells), key=natural_key)


def build_color_map(cells: list[str]) -> dict[str, tuple]:
    color_map: dict[str, tuple] = {}

    groups = [
        ([c for c in cells if c.startswith("25C")], plt.cm.Blues(np.linspace(0.75, 0.35, max(1, len([c for c in cells if c.startswith("25C")]))))),
        ([c for c in cells if c.startswith("35C")], plt.cm.Oranges(np.linspace(0.75, 0.45, max(1, len([c for c in cells if c.startswith("35C")]))))),
        ([c for c in cells if c.startswith("45C")], plt.cm.Reds(np.linspace(0.75, 0.45, max(1, len([c for c in cells if c.startswith("45C")]))))),
        ([c for c in cells if c.startswith("N10")], plt.cm.Blues(np.linspace(0.72, 0.35, max(1, len([c for c in cells if c.startswith("N10")]))))),
        ([c for c in cells if c.startswith("N20")], plt.cm.Reds(np.linspace(0.72, 0.35, max(1, len([c for c in cells if c.startswith("N20")]))))),
    ]

    assigned = set()
    for group_cells, palette in groups:
        for cell, color in zip(group_cells, palette):
            color_map[cell] = color
            assigned.add(cell)

    others = [c for c in cells if c not in assigned]
    if others:
        palette = plt.cm.Blues(np.linspace(0.72, 0.28, len(others))) if len(others) > 1 else [plt.cm.Blues(0.65)]
        for cell, color in zip(others, palette):
            color_map[cell] = color

    return color_map


def load_series(data_dir: Path, cell: str) -> tuple[np.ndarray, np.ndarray]:
    cap_path = data_dir / f"cap_{cell}.txt"
    cyc_path = data_dir / f"cyc_{cell}.txt"

    if cap_path.exists():
        cap = np.loadtxt(cap_path).astype(float).ravel()
        cyc = np.loadtxt(cyc_path).astype(float).ravel() if cyc_path.exists() else np.arange(len(cap), dtype=float)
    else:
        cache_key = str(data_dir.resolve())
        if cache_key not in CAMBRIDGE_CACHE:
            cap_map: dict[str, np.ndarray] = {}
            train_25 = data_dir / "Capacity_data_25C_train.txt"
            test_25 = data_dir / "Capacity_data_25C_test.txt"
            if train_25.exists():
                cap_map.update(split_concatenated_capacity(np.loadtxt(train_25).astype(float).ravel(), CAMBRIDGE_25_TRAIN))
            if test_25.exists():
                cap_map.update(split_concatenated_capacity(np.loadtxt(test_25).astype(float).ravel(), CAMBRIDGE_25_TEST))
            for cam_cell, filename in CAMBRIDGE_SINGLE_FILES.items():
                file_path = data_dir / filename
                if file_path.exists():
                    cap_map[cam_cell] = np.loadtxt(file_path).astype(float).ravel()
            CAMBRIDGE_CACHE[cache_key] = cap_map

        cap_map = CAMBRIDGE_CACHE[cache_key]
        if cell not in cap_map:
            raise FileNotFoundError(f"No capacity file found for cell {cell} in {data_dir}")
        cap = cap_map[cell]
        cyc = np.arange(len(cap), dtype=float)

    if len(cap) != len(cyc):
        n = min(len(cap), len(cyc))
        cap = cap[:n]
        cyc = cyc[:n]

    return cap, cyc


def plot_capacity_retention(
    data_dir: Path,
    cells: list[str],
    title: str,
    output: Path,
    ymin: float,
    dpi: int,
    mode: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = build_color_map(cells)

    fig, ax = plt.subplots(figsize=(10.8, 6.6))

    for cell in cells:
        cap, cyc = load_series(data_dir, cell)
        y = 100.0 * cap / cap[0] if mode == "retention" else cap
        ax.plot(cyc, y, linewidth=2.1, color=colors[cell], label=cell)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Cycle number", fontsize=12)
    ax.set_ylabel("Capacity retention (%)" if mode == "retention" else "Capacity (mAh)", fontsize=12)
    ax.grid(True, alpha=0.3)
    if mode == "retention":
        ax.set_ylim(ymin, 102)
    else:
        ax.set_ylim(bottom=ymin)
    ax.legend(ncol=min(4, max(2, len(cells))), fontsize=10, frameon=False, loc="lower left")

    plt.tight_layout()
    plt.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output}")
    for cell in cells:
        cap, _ = load_series(data_dir, cell)
        print(f"{cell}: n={len(cap):3d}, cap0={cap[0]:.1f}, final_cap={cap[-1]:.1f}, final_ret={100 * cap[-1] / cap[0]:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Plot full-cycle capacity-retention (%) or raw-capacity (mAh) curves from a dataset folder.")
    parser.add_argument("--data-dir", required=True, help="Folder containing cap_<CELL>.txt and cyc_<CELL>.txt files")
    parser.add_argument("--cells", nargs="*", default=None, help="Optional explicit list of cell IDs to plot")
    parser.add_argument("--prefix", nargs="*", default=None, help="Optional prefix filter, e.g. A CA N10 N20")
    parser.add_argument("--mode", choices=["retention", "capacity"], default="retention", help="Plot normalized retention (%) or raw capacity (mAh)")
    parser.add_argument("--title", default="Capacity retention curves", help="Figure title")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--ymin", type=float, default=70.0, help="Lower y-axis limit")
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    cells = args.cells if args.cells else discover_cells(data_dir)
    if args.prefix:
        prefixes = tuple(args.prefix)
        cells = [cell for cell in cells if cell.startswith(prefixes)]

    if not cells:
        raise ValueError("No cells found to plot. Check --data-dir, --cells, or --prefix.")

    plot_capacity_retention(
        data_dir=data_dir,
        cells=sorted(cells, key=natural_key),
        title=args.title,
        output=Path(args.output),
        ymin=args.ymin,
        dpi=args.dpi,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
