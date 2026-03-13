"""
Zenodo raw data preprocessor for Zhang et al. 2020

Converts raw EIS and capacity files from Zenodo into the N×120 / N-vector
format matching the GitHub preprocessed files.

Alignment convention (confirmed by comparison with GitHub files):
  EIS_data[i]      = state-V EIS at battery cycle (i+1)   [cycles 1, 2, 3, ...]
  Capacity_data[i] = discharge capacity at battery cycle i  [cycles 0, 1, 2, ...]
  => EIS[i] and Capacity[i] are paired (offset by 1 in cycle numbering)

RUL convention (confirmed from rul35C02.txt):
  - EIS measured every 2 battery cycles
  - EOL = first capacity index where capacity < 80% of initial (index 0)
  - RUL[i] = 2 * (EOL_index - i)  for i = 0 .. EOL_index
"""

import numpy as np
from pathlib import Path

EIS_DIR = Path(r"C:\Users\hithe\Downloads\paper reproduce\EIS data\EIS data")
CAP_DIR = Path(r"C:\Users\hithe\Downloads\paper reproduce\Capacity data\Capacity data")
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ── Core preprocessors ────────────────────────────────────────────────────────

def preprocess_eis(cell: str) -> np.ndarray:
    """
    Load EIS_state_V_{cell}.txt and return N×120 matrix.
    Columns 0-59: Re(Z) at 60 frequencies (high to low freq)
    Columns 60-119: -Im(Z) at same frequencies
    Skips incomplete cycles (not exactly 60 frequency points).
    """
    path = EIS_DIR / f"EIS_state_V_{cell}.txt"
    raw = np.loadtxt(path, skiprows=1)
    # cols: time(0), cycle(1), freq(2), Re(Z)(3), -Im(Z)(4), |Z|(5), Phase(6)
    cycles = np.unique(raw[:, 1])
    rows = []
    for c in cycles:
        mask = raw[:, 1] == c
        row = raw[mask]
        if len(row) != 60:
            continue  # skip incomplete measurement
        row = row[np.argsort(row[:, 2])[::-1]]  # sort by freq descending
        rows.append(np.concatenate([row[:, 3], row[:, 4]]))
    return np.array(rows)


def preprocess_capacity(cell: str) -> np.ndarray:
    """
    Load Data_Capacity_{cell}.txt and return discharge capacity per cycle.
    Returns vector of length N+1: caps[i] = max discharge capacity at cycle i
    (cycles 0, 1, 2, ...) where cycle 0 is the initial measurement.
    """
    path = CAP_DIR / f"Data_Capacity_{cell}.txt"
    raw = np.loadtxt(path, skiprows=1)
    # cols: time(0), cycle(1), ox/red(2), [Ewe/V(3), I/mA(4),] Capacity_mAh(-1)
    # File has either 4 or 6 columns; capacity is always the LAST column.
    # ox/red=0: discharge, ox/red=1: charge
    cycles = np.unique(raw[:, 1])
    caps = {}
    for c in cycles:
        d = raw[raw[:, 1] == c]
        dis = d[d[:, 2] == 0]  # discharge half
        if len(dis) > 0:
            caps[c] = dis[:, -1].max()  # last column = Capacity/mA.h
    # Return sorted by cycle number (0, 1, 2, ...)
    sorted_cycles = sorted(caps.keys())
    return np.array([caps[c] for c in sorted_cycles])


def find_eol(caps: np.ndarray, threshold: float = 0.80) -> int:
    """Return first index where capacity drops below threshold * initial."""
    init = caps[0]
    thr = init * threshold
    for i, c in enumerate(caps):
        if c < thr:
            return i
    return len(caps) - 1  # never reached EOL in data


def compute_rul(eol_index: int) -> np.ndarray:
    """
    Return RUL vector for a cell whose EOL is at index eol_index.
    RUL[i] = 2 * (eol_index - i)  for i = 0 .. eol_index
    (the ×2 because EIS is measured every 2 battery cycles)
    """
    return np.array([2 * (eol_index - i) for i in range(eol_index + 1)],
                    dtype=float)


def align(eis: np.ndarray, caps: np.ndarray):
    """
    Align EIS (N rows) and caps (M values) by trimming to min(N, M).
    Returns (eis_aligned, caps_aligned).
    """
    n = min(len(eis), len(caps))
    return eis[:n], caps[:n]


# ── Per-cell preprocessing ────────────────────────────────────────────────────

def process_cell(cell: str, verbose: bool = True):
    """Return (eis_N120, caps_N, rul_eol) aligned arrays for one cell."""
    eis = preprocess_eis(cell)
    caps = preprocess_capacity(cell)
    eis, caps = align(eis, caps)
    eol = find_eol(caps)
    if verbose:
        print(f"  {cell}: {len(eis)} EIS rows, EOL at idx {eol} "
              f"(cap={caps[eol]:.3f}, init={caps[0]:.3f}, "
              f"ratio={caps[eol]/caps[0]:.3f})")
    return eis, caps, eol


# ── Build datasets ────────────────────────────────────────────────────────────

def build_capacity_dataset(cells):
    """Concatenate EIS and capacity for a list of cells (all cycles)."""
    eis_list, cap_list = [], []
    for cell in cells:
        eis, caps, _ = process_cell(cell)
        eis_list.append(eis)
        cap_list.append(caps)
    return np.vstack(eis_list), np.concatenate(cap_list)


def build_rul_dataset(cells):
    """Concatenate EIS and RUL labels for a list of cells (cycles up to EOL)."""
    eis_list, rul_list = [], []
    for cell in cells:
        eis, caps, eol = process_cell(cell)
        rul = compute_rul(eol)
        # Trim to EOL (inclusive)
        eis_eol = eis[:eol + 1]
        eis_list.append(eis_eol)
        rul_list.append(rul)
    return np.vstack(eis_list), np.concatenate(rul_list)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Zenodo EIS/Capacity Preprocessor")
    print("=" * 60)

    # ── Verify against 35C02 ground truth ────────────────────────────────────
    print("\n[VERIFY] 35C02 ground truth check ...")
    eis_35c02_ref = np.loadtxt(DATA_DIR / "EIS_data_35C02.txt")
    cap_35c02_ref = np.loadtxt(DATA_DIR / "capacity35C02.txt")

    eis_35c02, cap_35c02, eol_35c02 = process_cell("35C02")
    print(f"  EIS shape: {eis_35c02.shape} (ref: {eis_35c02_ref.shape})")
    print(f"  Cap length: {len(cap_35c02)} (ref: {len(cap_35c02_ref)})")
    n_check = min(len(cap_35c02), len(cap_35c02_ref))
    cap_err = np.abs(cap_35c02[:n_check] - cap_35c02_ref[:n_check]).max()
    print(f"  Max capacity error (first {n_check} cycles): {cap_err:.6f} mAh")
    print(f"  Status: {'OK' if cap_err < 0.001 else 'WARNING - mismatch'}")

    # ── 25°C test cells (25C05–08) ────────────────────────────────────────────
    print("\n[BUILD] 25°C test cells (25C05-08) ...")
    eis_25test, cap_25test = build_capacity_dataset(["25C05","25C06","25C07","25C08"])
    np.savetxt(DATA_DIR / "EIS_data_25C_test.txt", eis_25test, fmt="%.5f")
    np.savetxt(DATA_DIR / "Capacity_data_25C_test.txt", cap_25test, fmt="%.5f")
    print(f"  Saved EIS_data_25C_test.txt       shape={eis_25test.shape}")
    print(f"  Saved Capacity_data_25C_test.txt  len={len(cap_25test)}")

    # ── 25°C training cells (25C01–04) ───────────────────────────────────────
    # Use the existing EIS_data.txt rows 0-759 (confirmed exact match above)
    # and Capacity_data.txt[0:760] to stay consistent with multi-T training.
    # For Fig 1 we need ONLY the 25°C portion.
    print("\n[BUILD] 25°C training cells (25C01-04) ...")
    eis_25train, cap_25train = build_capacity_dataset(["25C01","25C02","25C03","25C04"])
    np.savetxt(DATA_DIR / "EIS_data_25C_train.txt", eis_25train, fmt="%.5f")
    np.savetxt(DATA_DIR / "Capacity_data_25C_train.txt", cap_25train, fmt="%.5f")
    print(f"  Saved EIS_data_25C_train.txt      shape={eis_25train.shape}")
    print(f"  Saved Capacity_data_25C_train.txt len={len(cap_25train)}")

    # ── RUL: 25°C training (25C01–04) ────────────────────────────────────────
    print("\n[BUILD] RUL training cells (25C01-04) ...")
    eis_rul25_train, rul_25train = build_rul_dataset(["25C01","25C02","25C03","25C04"])
    np.savetxt(DATA_DIR / "EIS_data_25C_RUL_train.txt", eis_rul25_train, fmt="%.5f")
    np.savetxt(DATA_DIR / "RUL_25C_train.txt", rul_25train, fmt="%.5f")
    print(f"  Saved EIS_data_25C_RUL_train.txt  shape={eis_rul25_train.shape}")
    print(f"  Saved RUL_25C_train.txt           len={len(rul_25train)}")

    # ── RUL: 25°C test cells (25C05–08) individually ─────────────────────────
    print("\n[BUILD] RUL test cells (25C05-08) ...")
    for cell in ["25C05","25C06","25C07","25C08"]:
        eis, caps, eol = process_cell(cell)
        rul = compute_rul(eol)
        eis_eol = eis[:eol + 1]
        np.savetxt(DATA_DIR / f"EIS_rul_{cell}.txt", eis_eol, fmt="%.5f")
        np.savetxt(DATA_DIR / f"rul_{cell}.txt", rul, fmt="%.5f")
        print(f"  {cell}: EIS shape={eis_eol.shape}, RUL {rul[0]:.0f}->0")

    # ── 45C02 capacity test ───────────────────────────────────────────────────
    print("\n[BUILD] 45C02 test cell ...")
    eis_45c02, cap_45c02, eol_45c02 = process_cell("45C02")
    rul_45c02 = compute_rul(eol_45c02)
    eis_45c02_eol = eis_45c02[:eol_45c02 + 1]
    np.savetxt(DATA_DIR / "EIS_data_45C02.txt", eis_45c02, fmt="%.5f")
    np.savetxt(DATA_DIR / "capacity45C02.txt", cap_45c02, fmt="%.5f")
    np.savetxt(DATA_DIR / "EIS_rul_45C02.txt", eis_45c02_eol, fmt="%.5f")
    np.savetxt(DATA_DIR / "rul45C02.txt", rul_45c02, fmt="%.5f")
    print(f"  Saved EIS_data_45C02.txt    shape={eis_45c02.shape}")
    print(f"  Saved capacity45C02.txt     len={len(cap_45c02)}")
    print(f"  Saved EIS_rul_45C02.txt     shape={eis_45c02_eol.shape}")
    print(f"  Saved rul45C02.txt          RUL {rul_45c02[0]:.0f}->0")

    print("\n[DONE] All files saved to", DATA_DIR)
