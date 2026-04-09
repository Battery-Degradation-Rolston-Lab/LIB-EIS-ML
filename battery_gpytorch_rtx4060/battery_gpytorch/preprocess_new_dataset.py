"""
Preprocessor for new A1-A8 battery dataset (Battery data.zip)
Protocol: Ns=6 = PEIS after charge (10 kHz → 1 Hz, 33 freqs)
          Ns=8 = CC discharge → capacity

Why Ns=6: Closest analogue to paper's State V (15 min rest after full charge).
          17.80 Hz (paper's SEI feature) is directly measured at index [23].

Output: 66-feature vectors using the 33 NATIVE measured frequencies only.
        No interpolation, no extrapolation — all values are real measurements.
        Features 1-33: Re(Z) at 33 freqs (high→low)
        Features 34-66: Im(Z) at same 33 freqs (high→low)

        Measured range: 0.999 Hz → 10000 Hz
        NOT captured: diffusion region < 1 Hz (would need longer PEIS sweep)
"""

import zipfile
import numpy as np
from pathlib import Path

# Native Ns=6 frequencies (high → low), confirmed from A1 cycle 1.
# Consistent across all cycles and cells.
NATIVE_FREQS = np.array([
    10000.0, 7500.0, 5620.0, 4220.0, 3160.0, 2370.0, 1780.0, 1330.0,
    1000.0, 750.0, 564.0, 422.0, 316.0, 237.0, 178.0, 135.0, 102.0,
    75.0, 56.2, 42.2, 31.6, 23.7, 17.8, 13.3, 10.0, 7.5, 5.62,
    4.22, 3.16, 2.37, 1.78, 1.33, 0.999
])  # 33 values, high → low
N_FREQS = len(NATIVE_FREQS)  # 33


def parse_peis_cell(zip_path: str, cell: str) -> tuple:
    """
    Parse PEIS-HC-RT/{cell}.csv from zip file.

    Returns
    -------
    eis_matrix : ndarray (N_cycles, 66)
        Each row: [Re(Z) @ 33 native freqs high→low | Im(Z) @ 33 native freqs high→low]
        No interpolation — raw measured values only.
    capacities : ndarray (N_cycles,)
        Discharge capacity (mAh) per cycle.
    cycle_numbers : ndarray (N_cycles,)
        Battery cycle index.
    """
    with zipfile.ZipFile(zip_path) as z:
        with z.open(f'data/PEIS-HC-RT/{cell}.csv') as f:
            lines = f.read().decode('utf-8', errors='replace').splitlines()

    header = lines[0].split(',')
    ns_col   = header.index('Ns')
    freq_col = header.index('freq/Hz')
    re_col   = header.index('Re(Z)/Ohm')
    im_col   = header.index('#NAME?')   # -Im(Z)/Ohm (Excel corrupted name)
    cyc_col  = header.index('cycle number')
    cap_col  = header.index('Capacity/mA.h')

    data = [row.split(',') for row in lines[1:] if row.strip()]

    # ---- Extract Ns=6 EIS rows (after charge PEIS) ----
    eis_rows = {}   # {cycle: {freq: (re, im)}}
    for r in data:
        try:
            ns  = int(float(r[ns_col]))
            frq = float(r[freq_col])
        except ValueError:
            continue
        if ns == 6 and frq > 0:
            cy = int(float(r[cyc_col]))
            re = float(r[re_col])
            im = float(r[im_col])
            if cy not in eis_rows:
                eis_rows[cy] = {}
            eis_rows[cy][frq] = (re, im)

    # ---- Extract Ns=8 discharge capacity ----
    cap_rows = {}   # {cycle: max_capacity}
    for r in data:
        try:
            ns  = int(float(r[ns_col]))
            cap = float(r[cap_col])
        except ValueError:
            continue
        if ns == 8:
            cy = int(float(r[cyc_col]))
            if cy not in cap_rows or cap > cap_rows[cy]:
                cap_rows[cy] = cap

    # ---- Build aligned EIS + capacity matrix ----
    common_cycles = sorted(set(eis_rows.keys()) & set(cap_rows.keys()))

    eis_matrix = []
    capacities = []
    skipped    = 0

    for cy in common_cycles:
        freq_dict  = eis_rows[cy]
        # Sort measured frequencies descending (high → low), matching NATIVE_FREQS order
        meas_freqs = np.array(sorted(freq_dict.keys(), reverse=True))

        if len(meas_freqs) < N_FREQS:
            skipped += 1
            continue  # skip incomplete sweeps

        re_vals = np.array([freq_dict[f][0] for f in meas_freqs[:N_FREQS]])
        im_vals = np.array([freq_dict[f][1] for f in meas_freqs[:N_FREQS]])

        # 66 features: Re(Z) then Im(Z), both high→low freq — no interpolation
        feat = np.concatenate([re_vals, im_vals])
        eis_matrix.append(feat)
        capacities.append(cap_rows[cy])

    eis_matrix    = np.array(eis_matrix)
    capacities    = np.array(capacities)
    cycle_numbers = np.array(common_cycles[:len(eis_matrix)])

    if skipped:
        print(f'  [{cell}] Skipped {skipped} cycles with <{N_FREQS} EIS points')

    return eis_matrix, capacities, cycle_numbers


def compute_rul(capacities: np.ndarray, force_dnf: bool = False) -> tuple:
    """
    Compute RUL labels using paper's convention:
      EOL = first index where capacity < 80% of cap[0]
      RUL[i] = 2 * (EOL_index - i)  (EIS every 2 battery cycles)

    Returns (rul_labels, eol_index) or (None, None) if EOL not reached.

    force_dnf : set True for cells confirmed by experimenter to have NOT reached
                failure (overrides automatic 80%-of-initial detection).
    """
    if force_dnf:
        return None, None
    cap0    = capacities[0]
    thresh  = 0.8 * cap0
    eol_idx = next((i for i, c in enumerate(capacities) if c < thresh), None)
    if eol_idx is None:
        return None, None
    rul = np.array([2 * (eol_idx - i) for i in range(eol_idx + 1)],
                   dtype=float)
    return rul, eol_idx


# ---------------------------------------------------------------------------
# Main: process all 8 cells
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    ZIP_PATH  = str(Path(__file__).parents[2] / "raw_data" / "Battery data.zip")
    OUT_DIR   = Path(__file__).parent / "data" / "new_dataset"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cells = [f'A{i}' for i in range(1, 9)]

    # Cells confirmed by experimenter to NOT have reached failure.
    # A3: anomalously low initial capacity (3800 mAh vs ~4050 mAh fleet nominal);
    #     automatic 80%-of-initial EOL detection gives a false positive — excluded.
    # A6: capacity remained > 80% throughout the full recorded window.
    DNF_CELLS = {'A3', 'A6'}

    print('=' * 60)
    print('Preprocessing A1-A8  (Ns=6, after-charge EIS)')
    print(f'DNF cells (no EOL): {sorted(DNF_CELLS)}')
    print('=' * 60)

    all_eis = {}
    all_cap = {}
    all_cyc = {}
    all_rul = {}
    all_eol = {}

    for cell in cells:
        print(f'\n--- {cell} ---')
        eis, cap, cyc = parse_peis_cell(ZIP_PATH, cell)
        rul, eol = compute_rul(cap, force_dnf=(cell in DNF_CELLS))

        all_eis[cell] = eis
        all_cap[cell] = cap
        all_cyc[cell] = cyc

        print(f'  EIS matrix : {eis.shape}')
        print(f'  Capacity   : {cap[0]:.1f} → {cap[-1]:.1f} mAh  '
              f'(80% thresh = {0.8*cap[0]:.1f})')
        if eol is not None:
            print(f'  EOL index  : {eol}  (cycle {cyc[eol]}),  RUL_max = {rul[0]:.0f}')
            all_rul[cell] = rul
            all_eol[cell] = eol
        else:
            print(f'  EOL        : NOT REACHED in {len(cap)} cycles')

        # Save per-cell files
        np.savetxt(OUT_DIR / f'EIS_{cell}.txt', eis)
        np.savetxt(OUT_DIR / f'cap_{cell}.txt', cap)
        np.savetxt(OUT_DIR / f'cyc_{cell}.txt', cyc)
        if rul is not None:
            np.savetxt(OUT_DIR / f'rul_{cell}.txt', rul)
            np.savetxt(OUT_DIR / f'EIS_rul_{cell}.txt', eis[:eol + 1])

    # Save combined train sets (A1-A4) and test sets (A5-A8)
    # Capacity model
    train_cells = ['A1', 'A2', 'A3', 'A4']
    test_cells  = ['A5', 'A6', 'A7', 'A8']

    eis_train = np.vstack([all_eis[c] for c in train_cells])
    cap_train = np.concatenate([all_cap[c] for c in train_cells])
    eis_test  = np.vstack([all_eis[c] for c in test_cells])
    cap_test  = np.concatenate([all_cap[c] for c in test_cells])

    np.savetxt(OUT_DIR / 'EIS_train.txt', eis_train)
    np.savetxt(OUT_DIR / 'cap_train.txt', cap_train)
    np.savetxt(OUT_DIR / 'EIS_test.txt',  eis_test)
    np.savetxt(OUT_DIR / 'cap_test.txt',  cap_test)

    # RUL model (only cells that reached EOL)
    rul_train_cells = [c for c in train_cells if c in all_rul]
    rul_test_cells  = [c for c in test_cells  if c in all_rul]

    if rul_train_cells:
        eis_rul_train = np.vstack([all_eis[c][:all_eol[c]+1] for c in rul_train_cells])
        rul_train     = np.concatenate([all_rul[c] for c in rul_train_cells])
        np.savetxt(OUT_DIR / 'EIS_rul_train.txt', eis_rul_train)
        np.savetxt(OUT_DIR / 'rul_train.txt',     rul_train)

    print('\n' + '=' * 60)
    print('Summary')
    print('=' * 60)
    print(f'Train EIS (A1-A4): {eis_train.shape}')
    print(f'Test  EIS (A5-A8): {eis_test.shape}')
    print(f'RUL training cells: {rul_train_cells}')
    print(f'RUL test cells:     {rul_test_cells}')
    print(f'\nFiles saved to: {OUT_DIR}')
    print('\nNext: run run_new_dataset.py to apply the paper GPR model')
