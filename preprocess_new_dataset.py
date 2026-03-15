"""
Preprocessor for new A1-A8 battery dataset (Battery data.zip)
Protocol: Ns=6 = PEIS after charge (10 kHz → 1 Hz, 33 freqs)
          Ns=8 = CC discharge → capacity

Why Ns=6: Closest analogue to paper's State V (15 min rest after full charge).
          Feature #91 (17.80 Hz) is exactly captured in Ns=6.

Output: Feature vectors interpolated to paper's 60-frequency grid,
        aligned with per-cycle discharge capacity.
"""

import zipfile
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Paper's 60 frequencies (Zenodo, ascending order for interpolation)
# Extracted from EIS_state_V_25C01.txt cycle 1
# ---------------------------------------------------------------------------
ZENODO_FREQS = np.array([
    0.02, 0.0253, 0.0319, 0.0404, 0.051, 0.0644, 0.0815, 0.1031, 0.1301,
    0.1645, 0.2079, 0.2626, 0.3318, 0.4198, 0.5307, 0.6707, 0.8473, 1.0708,
    1.3535, 1.7095, 2.1605, 2.7355, 3.4569, 4.3694, 5.5173, 6.9754, 8.8177,
    11.1448, 14.0681, 17.7961, 22.482, 28.4091, 35.9313, 45.3629, 57.3682,
    72.517, 91.6721, 115.778, 146.3582, 185.0592, 233.8774, 295.4728,
    373.2086, 471.9634, 596.7186, 754.2756, 952.8659, 1203.8446, 1522.4358,
    1923.1537, 2430.7778, 3070.9827, 3881.2737, 4905.291, 6217.2461,
    7835.48, 9909.4424, 12516.703, 15829.126, 20004.453
])  # 60 values, ascending


def parse_peis_cell(zip_path: str, cell: str) -> tuple:
    """
    Parse PEIS-HC-RT/{cell}.csv from zip file.

    Returns
    -------
    eis_matrix : ndarray (N_cycles, 120)
        Each row: [Re(Z) @ 60 zenodo freqs | Im(Z) @ 60 zenodo freqs]
        Interpolated from 33 Ns=6 measurements to 60 Zenodo freqs.
    capacities : ndarray (N_cycles,)
        Discharge capacity (mAh) per cycle.
    cycle_numbers : ndarray (N_cycles,)
        Battery cycle index.
    """
    with zipfile.ZipFile(zip_path) as z:
        with z.open(f'data/PEIS-HC-RT/{cell}.csv') as f:
            lines = f.read().decode('utf-8', errors='replace').splitlines()

    header = lines[0].split(',')
    ns_col  = header.index('Ns')
    freq_col = header.index('freq/Hz')
    re_col  = header.index('Re(Z)/Ohm')
    im_col  = header.index('#NAME?')   # -Im(Z)/Ohm (Excel corrupted name)
    cyc_col = header.index('cycle number')
    cap_col = header.index('Capacity/mA.h')

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
            cy  = int(float(r[cyc_col]))
            re  = float(r[re_col])
            im  = float(r[im_col])   # already -Im(Z) (positive in inductive region)
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
    # Use cycles that have BOTH Ns=6 EIS and Ns=8 capacity
    common_cycles = sorted(set(eis_rows.keys()) & set(cap_rows.keys()))

    eis_matrix  = []
    capacities  = []
    skipped     = 0

    for cy in common_cycles:
        freq_dict = eis_rows[cy]
        meas_freqs = np.array(sorted(freq_dict.keys()))   # ascending
        re_vals    = np.array([freq_dict[f][0] for f in meas_freqs])
        im_vals    = np.array([freq_dict[f][1] for f in meas_freqs])

        if len(meas_freqs) < 5:
            skipped += 1
            continue  # skip incomplete sweeps

        # Log-linear interpolation to Zenodo 60-freq grid
        # Extrapolate beyond measurement range using boundary values
        f_re = interp1d(np.log(meas_freqs), re_vals,
                        kind='linear', fill_value='extrapolate')
        f_im = interp1d(np.log(meas_freqs), im_vals,
                        kind='linear', fill_value='extrapolate')

        re_interp = f_re(np.log(ZENODO_FREQS))
        im_interp = f_im(np.log(ZENODO_FREQS))

        # Paper ordering: features 0-59 = Re(Z) high→low freq,
        #                 features 60-119 = Im(Z) high→low freq
        feat = np.concatenate([re_interp[::-1], im_interp[::-1]])
        eis_matrix.append(feat)
        capacities.append(cap_rows[cy])

    eis_matrix  = np.array(eis_matrix)
    capacities  = np.array(capacities)
    cycle_numbers = np.array(common_cycles[:len(eis_matrix)])

    if skipped:
        print(f'  [{cell}] Skipped {skipped} cycles with <5 EIS points')

    return eis_matrix, capacities, cycle_numbers


def compute_rul(capacities: np.ndarray) -> tuple:
    """
    Compute RUL labels using paper's convention:
      EOL = first index where capacity < 80% of cap[0]
      RUL[i] = 2 * (EOL_index - i)  (EIS every 2 battery cycles)

    Returns (rul_labels, eol_index) or (None, None) if EOL not reached.
    """
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

    ZIP_PATH  = r'C:\Users\hithe\Downloads\paper reproduce\Battery data.zip'
    OUT_DIR   = Path(r'C:\Users\hithe\Downloads\paper reproduce'
                     r'\battery_gpytorch_rtx4060\battery_gpytorch\data\new_dataset')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cells = [f'A{i}' for i in range(1, 9)]

    print('=' * 60)
    print('Preprocessing A1-A8  (Ns=6, after-charge EIS)')
    print('=' * 60)

    all_eis = {}
    all_cap = {}
    all_cyc = {}
    all_rul = {}
    all_eol = {}

    for cell in cells:
        print(f'\n--- {cell} ---')
        eis, cap, cyc = parse_peis_cell(ZIP_PATH, cell)
        rul, eol = compute_rul(cap)

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
