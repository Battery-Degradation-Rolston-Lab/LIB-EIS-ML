"""
Preprocessor for CA1-CA8 battery dataset.

Source files: Battery data/100 cycles/*.mpt  (BioLogic EC-Lab ASCII)
Chemistry   : MOLICELL 21700 P42A NMC, Room Temperature
Protocol    : Ns=6 = PEIS after charge (10 kHz → 1 Hz, 33 freqs, every cycle)
              Ns=8 = CC discharge → capacity

EIS cadence: 1 EIS per battery cycle  (vs A1-A8 which was every 2 cycles)
             → RUL factor = 1  (RUL[i] = 1 * (EOL_index - i))

Output format: identical to A1-A8 (preprocess_new_dataset.py)
  EIS_CA{n}.txt     — (N_cycles, 66)  [Re(Z) @ 33 freqs | Im(Z) @ 33 freqs]
  cap_CA{n}.txt     — (N_cycles,)     discharge capacity in mAh
  cyc_CA{n}.txt     — (N_cycles,)     BioLogic cycle number
  rul_CA{n}.txt     — (EOL_idx+1,)    RUL in battery cycles  [EOL cells only]
  EIS_rul_CA{n}.txt — (EOL_idx+1, 66) EIS rows up to EOL      [EOL cells only]

DNF cells:
  CA6: capacity remained > 80% throughout (genuine DNF, cap_last=3399/4066=83.6%)
  CA3: anomalously low cap0=3813 mAh (fleet nominal ~4050 mAh), similar to A3 pattern.
       INCLUDED as EOL cell since it genuinely degraded to near-zero — flag in analysis.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path

# Same 33-frequency grid as A1-A8 (Ns=6, 10 kHz → ~1 Hz)
NATIVE_FREQS = np.array([
    10000.0, 7500.0, 5620.0, 4220.0, 3160.0, 2370.0, 1780.0, 1330.0,
    1000.0,  750.0,  564.0,  422.0,  316.0,  237.0,  178.0,  135.0,
    102.0,   75.0,   56.2,   42.2,   31.6,   23.7,   17.8,   13.3,
    10.0,    7.5,    5.62,   4.22,   3.16,   2.37,   1.78,   1.33, 0.999
])  # 33 values, high → low
N_FREQS = len(NATIVE_FREQS)  # 33

# RUL factor: EIS every 1 battery cycle for CA cells
RUL_FACTOR = 1


def find_col_line(lines):
    """Return index of the tab-separated column header line."""
    for i, line in enumerate(lines):
        if 'mode' in line and 'Ns' in line and 'freq/Hz' in line:
            return i
    raise ValueError("Column header line not found in .mpt file")


def parse_mpt(path: Path) -> tuple:
    """
    Parse a BioLogic .mpt file.

    Returns
    -------
    eis_matrix   : ndarray (N_cycles, 66)
    capacities   : ndarray (N_cycles,)
    cycle_numbers: ndarray (N_cycles,)
    """
    with open(path, encoding='latin-1') as f:
        lines = f.readlines()

    col_idx = find_col_line(lines)
    cols = lines[col_idx].strip().split('\t')
    data = [l.strip().split('\t') for l in lines[col_idx + 1:] if l.strip()]

    ns_i   = cols.index('Ns')
    freq_i = cols.index('freq/Hz')
    cyc_i  = cols.index('cycle number')
    rez_i  = cols.index('Re(Z)/Ohm')
    imz_i  = cols.index('-Im(Z)/Ohm')
    cap_i  = cols.index('Capacity/mA.h')

    # ---- Ns=6: post-charge PEIS (10 kHz → 1 Hz) ----
    eis_rows = {}   # {cycle_int: {freq: (re, im)}}
    for r in data:
        try:
            ns  = int(float(r[ns_i]))
            frq = float(r[freq_i])
        except (ValueError, IndexError):
            continue
        if ns == 6 and frq > 0:
            cy = int(float(r[cyc_i]))
            re = float(r[rez_i])
            im = float(r[imz_i])
            if cy not in eis_rows:
                eis_rows[cy] = {}
            eis_rows[cy][frq] = (re, im)

    # ---- Ns=8: CC discharge capacity ----
    cap_rows = {}   # {cycle_int: max_capacity}
    for r in data:
        try:
            ns  = int(float(r[ns_i]))
            cap = float(r[cap_i])
        except (ValueError, IndexError):
            continue
        if ns == 8 and cap > 0:
            cy = int(float(r[cyc_i]))
            if cy not in cap_rows or cap > cap_rows[cy]:
                cap_rows[cy] = cap

    # ---- Align EIS + capacity ----
    common = sorted(set(eis_rows) & set(cap_rows))

    eis_matrix = []
    capacities = []
    skipped    = 0

    for cy in common:
        freq_dict = eis_rows[cy]
        meas_freqs = np.array(sorted(freq_dict.keys(), reverse=True))

        if len(meas_freqs) < N_FREQS:
            skipped += 1
            continue

        re_vals = np.array([freq_dict[f][0] for f in meas_freqs])
        im_vals = np.array([freq_dict[f][1] for f in meas_freqs])
        eis_matrix.append(np.concatenate([re_vals, im_vals]))
        capacities.append(cap_rows[cy])

    if skipped:
        print(f'    Skipped {skipped} incomplete EIS sweeps')

    eis_matrix    = np.array(eis_matrix)
    capacities    = np.array(capacities)
    cycle_numbers = np.array(common[:len(eis_matrix)])

    return eis_matrix, capacities, cycle_numbers


def compute_rul(capacities: np.ndarray, force_dnf: bool = False) -> tuple:
    """
    EOL  = first index where capacity < 80% of cap[0]
    RUL[i] = RUL_FACTOR * (EOL_index - i)

    CA cells: RUL_FACTOR=1  (EIS every 1 battery cycle)
    A1-A8:    RUL_FACTOR=2  (EIS every 2 battery cycles)
    """
    if force_dnf:
        return None, None
    cap0    = capacities[0]
    thresh  = 0.8 * cap0
    eol_idx = next((i for i, c in enumerate(capacities) if c < thresh), None)
    if eol_idx is None:
        return None, None
    rul = np.array([RUL_FACTOR * (eol_idx - i) for i in range(eol_idx + 1)],
                   dtype=float)
    return rul, eol_idx


# ---------------------------------------------------------------------------
# File map: cell name → .mpt filename (under Battery data/100 cycles/)
# ---------------------------------------------------------------------------
CA_FILES = {
    'CA1': '2403060001Li-ion_RT_CA1.mpt',
    'CA2': '2403060002Li-ion_RT_CA2.mpt',
    'CA3': '2403060003Li-ion_RT_CA3.mpt',
    'CA4': '2403060004Li-ion_RT_CA4.mpt',
    'CA5': '2403060005Li-ion_RT_CA5.mpt',
    'CA6': '2403060006Li-ion_RT_CA6.mpt',
    'CA7': '2403060007Li-ion_RT_CA7.mpt',
    'CA8': '2403060008Li-ion_RT_CA8.mpt',
}

# CA6: genuine DNF (never reached 80% EOL threshold)
# CA3: low cap0 anomaly — included as EOL, flagged for review
DNF_CELLS = {'CA6'}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    SRC_DIR = Path(__file__).parents[2] / "Battery data" / "100 cycles"
    OUT_DIR = Path(__file__).parent / "data" / "ca_dataset"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cells = list(CA_FILES.keys())

    print('=' * 60)
    print('Preprocessing CA1-CA8  (Ns=6, after-charge EIS, RT)')
    print(f'DNF cells (no EOL): {sorted(DNF_CELLS)}')
    print(f'RUL factor: {RUL_FACTOR}  (EIS every 1 battery cycle)')
    print('=' * 60)

    all_eis = {}
    all_cap = {}
    all_cyc = {}
    all_rul = {}
    all_eol = {}

    for cell in cells:
        path = SRC_DIR / CA_FILES[cell]
        print(f'\n--- {cell} ---')
        print(f'  File: {path.name}')

        eis, cap, cyc = parse_mpt(path)
        rul, eol = compute_rul(cap, force_dnf=(cell in DNF_CELLS))

        all_eis[cell] = eis
        all_cap[cell] = cap
        all_cyc[cell] = cyc

        print(f'  EIS matrix  : {eis.shape}')
        print(f'  Capacity    : {cap[0]:.1f} → {cap[-1]:.1f} mAh  '
              f'(80% thresh = {0.8 * cap[0]:.1f})')
        if eol is not None:
            print(f'  EOL index   : {eol}  (cycle {cyc[eol]:.0f}),  '
                  f'RUL_max = {rul[0]:.0f} bat-cycles')
            if cell == 'CA3':
                print(f'  [NOTE] CA3 cap0={cap[0]:.0f} mAh is below fleet nominal '
                      f'(~4050 mAh) — review before including in RUL training')
            all_rul[cell] = rul
            all_eol[cell] = eol
        else:
            print(f'  EOL         : NOT REACHED  ({len(cap)} cycles recorded)')

        # Save per-cell files
        np.savetxt(OUT_DIR / f'EIS_{cell}.txt',  eis)
        np.savetxt(OUT_DIR / f'cap_{cell}.txt',  cap)
        np.savetxt(OUT_DIR / f'cyc_{cell}.txt',  cyc)
        if rul is not None:
            np.savetxt(OUT_DIR / f'rul_{cell}.txt',     rul)
            np.savetxt(OUT_DIR / f'EIS_rul_{cell}.txt', eis[:eol + 1])

    # ---- Summary ----
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    eol_cells = [c for c in cells if c in all_rul]
    dnf_cells = [c for c in cells if c in DNF_CELLS or c not in all_rul]

    print(f'\n{"Cell":<6}  {"cap0":>7}  {"EOL@":>6}  {"RUL_max":>8}  {"n_cycles":>9}  {"Status"}')
    print('-' * 55)
    for c in cells:
        cap0 = all_cap[c][0]
        n    = len(all_cap[c])
        if c in all_rul:
            eol = all_eol[c]
            rul_max = int(all_rul[c][0])
            tag = '[LOW cap0]' if c == 'CA3' else ''
            print(f'{c:<6}  {cap0:>7.0f}  {eol:>6}  {rul_max:>8}  {n:>9}  EOL {tag}')
        else:
            print(f'{c:<6}  {cap0:>7.0f}  {"--":>6}  {"--":>8}  {n:>9}  DNF')

    print(f'\nEOL cells ({len(eol_cells)}): {eol_cells}')
    print(f'DNF cells ({len(dnf_cells)}): {dnf_cells}')
    print(f'\nFiles saved to: {OUT_DIR}')
    print('\nNext: run run_loocv_combined.py to merge CA + A1-A8 for LOOCV')
