"""
Preprocessor for multi-temperature dataset: -10°C and -20°C CB cells.

Source : Battery data/21700 Molicell Cycling Data.zip
Chemistry: MOLICELL 21700 P42A NMC
Protocol : Ns=6 = PEIS after charge (10 kHz → ~1 Hz, 33 freqs, every cycle)
           Ns=8 = CC discharge → capacity
C-rate  : 0.375C charge / 0.375C discharge  (both temp groups)

Temperature groups
------------------
  N10_CB1–CB4  →  -10°C  (Low Temperature/-10Celsius …/)
  N20_CB1–CB4  →  -20°C  (Low Temperature/-20Celsius …/)

RT (CA1-CA8) is already in data/ca_dataset/ — not duplicated here.

Output format: identical to ca_dataset
  EIS_{cell}.txt     — (N_cycles, 66)  [Re(Z) @ 33 freqs | -Im(Z) @ 33 freqs]
  cap_{cell}.txt     — (N_cycles,)     discharge capacity in mAh
  cyc_{cell}.txt     — (N_cycles,)     BioLogic cycle number
  rul_{cell}.txt     — (EOL_idx+1,)    RUL in battery cycles  [EOL cells only]
  EIS_rul_{cell}.txt — (EOL_idx+1, 66) EIS rows up to EOL      [EOL cells only]

EOL  = first index where capacity < 80% of cap[0]
RUL[i] = 1 * (EOL_index - i)   (EIS every 1 battery cycle, same as CA series)

DNF cells (confirmed post-hoc from capacity traces):
  None confirmed yet — all CB cells reached EOL in initial inspection.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import zipfile
import io
import numpy as np
from pathlib import Path

# ── Frequency grid ────────────────────────────────────────────────────────────
# Same 33-point grid as CA / A1-A8 datasets (Ns=6, 10 kHz → ~1 Hz)
NATIVE_FREQS = np.array([
    10000.0, 7500.0, 5620.0, 4220.0, 3160.0, 2370.0, 1780.0, 1330.0,
    1000.0,  750.0,  564.0,  422.0,  316.0,  237.0,  178.0,  135.0,
    102.0,   75.0,   56.2,   42.2,   31.6,   23.7,   17.8,   13.3,
    10.0,    7.5,    5.62,   4.22,   3.16,   2.37,   1.78,   1.33, 0.999
])
N_FREQS = len(NATIVE_FREQS)  # 33

RUL_FACTOR = 1  # EIS every 1 battery cycle

# ── File map ──────────────────────────────────────────────────────────────────
# zip_path_prefix → (cell_name, mpt_filename)
CELL_MAP = {
    # -10°C group  (prefix inside zip)
    'N10_CB1': (
        '21700 Molicell Cycling Data/Low Temperature/'
        '-10Celsius 0.375C charge and discharge/'
        '2403220001Li-ion_RT_CB1.mpt'
    ),
    'N10_CB2': (
        '21700 Molicell Cycling Data/Low Temperature/'
        '-10Celsius 0.375C charge and discharge/'
        '2403220002Li-ion_RT_CB2.mpt'
    ),
    'N10_CB3': (
        '21700 Molicell Cycling Data/Low Temperature/'
        '-10Celsius 0.375C charge and discharge/'
        '2403220003Li-ion_RT_CB3.mpt'
    ),
    'N10_CB4': (
        '21700 Molicell Cycling Data/Low Temperature/'
        '-10Celsius 0.375C charge and discharge/'
        '2403220004Li-ion_RT_CB4.mpt'
    ),
    # -20°C group
    'N20_CB1': (
        '21700 Molicell Cycling Data/Low Temperature/'
        '-20Celsius 0.375C charge and discharge/'
        '2403080010Li-ion_RT_CB1.mpt'
    ),
    'N20_CB2': (
        '21700 Molicell Cycling Data/Low Temperature/'
        '-20Celsius 0.375C charge and discharge/'
        '2403080005Li-ion_RT_CB2.mpt'
    ),
    'N20_CB3': (
        '21700 Molicell Cycling Data/Low Temperature/'
        '-20Celsius 0.375C charge and discharge/'
        '2403080006Li-ion_RT_CB3.mpt'
    ),
    'N20_CB4': (
        '21700 Molicell Cycling Data/Low Temperature/'
        '-20Celsius 0.375C charge and discharge/'
        '2403080011Li-ion_RT_CB4.mpt'
    ),
}

# Cells confirmed as DNF (set empty unless confirmed after reviewing output)
DNF_CELLS: set = set()


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_mpt_from_zip(zf: zipfile.ZipFile, zip_path: str) -> tuple:
    """
    Parse one BioLogic .mpt file from an open ZipFile.

    Returns
    -------
    eis_matrix    : ndarray (N_cycles, 66)
    capacities    : ndarray (N_cycles,)
    cycle_numbers : ndarray (N_cycles,)
    """
    with zf.open(zip_path) as fp:
        lines = [line.decode('latin-1', errors='replace') for line in fp]

    # Find column header line (contains 'mode', 'Ns', 'freq/Hz')
    col_idx = None
    for i, line in enumerate(lines):
        if 'mode' in line and 'Ns' in line and 'freq/Hz' in line:
            col_idx = i
            break
    if col_idx is None:
        raise ValueError(f"Column header not found in {zip_path}")

    cols = lines[col_idx].strip().split('\t')
    ns_i   = cols.index('Ns')
    freq_i = cols.index('freq/Hz')
    cyc_i  = cols.index('cycle number')
    rez_i  = cols.index('Re(Z)/Ohm')
    imz_i  = cols.index('-Im(Z)/Ohm')
    cap_i  = cols.index('Capacity/mA.h')

    eis_rows  = {}   # {cycle: {freq: (re, im)}}
    cap_rows  = {}   # {cycle: max_capacity}

    for line in lines[col_idx + 1:]:
        parts = line.strip().split('\t')
        if len(parts) <= max(ns_i, freq_i, cyc_i, rez_i, imz_i, cap_i):
            continue
        try:
            ns  = int(float(parts[ns_i]))
            cy  = int(float(parts[cyc_i]))
        except (ValueError, IndexError):
            continue

        # Ns=6: PEIS after charge
        if ns == 6:
            try:
                frq = float(parts[freq_i])
                re  = float(parts[rez_i])
                im  = float(parts[imz_i])
                if frq > 0:
                    if cy not in eis_rows:
                        eis_rows[cy] = {}
                    eis_rows[cy][frq] = (re, im)
            except (ValueError, IndexError):
                pass

        # Ns=8: CC discharge → capacity
        if ns == 8:
            try:
                cap = float(parts[cap_i])
                if cap > 0:
                    cap_rows[cy] = max(cap_rows.get(cy, 0.0), cap)
            except (ValueError, IndexError):
                pass

    # Align EIS + capacity on common cycle numbers
    common  = sorted(set(eis_rows) & set(cap_rows))
    eis_out = []
    cap_out = []
    skipped = 0

    for cy in common:
        freq_dict  = eis_rows[cy]
        meas_freqs = np.array(sorted(freq_dict.keys(), reverse=True))

        if len(meas_freqs) < N_FREQS:
            skipped += 1
            continue

        re_vals = np.array([freq_dict[f][0] for f in meas_freqs[:N_FREQS]])
        im_vals = np.array([freq_dict[f][1] for f in meas_freqs[:N_FREQS]])
        eis_out.append(np.concatenate([re_vals, im_vals]))
        cap_out.append(cap_rows[cy])

    if skipped:
        print(f'    Skipped {skipped} incomplete EIS sweeps')

    eis_matrix    = np.array(eis_out)
    capacities    = np.array(cap_out)
    cycle_numbers = np.array(common[:len(eis_out)])

    return eis_matrix, capacities, cycle_numbers


def compute_rul(capacities: np.ndarray, force_dnf: bool = False) -> tuple:
    """
    EOL  = first index where capacity < 80% of cap[0]
    RUL[i] = RUL_FACTOR * (EOL_index - i)
    """
    if force_dnf:
        return None, None
    cap0   = capacities[0]
    thresh = 0.8 * cap0
    eol    = next((i for i, c in enumerate(capacities) if c < thresh), None)
    if eol is None:
        return None, None
    rul = np.array([RUL_FACTOR * (eol - i) for i in range(eol + 1)], dtype=float)
    return rul, eol


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ZIP_PATH = Path(__file__).parents[2] / 'Battery data' / '21700 Molicell Cycling Data.zip'
    OUT_DIR  = Path(__file__).parent / 'data' / 'multitemp_dataset'
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        sys.exit(f'ERROR: zip not found at {ZIP_PATH}')

    cells = list(CELL_MAP.keys())

    print('=' * 65)
    print('Preprocessing multi-temperature CB cells')
    print('  -10°C : N10_CB1–CB4   (0.375C chg/dchg)')
    print('  -20°C : N20_CB1–CB4   (0.375C chg/dchg)')
    print(f'DNF cells: {sorted(DNF_CELLS) or "none"}')
    print(f'RUL factor: {RUL_FACTOR}  (EIS every 1 battery cycle)')
    print('=' * 65)

    all_eis = {}
    all_cap = {}
    all_cyc = {}
    all_rul = {}
    all_eol = {}

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for cell in cells:
            zip_path = CELL_MAP[cell]
            print(f'\n--- {cell} ---')
            print(f'  {zip_path.split("/")[-1]}')

            eis, cap, cyc = parse_mpt_from_zip(zf, zip_path)
            rul, eol      = compute_rul(cap, force_dnf=(cell in DNF_CELLS))

            all_eis[cell] = eis
            all_cap[cell] = cap
            all_cyc[cell] = cyc

            print(f'  EIS matrix  : {eis.shape}')
            print(f'  Capacity    : {cap[0]:.1f} → {cap[-1]:.1f} mAh'
                  f'  (80% thresh = {0.8 * cap[0]:.1f})')

            if eol is not None:
                all_rul[cell] = rul
                all_eol[cell] = eol
                print(f'  EOL index   : {eol}  (cycle {cyc[eol]:.0f})'
                      f'  RUL_max = {rul[0]:.0f} bat-cycles')
            else:
                print(f'  EOL         : NOT REACHED  ({len(cap)} cycles recorded)')

            # Save per-cell files
            np.savetxt(OUT_DIR / f'EIS_{cell}.txt',  eis)
            np.savetxt(OUT_DIR / f'cap_{cell}.txt',  cap)
            np.savetxt(OUT_DIR / f'cyc_{cell}.txt',  cyc)
            if rul is not None:
                np.savetxt(OUT_DIR / f'rul_{cell}.txt',      rul)
                np.savetxt(OUT_DIR / f'EIS_rul_{cell}.txt',  eis[:eol + 1])

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('SUMMARY')
    print('=' * 65)

    print(f'\n{"Cell":<10}  {"Temp":>5}  {"cap0":>7}  {"EOL@":>6}'
          f'  {"RUL_max":>8}  {"n_cycles":>9}  Status')
    print('-' * 65)

    for cell in cells:
        temp_str = '-10°C' if cell.startswith('N10') else '-20°C'
        cap0 = all_cap[cell][0]
        n    = len(all_cap[cell])
        if cell in all_rul:
            eol     = all_eol[cell]
            rul_max = int(all_rul[cell][0])
            print(f'{cell:<10}  {temp_str:>5}  {cap0:>7.0f}  {eol:>6}'
                  f'  {rul_max:>8}  {n:>9}  EOL')
        else:
            print(f'{cell:<10}  {temp_str:>5}  {cap0:>7.0f}  {"--":>6}'
                  f'  {"--":>8}  {n:>9}  DNF')

    eol_cells = [c for c in cells if c in all_rul]
    dnf_cells = [c for c in cells if c not in all_rul]
    print(f'\nEOL cells ({len(eol_cells)}): {eol_cells}')
    print(f'DNF cells ({len(dnf_cells)}): {dnf_cells}')
    print(f'\nFiles saved to: {OUT_DIR}')
    print('\nNext: run run_multitemp_rul.py for multi-T GPR')
