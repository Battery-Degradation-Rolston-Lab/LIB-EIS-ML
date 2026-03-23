# Data Reference — Battery Degradation GPR

## GitHub Preprocessed Data (paper reproduction)

Located at `battery_gpytorch_rtx4060/battery_gpytorch/data/`:

| File | Shape | Description |
|------|-------|-------------|
| `EIS_data.txt` | 1358×120 | Multi-T training EIS (25C01–04, 35C01, 45C01), state V |
| `Capacity_data.txt` | 1358 | Corresponding capacities |
| `EIS_data_35.txt` | 299×120 | 35°C cell EIS for ARD training |
| `Capacity_data_35.txt` | 299 | 35°C capacities |
| `EIS_data_35C02.txt` | 299×120 | Test cell 35C02 EIS |
| `capacity35C02.txt` | 299 | Test cell 35C02 capacities |
| `EIS_data_RUL.txt` | 525×120 | RUL training EIS |
| `RUL.txt` | 525 | RUL labels |
| `rul35C02.txt` | 127 | Test cell 35C02 RUL (252→0) |

EIS_data.txt row layout: 25C01 rows 0-199, 25C02 rows 200-449, 25C03 rows 450-678, 25C04 rows 679-759, 35C01+45C01 rows 760-1357.

---

## New A1-A8 Dataset

Raw zip: `Battery data.zip`
Preprocessed: `battery_gpytorch_rtx4060/battery_gpytorch/data/new_dataset/`

| File pattern | Description |
|-------------|-------------|
| `EIS_{cell}.txt` | 66-feature EIS per cycle (N_cycles, 66) |
| `cap_{cell}.txt` | Discharge capacity per cycle (mAh) |
| `cyc_{cell}.txt` | Battery cycle numbers |
| `rul_{cell}.txt` | RUL labels (EOL cells only) |
| `EIS_rul_{cell}.txt` | EIS rows up to EOL (for RUL training) |
| `EIS_train.txt` | A1-A4 combined EIS |
| `cap_train.txt` | A1-A4 combined capacity |
| `EIS_rul_train.txt` | A1-A4 RUL EIS (A1, A2, A4 only — A3 DNF) |
| `rul_train.txt` | A1-A4 RUL labels |

### Cell Properties

| Cell | Cycles | EoL index | RUL_max | Notes |
|------|--------|-----------|---------|-------|
| A1 | 268 | 174 | 348 | EOL |
| A2 | 268 | 168 | 336 | EOL |
| A3 | 267 | — | — | DNF — anomalous low initial cap |
| A4 | 270 | 140 | 280 | EOL |
| A5 | 299 | 116 | 232 | EOL |
| A6 | 267 | — | — | DNF — never reached 80% threshold |
| A7 | 301 | 95 | 190 | EOL — shortest life |
| A8 | 267 | 224 | 448 | EOL — longest life |

---

## EIS Feature Format (66-feature native grid)

33 native Ns=6 frequencies (high → low):
```
10000, 7500, 5620, 4220, 3160, 2370, 1780, 1330, 1000, 750, 564, 422, 316,
237, 178, 135, 102, 75, 56.2, 42.2, 31.6, 23.7, 17.8, 13.3, 10.0, 7.5,
5.62, 4.22, 3.16, 2.37, 1.78, 1.33, 0.999  (Hz)
```

Feature layout: [Re(Z) @ freqs 1-33 high→low | Im(Z) @ freqs 34-66 high→low]
- Features 1-33: Re(Z), feature #1 = 10 kHz
- Features 34-66: Im(Z), feature #34 = 10 kHz, feature #56 = 17.8 Hz, feature #66 = 0.999 Hz

Measurement range: 0.999 Hz → 10000 Hz. Diffusion region (<1 Hz) NOT captured.

---

## Zenodo Raw Data Format

EIS files: `EIS data/EIS data/EIS_state_V_{cell}.txt`
Capacity files: `Capacity data/Capacity data/Data_Capacity_{cell}.txt`

EIS file structure (tab-separated):
```
time/s  cycle_number  freq/Hz  Re(Z)/Ohm  -Im(Z)/Ohm  |Z|/Ohm  Phase(Z)/deg
```
- 60 frequencies per cycle (0.02 Hz to 20004 Hz)
- Paper uses State V = 15 min rest after full charge (most electrochemically stable)

Capacity file: last column = capacity (mAh). Some cells 4-column, some 6-column.
Always use last column (-1 index). ox/red=0 → discharge; take max per cycle.

---

## Alignment Convention (CONFIRMED)

- Capacity[i] = discharge max at battery cycle i (starts at cycle 0)
- EIS[i] = state V EIS at cycle i+1 (starts at cycle 1, no EIS at cycle 0)
- They are PAIRED: EIS[i] predicts Capacity[i] (offset by 1 in cycle number)

---

## RUL Convention

- EIS measured every 2 battery cycles
- EOL = first capacity index where cap < 80% of cap[0]
- RUL[i] = 2 × (EOL_index − i) for i = 0..EOL_index
- Training uses only EIS cycles 0..EOL_index (inclusive)
- DNF cells: `force_dnf=True` in `compute_rul()` overrides automatic detection
