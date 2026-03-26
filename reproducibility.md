# Reproducibility & Dataset Compatibility Framework

Two parts:
1. **Universal checklist** — run through this for any new dataset before modelling
2. **Existing dataset records** — completed records for our current datasets

---

# Part 1 — Universal New-Dataset Checklist

Work through each section in order. Each section has a **verdict** that gates the next.

---

## Step 1 — Raw Data Format

Answer these before writing any preprocessing code.

| Check | Question | Pass condition |
|-------|----------|----------------|
| 1.1 | Are EIS measurements available as Re(Z) and Im(Z) per frequency? | Both Re and Im present |
| 1.2 | Is the frequency grid consistent across all cells and all cycles? | Same N frequencies per measurement |
| 1.3 | What is the frequency range? | Covers at least part of 1 Hz–10 kHz |
| 1.4 | Are any frequencies below 1 Hz? | If yes: flag — extrapolation needed or exclude |
| 1.5 | Is discharge capacity available per cycle? | One value per cycle |
| 1.6 | Are EIS and capacity measurements from the same cycle? | Must be alignable (may be offset by 1) |
| 1.7 | At what state is EIS measured? | Should be consistent (e.g. after full charge) |
| 1.8 | How many frequencies per measurement? | Record: N_freq = ___ |

**Verdict:** If 1.1, 1.2, 1.5, 1.6 all pass → continue. Otherwise stop and fix data format.

---

## Step 2 — Dataset Characteristics

| Check | Question | Record |
|-------|----------|--------|
| 2.1 | How many cells total? | N_cells = ___ (need ≥ 3 for LOOCV, ≥ 6 recommended) |
| 2.2 | How many cycles per cell (min / max)? | min = ___ , max = ___ |
| 2.3 | EIS cadence: every cycle or every N cycles? | cadence = ___ (sets RUL factor) |
| 2.4 | How many cells reach 80% EOL? | N_eol = ___ |
| 2.5 | How many cells are DNF (never reach 80%)? | N_dnf = ___ |
| 2.6 | Any cells with anomalously low initial capacity? | Flag them — likely false EOL triggers |
| 2.7 | What is the RUL_max spread: max(RUL_max) / min(RUL_max)? | spread = ___ |
| 2.8 | Single temperature or multi-temperature DOE? | Single-T / Multi-T |
| 2.9 | If multi-T: do higher-T cells have shorter lifetimes? | Yes / No / Unknown |

---

## Step 3 — Capacity Prediction Compatibility

**Capacity GPR almost always works if EIS quality is reasonable.**

| Check | Condition | Action |
|-------|-----------|--------|
| 3.1 | N_cells ≥ 3 | LOOCV possible |
| 3.2 | Cell-to-cell impedance offset visible? | Use **joint normalisation** per fold |
| 3.3 | All cells same temperature/conditions? | Joint norm removes systematic offset |
| 3.4 | Cells from different conditions (T, C-rate)? | Training-only norm may be better; test both |
| 3.5 | Fixed RBF length-scale calibrated? | Grid-search l on validation fold; typical range 10–100 |

**Expected outcome:** R² > 0.90 if EIS has reasonable SNR and captures degradation.

If R² < 0.7: check alignment (Step 1.6), check normalisation (Step 3.2), check EIS state (Step 1.7).

---

## Step 4 — Direct RUL Prediction Compatibility

**This is where most datasets fail. Evaluate before attempting.**

| Check | Question | Verdict |
|-------|----------|---------|
| 4.1 | RUL_max spread (from 2.7) > 2×? | If yes → direct RUL will likely **fail** |
| 4.2 | Are all cells same temperature? | If yes → direct RUL will likely **fail** |
| 4.3 | Multi-T DOE with Arrhenius relationship? | If yes → direct RUL **may work** |
| 4.4 | Do higher-T cells show distinct EIS signatures? | If yes → direct RUL **may work** |
| 4.5 | Training RUL range covers test RUL range? | If no → direct RUL will **fail** (extrapolation) |

**Decision tree:**

```
Same-T dataset?
├── Yes → Skip direct RUL. Go to Step 5 (capacity-derived RUL).
└── No (multi-T)
    ├── Higher-T cells shorter-lived AND distinct EIS? → Try direct RUL (Step 4 passes)
    └── No clear T-lifetime-EIS link → Skip direct RUL. Go to Step 5.
```

**Why direct RUL fails for same-T data:** EIS encodes current State of Health, not total lifespan.
Two cells at 90% SOH have identical EIS regardless of whether they will last 200 or 400 more cycles.
The model has no signal to distinguish them.

---

## Step 5 — Capacity-Derived RUL Compatibility

Use this when direct RUL fails (Step 4 verdict = skip).

| Check | Condition | Required |
|-------|-----------|----------|
| 5.1 | Capacity LOOCV R² > 0.90? | GPR must predict capacity accurately |
| 5.2 | Capacity trajectory is monotonically decreasing? | Extrapolation to 80% is valid |
| 5.3 | Cells reach EOL within the recorded dataset? | Need ground-truth EOL to evaluate |
| 5.4 | Sufficient cycles after ~50% of life for extrapolation? | Short recordings will give poor EOL estimates |

**Approach:** Train capacity GPR → predict full trajectory → fit linear trend → extrapolate to 80% → EOL → RUL.

---

## Step 6 — ARD / Frequency Importance Compatibility

| Check | Condition | Note |
|-------|-----------|------|
| 6.1 | N_train ≥ 100 per fold? | ARD with too few points overfits |
| 6.2 | EIS has ≥ 10 frequencies? | Fewer → ARD not meaningful |
| 6.3 | Consistent EIS state across cycles? | Mixed states confound ARD |
| 6.4 | Are Re and Im physically meaningful pairs? | Consider coupled ARD (1 l per frequency) |

---

## Step 7 — Summary Scorecard

Fill this in after completing Steps 1–6:

| Experiment | Compatible? | Expected R² | Notes |
|-----------|-------------|-------------|-------|
| Capacity prediction (train/test) | ✅ / ⚠️ / ❌ | | |
| Capacity LOOCV | ✅ / ⚠️ / ❌ | | |
| Direct RUL prediction | ✅ / ⚠️ / ❌ | | |
| Capacity-derived RUL | ✅ / ⚠️ / ❌ | | |
| ARD frequency importance | ✅ / ⚠️ / ❌ | | |

---

---

# Part 2 — Existing Dataset Records

Completed scorecards for our three datasets.

---

## Record: Cambridge / Zenodo (Zhang et al. 2020)

**Cells:** 25C01–08, 35C01–02, 45C01–02 | **Features:** 120 (60 freqs × Re + Im) | **EIS cadence:** every 2 cycles

### Step 1 — Format
| Check | Result |
|-------|--------|
| Re(Z) + Im(Z) available | ✅ |
| Consistent frequency grid | ✅ 60 frequencies per measurement |
| Frequency range | 1 Hz – 10 kHz (state V after charge) |
| Sub-1 Hz | ❌ None (Zenodo) — some GitHub files extrapolated; excluded |
| Discharge capacity per cycle | ✅ |
| EIS–capacity alignment | ✅ EIS[i] ↔ Capacity[i], cycle offset confirmed |
| EIS state | State V (after full charge + rest) |

### Step 2 — Characteristics
| Check | Value |
|-------|-------|
| N_cells | 12 (6 train, 6 test) |
| Cycles per cell | 150–330 |
| EIS cadence | Every 2 cycles |
| N_eol | 10 |
| N_dnf | 2 (cells never cycled to failure in dataset) |
| Anomalous initial cap | 25C08 — 33.9 mAh vs fleet ~37 mAh |
| RUL_max spread | 25C07: 30, 25C05: 152 → **5× spread within 25°C cells** |
| Temperature DOE | **Multi-T: 25 / 35 / 45°C** |
| Higher-T = shorter life | ✅ Yes |

### Step 7 — Scorecard
| Experiment | Compatible? | Our R² | Notes |
|-----------|-------------|--------|-------|
| Capacity (multi-T train → 35C02 test) | ✅ | 0.91 | Beat paper target 0.81 |
| Capacity (multi-T train → 45C02 test) | ✅ | 0.94 | Beat paper target 0.72 |
| Direct RUL (multi-T, 35C02) | ✅ | 0.85 | Multi-T DOE designed for this |
| Direct RUL (multi-T, 45C02) | ✅ | 0.91 | |
| Direct RUL per cell (25C, single-T) | ⚠️ | 0.84/0.95/0.76/**−0.33** | 25C08 anomalous (37 cycles total in Zenodo); paper data not released |
| ARD (35°C, Fig 3c) | ✅ | top = #91 | Exact match |
| Capacity-derived RUL | N/A | — | Not needed; direct works for multi-T |

**Key insight:** Success depends on the multi-temperature DOE. Temperature simultaneously shifts EIS (Arrhenius) and accelerates degradation — this is the learnable signal for RUL.

---

## Record: A1-A8 (Partial Lifecycle, ~268 cycles)

**Cells:** A1–A8 | **Features:** 66 (33 freqs × Re + Im, native Ns=6) | **EIS cadence:** every 2 cycles | **Source:** `Battery data.zip`

### Step 1 — Format
| Check | Result |
|-------|--------|
| Re(Z) + Im(Z) available | ✅ |
| Consistent frequency grid | ✅ 33 native frequencies (Ns=6) |
| Frequency range | 1 Hz – 10 kHz |
| Sub-1 Hz | ❌ None — Ns=6 only goes to 0.999 Hz |
| Discharge capacity per cycle | ✅ |
| EIS–capacity alignment | ✅ confirmed |
| EIS state | Ns=6 after full charge (State V analogue) |

### Step 2 — Characteristics
| Check | Value |
|-------|-------|
| N_cells | 8 |
| Cycles per cell | ~267–301 (partial export) |
| EIS cadence | Every 2 cycles (RUL factor = 2) |
| N_eol | 6 (A1, A2, A4, A5, A7, A8) |
| N_dnf | 2 (A3, A6) |
| Anomalous initial cap | A3 — 3800 mAh vs fleet ~4050 mAh → false EOL at cycle 216 |
| RUL_max spread | A7: 190, A8: 448 → **2.4× spread** |
| Temperature DOE | **Single-T (25°C, room temperature)** |
| Higher-T = shorter life | N/A — single temperature |

### Step 7 — Scorecard
| Experiment | Compatible? | R² | Notes |
|-----------|-------------|-----|-------|
| Capacity LOOCV | ✅ | **0.964** (mean, 8 cells) | Joint norm essential |
| Direct RUL LOOCV | ❌ | −0.33 linear, −1.24 RBF | Single-T, 2.4× lifetime spread — Step 4 predicts failure |
| Frequency subset LOOCV | ✅ | 0.87–0.96 by band | Capacity encoded redundantly across spectrum |
| Coupled ARD LOOCV | ✅ | comparable to decoupled | Physically interpretable importance scores |
| Capacity-derived RUL | ✅ | see `fig_cap_rul_scatter.png` | Uses capacity GPR → extrapolate to 80% threshold |

**Key insight:** Step 4.2 immediately flags this dataset (single-T → skip direct RUL). Capacity LOOCV succeeds because joint normalisation removes inter-cell impedance offset.

---

## Record: CA1-CA8 (Complete Lifecycle, 470+ cycles)

**Cells:** CA1–CA8 | **Features:** 66 (33 freqs × Re + Im) | **EIS cadence:** every cycle | **Source:** `Battery data/` (.mpt files)

### Step 1 — Format
| Check | Result |
|-------|--------|
| Re(Z) + Im(Z) available | ✅ |
| Consistent frequency grid | ✅ 33 native frequencies |
| Frequency range | 1 Hz – 10 kHz |
| Sub-1 Hz | ❌ None |
| Discharge capacity per cycle | ✅ |
| EIS–capacity alignment | ✅ |
| EIS state | Ns=6 after full charge |

### Step 2 — Characteristics
| Check | Value |
|-------|-------|
| N_cells | 8 (same physical cells as A1-A8, full runs) |
| Cycles per cell | 414–529 |
| EIS cadence | Every cycle (RUL factor = 1) |
| N_eol | 7 (CA1–CA5, CA7, CA8; CA3 included despite low initial cap) |
| N_dnf | 1 (CA6 — 83.6% final capacity, genuine DNF) |
| Anomalous initial cap | CA3 — 3813 mAh vs fleet ~4066 mAh; genuine EOL confirmed |
| RUL_max spread | CA7: 190, CA8: 448 → **2.4× spread** |
| Temperature DOE | **Single-T (25°C)** |

### Step 7 — Scorecard
| Experiment | Compatible? | R² | Notes |
|-----------|-------------|-----|-------|
| Capacity LOOCV | ✅ | see `fig_cap_loocv.png` | Full lifecycle improves trajectory coverage |
| Direct RUL LOOCV (linear) | ❌ | see `fig_rul_loocv.png` | Same single-T problem as A1-A8 |
| Direct RUL LOOCV (RBF) | ❌ | see `fig_rul_loocv.png` | Worse than linear |
| Capacity-derived RUL | ✅ | see `fig_cap_rul_scatter.png` | Main RUL approach for this dataset |
| ARD LOOCV (fold-averaged) | ✅ | — | `fig_ARD_loocv_folds.png` |

---

## Quick Comparison Across Datasets

| Property | Cambridge | A1-A8 | CA1-CA8 |
|----------|-----------|-------|---------|
| N cells | 12 | 8 | 8 |
| Features | 120 | 66 | 66 |
| Temperature DOE | Multi-T | Single-T | Single-T |
| EIS cadence | every 2 cycles | every 2 cycles | every cycle |
| Capacity LOOCV R² | N/A (train/test split) | **0.964** | see output |
| Direct RUL works? | ✅ (multi-T) | ❌ | ❌ |
| Cap-derived RUL | not needed | ✅ | ✅ |
| Data public? | Partially (GitHub + Zenodo) | 🔒 in-house | 🔒 in-house |
