# Lessons Learned — Self-Correction Log

Append to this file after every user correction. Review at the start of complex sessions.

---

## 2026-03-14

**A3 EOL detection**: A3 initial cap is anomalously low (~3800 mAh vs fleet ~4050 mAh).
The automatic 80%-of-initial threshold gives a false positive at cycle 216.
User confirmed A3 did NOT fail. Always use `DNF_CELLS = {'A3', 'A6'}` with `force_dnf=True`
to override automatic detection for these cells.

**No extrapolation below 1 Hz**: The Ns=6 EIS sweep only goes to 0.999 Hz.
Previous 120-feature Zenodo grid included extrapolated sub-1Hz values — not real measurements.
Use native 66-feature grid (33 measured freqs × Re+Im) — no interpolation, no extrapolation.

**RUL absolute count doesn't transfer**: Direct EIS → absolute RUL mapping fails LOOCV
(mean R²=-0.33 linear, -1.24 RBF) because A1-A8 span RUL_max 190–448 cycles (2.4× range).
EIS encodes current health, not remaining life. Use capacity-derived RUL instead.

**Joint vs training-only normalisation**:
- Capacity LOOCV: use JOINT norm (train+test pooled per fold) — removes cell-to-cell
  impedance offset. This is key to getting R²=0.964.
- RUL model: use TRAINING-ONLY norm (paper's stated approach). Joint norm hurts RUL
  because test RUL cells span different cycle ranges.

**ARD weight formula**: Paper eq. is `w = exp(−σ_m)` then normalise. NOT `exp(-10^log(σ))`.
The latter is a MATLAB approximation artifact. Always use the linear-space formula.

---

## 2026-03-20

**Do not infer degradation mechanism from Re(Z) trend alone**: Observing that Re(Z) rises
monotonically with cycling does NOT mean degradation is ohmic/electrolyte-driven. Re(Z) and
Im(Z) at the same frequency are coupled by the Kramers-Kronig relations — a change in Im(Z)
at the SEI frequency necessarily perturbs Re(Z) at all other frequencies. The Re(Z) increase
could be a K-K shadow of SEI growth, not an independent ohmic process. Never attribute a
physical mechanism to Re(Z) alone without considering the coupled (Re, Im) structure.

**ARD on decoupled Re/Im features is physically ambiguous**: The current feature vector
[Re(ω₁)...Re(ωₙ), Im(ω₁)...Im(ωₙ)] treats Re and Im as independent features. ARD
importance scores on this representation are misleading — predictive power attributed to
Re(ω) at one frequency may actually belong to the coupled (Re, Im) structure at another
frequency via K-K. The correct representation pairs (Re, Im) per frequency point and assigns
one ARD length-scale per ω.

---

## 2026-04-16

**EOL reference for Cambridge cells is cap[30], not cap[0]**: The paper states "80% of its
initial value after undergoing 30 pre-cycles at 25°C". The Zenodo capacity data INCLUDES
these 30 pre-cycles (formation cycling), causing a steep initial capacity drop (~5–20% in
first 30 cycles). Using cap[0] as reference gives artificially high EOL thresholds, triggering
false EOL detections as early as cycle 6. Fix: `find_eol` in `preprocess_zenodo.py` now uses
`caps[min(30, len(caps)-1)]` as the reference.
Impact: 25C02 and 25C03 are now correctly DNF (no EOL); 25C05/06/07 EOL indices shifted
+50–100 cycles later. Only applies to Cambridge/Zenodo cells. CA/CB/A cells use cap[0]
(no confirmed pre-cycling in their MPT protocol).
