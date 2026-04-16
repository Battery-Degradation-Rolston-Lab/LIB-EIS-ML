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

---

## 2026-04-16 (session 2)

**sklearn GPR with L-BFGS-B always converges to dead-kernel local min (l~3)**:
For the isotropic RBF on the 1358-sample 120-D z-scored dataset, L-BFGS-B with any
number of restarts or any starting point converges to l≈3 (dead kernel, R²≈0.30).
The fix is a FIXED length-scale found by grid-search: l=1500 for multi-T (1358-sample),
l=1000 for single-T 25°C (760-sample). MATLAB's `minimize` with 10000 steps stops at
an intermediate l≈1500 that generalises well but is not the MLL optimum. Do NOT use
`make_rbf_kernel()` with L-BFGS-B optimization for these models.

**Linear RUL model: normalize_y=False + fixed DotProduct + alpha=0.1**:
With normalize_y=True and the DotProduct kernel, the centred y + tiny alpha makes the
kernel matrix ill-conditioned and the model collapses to the mean. Use normalize_y=False
with alpha=0.1 (multi-T) or alpha=0.4 (single-T 25°C). No WhiteKernel wrapper needed.

**EOL fix (cap[30]) breaks paper reproduction for RUL test files**:
run_gpytorch.py reproduces the paper which used cap[0] for Cambridge cell EOL.
After the cap[30] fix, rul45C02.txt grew from 195→267 rows and 25C05 RUL files
grew from 77→184 rows. This caused RUL scale mismatch with GitHub training data
(trained on cap[0] RUL labels). For paper reproduction, restore the pre-fix RUL
files (7776feb commit). The cap[30] correction is correct for standalone analysis
but must not be used as run_gpytorch.py test data.

**Fig 1a 25°C capacity: use GitHub EIS_data.txt[:760] for training**:
Zenodo 25C04 has only 35 EIS cycles (vs 190 in GitHub) due to data truncation.
Using Zenodo EIS_data_25C_train.txt (679 rows) gives R²≈-0.06; switching to
GitHub EIS_data.txt[:760] (4×190 = 760 rows) gives R²≈0.88 for 25C05.

**25C08 RUL is a known anomaly**: 25C08 has anomalous EIS that barely changes
over its short cycle life (different degradation mechanism from training cells).
Negative R² for 25C08 is expected and documented in the original code.
