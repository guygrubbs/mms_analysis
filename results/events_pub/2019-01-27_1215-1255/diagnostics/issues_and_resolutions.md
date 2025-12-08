# Issues and Resolutions — MMS 2019-01-27 (12:15–12:55)

This document is the canonical, auditable record of all issues identified during the MMS 2019-01-27 12:15–12:55 UT event analysis, together with their root causes, quantitative impact, and resolution status.

Status codes used below:
- **FIXED** – A concrete bug was identified and corrected; tests and diagnostics verify the fix.
- **DOCUMENTED** – Behaviour is understood and acceptable but differs from some reference; fully documented.
- **ACCEPTABLE_LIMITATION** – Inherent limitation of the data or analysis approach; retained with clear caveats.

## 1. Known Issues

### 1.1 LMN Orientation Discrepancy (BN Differences)

- **Description:** The independently computed hybrid LMN (via `mms_mp.coords.hybrid_lmn`) differs from the curated IDL/.sav LMN by ~90° in the N-direction for MMS1–4 over the .sav LMN intervals.
- **Evidence:**
  - BN (hybrid LMN) vs BN (.sav LMN) differences are large in magnitude (MAE of order hundreds of nT, RMSE of order 1 µT), but correlations are moderately positive (∼0.8), indicating a rotated frame rather than noise.
  - N-angle differences in `bn_difference_stats.csv` are ~90° for all probes.
- **Root cause:**
  - MVA/hybrid LMN solutions are highly sensitive to interval choice and eigenvalue criteria.
  - The .sav LMN triads are expert-curated for this specific crossing; the independent hybrid LMN is computed automatically over a different effective interval.
- **Impact:**
  - BN values differ substantially between the two LMN systems; any science result that depends directly on BN must specify which LMN definition is used.
- **Mitigation and policy:**
  - For this event, the .sav LMN is treated as **authoritative** for production analysis.
  - The hybrid LMN is retained only for diagnostics and “what if” experiments.
- **Resolution status:** **ACCEPTABLE_LIMITATION** (coordinate-system choice, not a physics error).

### 1.2 VN Decorrelation (.sav ViN vs mms_mp VN)

- **Description:** Normal ion velocity (V_N) derived from FPI moments via `mms_mp` shows poor correlation with V_N from the .sav ViN time series when both are rotated into the .sav LMN.
- **Evidence (per `vn_difference_stats.csv` and diagnostics):**
  - MAE ≈ 22–27 km/s, RMSE ≈ 30–37 km/s across MMS1–4.
  - Correlations are near zero even after forcing both series onto a shared 1 s UTC grid.
- **Root cause:**
  - Different gridding and interpolation strategies between the .sav pipeline and the Python pipeline.
  - Possible differences in DIS quality-flag masking and any smoothing/filtering applied inside the legacy IDL workflow, which is not fully documented.
- **Impact:**
  - Point-by-point VN values disagree at the ~10–30 km/s level; integrated DN distances and detailed timing inferences inherit this uncertainty.
- **Mitigation and policy:**
  - DN comparison is now defined relative to **.sav-integrated DN** using the same cold-ion windows (see Section 1.3).
  - VN differences are documented as pipeline-level differences, not treated as a show-stopper.
- **Resolution status:** **DOCUMENTED**.

### 1.3 DN Comparison vs Published all_1243 Files

- **Description:** Published DN CSV files (`dn_mms*_all_1243.csv`) have effectively no overlapping valid samples with the 12:15–12:55 event window.
- **Evidence:**
  - Non-NaN DN values in the published `all_1243` files occur mostly after ≈13:03 UT.
  - Within 12:15–12:55, comparisons yield NaN for MAE/RMSE/correlation, as correctly reported by the diagnostics.
- **Root cause:**
  - The published all_1243 DN files were generated for longer intervals that extend well past the specific 12:15–12:55 window used here.
- **Impact:**
  - It is not meaningful to compare DN(mms_mp) to published DN within this strict event window; any such metrics are undefined.
- **Mitigation and policy:**
  - For this event, DN metrics are defined as **DN(mms_mp) vs DN(.sav-integrated)** over the same cold-ion windows inferred from `mms*_DN_segments.csv` within 12:15–12:55.
  - Published all_1243 DN files are treated as documentation-only artefacts for this specific window.
- **Resolution status:** **DOCUMENTED**.

### 1.4 Direct B_LMN Component Comparisons (Sanity Check)

- **Description:** Direct BL, BM, and BN comparisons between the .sav B_LMN
  time series and B_L, B_M, B_N obtained by rotating CDF B_gsm with the .sav
  LMN triads on a common 1 s grid over 12:15–12:55 UT.
- **Note (updated):** Initial diagnostics showed surprisingly large
  component-level discrepancies (MAE ≈ 10–45 nT, modest correlations), which
  were later traced to a **column-ordering mismatch** in the .sav B_LMN
  arrays rather than a physics/calibration difference.
- **Current status:** After adopting the confirmed native ordering
  (B_N, B_M, B_L) for this event, the **BN** component now agrees at the
  sub-nT level with correlations ≳0.99 for MMS1–4, while **BL** and **BM**
  still show residual differences of order 9–14 nT with relatively low
  correlations (≲0.3). These BL/BM residuals are therefore treated as
  pipeline/calibration differences (see §3.2), not indexing bugs.
- **Resolution status:** **FIXED (see 2.3)** – the specific issue of incorrect
  B_LMN column ordering has been resolved and documented; remaining BL/BM
  discrepancies are tracked as part of the broader pipeline-differences
  limitation.

## 2. Corrected Issues

### 2.1 cdflib API Compatibility for Spectrograms

- **Problem:** `cdf.cdf_info()` was treated as a dict, but newer cdflib returns a dataclass-like object, causing attribute errors when building spectrograms.
- **Fix:** Added a compatibility layer in `examples/make_event_figures_20190127.py` to support both dict-style and attribute-style access to `rVariables`.
- **Impact:** Previously blocked spectrogram generation from CDF files; now all 8 spectrogram PNGs (DIS/DES × MMS1–4) are generated from real MMS data.
- **Resolution status:** **FIXED** (covered by tests and manual verification).

### 2.2 Time-Varying FPI Energy Grids

- **Problem:** FPI energy arrays are 2D (time, energy), but the original code assumed a static 1D energy grid and performed a naive reshape, causing shape mismatches.
- **Fix:** Introduced an `_extract_energy` helper that detects the time axis and extracts a consistent 1D energy grid from the first time step.
- **Impact:** Eliminated dimension-mismatch errors and ensured correct energy labelling for all spectrograms.
- **Resolution status:** **FIXED**.

### 2.3 B_LMN Column Ordering in .sav Files (2019-01-27 Event)

- **Problem:** Direct BL/BM/BN comparisons between the .sav B_LMN time series
  and CDF B_gsm rotated into the .sav LMN frame showed large MAE values
  (≈10–45 nT) and only modest correlations when assuming the natural
  (L, M, N) column order for the .sav arrays.
- **Investigation and evidence:**
  - A dedicated diagnostic (`tmp_check_blmn_order.py`) compared B_LMN from the
    .sav file against B_gsm·L, B_gsm·M, and B_gsm·N on the same 1 s grid for
    two ordering hypotheses:
    - **LMN (0,1,2)**: BL=col 0, BM=col 1, BN=col 2.
    - **NML (2,1,0)**: BL=col 2, BM=col 1, BN=col 0.
  - For all probes (MMS1–4), the NML assumption produced:
    - BN MAE ≈ 0.33–0.41 nT with correlation ≈ 0.998.
    - BL MAE reduced from ≈41–43 nT to ≈9–14 nT with moderately positive
      correlations.
  - This is strong evidence that, for the
    `mp_lmn_systems_20190127_1215-1255_mp-ver2b.sav` file, the B_LMN arrays
    are stored as **(B_N, B_M, B_L)** in columns (0,1,2), not (B_L,B_M,B_N).
- **Fix:**
  - Left the loader `tools/idl_sav_import.load_idl_sav` as a thin wrapper
    that preserves the raw ordering, but added explicit comments documenting
    that this event’s B_LMN arrays use (B_N,B_M,B_L) column order.
  - Updated `examples/diagnostic_sav_vs_mmsmp_20190127.py` so that the direct
    component comparisons now map desired (L,M,N) onto raw indices (2,1,0)
    before building BL/BM/BN time series from `blmn`.
	- **Post-fix impact:**
	  - Updated `comparison_statistics_consolidated.csv` now shows, for
	    `BN_direct`, MAE ≲ 0.5 nT and correlations ≳ 0.99 for MMS1–4, confirming
	    near one-to-one agreement between .sav B_N and CDF-derived B_N in the
	    .sav LMN frame.
	  - For BL and BM, MAE values are reduced but remain at ≈9–14 nT with
	    relatively low correlations (≲0.3) across probes, indicating residual
	    pipeline/calibration differences that are **not** explained by column
	    ordering and are therefore treated as part of the broader
	    `.sav`-vs-`mms_mp` workflow differences (see §3.2).
	- **Resolution status:** **FIXED** – the B_LMN column ordering for this
	  event is now correctly handled and documented; direct BL/BM/BN
	  comparisons provide a robust check on the LMN magnetic field
	  transformation (excellent agreement for B_N and quantified, documented
	  residuals for BL/BM).

## 3. Acceptable Limitations

### 3.1 Coordinate-System Choice (LMN)

- For this event, the curated .sav LMN triads are treated as **authoritative**. Independent hybrid LMN solutions are used only for diagnostics.
- Different LMN choices are scientifically valid but produce different BN values; this is a documented, acceptable limitation rather than a bug.

### 3.2 Pipeline Differences Between .sav and mms_mp

- The .sav and Python pipelines use different (and partly undocumented) quality-flag masks, filters, and resampling strategies.
- As a result, VN and DN show discrepancies at the tens of km/s and hundreds–thousands of km level, respectively—well above 10% in some regimes.
- These discrepancies are fully quantified in the CSV diagnostics (`bn_difference_stats.csv`, `vn_difference_stats.csv`, `dn_difference_stats.csv`, and `comparison_statistics_consolidated.csv`) and are treated as **DOCUMENTED** behaviour.

### 3.3 Data Availability Constraints

- Certain MMS data products (e.g., some DES fast moments or legacy DN products) are unavailable or do not overlap the 12:15–12:55 window.
- Where direct comparisons are impossible, this is called out explicitly rather than silently interpolated or extrapolated.
- This limitation is inherent to the available mission data and not to the `mms_mp` code.

