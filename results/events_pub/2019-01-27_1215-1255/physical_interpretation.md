# Physical interpretation: MMS 2019-01-27 12:15–12:55 UT (two LMN systems)

Inputs and methods
- LMN sets and .sav sources (canonical vs legacy):
  - mixed_1230_1243 (canonical): mp_lmn_systems_20190127_1215-1255_mp-ver3b.sav (MMS1–2 from 12:43; MMS3–4 from ~12:30 MVAB)
  - all_1243 (legacy comparison): mp_lmn_systems_20190127_1215-1255_mp-ver2b.sav (MMS1–4 LMN from the post-12:43 UT crossing)
- Option 2 executed with strict local caching: DIS (ion) moments loaded from local pydata/ cache; no downloads performed.
- Cold-ion windows: inferred from local DIS when possible; in this run the DIS quality flags were not present in the local notplot cache, so .sav-derived VI_LMN quantiles were used as a fallback (logged by the script). Windows require ≥30 s contiguous intervals.
- Boundary detection: |dBN/dt| threshold ≥ 0.4 with ≥30 s separation (user preference). Crossings restricted to 12:15–12:55 UT.
- DN integration: trapezoidal integration of VN within cold-ion windows; LMN rotations from each .sav file.

Key quantitative results (summary_metrics.csv)
- Magnetic shear at 12:18 UT (deg, 1σ):
  - all_1243: MMS1 150.4±9.9, MMS2 150.7±11.8, MMS3 145.1±15.3, MMS4 146.1±15.1
  - mixed_1230_1243: MMS1 150.4±9.9, MMS2 150.7±11.8, MMS3 145.1±15.3, MMS4 145.1±17.1
  - Interpretation: high-shear magnetopause (≈145–151°) consistent with strong antiparallel configuration favorable for reconnection; cross-probe spread 10–17° reflects spatial and temporal variability.
- DN medians and maxima |DN| (km):
  - all_1243: MMS1 −14.4 (610.7), MMS2 −8.4 (331.4), MMS3 +133.0 (594.7), MMS4 ~0.0 (301.5)
  - mixed_1230_1243: MMS1 −14.4 (610.7), MMS2 −8.4 (331.4), MMS3 +152.5 (614.6), MMS4 +3.7 (378.8)
  - Interpretation: MMS3 shows larger positive DN in the mixed set, consistent with using an earlier (~12:30) MVAB normal that slightly rotates N for MMS3/4; MMS4 shifts from ~0 to small +DN, within uncertainties.
- DN-based timing prediction errors (pairwise, seconds):
  - all_1243: typical |median error| ≈ 2.6 s
  - mixed_1230_1243: typical |median error| ≈ 1.3 s
  - Interpretation: mixed_1230_1243 modestly improves pairwise timing consistency, especially for pairs involving MMS3/4 whose LMN comes from ~12:30 MVAB.

Crossing times and positions (crossings_*.csv)
- Automatic picks are now limited to 12:15–12:55 UT as requested. Each row includes the LMN set, probe, crossing time (UTC), and rN position (km) at the crossing.
- The density of picks reflects a permissive ≥0.4 |dBN/dt| threshold with hysteresis and ≥30 s spacing; use these picks for relative timing and clustering rather than as a single “the” crossing.

Physical implications and comparison of LMN choices
- High shear at 12:18 UT for all probes indicates a strongly sheared (near antiparallel) magnetopause favorable for reconnection. The similarity of shear between LMN sets shows that global topology conclusions are robust to the LMN choice here.
- Mixed LMN (MMS3/4 from ~12:30 MVAB) rotates N slightly for MMS3/4, yielding:
  - Larger +DN for MMS3 and small +DN for MMS4 (vs near‑zero in all_1243), consistent with a geometry that better aligns with the motion observed earlier in the interval.
  - Reduced median timing errors across spacecraft pairs (≈1.3 s vs 2.6 s), suggesting improved internal consistency for DN-based timing when MMS3/4 use ~12:30 MVAB normals.
- Recommendation: For production analyses in this repository (inter-spacecraft timing and DN consistency around 12:18–12:45 UT), we adopt mixed_1230_1243 (mp-ver3b) as the canonical LMN system. The earlier all_1243 set (mp-ver2b) is retained as a legacy comparison; key conclusions (high shear, presence of clear crossings) are unchanged between them.

Data gaps, uncertainties, and limitations
- Cold-ion windows: DIS quality flags were not accessible from the local notplot cache for this run; the script fell back to .sav-derived VI_LMN quantiles for window inference. This is documented in the logs. If local DIS quality flags become available, the pipeline will use them automatically.
- Shear at 12:25 and 12:45 UT: not reported because the BN-based side classification did not yield sufficiently clean sheath vs sphere averages within ±60 s for every probe (windowing and sign criteria). This is a data-availability limitation, not a failure; extending windows or relaxing side criteria may recover these metrics at the expense of purity.
- Crossing detection uses a single parameter (|dBN/dt|) threshold; spurious picks are mitigated by ≥30 s spacing but may remain. The predictions_*.csv and BN/DN figures help identify and filter outliers.

Reproducibility
- Run: `py -3.11 examples/analyze_20190127_dn_shear.py`
- Outputs: results/events_pub/2019-01-27_1215-1255/
- Strict local caching: the script only reads existing local CDFs (pydata/) and .sav files in the repo root.

