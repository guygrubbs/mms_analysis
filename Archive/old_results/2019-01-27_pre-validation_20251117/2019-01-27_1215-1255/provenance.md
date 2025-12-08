# Data provenance and reproducibility (2019-01-27 12:15–12:55)

This document records the local inputs, processing chain, and reproducibility status for quantities used in the 2019‑01‑27 analysis.

## Local inputs (strict local caching)
All inputs were loaded from the local pySPEDAS cache; no network downloads occurred.

FGM (GSM, survey L2):
- mms1: pydata/mms1/fgm/srvy/l2/2019/01/mms1_fgm_srvy_l2_20190127_v5.177.0.cdf
- mms2: pydata/mms2/fgm/srvy/l2/2019/01/mms2_fgm_srvy_l2_20190127_v5.177.0.cdf
- mms3: pydata/mms3/fgm/srvy/l2/2019/01/mms3_fgm_srvy_l2_20190127_v5.177.0.cdf
- mms4: pydata/mms4/fgm/srvy/l2/2019/01/mms4_fgm_srvy_l2_20190127_v5.177.0.cdf

FPI Ion moments (fast L2, dis-moms):
- mms1: pydata/mms1/fpi/fast/l2/dis-moms/2019/01/mms1_fpi_fast_l2_dis-moms_20190127120000_v3.4.0.cdf
- mms2: pydata/mms2/fpi/fast/l2/dis-moms/2019/01/mms2_fpi_fast_l2_dis-moms_20190127120000_v3.4.0.cdf
- mms3: pydata/mms3/fpi/fast/l2/dis-moms/2019/01/mms3_fpi_fast_l2_dis-moms_20190127120000_v3.4.0.cdf
- mms4: pydata/mms4/fpi/fast/l2/dis-moms/2019/01/mms4_fpi_fast_l2_dis-moms_20190127120000_v3.4.0.cdf

MEC ephemeris (GSM, srvy L2, epht89q):
- mms1: pydata/mms1/mec/srvy/l2/epht89q/2019/01/mms1_mec_srvy_l2_epht89q_20190127_v2.2.2.cdf
- mms2: pydata/mms2/mec/srvy/l2/epht89q/2019/01/mms2_mec_srvy_l2_epht89q_20190127_v2.2.2.cdf
- mms3: pydata/mms3/mec/srvy/l2/epht89q/2019/01/mms3_mec_srvy_l2_epht89q_20190127_v2.2.2.cdf
- mms4: pydata/mms4/mec/srvy/l2/epht89q/2019/01/mms4_mec_srvy_l2_epht89q_20190127_v2.2.2.cdf

Authoritative IDL .sav inputs for LMN and ViN (provided in repo root):
- mp_lmn_systems_20190127_1215-1255_mp-ver2b.sav ("all_1243")
- mp_lmn_systems_20190127_1215-1255_mp-ver3b.sav ("mixed_1230_1243")

## Derivation chain (CDF → mms_mp → outputs)
- CDF → pySPEDAS (notplot=True) → pandas DataFrames at 1 s cadence (UTC)
- LMN: mms_mp.coords.hybrid_lmn(B_gsm[, POS_gsm]) over analysis window(s)
- B_N: rotate B_gsm using LMN: BN = B_gsm @ R[:,2]
- V_N: rotate ion bulk velocity V_i_gse using the chosen LMN
- DN: integrate V_N over cold‑ion windows (event script; diagnostic script uses 1 s running integral for baseline)

The above steps are encoded in:
- examples/analyze_20190127_dn_shear.py (event analysis; Option 2 masks)
- examples/diagnostic_sav_vs_mmsmp_20190127.py (explicit diagnostics)

## Quantitative comparison (.sav vs mms_mp)
Outputs are saved under results/events_pub/2019-01-27_1215-1255/diagnostics/.

- BN differences: see bn_difference_stats.csv (includes RMS |ΔBN| and N‑vector angle differences).
- DN differences: see dn_difference_stats.csv (RMSE vs published DN for all_1243 windows). NaNs indicate no overlapping non‑NaN samples on the shared 1 s grid.
- VN overlays: see vn_overlay_mms{1–4}.png in results/events_pub/2019-01-27_1215-1255/ (from earlier comparison run). Where coverage overlaps, mms_mp V_N rotated by .sav LMN tracks .sav ViN within tens of km/s.

## Provenance statement
- LMN vectors: Independently derivable from raw FGM (+optional MEC) via mms_mp.hybrid_lmn. However, values depend on the selected interval and method; for this event, the publication uses LMN triads curated in the .sav files. Our independent LMN differs substantially in direction (see N‑angle ≈90°), so for publication we treat .sav LMN as authoritative.
- B_N: Fully reproducible from raw FGM once a specific LMN is chosen. Using the .sav LMN, BN curves generated from raw FGM match those implicit in the .sav basis.
- V_N: Reproducible from raw FPI (DIS moments) by rotation with the chosen LMN. Agreement with .sav ViN is good where FPI coverage overlaps; gaps are documented.
- DN: Reproducible using mms_mp motion integration when the same cold‑ion windows are applied. Published DN in summary_metrics.csv was generated under strict local caching using Option 2 masks; diagnostic DN (1 s baseline) intentionally omits masks and will differ.

## Limitations and gaps
- FPI coverage and quality flags may limit exact overlap with .sav ViN; overlays and diagnostics document where comparisons are possible.
- Independent LMN (MVA/hybrid) depends on interval choice; using .sav LMN ensures consistency with the curated event coordinate system.

## How to reproduce
1) Run the event analysis (regenerates all CSV/PNGs under events_pub):
   - py -3.11 examples/analyze_20190127_dn_shear.py
2) Run diagnostics comparing .sav vs mms_mp (saves to diagnostics/):
   - py -3.11 examples/diagnostic_sav_vs_mmsmp_20190127.py

