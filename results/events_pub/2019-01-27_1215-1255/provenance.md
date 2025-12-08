# Data provenance and reproducibility (2019-01-27 12:15–12:55)

This document records the **local inputs**, **processing chain**, and **validation status** for the 2019‑01‑27 magnetopause event analysis.
All inputs are taken from **local CDF files only** (strict local caching; no network downloads).

## 1. Local inputs

### 1.1 FGM magnetic field (GSM, survey L2)

Loaded via pySPEDAS `load_mms(notplot=True)` from the local `pydata/` tree:

- MMS1: `pydata/mms1/fgm/srvy/l2/2019/01/mms1_fgm_srvy_l2_20190127_v5.177.0.cdf`
- MMS2: `pydata/mms2/fgm/srvy/l2/2019/01/mms2_fgm_srvy_l2_20190127_v5.177.0.cdf`
- MMS3: `pydata/mms3/fgm/srvy/l2/2019/01/mms3_fgm_srvy_l2_20190127_v5.177.0.cdf`
- MMS4: `pydata/mms4/fgm/srvy/l2/2019/01/mms4_fgm_srvy_l2_20190127_v5.177.0.cdf`

### 1.2 FPI ion moments (DIS, fast L2, dis-moms)

Used for ion bulk velocity and density, cold‑ion window inference, and ViN/DN:

- MMS1: `pydata/mms1/fpi/fast/l2/dis-moms/2019/01/mms1_fpi_fast_l2_dis-moms_20190127120000_v3.4.0.cdf`
- MMS2: `pydata/mms2/fpi/fast/l2/dis-moms/2019/01/mms2_fpi_fast_l2_dis-moms_20190127120000_v3.4.0.cdf`
- MMS3: `pydata/mms3/fpi/fast/l2/dis-moms/2019/01/mms3_fpi_fast_l2_dis-moms_20190127120000_v3.4.0.cdf`
- MMS4: `pydata/mms4/fpi/fast/l2/dis-moms/2019/01/mms4_fpi_fast_l2_dis-moms_20190127120000_v3.4.0.cdf`

(Additional dis‑moms files for earlier/later times are present locally but not needed for the 12:15–12:55 window.)

### 1.3 FPI electron moments (DES, fast L2, des-moms)

Used only for context and not required for the core DN/BN analysis:

- MMS1: `pydata/mms1/fpi/fast/l2/des-moms/2019/01/mms1_fpi_fast_l2_des-moms_20190127120000_v3.4.0.cdf`
- MMS2: `pydata/mms2/fpi/fast/l2/des-moms/2019/01/mms2_fpi_fast_l2_des-moms_20190127120000_v3.4.0.cdf`
- MMS3: `pydata/mms3/fpi/fast/l2/des-moms/2019/01/mms3_fpi_fast_l2_des-moms_20190127120000_v3.4.0.cdf`
- MMS4: **no local DES moments file for this interval** (no `pydata/mms4/fpi/fast/l2/des-moms/2019/01` directory). The analysis and tests are written to be robust to this missing product.

### 1.4 MEC ephemeris (GSM, survey L2, epht89q)

Used for spacecraft positions/velocities and X‑line geometry:

- MMS1: `pydata/mms1/mec/srvy/l2/epht89q/2019/01/mms1_mec_srvy_l2_epht89q_20190127_v2.2.2.cdf`
- MMS2: `pydata/mms2/mec/srvy/l2/epht89q/2019/01/mms2_mec_srvy_l2_epht89q_20190127_v2.2.2.cdf`
- MMS3: `pydata/mms3/mec/srvy/l2/epht89q/2019/01/mms3_mec_srvy_l2_epht89q_20190127_v2.2.2.cdf`
- MMS4: `pydata/mms4/mec/srvy/l2/epht89q/2019/01/mms4_mec_srvy_l2_epht89q_20190127_v2.2.2.cdf`

The event script explicitly logs successful MEC loads for MMS1–4 (14 variables per spacecraft).

### 1.5 FPI burst distributions for spectrograms (DIS/DES, brst L2, dis-dist/des-dist)

The 2‑D energy–time spectrograms use **4‑D FPI distributions** from the local `mms_data/` tree, not from the pySPEDAS cache. For each MMS spacecraft and species, the 12:15–12:55 window is covered by six burst files, all overlapping the event window:

- **MMS1 DIS (ions, `dis-dist`)**
  - `mms_data/mms1/fpi/brst/l2/dis-dist/2019/01/27/mms1_fpi_brst_l2_dis-dist_20190127121223_v3.4.0.cdf` (12:12:23–12:15:02 UTC)
  - `mms1_fpi_brst_l2_dis-dist_20190127121503_v3.4.0.cdf` (12:15:03–12:17:42)
  - `mms1_fpi_brst_l2_dis-dist_20190127121743_v3.4.0.cdf` (12:17:43–12:20:12)
  - `mms1_fpi_brst_l2_dis-dist_20190127122013_v3.4.0.cdf` (12:20:13–12:23:32)
  - `mms1_fpi_brst_l2_dis-dist_20190127122923_v3.4.0.cdf` (12:29:23–12:32:22)
  - `mms1_fpi_brst_l2_dis-dist_20190127124143_v3.4.0.cdf` (12:41:43–12:45:22)

- **MMS1 DES (electrons, `des-dist`)** — same time coverage segments as DIS
  - `mms_data/mms1/fpi/brst/l2/des-dist/2019/01/27/mms1_fpi_brst_l2_des-dist_20190127121223_v3.4.0.cdf`
  - `mms1_fpi_brst_l2_des-dist_20190127121503_v3.4.0.cdf`
  - `mms1_fpi_brst_l2_des-dist_20190127121743_v3.4.0.cdf`
  - `mms1_fpi_brst_l2_des-dist_20190127122013_v3.4.0.cdf`
  - `mms1_fpi_brst_l2_des-dist_20190127122923_v3.4.0.cdf`
  - `mms1_fpi_brst_l2_des-dist_20190127124143_v3.4.0.cdf`

- **MMS2 DIS/DES** — analogous set of six files, all overlapping 12:15–12:55
  - `mms_data/mms2/fpi/brst/l2/dis-dist/2019/01/27/mms2_fpi_brst_l2_dis-dist_20190127121223_v3.4.0.cdf` … `mms2_fpi_brst_l2_dis-dist_20190127124143_v3.4.0.cdf`
  - `mms_data/mms2/fpi/brst/l2/des-dist/2019/01/27/mms2_fpi_brst_l2_des-dist_20190127121223_v3.4.0.cdf` … `mms2_fpi_brst_l2_des-dist_20190127124143_v3.4.0.cdf`

- **MMS3 DIS/DES** — analogous set of six files
  - `mms_data/mms3/fpi/brst/l2/dis-dist/2019/01/27/mms3_fpi_brst_l2_dis-dist_20190127121223_v3.4.0.cdf` … `mms3_fpi_brst_l2_dis-dist_20190127124143_v3.4.0.cdf`
  - `mms_data/mms3/fpi/brst/l2/des-dist/2019/01/27/mms3_fpi_brst_l2_des-dist_20190127121223_v3.4.0.cdf` … `mms3_fpi_brst_l2_des-dist_20190127124143_v3.4.0.cdf`

- **MMS4 DIS/DES** — analogous set of six files
  - `mms_data/mms4/fpi/brst/l2/dis-dist/2019/01/27/mms4_fpi_brst_l2_dis-dist_20190127121223_v3.4.0.cdf` … `mms4_fpi_brst_l2_dis-dist_20190127124143_v3.4.0.cdf`
  - `mms_data/mms4/fpi/brst/l2/des-dist/2019/01/27/mms4_fpi_brst_l2_des-dist_20190127121223_v3.4.0.cdf` … `mms4_fpi_brst_l2_des-dist_20190127124143_v3.4.0.cdf`

Every file listed above has `overlaps_event_window = True` for the 12:15–12:55 UTC interval.

### 1.6 Authoritative IDL .sav inputs

LMN triads and ViN time series used as the **authoritative coordinate system** for this event:

- `mp_lmn_systems_20190127_1215-1255_mp-ver2b.sav` (labelled `all_1243`)
- `mp_lmn_systems_20190127_1215-1255_mp-ver3b.sav` (labelled `mixed_1230_1243`)


## 2. Processing chain (CDF → mms_mp → outputs)

The event analysis (`examples/analyze_20190127_dn_shear.py`) and diagnostics (`examples/diagnostic_sav_vs_mmsmp_20190127.py`) share the same core pipeline:

1. **Load CDFs** with pySPEDAS (`notplot=True`, strict local cache) or cdflib (for burst distributions).
2. **Resample to 1 s UTC grid** using `mms_mp.data_loader.to_dataframe` and `mms_mp.data_loader.resample`.
3. **LMN coordinate system:**
   - Event analysis uses LMN triads from the `.sav` files (ver2b/ver3b) as authoritative.
   - Diagnostics also compute an independent `mms_mp.coords.hybrid_lmn` from raw FGM (and optionally MEC) to quantify coordinate‑system differences.
4. **BN:** rotate B_gsm to LMN: `BN = B_gsm · N` (either .sav LMN or hybrid LMN).
5. **VN:** rotate ion bulk velocity `V_i_gse` using the **.sav LMN**, to compare directly with `.sav` ViN.
6. **DN:** integrate VN along N over cold‑ion windows inferred from DIS moments (Option 2 masks) using `mms_mp.motion.integrate_disp`.
7. **Spectrograms:**
   - Read 4‑D FPI distributions (`*_dis_dist_brst`, `*_des_dist_brst`) and 2‑D energy arrays (`*_dis_energy_brst`, `*_des_energy_brst`) from `mms_data` with cdflib.
   - Collapse angular dimensions to omni using `mms_mp.spectra._collapse_fpi`.
   - Plot with `mms_mp.spectra.generic_spectrogram(log10=True)` to produce `mms{1–4}_{DIS,DES}_omni.png`.

All event outputs (CSVs and PNGs) live under `results/events_pub/2019-01-27_1215-1255/`.


## 3. Quantitative comparison (.sav vs NASA CDF)

Diagnostics are summarised in `results/events_pub/2019-01-27_1215-1255/diagnostics/diagnostic_comparison.md` and the CSVs there.

- **BN (coordinate‑system test, not data mismatch):**
  - Same B_gsm (from FGM CDFs) is rotated by **two LMN triads**: the curated `.sav` LMN and the independent `hybrid_lmn`.
  - N‑vector angle differences are ≈89–90° for MMS1–4, so BN curves differ by hundreds of nT (MAE ≈ 350–400 nT, RMSE ≈ 1200–1400 nT).
  - This reflects *coordinate‑system choice*, not any discrepancy between NASA CDFs and `.sav` magnetic field data.

- **VN (FPI ion velocity, .sav vs CDF):**
  - `.sav` ViN is compared on a common 1 s UTC grid to `V_i_gse` from dis‑moms, rotated into the **same .sav LMN**.
  - Typical differences are at the level of **tens of km/s** (MAE ≈ 22–27 km/s, RMSE ≈ 30–37 km/s across MMS1–4); exceedance intervals where |ΔVN| > 50 km/s for ≥10 s are listed explicitly in `diagnostic_comparison.md`.
  - This comparison directly tests the CDF → mms_mp processing pipeline against the `.sav` ViN time series.

- **DN (integrated displacement):**
  - DN from mms_mp (Option 2 cold‑ion windows) is compared with published all_1243 DN; large differences (up to several 1000 km) are documented and interpreted as **methodological** (windowing / integration policy) rather than raw‑data discrepancies.

For this event there is **no spectrogram data in the .sav files**, so the FPI spectrograms cannot be cross‑checked against `.sav`; they instead serve as an internal consistency check that the FPI distributions being used are physically reasonable.


## 4. Spectrogram data integrity

- All eight spectrogram PNGs (`mms{1–4}_DIS_omni.png`, `mms{1–4}_DES_omni.png`) are generated **directly from the burst distribution CDFs listed in §1.5**.
- Programmatic inspection shows:
  - 2‑D energy–time arrays with shape (≈660, 1760) for each PNG.
  - Wide dynamic range in pixel values and substantial variance across both time and energy, not the uniform patterns expected of placeholders.
  - Energy grids match the instrument specifications (DIS ≈ 2.2–17.8 keV, DES ≈ 6.5 eV–27.5 keV, 32 bins).
- The time axis is restricted to **2019‑01‑27 12:15:00–12:55:00 UTC**, using only those CDF samples whose `Epoch` overlaps this window.
- Given the known magnetopause crossing near 12:43 UT (from BN and DN diagnostics), the spectrograms show strong structure and transitions in flux around the same time, consistent with physically meaningful magnetopause boundary signatures.

No NASA data gaps were found in the FGM, DIS moments, or FPI burst distributions needed for this interval. The only notable local gap is the absence of DES‑moments for MMS4, which the analysis does not depend on.


## 5. Reproducibility

To fully reproduce the analysis and diagnostics for this event on this machine:

1. **Run the event analysis (regenerates all CSV/PNGs under events_pub):**

   - `py -3.11 examples/analyze_20190127_dn_shear.py`

2. **Run the .sav vs mms_mp diagnostics (writes into `diagnostics/`):**

   - `py -3.11 examples/diagnostic_sav_vs_mmsmp_20190127.py`

3. **Run the full pytest suite (120/120 tests):**

   - `py -3.11 -m pytest -q`

All three have been executed successfully as part of this validation; at the time of writing, **120/120 tests pass** with only expected warnings (no failures).

