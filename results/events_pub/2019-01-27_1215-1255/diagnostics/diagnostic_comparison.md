# Diagnostic Comparison (.sav vs mms_mp): 2019-01-27 12:15–12:55

- Strict local caching: all inputs loaded from local CDFs only (no re-downloads).
- DN integration uses the same cold-ion windowing as published all_1243 outputs.

## LMN methods summary (.sav vs hybrid vs algorithmic)

| Method          | N-angle vs .sav (deg, MMS1–4) | BN corr vs .sav (MMS1–4) | BN RMSE vs .sav (nT, MMS1–4) | Notes                                  |
|-----------------|-------------------------------|---------------------------|------------------------------|----------------------------------------|
| .sav (reference)| 0                             | 1.00                      | 0                            | Expert-curated LMN from original study |
| hybrid_lmn      | ≈89–90                        | ≈0.80                     | ≳1200                        | N rotated/flipped; not recommended     |
| algorithmic_lmn | ≈6–8                          | 1.00                      | ≈60–70                       | Small rotation; good quantitative match|

## BN differences vs .sav LMN (hybrid and algorithmic)
### Source: algorithmic
- MMS1: MAE=11.578 nT, RMSE=67.772 nT, corr=1.000, count(|Δ|>0.5 nT)=25795, N-angle diff≈5.6°
- MMS2: MAE=9.968 nT, RMSE=51.709 nT, corr=1.000, count(|Δ|>0.5 nT)=41815, N-angle diff≈6.7°
- MMS3: MAE=11.294 nT, RMSE=57.660 nT, corr=1.000, count(|Δ|>0.5 nT)=51244, N-angle diff≈8.5°
- MMS4: MAE=13.391 nT, RMSE=80.621 nT, corr=1.000, count(|Δ|>0.5 nT)=28312, N-angle diff≈6.3°
### Source: hybrid
- MMS1: MAE=395.219 nT, RMSE=1421.218 nT, corr=0.798, count(|Δ|>0.5 nT)=80644, N-angle diff≈89.7°
- MMS2: MAE=355.806 nT, RMSE=1249.658 nT, corr=0.863, count(|Δ|>0.5 nT)=81598, N-angle diff≈89.6°
- MMS3: MAE=351.808 nT, RMSE=1232.849 nT, corr=0.873, count(|Δ|>0.5 nT)=81939, N-angle diff≈89.7°
- MMS4: MAE=396.521 nT, RMSE=1422.198 nT, corr=0.790, count(|Δ|>0.5 nT)=81017, N-angle diff≈89.2°
  Exceedance intervals where |ΔBN|>0.5 nT for ≥10 s (hybrid LMN):
  - MMS1: 2019-01-27 00:00:04+00:00 → 2019-01-27 00:07:24+00:00 (dur=440 s), max |ΔBN|=3.52 nT
  - MMS1: 2019-01-27 00:07:55+00:00 → 2019-01-27 00:08:05+00:00 (dur=10 s), max |ΔBN|=0.87 nT
  - MMS1: 2019-01-27 00:08:06+00:00 → 2019-01-27 00:08:27+00:00 (dur=21 s), max |ΔBN|=0.97 nT
  - MMS1: 2019-01-27 00:09:05+00:00 → 2019-01-27 00:09:22+00:00 (dur=17 s), max |ΔBN|=0.98 nT
  - MMS1: 2019-01-27 00:09:36+00:00 → 2019-01-27 00:10:33+00:00 (dur=57 s), max |ΔBN|=1.50 nT
  - MMS2: 2019-01-27 00:00:02+00:00 → 2019-01-27 00:07:26+00:00 (dur=444 s), max |ΔBN|=3.46 nT
  - MMS2: 2019-01-27 00:07:27+00:00 → 2019-01-27 00:07:54+00:00 (dur=27 s), max |ΔBN|=0.81 nT
  - MMS2: 2019-01-27 00:07:55+00:00 → 2019-01-27 00:08:29+00:00 (dur=34 s), max |ΔBN|=1.12 nT
  - MMS2: 2019-01-27 00:08:35+00:00 → 2019-01-27 00:08:50+00:00 (dur=15 s), max |ΔBN|=0.73 nT
  - MMS2: 2019-01-27 00:08:51+00:00 → 2019-01-27 00:09:27+00:00 (dur=36 s), max |ΔBN|=1.13 nT
  - MMS3: 2019-01-27 00:00:07+00:00 → 2019-01-27 00:10:33+00:00 (dur=626 s), max |ΔBN|=3.56 nT
  - MMS3: 2019-01-27 00:10:45+00:00 → 2019-01-27 00:10:59+00:00 (dur=14 s), max |ΔBN|=1.32 nT
  - MMS3: 2019-01-27 00:12:32+00:00 → 2019-01-27 00:17:29+00:00 (dur=297 s), max |ΔBN|=1.56 nT
  - MMS3: 2019-01-27 00:17:46+00:00 → 2019-01-27 00:35:55+00:00 (dur=1089 s), max |ΔBN|=3.33 nT
  - MMS3: 2019-01-27 00:36:11+00:00 → 2019-01-27 00:37:34+00:00 (dur=83 s), max |ΔBN|=2.98 nT
  - MMS4: 2019-01-27 00:00:05+00:00 → 2019-01-27 00:07:26+00:00 (dur=441 s), max |ΔBN|=3.70 nT
  - MMS4: 2019-01-27 00:07:27+00:00 → 2019-01-27 00:09:26+00:00 (dur=119 s), max |ΔBN|=1.30 nT
  - MMS4: 2019-01-27 00:09:29+00:00 → 2019-01-27 00:10:34+00:00 (dur=65 s), max |ΔBN|=1.74 nT
  - MMS4: 2019-01-27 00:10:35+00:00 → 2019-01-27 00:11:04+00:00 (dur=29 s), max |ΔBN|=1.58 nT
  - MMS4: 2019-01-27 00:12:45+00:00 → 2019-01-27 00:16:48+00:00 (dur=243 s), max |ΔBN|=1.42 nT

## VN differences (.sav ViN vs mms_mp V_i·N_sav)
- MMS1: MAE=21.7 km/s, RMSE=30.0 km/s, corr=nan
- MMS2: MAE=21.8 km/s, RMSE=30.6 km/s, corr=nan
- MMS3: MAE=26.6 km/s, RMSE=36.5 km/s, corr=0.000
- MMS4: MAE=21.8 km/s, RMSE=30.2 km/s, corr=-0.000
  Exceedance intervals where |ΔVN|>50 km/s for ≥10 s:
  - MMS1: 2019-01-27 12:15:27+00:00 → 2019-01-27 12:15:41+00:00 (dur=14 s), max |ΔVN|=57.5 km/s
  - MMS1: 2019-01-27 12:18:09+00:00 → 2019-01-27 12:18:23+00:00 (dur=14 s), max |ΔVN|=96.2 km/s
  - MMS1: 2019-01-27 12:19:35+00:00 → 2019-01-27 12:20:38+00:00 (dur=63 s), max |ΔVN|=73.0 km/s
  - MMS1: 2019-01-27 12:44:06+00:00 → 2019-01-27 12:44:29+00:00 (dur=23 s), max |ΔVN|=61.9 km/s
  - MMS1: 2019-01-27 12:44:33+00:00 → 2019-01-27 12:45:27+00:00 (dur=54 s), max |ΔVN|=102.5 km/s
  - MMS2: 2019-01-27 12:19:40+00:00 → 2019-01-27 12:20:34+00:00 (dur=54 s), max |ΔVN|=98.0 km/s
  - MMS2: 2019-01-27 12:44:16+00:00 → 2019-01-27 12:46:00+00:00 (dur=104 s), max |ΔVN|=105.7 km/s
  - MMS2: 2019-01-27 12:46:04+00:00 → 2019-01-27 12:46:49+00:00 (dur=45 s), max |ΔVN|=65.6 km/s
  - MMS3: 2019-01-27 12:15:56+00:00 → 2019-01-27 12:16:28+00:00 (dur=32 s), max |ΔVN|=119.9 km/s
  - MMS3: 2019-01-27 12:17:53+00:00 → 2019-01-27 12:18:07+00:00 (dur=14 s), max |ΔVN|=146.5 km/s
  - MMS3: 2019-01-27 12:18:52+00:00 → 2019-01-27 12:19:10+00:00 (dur=18 s), max |ΔVN|=67.6 km/s
  - MMS3: 2019-01-27 12:19:14+00:00 → 2019-01-27 12:19:28+00:00 (dur=14 s), max |ΔVN|=67.6 km/s
  - MMS3: 2019-01-27 12:19:32+00:00 → 2019-01-27 12:19:59+00:00 (dur=27 s), max |ΔVN|=83.4 km/s
  - MMS4: 2019-01-27 12:16:26+00:00 → 2019-01-27 12:16:39+00:00 (dur=13 s), max |ΔVN|=67.2 km/s
  - MMS4: 2019-01-27 12:18:05+00:00 → 2019-01-27 12:18:23+00:00 (dur=18 s), max |ΔVN|=113.6 km/s
  - MMS4: 2019-01-27 12:19:35+00:00 → 2019-01-27 12:20:42+00:00 (dur=67 s), max |ΔVN|=80.5 km/s
  - MMS4: 2019-01-27 12:44:06+00:00 → 2019-01-27 12:44:24+00:00 (dur=18 s), max |ΔVN|=62.6 km/s
  - MMS4: 2019-01-27 12:44:29+00:00 → 2019-01-27 12:45:27+00:00 (dur=58 s), max |ΔVN|=109.6 km/s

## DN differences (mms_mp vs published all_1243)
- MMS1: MAE=4539 km, RMSE=4943 km, corr=nan
- MMS2: MAE=5468 km, RMSE=6513 km, corr=nan
- MMS3: MAE=2373 km, RMSE=2745 km, corr=nan
- MMS4: MAE=3989 km, RMSE=4415 km, corr=nan
  Exceedance intervals where |ΔDN|>200 km for ≥10 s:
  - MMS1: 2019-01-27 12:21:14+00:00 → 2019-01-27 12:30:32+00:00 (dur=558 s), max |ΔDN|=7123 km
  - MMS2: 2019-01-27 12:21:23+00:00 → 2019-01-27 12:43:22+00:00 (dur=1319 s), max |ΔDN|=14777 km
  - MMS3: 2019-01-27 12:21:44+00:00 → 2019-01-27 12:27:05+00:00 (dur=321 s), max |ΔDN|=5170 km
  - MMS4: 2019-01-27 12:21:28+00:00 → 2019-01-27 12:30:29+00:00 (dur=541 s), max |ΔDN|=6413 km

Notes:
- BN comparison isolates coordinate-system differences (two LMN triads applied to the same B_gsm).
- VN comparison uses .sav LMN for rotation to isolate instrument/pipeline differences.
- DN comparisons use the cold-ion windows from mms*_DN_segments.csv to match the published methodology.