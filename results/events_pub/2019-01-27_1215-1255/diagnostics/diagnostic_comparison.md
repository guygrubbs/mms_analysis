# Diagnostic Comparison (.sav vs mms_mp): 2019-01-27 12:15–12:55

- Strict local caching: all inputs loaded from local CDFs only (no re-downloads).
- DN integration for the comparison uses the same cold-ion windowing as the originally published all_1243 outputs; canonical publication results in this repository use the mixed_1230_1243 (mp-ver3b) LMN set.

## BN differences vs .sav LMN (hybrid and algorithmic)
### Source: algorithmic
- MMS1: MAE=11.578 nT, RMSE=67.772 nT, corr=1.000, count(|Δ|>0.5 nT)=25795, N-angle diff≈5.6°
- MMS2: MAE=9.968 nT, RMSE=51.709 nT, corr=1.000, count(|Δ|>0.5 nT)=41815, N-angle diff≈6.7°
- MMS3: MAE=13.851 nT, RMSE=70.186 nT, corr=0.999, count(|Δ|>0.5 nT)=55042, N-angle diff≈12.1°
- MMS4: MAE=43.740 nT, RMSE=245.757 nT, corr=0.997, count(|Δ|>0.5 nT)=69473, N-angle diff≈24.2°
### Source: hybrid
- MMS1: MAE=395.219 nT, RMSE=1421.218 nT, corr=0.798, count(|Δ|>0.5 nT)=80644, N-angle diff≈89.7°
- MMS2: MAE=355.806 nT, RMSE=1249.658 nT, corr=0.863, count(|Δ|>0.5 nT)=81598, N-angle diff≈89.6°
- MMS3: MAE=273.372 nT, RMSE=919.178 nT, corr=0.922, count(|Δ|>0.5 nT)=82954, N-angle diff≈89.6°
- MMS4: MAE=546.697 nT, RMSE=2228.007 nT, corr=-0.928, count(|Δ|>0.5 nT)=83628, N-angle diff≈89.9°
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
  - MMS3: 2019-01-27 00:00:07+00:00 → 2019-01-27 00:10:33+00:00 (dur=626 s), max |ΔBN|=3.65 nT
  - MMS3: 2019-01-27 00:12:15+00:00 → 2019-01-27 00:12:30+00:00 (dur=15 s), max |ΔBN|=1.13 nT
  - MMS3: 2019-01-27 00:12:31+00:00 → 2019-01-27 00:36:10+00:00 (dur=1419 s), max |ΔBN|=3.53 nT
  - MMS3: 2019-01-27 00:36:11+00:00 → 2019-01-27 00:37:30+00:00 (dur=79 s), max |ΔBN|=3.30 nT
  - MMS3: 2019-01-27 00:37:37+00:00 → 2019-01-27 00:37:47+00:00 (dur=10 s), max |ΔBN|=1.32 nT
  - MMS4: 2019-01-27 00:00:05+00:00 → 2019-01-27 00:37:27+00:00 (dur=2242 s), max |ΔBN|=6.36 nT
  - MMS4: 2019-01-27 00:37:29+00:00 → 2019-01-27 00:37:40+00:00 (dur=11 s), max |ΔBN|=2.53 nT
  - MMS4: 2019-01-27 00:37:58+00:00 → 2019-01-27 00:38:11+00:00 (dur=13 s), max |ΔBN|=2.55 nT
  - MMS4: 2019-01-27 00:38:55+00:00 → 2019-01-27 00:42:37+00:00 (dur=222 s), max |ΔBN|=4.33 nT
  - MMS4: 2019-01-27 00:42:40+00:00 → 2019-01-27 00:47:07+00:00 (dur=267 s), max |ΔBN|=4.78 nT

## VN differences (.sav ViN vs mms_mp V_i·N_sav)
- MMS1: MAE=37.6 km/s, RMSE=53.7 km/s, corr=-0.206
- MMS2: MAE=39.4 km/s, RMSE=55.1 km/s, corr=-0.202
- MMS3: MAE=42.0 km/s, RMSE=52.5 km/s, corr=0.144
- MMS4: MAE=37.7 km/s, RMSE=51.6 km/s, corr=-0.127
  Exceedance intervals where |ΔVN|>50 km/s for ≥10 s:
  - MMS1: 2019-01-27 12:15:24+00:00 → 2019-01-27 12:16:00+00:00 (dur=36 s), max |ΔVN|=115.8 km/s
  - MMS1: 2019-01-27 12:16:23+00:00 → 2019-01-27 12:17:08+00:00 (dur=45 s), max |ΔVN|=191.8 km/s
  - MMS1: 2019-01-27 12:17:35+00:00 → 2019-01-27 12:18:11+00:00 (dur=36 s), max |ΔVN|=116.8 km/s
  - MMS1: 2019-01-27 12:18:15+00:00 → 2019-01-27 12:18:38+00:00 (dur=23 s), max |ΔVN|=70.6 km/s
  - MMS1: 2019-01-27 12:18:56+00:00 → 2019-01-27 12:20:30+00:00 (dur=94 s), max |ΔVN|=205.9 km/s
  - MMS2: 2019-01-27 12:15:25+00:00 → 2019-01-27 12:16:01+00:00 (dur=36 s), max |ΔVN|=125.0 km/s
  - MMS2: 2019-01-27 12:16:19+00:00 → 2019-01-27 12:17:08+00:00 (dur=49 s), max |ΔVN|=203.1 km/s
  - MMS2: 2019-01-27 12:17:35+00:00 → 2019-01-27 12:18:11+00:00 (dur=36 s), max |ΔVN|=111.3 km/s
  - MMS2: 2019-01-27 12:18:20+00:00 → 2019-01-27 12:18:47+00:00 (dur=27 s), max |ΔVN|=98.7 km/s
  - MMS2: 2019-01-27 12:18:56+00:00 → 2019-01-27 12:20:35+00:00 (dur=99 s), max |ΔVN|=207.4 km/s
  - MMS3: 2019-01-27 12:15:49+00:00 → 2019-01-27 12:16:29+00:00 (dur=40 s), max |ΔVN|=194.3 km/s
  - MMS3: 2019-01-27 12:16:47+00:00 → 2019-01-27 12:17:10+00:00 (dur=23 s), max |ΔVN|=96.0 km/s
  - MMS3: 2019-01-27 12:18:22+00:00 → 2019-01-27 12:18:53+00:00 (dur=31 s), max |ΔVN|=94.7 km/s
  - MMS3: 2019-01-27 12:18:58+00:00 → 2019-01-27 12:20:14+00:00 (dur=76 s), max |ΔVN|=146.9 km/s
  - MMS3: 2019-01-27 12:21:35+00:00 → 2019-01-27 12:22:07+00:00 (dur=32 s), max |ΔVN|=116.7 km/s
  - MMS4: 2019-01-27 12:15:02+00:00 → 2019-01-27 12:16:59+00:00 (dur=117 s), max |ΔVN|=212.9 km/s
  - MMS4: 2019-01-27 12:17:35+00:00 → 2019-01-27 12:18:29+00:00 (dur=54 s), max |ΔVN|=171.9 km/s
  - MMS4: 2019-01-27 12:20:35+00:00 → 2019-01-27 12:21:15+00:00 (dur=40 s), max |ΔVN|=117.1 km/s
  - MMS4: 2019-01-27 12:21:24+00:00 → 2019-01-27 12:21:51+00:00 (dur=27 s), max |ΔVN|=93.4 km/s
  - MMS4: 2019-01-27 12:21:56+00:00 → 2019-01-27 12:22:14+00:00 (dur=18 s), max |ΔVN|=67.0 km/s

## DN differences (mms_mp vs published all_1243)
- MMS1: MAE=5141 km, RMSE=5416 km, corr=0.701
- MMS2: MAE=6055 km, RMSE=6991 km, corr=0.157
- MMS3: MAE=2114 km, RMSE=2862 km, corr=-0.912
- MMS4: MAE=1413 km, RMSE=1641 km, corr=0.939
  Exceedance intervals where |ΔDN|>200 km for ≥10 s:
  - MMS1: 2019-01-27 12:21:14+00:00 → 2019-01-27 12:30:32+00:00 (dur=558 s), max |ΔDN|=7394 km
  - MMS2: 2019-01-27 12:21:21+00:00 → 2019-01-27 12:43:22+00:00 (dur=1321 s), max |ΔDN|=14510 km
  - MMS3: 2019-01-27 12:22:02+00:00 → 2019-01-27 12:22:28+00:00 (dur=26 s), max |ΔDN|=288 km
  - MMS3: 2019-01-27 12:23:04+00:00 → 2019-01-27 12:27:05+00:00 (dur=241 s), max |ΔDN|=5932 km
  - MMS4: 2019-01-27 12:21:33+00:00 → 2019-01-27 12:24:46+00:00 (dur=193 s), max |ΔDN|=1303 km
  - MMS4: 2019-01-27 12:25:13+00:00 → 2019-01-27 12:30:29+00:00 (dur=316 s), max |ΔDN|=3122 km

Notes:
- BN comparison isolates coordinate-system differences (two LMN triads applied to the same B_gsm).
- VN comparison uses .sav LMN for rotation to isolate instrument/pipeline differences.
- DN comparisons use the cold-ion windows from mms*_DN_segments.csv to match the published methodology.