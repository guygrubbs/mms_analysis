# MMS Magnetopause Analysis: mms_mp vs pySPEDAS vs IDL (+ .sav)

This document summarizes similarities and differences among:
- mms_mp (Python toolkit in this repo)
- pySPEDAS (references/pyspedas-master)
- IDL example + .sav (references/IDL_Code)

It includes the event .sav files for 2019-01-27: the canonical LMN set `mp_lmn_systems_20190127_1215-1255_mp-ver3b.sav` ("mixed_1230_1243") and the legacy set `mp_lmn_systems_20190127_1215-1255_mp-ver2b.sav` ("all_1243").

## Scope and orientation
- mms_mp: end-to-end MP workflow (load → ephemeris → LMN → boundary detection → motion/timing → thickness → plots/outputs)
- pySPEDAS: foundational MMS loaders, transforms and plotting (tplot), used by mms_mp as a data/transform backend
- IDL + .sav: curated event workflow with precomputed LMN and Vi_LMN, manual interval selection, and tplot visualization

## Data access
- All use MMS L2 and observe data rates (fast/srvy/brst)
- mms_mp: central loader, cadence fallbacks, MEC-first ephemeris, aggressive FPI spectrogram discovery
- pySPEDAS: per-instrument loaders, tplot registry, extensive tests
- IDL: event-specific mms_load_* and varformat filters; .sav anchors LMN/Vi

## Ephemeris/coordinates
- mms_mp: EphemerisManager (MEC authoritative); hybrid LMN (MVA → pySPEDAS lmn_matrix_make → Shue)
- pySPEDAS: cotrans/mms_qcotrans/mms_cotrans_lmn; broad coordinate support
- IDL: .sav provides LHAT/MHAT/NHAT for event; script rotates B→LMN with fixed matrix

## Boundary detection
- mms_mp: multi-parameter detector (He+, BN, hysteresis, layer tagging)
- pySPEDAS: user-assembled; no packaged MP-detector in MMS directory
- IDL: manual vt_mms# intervals (timebars), visual validation

## E×B and VN selection
- mms_mp: exb_velocity, resample sync, VN strategy blender (bulk/ExB/avg)
- pySPEDAS: EDP available; E×B in user scripts
- IDL: integrates ViN from .sav or computed LMN velocities

## Motion/thickness
- mms_mp: integrate_disp (trap/simpson/rect, masks, σ propagation); thickness from crossings
- pySPEDAS: user scripts
- IDL: DN via tsum over curated intervals

## Multi-spacecraft timing/formation
- mms_mp: SVD timing for n̂, V_phase + uncertainty; formation detection (tetrahedral/string-of-pearls/…)
- pySPEDAS: multi-point tools (e.g., curlometer); timing normal assembled by users
- IDL: script focuses on ViN/DN; no timing solver in this file

## Spectrograms and plotting
- mms_mp: 2D time-energy plots with log-safe pipeline, mask overlay, optional resample; force-load FPI omni/dist products
- pySPEDAS: tplot-centric; energyspectr_omni variables
- IDL: tplot spectrograms, ViN overlays, timebars

## Quality/masks
- mms_mp: DIS/DES/HPCA masks, resampling of masks, gaps/outliers utilities
- pySPEDAS: status/metadata and flags are available; application left to users
- IDL: relies on product selection and expert screening

## Role of the .sav
- Provides authoritative event LMN (LHAT/MHAT/NHAT) and Vi_LMN#.x/.y `[VL, VM, VN]`
- Enables direct cross-checks:
  - Compare mms_mp hybrid LMN vectors to .sav LMN (angles)
  - Compare mms_mp VN (bulk or ExB) to .sav VN series
  - Replicate IDL DN integration on the same intervals and quantify differences

## Validation helpers added
- tools/idl_sav_import.py — loads .sav (SciPy), exposes LMN and Vi_LMN
- examples/compare_sav_vs_mms_mp.py — runs mms_mp on the same window, computes LMN/VN, compares to .sav, and writes CSV/JSON summaries to `results/comparison/`

## Suggested usage
```bash
python examples/compare_sav_vs_mms_mp.py
```
This will produce:
- results/comparison/sav_vs_mms_mp_summary.csv
- results/comparison/sav_vs_mms_mp_summary.json

## Notes
- SciPy is required to read .sav; we did not auto-install packages.
- For DES data gaps (e.g., MMS4 L2 post-2018), mms_mp force-load utilities and broad pattern matching mimic the IDL contingency (switching to available/QL-like sources).

## References
- mms_mp modules: data_loader.py, coords.py, boundary.py, electric.py, motion.py, multispacecraft.py, spectra.py, ephemeris.py
- pySPEDAS MMS project: references/pyspedas-master/pyspedas/projects/mms
- IDL example: references/IDL_Code/requested_mp_motion_givenlmn_vion.pro and the .sav listed above

