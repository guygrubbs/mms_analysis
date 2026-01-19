# MMS Magnetopause Analysis Toolkit – Detailed Overview

This guide explains the **end-to-end workflow** for the MMS Magnetopause
Analysis Toolkit (`mms_mp`): from loading MMS data to producing scientific
results (LMN, BN, VN, DN, boundaries, and figures). It is written for
magnetopause scientists, including those **new to Python**.

See also:
- [Variable reference](variable-reference-mms-mp.md)
- API overview: [docs/api/README.md](api/README.md)

---

## 1. Big-picture workflow

At a high level, a typical analysis (e.g. 2019‑01‑27 12:15–12:55 UT) follows:

1. **Load MMS data** with `mms_mp.data_loader.load_event`.
2. **Build or choose an LMN system** (curated `.sav` or algorithmic LMN).
3. **Rotate B and V into LMN** to obtain BL, BM, BN and VN.
4. **Integrate VN → DN** to track boundary motion and thickness.
5. **Detect boundary regions** using He⁺, density contrast, and BN.
6. **Visualise** fields, VN/DN, and spectrograms with `mms_mp.visualize`
   and `mms_mp.spectra`.
7. **Compare with IDL** using `tools.idl_sav_import` and the `.sav` files
   for LMN and ViN.

Each step is backed by tests in `tests/` that validate both the **physics**
and the **numerics** against analytic expectations and the IDL reference.

---

## 2. Loading data: `data_loader`

**Goal:** obtain a single "event dictionary" with all needed MMS variables.

Key function: `mms_mp.data_loader.load_event`  
API docs: [api/data_loader.md](api/data_loader.md)

Example (2019‑01‑27 event):

```python
import mms_mp as mp

TRANGE = ['2019-01-27/12:15:00', '2019-01-27/12:55:00']
PROBES = ['1', '2', '3', '4']

evt = mp.data_loader.load_event(TRANGE, probes=PROBES,
                                include_hpca=True,
                                include_edp=True,
                                include_ephem=True)
```

This returns:
- `evt[probe][var] -> (t, data)` for variables like `B_gsm`, `N_tot`, `N_he`,
  `V_i_gse`, `E_gse`, `POS_gsm`, `VEL_gsm`.
- `evt['__meta__']` with provenance (CDF sources, cadences, coverage, etc.).

The **variable reference** document explains each key (units, frame, CDF name).

---

## 3. Choosing an LMN coordinate system

### 3.1 Canonical `.sav` LMN (ver3b)

For the 2019‑01‑27 event, the canonical LMN frame is stored in:

- `mp_lmn_systems_20190127_1215-1255_mp-ver3b.sav`

Use the helper in `tools.idl_sav_import`:

```python
from tools.idl_sav_import import load_idl_sav

sav = load_idl_sav('mp_lmn_systems_20190127_1215-1255_mp-ver3b.sav')
# sav['lmn']['1']['L'], ['M'], ['N'] are MMS1 unit vectors
```

These LMN triads reflect expert judgment and are treated as **authoritative**
for this event. BN/VN comparisons in the repository all reference this frame.

### 3.2 Algorithmic LMN from CDF data

For events without curated `.sav` files, or for diagnostics, you can build LMN
purely from CDF data using `mms_mp.coords.algorithmic_lmn` (see tests and
`examples/algorithmic_lmn_param_sweep_20190127.py`). The optimised defaults
are tuned so that, for 2019‑01‑27, algorithmic LMN nearly reproduces the
curated `.sav` normals.

---

## 4. Rotating into LMN and computing BN, VN, DN

### 4.1 BN: magnetic field normal component

**Why:** BN identifies the current sheet and enters boundary detection.

1. From `evt[probe]['B_gsm']`, build a time-indexed DataFrame.
2. Apply the LMN rotation to obtain BL, BM, BN.

This is encapsulated in the DN/shear example:
- `examples/analyze_20190127_dn_shear.py` (function `build_timeseries`).

### 4.2 VN: normal velocity

**Why:** VN measures how fast the boundary moves along N.

Two main sources:

- **Bulk ion flow**: rotate `V_i_gse` into LMN.
- **E×B drift**: use `mms_mp.electric.exb_velocity` and rotate into N.

The helper `mms_mp.electric.normal_velocity` blends these sources, applying
quality masks and B-field thresholds. See its docstring for details.

### 4.3 DN: normal displacement

**Why:** DN(t) tracks the position of the boundary relative to a reference
point and is used to measure layer thickness and shear.

Use `mms_mp.motion.integrate_disp`:

```python
from mms_mp.motion import integrate_disp

# t_sec: float seconds since epoch, vn_km_s: VN in km/s
res = integrate_disp(t_sec, vn_km_s, scheme='trap')
DN_km = res.disp_km  # displacement along N
```

For the 2019‑01‑27 event, the DN/shear story in
`examples/analyze_20190127_dn_shear.py` reconstructs DN for each probe in
the cold-ion intervals around the 12:43 UT crossing.

---

## 5. Boundary detection and region classification

Module: `mms_mp.boundary`  
Key class/function: `DetectorCfg`, `detect_crossings_multi`

Inputs (per time sample):
- He⁺ density and fraction (from HPCA, `N_he` and `N_tot`).
- BN and its gradients.
- Total ion density contrast.

`detect_crossings_multi` returns labelled intervals (sheath, boundary layer,
magnetosphere) that correspond to physically distinct regions in the crossing.
Its docstring explains the scoring and hysteresis logic.

These classifications are used in the 2019‑01‑27 storyboards and summary
figures under `results/events_pub/2019-01-27_1215-1255/`.

---

## 6. Spectrograms and visualisation

Energy–time spectrograms and summary panels are handled by:

- `mms_mp.spectra.generic_spectrogram`, `fpi_ion_spectrogram`,
  `fpi_electron_spectrogram`, `hpca_ion_spectrogram`.
- `mms_mp.visualize.summary_single`, `overlay_multi`, etc.

Examples for 2019‑01‑27:
- `examples/make_spectrograms_20190127_algorithmic.py`
- `examples/make_event_figures_20190127.py`

These scripts are good starting templates for new events: update `TRANGE`,
`PROBES`, and any event-specific file paths.

---

## 7. Comparing to IDL workflows

To relate Python results to the original IDL analysis:

1. Use `tools.idl_sav_import.load_idl_sav` to read LMN and ViN from the
   `.sav` files.
2. Refer to [variable-reference-mms-mp.md](variable-reference-mms-mp.md),
   Section 5, for **Python ↔ IDL field mappings**.
3. Use `examples/diagnostic_sav_vs_mmsmp_20190127.py` to reproduce and extend
   the repository’s own CDF vs `.sav` comparisons for B_LMN and VN.

This allows IDL users to treat the Python toolkit as a fully documented,
reproducible reimplementation of the 2019‑01‑27 magnetopause analysis, and as
a template for new events.

