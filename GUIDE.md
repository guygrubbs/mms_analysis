Thought for 4 seconds


## Quick-Start Guide

*(companion to the `README.md`)*

This document shows **exactly what to type** and how to interpret the outputs when using the MMS-Magnetopause Python toolkit.  It is intentionally hands-on and complements the high-level reference information in the README.

---

### 1  Prerequisites in one glance

| Requirement | Tested version | Notes                                       |
| ----------- | -------------- | ------------------------------------------- |
| Python      | ≥ 3.9, ≤ 3.12  | 3.13 not yet supported by PySPEDAS          |
| PySPEDAS    | 1.7.20         | Installs `pytplot`, `cdflib` automatically  |
| SciPy       | 1.12 +         | Used for Simpson integration & FFT resample |
| Matplotlib  | 3.8 +          | Publication plots                           |
| pandas      | 2.x            | Resampling & CSV output                     |

```bash
# fresh environment (example)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # ships with the repo
```

---

## 2  Core concepts

| Term             | Meaning in the code                                                                             |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| **event**        | A single time interval (list of two UTC strings) analysed for all probes.                       |
| **probe**        | Spacecraft number `'1'…'4'`; toolkit always refers to MMS1-4.                                   |
| **variable key** | Friendly alias (`'N_e'`, `'B_gsm'`, …) used **after** loading.                                  |
| **good mask**    | Boolean array returned by `resample.merge_vars`; `True` = sample accepted after quality checks. |
| **layer**        | Tuple `(layer_type, i_start, i_end)` produced by `boundary.detect_crossings_multi`.             |

---

## 3  Running the out-of-the-box demos

### 3.1 Notebook-style (`main_analysis.py`)

```bash
python main_analysis.py
```

* Downloads all burst CDFs for 27 Jan 2019, 11:00–13:00 UT.
* Opens interactive windows:
  • 4-panel quick-look for each probe
  • displacement curve
* Prints boundary-normal, phase speed, layer thickness.

### 3.2 Batch / scripting (`cli.py`)

```bash
python -m mms_mp.cli \
  --start 2019-11-12T04:00 --end 2019-11-12T05:00 \
  --probes 1 2 3 4 \
  --cadence 150ms \
  --edp        # include ExB drift
  --plot       # save PNGs
```

Outputs:

```
results/
 ├─ mp_analysis.json          # all numeric results, easy to parse
 ├─ layer_thickness.csv       # tidy table
 └─ figures/                  # PNG quick-look per spacecraft
```

---

## 4  Typical workflow in a notebook

```python
import mms_mp as mp

# 1) Load an event
trange = ['2020-02-07/12:00', '2020-02-07/12:40']
evt = mp.data_loader.load_event(trange, include_edp=True)

# 2) Pick a probe & resample
d = evt['2']
t, v, good = mp.resample.merge_vars({
    'Ni': (d['N_tot'][0], d['N_tot'][1]),
    'Ne': (d['N_e'][0],   d['N_e'][1]),
    'B' : (d['B_gsm'][0], d['B_gsm'][1]),
    'E' : (d['E_gse'][0], d['E_gse'][1])
}, cadence='100ms')

# 3) Build LMN & detect boundary
lm  = mp.coords.hybrid_lmn(d['B_gsm'][1])
BN  = lm.to_lmn(v['B'])[:,2]
layers = mp.boundary.detect_crossings_multi(t, v['Ni'], BN, good_mask=good['Ni'])

# 4) Plot
mp.visualize.summary_single(t, lm.to_lmn(v['B']),
                            v['Ni'], v['Ne'], d['N_he'][1],
                            lm.to_lmn(d['V_i_gse'][1])[:,2],   # V_N ions
                            lm.to_lmn(d['V_e_gse'][1])[:,2],   # V_N e-
                            lm.to_lmn(d['V_he_gsm'][1])[:,2],  # V_N He+
                            layers=layers,
                            title='MMS2 quick look')
```

---

## 5  Configuring the algorithms

| Knob                       | Where                                          | Effect                                              |
| -------------------------- | ---------------------------------------------- | --------------------------------------------------- |
| **Hybrid-LMN eigen-ratio** | `coords.hybrid_lmn(..., eig_ratio_thresh=7)`   | Stricter MVA acceptance.                            |
| **Boundary thresholds**    | `boundary.DetectorCfg`                         | `he_in`, `he_out`, `BN_tol` and hysteresis lengths. |
| **Velocity blend**         | `electric.normal_velocity(strategy='average')` | `'prefer_exb'`, `'prefer_bulk'`, `'average'`.       |
| **Integration scheme**     | `motion.integrate_disp(..., scheme='simpson')` | `'trap'`, `'simpson'`, `'rect'`.                    |

---

## 6  Troubleshooting checklist

| Symptom                                                           | Likely cause                                                    | Fix                                                                                            |
| ----------------------------------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `TypeError: mms_load_state() got an unexpected keyword 'notplot'` | Older/newer PySPEDAS changed API.                               | Patched in `data_loader.load_ephemeris()` – no extra flags except `downloadonly` if requested. |
| `Missing var mms4_des_numberdensity_fast`                         | MMS4 has no DES moments at that cadence/date.                   | Toolkit now falls back to `srvy`, then inserts NaNs with a warning.                            |
| Many “**compressionloss** not in pytplot” warnings                | Loader attempted to read optional FPI distribution diagnostics. | Harmless; ignore or add `varformat='*-moms*'` in loader.                                       |
| HPCA variables all fillval –1e31                                  | Instrument had no counts in interval.                           | Detector auto-masks; consider skipping He⁺ logic for that probe/time.                          |

---

## 7  Extending the toolkit

* **Add your own detector:** create a new module (e.g., `boundary_custom.py`) that imports `boundary.detect_crossings_multi`, tweaks thresholds, or substitutes different signatures, and call it from your driver script.
* **Parallel batch:** wrap `cli.py` inside GNU `parallel` or a Slurm job array to process dozens of events (each run is self-contained).
* **Interactive dashboards:** the resampled pandas DataFrames integrate cleanly with Plotly or Holoviews—simply convert `vars_grid` values to DataFrames.

---

## 8  Uninstall / clean cache

```bash
rm -rf ~/.pyspedas   # removes downloaded CDFs
deactivate           # leave venv
rm -rf venv
```

*(Cached CDFs can occupy many GB—delete when done.)*

---

### Happy magnetopause hunting 🌌!

Feel free to open an issue or pull request if you hit an unexpected dataset or need a new feature.
