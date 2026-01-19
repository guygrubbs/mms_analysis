# MMS-MP Variable Reference

This document summarises the core variables produced and consumed by the
`mms_mp` toolkit, with emphasis on the 2019-01-27 12:15–12:55 UT
magnetopause event. It links **Python names**, **MMS CDF/tplot names** and
**physical meaning/units**.

---

## 1. Event dictionary structure

`mms_mp.data_loader.load_event(trange, probes, ...)` returns:

- `event[probe][var] -> (t, data)` for each spacecraft, where `probe` is
  `'1'`–`'4'`.
- `event['__meta__']` – metadata (requested trange, sources, warnings,
  ephemeris provenance, etc.).

Time arrays `t` are typically float epoch seconds as returned by
`pyspedas.get_data`. Data arrays are NumPy ndarrays.

---

## 2. Core per-probe variables

### `B_gsm`
- **Physical quantity**: Magnetic field vector in GSM (or GSE treated as GSM by
  convention).
- **Units**: nT.
- **Shape**: `(N, 3)` → `[Bx, By, Bz]`.
- **MMS CDF / tplot**: `mms{p}_fgm_b_gsm_{rate}_l2_bvec` or
  `mms{p}_fgm_b_gse_{rate}_l2_bvec`.
- **Usage**: Source for LMN construction, BN profiles, and E×B drift when
  combined with `E_gse`.

### `N_tot`
- **Physical quantity**: Ion number density from FPI-DIS.
- **Units**: cm⁻³.
- **Shape**: `(N,)`.
- **MMS CDF / tplot**: `mms{p}_dis_numberdensity_*`.
- **Usage**: Total plasma density for boundary classification and VN/DN
  diagnostics.

### `N_e`
- **Physical quantity**: Electron number density from FPI-DES.
- **Units**: cm⁻³.
- **Shape**: `(N,)`.
- **MMS CDF / tplot**: `mms{p}_des_numberdensity_*`.
- **Usage**: Charge balance checks and comparison with `N_tot`.

### `N_he`
- **Physical quantity**: He⁺ number density from HPCA.
- **Units**: cm⁻³.
- **Shape**: `(N,)`.
- **MMS CDF / tplot**: `mms{p}_hpca_heplus_number_density_*`.
- **Usage**: Composition-sensitive tracer for magnetosphere vs sheath; key
  input to the boundary detector (`DetectorCfg.he_*`).

### `V_i_gse`
- **Physical quantity**: Ion bulk velocity (FPI-DIS) in GSE.
- **Units**: km s⁻¹.
- **Shape**: `(N, 3)`.
- **MMS CDF / tplot**: `mms{p}_dis_bulkv_gse_*`.
- **Usage**: Bulk flow used for normal-velocity estimates and comparison with
  E×B drift (`exb_velocity`).

### `V_e_gse`
- **Physical quantity**: Electron bulk velocity (FPI-DES) in GSE (when
  available).
- **Units**: km s⁻¹.
- **Shape**: `(N, 3)`.
- **MMS CDF / tplot**: `mms{p}_des_bulkv_gse_*`.

### `V_he_gsm`
- **Physical quantity**: He⁺ bulk velocity from HPCA in GSM.
- **Units**: km s⁻¹.
- **Shape**: `(N, 3)`.
- **MMS CDF / tplot**: species-specific HPCA bulk-velocity variables.
- **Usage**: Alternative bulk-flow proxy for boundary motion when FPI data are
  poor.

### `E_gse`
- **Physical quantity**: DC electric field from EDP in GSE.
- **Units**: mV m⁻¹.
- **Shape**: `(N, 3)`.
- **MMS CDF / tplot**: `mms{p}_edp_dce_*_l2`.
- **Usage**: Input to `exb_velocity` and `exb_velocity_sync` for E×B drift
  estimates and convection-field calculations.

### `SC_pot`
- **Physical quantity**: Spacecraft potential.
- **Units**: V.
- **Shape**: `(N,)`.
- **MMS CDF / tplot**: `mms{p}_edp_scpot_*_l2`.
- **Usage**: Optional E-field correction via `correct_E_for_scpot`.

### `POS_gsm`, `VEL_gsm`
- **Physical quantity**: Spacecraft position and velocity (MEC ephemeris).
- **Units**: km (position), km s⁻¹ (velocity).
- **Shape**: `(N, 3)`.
- **MMS CDF / tplot**: `mms{p}_mec_r_gsm`, `mms{p}_mec_v_gsm` (with GSE
  fallbacks recorded in metadata when necessary).
- **Usage**: Geometry for timing normals (`timing_normal`), formation analysis,
  and mapping VN/DN to physical distances.

---

## 3. Derived LMN / boundary variables

These are not stored directly in the event dict but are produced by the
analysis pipeline and IDL `.sav` reference files.

### LMN systems
- **Physical quantity**: Boundary-aligned orthonormal triads (L, M, N).
- **Code**: `mms_mp.coords.hybrid_lmn`, `mms_mp.coords.algorithmic_lmn`.
- **Units / type**: Dimensionless 3×3 rotation matrices (rows = L, M, N).
- **IDL `.sav`**: `mp_lmn_systems_20190127_1215-1255_mp-ver3b.sav` etc.

### `BN`
- **Physical quantity**: N-component of B in LMN.
- **Units**: nT.
- **Usage**: Identifies current-sheet structure and enters the boundary
  classifier (`DetectorCfg.BN_tol`, `bn_grad_min`, etc.).

### `VN`
- **Physical quantity**: Boundary-normal velocity.
- **Units**: km s⁻¹.
- **Code**: `mms_mp.motion.normal_velocity` (LMN rotation) and
  `mms_mp.electric.normal_velocity` (blend of bulk and E×B VN estimates).
- **Usage**: Integrated to obtain DN; compared with IDL ViN from `.sav` files.

### `DN`
- **Physical quantity**: Boundary-normal displacement.
- **Units**: km.
- **Code**: `mms_mp.motion.integrate_disp`.
- **Usage**: Thickness of boundary layer and position along the crossing; used
  extensively in the 2019-01-27 DN/shear analyses.

---

## 4. Canonical configuration variables

### `TRANGE`
- **Example**: `['2019-01-27/12:15:00', '2019-01-27/12:55:00']`.
- **Meaning**: UTC analysis window passed to `load_event` and used
  consistently across examples and diagnostics.

### `PROBES`
- **Example**: `['1', '2', '3', '4']`.
- **Meaning**: MMS spacecraft included in the event analysis. All core
  2019-01-27 workflows assume all four probes when available.

---

## 5. Python ↔ IDL `.sav` mappings (2019-01-27 canonical event)

The reference IDL files for this event are:

- **Canonical LMN set (`mixed_1230_1243`)**:
  `mp_lmn_systems_20190127_1215-1255_mp-ver3b.sav`
- **Legacy comparison set (`all_1243`)**:
  `mp_lmn_systems_20190127_1215-1255_mp-ver2b.sav`

Python comparisons use the thin wrapper
`tools.idl_sav_import.load_idl_sav(path)`, which returns a dict with keys
`'lmn'`, `'vi_lmn'`, and `'b_lmn'` that expose the most important IDL
structures.

### 5.1 LMN rotation matrices

**Concept:** boundary-aligned unit vectors L, M, N for each probe.

- **Python (via loader):**
  - `sav = load_idl_sav('...mp-ver3b.sav')`
  - `L = sav['lmn'][probe]['L']`  (shape `(3,)`)
  - `M = sav['lmn'][probe]['M']`
  - `N = sav['lmn'][probe]['N']`
- **IDL fields inside `.sav`:**
  - `LHAT{p}`, `MHAT{p}`, `NHAT{p}` (e.g. `LHAT1`, `MHAT1`, `NHAT1` for MMS1)
- **Physical meaning:** rows of the LMN rotation matrix that map from GSM/GSE
  into the magnetopause-aligned frame (L approx tangential field, N approx
  boundary normal).

**Example (2019-01-27, MMS1, canonical ver3b):**

- `L`, `M`, `N` from `sav['lmn']['1']` are used as the **authoritative** LMN
  triad around the 12:43 UT crossing in
  `examples/analyze_20190127_dn_shear.py`, and BN/VN/DN are constructed by
  rotating CDF-based `B_gsm` and `V_i_gse` into this frame.

### 5.2 BN and B_LMN

**Concept:** magnetic field components in LMN and the normal component BN.

- **Python (CDF-driven):**
  - `B_gsm` from `load_event` is rotated with `L,M,N` to produce BL, BM, BN time
    series.
- **IDL `.sav` field:**
  - `B_LMN{p}` structures (e.g. `B_LMN1`), read as
    `sav['b_lmn'][probe] = {'t': t_sec, 'blmn': blmn}`.
  - `blmn[:, 0:3]` are the LMN components in the `.sav` frame.

**Special case (ver2b "all_1243" file):**

- For `mp_lmn_systems_20190127_1215-1255_mp-ver2b.sav`, diagnostics show that
  the stored columns correspond to **(B_N, B_M, B_L)** rather than (B_L,B_M,B_N).
- The loader deliberately preserves this raw ordering; downstream code
  (e.g. `examples/diagnostic_sav_vs_mmsmp_20190127.py`) remaps indices so that
  comparisons to CDF-rotated BL/BM/BN are physically meaningful.

### 5.3 VN from IDL ViN series

**Concept:** boundary-normal ion velocity Vi · N ("ViN").

- **Python (analysis variables):**
  - `VN` is built from either rotated DIS velocities or a blend with E×B drift
    using `mms_mp.electric.normal_velocity`, then integrated to DN via
    `mms_mp.motion.integrate_disp`.
- **Python (IDL-based comparison):**
  - `sav = load_idl_sav('...mp-ver3b.sav')`
  - `vi_lmn = sav['vi_lmn'][probe]` with keys:
    - `vi_lmn['t']` → float seconds since epoch
    - `vi_lmn['vlmn']` → array `(N,3)` of (Vi_L, Vi_M, Vi_N)
  - `tools.idl_sav_import.extract_vn_series(sav)` returns
    `{probe: (t, vn)}` where `vn` is the N-component of `vlmn`.
- **IDL fields inside `.sav`:**
  - `VI_LMN{p}` (e.g. `VI_LMN2`), whose `y` field contains the LMN velocity
    components.

**Example (2019-01-27 DN/shear analysis):**

- In `examples/analyze_20190127_dn_shear.py`, `build_timeseries()` prefers
  DIS-based VN but will fall back to `.sav` ViN when FPI data are missing.
  The fallback path uses `sav['vi_lmn'][probe]['vlmn'][:, 2]` as VN, resampled
  to a 1 s grid, before feeding it into `integrate_disp`.

### 5.4 DN (displacement) in Python vs IDL

The canonical `.sav` files used here do **not** store DN explicitly. In the
original IDL workflow, DN is computed by integrating ViN in LMN; in Python we
recompute DN with full control over masks, gaps, and error estimates:

- **Python DN:**
  - `from mms_mp.motion import integrate_disp`
  - `res = integrate_disp(t_sec, vn_km_s, scheme='trap')`
  - `res.disp_km` is the cumulative normal displacement relative to the start
    of each integration window.

When comparing to legacy IDL DN, treat the Python DN as a **fresh numerical
integration** of the same physical quantity (ViN), not as a direct copy of any
single `.sav` field.
