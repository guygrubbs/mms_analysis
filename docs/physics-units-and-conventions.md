# Physics Units and Conventions

This document summarizes the units, coordinate conventions, and sign rules used throughout the MMS Magnetopause Analysis Toolkit (mms_mp). It is intended to make unit expectations explicit and to help compare our analysis and figures with published literature.

---

## Coordinate Systems

- Global frames: GSM and GSE are used by MMS instruments and ephemeris.
- Local boundary frame: LMN (L = maximum variance, M = intermediate, N = minimum variance/normal)
  - Built by Minimum Variance Analysis (MVA) with right‑hand enforcement: (L × M) · N > 0
  - Fallbacks: pySPEDAS LMN matrix (if position available), then Shue (1997) model outward normal
- Transformations:
  - `LMN.to_lmn(XYZ)` rotates vectors from input XYZ into LMN
  - The rotation matrix R has rows [L; M; N]

Notes:
- LMN is local to the interval; check `LMN.method` to know whether the frame came from MVA, PySPEDAS, or Shue fallback.
- The sign of N may flip depending on eigenvector orientation; physical interpretation (e.g., sheath→sphere) may require matching with field rotation/density.

---

## Base Units

- Time: seconds (s) for arrays of time in numerics; timestamps as numpy datetime64[ns] when resampled.
- Position: kilometers (km) in GSM/GSE (MEC position is km; some sources may be in Earth radii RE and are converted internally when detected).
- Velocity: kilometers per second (km/s).
- Magnetic field B: nanotesla (nT) in typical instrument outputs; functions accept `unit_B` ('nT' or 'T') and convert internally.
- Electric field E: millivolts per meter (mV/m) commonly; functions accept `unit_E` ('mV/m' or 'V/m') and convert internally.
- Densities: particles per cubic centimeter (cm⁻3) for FPI/HPCA outputs.

---

## Core Formulas and Conventions

### E×B drift (electric.py)

- Formula: v = (E × B) / |B|²
- Inputs:
  - E: in V/m or mV/m (set unit_E accordingly)
  - B: in T or nT (set unit_B accordingly)
- Output: v in km/s (vector)
- Direction: right‑hand rule (E × B)
- Numerical notes: For |B| ≪ 1 nT, drift can become very large; we do not automatically clip but downstream logic may flag/ignore with data quality masks.

### Motional (convection) electric field (electric.py)

- Formula: E = − v × B
- Inputs: v in km/s, B in nT
- Output: E in mV/m

### LMN construction (coords.py)

- MVA (Sonnerup & Cahill, 1967) → eigen‑decomposition of B covariance
- Sorting: λ_max ≥ λ_mid ≥ λ_min → rows of R are eigenvectors [L; M; N]
- Right‑handedness enforced: if (L × M) · N < 0 then N ↦ −N
- Ratios reported: r_max_mid = λ_max/λ_mid, r_mid_min = λ_mid/λ_min
- Fallbacks: PySPEDAS LMN matrix (needs position), then Shue (1997) outward normal (direction only)

### Multi‑spacecraft timing (multispacecraft.py)

- Model: n · (r_i − r_0) = V (t_i − t_0) for spacecraft i
- Solver: SVD on stacked rows [Δr_x, Δr_y, Δr_z, −Δt]
- Output: unit normal n̂ and phase speed V (km/s), plus σ_V from singular values
- Assumptions: planar boundary with constant speed over the interval; collinear geometries degrade the solution.

### Displacement from normal velocity (motion.py)

- `integrate_disp(t, vN, scheme='trap'|'simpson'|'rect')`
- Inputs: t in seconds (monotonic), vN in km/s
- Output: displacement Δs in km, optional 1σ uncertainty propagation

---

## Function I/O Expectations (quick reference)

- `electric.exb_velocity(E_xyz, B_xyz, unit_E='mV/m', unit_B='nT') → v_exb (km/s)`
- `electric.calculate_convection_field(v_km_s, B_nT) → E_mV/m`
- `coords.hybrid_lmn(B_xyz[, pos_gsm_km]) → LMN` with `LMN.R` rows [L; M; N]
- `multispacecraft.timing_normal(pos_gsm, t_cross) → (n_hat, V_km_s, sigma_V)`
- `motion.integrate_disp(t_seconds, vN_km_s[, sigma_v]) → disp_km`

---

## Sign Conventions and Ambiguities

- E×B drift is charge‑independent; sign is set solely by E and B orientation.
- MVA eigenvectors are defined up to a sign; we enforce right‑handedness but N may still be opposite to a particular physical outward/inward convention. Use additional context (e.g., density/BN changes) to set the sign when comparing with literature.
- Timing normal sign may be ambiguous; interpretation should be based on event context.

---

## Common Pitfalls

- Mixed units (e.g., feeding E in V/m with unit_E='mV/m'). Always set unit flags.
- Near‑zero |B| causing unrealistically large E×B drift. Consider masking such intervals.
- Using mismatched coordinate frames for E and B; E and B must be in the same frame for E×B.
- Assuming LMN persists across long windows; recompute LMN for each boundary interval.

---

## References

- Chen, F. F. (2016), Introduction to Plasma Physics and Controlled Fusion
- Baumjohann, W. & Treumann, R. A. (1996), Basic Space Plasma Physics
- Kivelson, M. G. & Russell, C. T. (1995), Introduction to Space Physics
- Shue, J.-H., et al. (1997), A new functional form to study the solar wind control of the magnetopause size and shape
