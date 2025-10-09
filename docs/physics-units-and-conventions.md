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
- LMN is local to the interval; check `LMN.method` to know whether the frame came from MVA (`"mva"`), PySPEDAS (`"pyspedas"`), or Shue fallback (`"shue"`).
- `LMN.meta["formation_type"]` records the formation context (`"auto"`, `"planar"`, etc.) that set the eigenvalue thresholds, while `LMN.meta["eig_ratio_thresholds"]` stores the exact λ ratios required for acceptance.
- The sign of N may flip depending on eigenvector orientation; physical interpretation (e.g., sheath→sphere) may require matching with field rotation/density.

Threshold guidance for accepting an MVA solution:

| Formation context | λ<sub>max</sub>/λ<sub>mid</sub> | λ<sub>mid</sub>/λ<sub>min</sub> | Notes |
|-------------------|-----------------------------|------------------------------|-------|
| auto / tetrahedral | ≥ 2.0 | ≥ 2.0 | Matches Paschmann & Daly (1998) recommendation for well-separated eigenvalues. |
| planar | ≥ 3.0 | ≥ 2.5 | Requires stronger separation before trusting a quasi-2D sheet. |
| string_of_pearls | ≥ 3.5 | ≥ 2.5 | String formations tolerate more anisotropy; keep higher threshold for λ<sub>max</sub>. |
| linear | ≥ 4.0 | ≥ 3.0 | Forces fallback unless the maximum direction is dominant, typical of nearly 1-D strings. |
| irregular | ≥ 2.5 | ≥ 2.0 | Slightly stricter than auto to guard against noisy orientations. |
| collapsed | fallback | fallback | Degenerate formations always trigger the Shue model normal. |

Passing an explicit `eig_ratio_thresh` overrides these defaults and can be either a scalar (applied to both ratios) or a `(λ_max/λ_mid, λ_mid/λ_min)` tuple.

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
- Numerical notes: `exb_velocity` can optionally return a boolean quality mask (`return_quality=True`) and treat magnetic magnitudes below `min_b` as invalid so that vanishing |B| does not yield unphysical drift speeds.
- Verification: `tests/test_electric_physics_validation.py` regresses against hand-derived analytic solutions and the IDL quicklook workflow for the 2019‑01‑27 interval, enforcing ≤0.15 km/s agreement.

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
- Output: unit normal n̂, phase speed V (km/s), σ_V from singular values, and optional diagnostics (condition number, residual norm, degeneracy flag)
- Assumptions: planar boundary with constant speed over the interval; collinear geometries degrade the solution. Use `return_diagnostics=True` to quantify conditioning.

### Displacement from normal velocity (motion.py)

- `integrate_disp(t, vN, scheme='trap'|'simpson'|'rect', max_step_s=None)`
- Inputs: t in seconds (monotonic), vN in km/s; optional `max_step_s` enforces adaptive sub-steps when sparse cadence would bias the integral.
- Output: displacement Δs in km, optional 1σ uncertainty propagation, plus metadata (`n_gaps_filled`, `segment_count`, `max_step_s`) describing gap handling.

### Dynamic pressure & charge balance panels (visualize.py)

- Ion dynamic pressure is plotted as $P_{dyn} = n_i m_p v_{N,i}^2$, converted to nano-Pascals via the $10^6$ cm⁻³→m⁻³ and $10^3$ km/s→m/s factors.  The He⁺ curve multiplies the proton mass term by four.
- Charge-balance diagnostics display ΔN = Nₑ − Nᵢ (cm⁻³) with a zero reference line and overlay the He⁺ fraction `N_he / N_i` on a twin axis.
- Multi-probe overlays (`visualize.overlay_multi`) expect stacked `[Δt, value]` arrays relative to a reference probe and label the x-axis in seconds, matching the timing diagnostics used in publication figures.

---

## Function I/O Expectations (quick reference)

- `electric.exb_velocity(E_xyz, B_xyz, unit_E='mV/m', unit_B='nT', min_b=None, return_quality=False) → v_exb (km/s[, quality])`
- `electric.calculate_convection_field(v_km_s, B_nT) → E_mV/m`
- `electric.normal_velocity(v_bulk_lmn, v_exb_lmn, ..., return_metadata=False) → vN (km/s[, metadata])`
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
- Near‑zero |B| causing unrealistically large E×B drift. Use `min_b` or the returned quality mask to filter such samples.
- Using mismatched coordinate frames for E and B; E and B must be in the same frame for E×B.
- Assuming LMN persists across long windows; recompute LMN for each boundary interval.

---

## References

- Chen, F. F. (2016), Introduction to Plasma Physics and Controlled Fusion
- Baumjohann, W. & Treumann, R. A. (1996), Basic Space Plasma Physics
- Kivelson, M. G. & Russell, C. T. (1995), Introduction to Space Physics
- Shue, J.-H., et al. (1997), A new functional form to study the solar wind control of the magnetopause size and shape
