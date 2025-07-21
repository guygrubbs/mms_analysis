"""
Multi-spacecraft timing analysis for boundary normal determination.

This module implements multi-spacecraft timing methods to determine boundary
orientation and motion using crossing times observed by multiple spacecraft.
The primary application is magnetopause boundary analysis using MMS data.

Key Functions:
    timing_normal: SVD-based solver for boundary normal and phase velocity
    stack_aligned: Overlay data from multiple spacecraft for comparison

The timing method solves the fundamental equation: n⃗ · (r⃗ᵢ - r⃗₀) = V(tᵢ - t₀)
where n⃗ is the boundary normal, V is the phase velocity, r⃗ᵢ are spacecraft
positions, and tᵢ are crossing times.
"""

# mms_mp/multispacecraft.py
# ------------------------------------------------------------
# Multi-point timing & comparison utilities  (upgraded)
# ------------------------------------------------------------
# New capabilities
# ----------------
# 1. Generalised *timing* solver via SVD that works for 2–4 spacecraft
#    and returns both boundary normal **n̂** and phase velocity **V_ph**,
#    including uncertainty estimates from singular values.
# 2. Helper to *align* every spacecraft’s timeline so that a chosen
#    reference crossing (e.g. magnetopause entry) occurs at Δt = 0.
# 3. Simple plot-ready dictionary builder to stack variables from all
#    probes after alignment (for overlay plots).
# ------------------------------------------------------------
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, List, Iterable, Optional


# ------------------------------------------------------------
# Timing solver
# ------------------------------------------------------------
def timing_normal(pos_gsm: Dict[str, np.ndarray],
                  t_cross: Dict[str, float],
                  *,
                  normalise: bool = True,
                  min_sc: int = 2
                  ) -> Tuple[np.ndarray, float, float]:
    """
    Determine boundary normal vector and phase velocity using multi-spacecraft timing analysis.

    This function implements the multi-spacecraft timing method to determine the
    orientation and motion of planar boundaries (such as the magnetopause) using
    crossing times observed by multiple spacecraft. The method solves the fundamental
    timing equation: n⃗ · (r⃗ᵢ - r⃗₀) = V(tᵢ - t₀) using Singular Value Decomposition.

    The physics assumes a planar boundary moving with constant velocity V in the
    direction of the normal vector n⃗. When spacecraft i crosses the boundary at
    time tᵢ and position r⃗ᵢ, the timing relationship provides constraints that
    allow determination of both n⃗ and V.

    Args:
        pos_gsm: Dictionary mapping spacecraft identifiers to position vectors.
            Keys: Spacecraft identifiers (e.g., '1', '2', '3', '4' for MMS)
            Values: Position vectors in GSM coordinates, shape (3,)
            Units: km. Positions should be at or near the crossing times.

        t_cross: Dictionary mapping spacecraft identifiers to crossing times.
            Keys: Spacecraft identifiers (must match pos_gsm keys)
            Values: Crossing times in seconds since epoch (Unix timestamp)
            Units: seconds. Times when each spacecraft crossed the boundary.

        normalise: Whether to normalize the boundary normal to unit length.
            Default: True. When True, returns unit normal and scales velocity accordingly.
            When False, returns unnormalized solution vector.

        min_sc: Minimum number of spacecraft required for analysis.
            Default: 2. Must be ≥ 2. More spacecraft provide better constraints
            and uncertainty estimates.

    Returns:
        tuple: A 3-element tuple containing:

        n_hat (np.ndarray): Boundary normal vector, shape (3,).
            If normalise=True: Unit vector pointing in boundary normal direction
            If normalise=False: Unnormalized solution vector
            Direction convention: From reference region to target region

        V_phase (float): Boundary phase velocity in km/s.
            Positive values indicate motion in the +n⃗ direction
            Negative values indicate motion in the -n⃗ direction
            Magnitude represents speed of boundary motion

        sigma_V (float): Estimated 1-σ uncertainty in phase velocity, km/s.
            Based on the smallest singular value from SVD decomposition
            Larger values indicate less reliable velocity determination
            NaN if uncertainty cannot be estimated

    Raises:
        ValueError: If fewer than min_sc spacecraft are provided, or if
                   spacecraft positions/times are inconsistent.
        RuntimeError: If the SVD solution is degenerate (boundary normal = 0).

    Examples:
        >>> import numpy as np
        >>> from mms_mp import timing_normal

        # Example with 4 MMS spacecraft
        >>> positions = {
        ...     '1': np.array([10000, 5000, -2000]),  # km, GSM
        ...     '2': np.array([10100, 5100, -1900]),
        ...     '3': np.array([9900, 4900, -2100]),
        ...     '4': np.array([10050, 5050, -1950])
        ... }
        >>> crossing_times = {
        ...     '1': 1609459200.0,  # 2021-01-01 00:00:00 UTC
        ...     '2': 1609459202.5,  # 2.5 seconds later
        ...     '3': 1609459198.0,  # 2 seconds earlier
        ...     '4': 1609459201.0   # 1 second later
        ... }
        >>> n_hat, V_phase, sigma_V = timing_normal(positions, crossing_times)
        >>> print(f"Normal: [{n_hat[0]:.3f}, {n_hat[1]:.3f}, {n_hat[2]:.3f}]")
        >>> print(f"Velocity: {V_phase:.1f} ± {sigma_V:.1f} km/s")

        # Minimum case with 2 spacecraft
        >>> pos_2sc = {'1': positions['1'], '2': positions['2']}
        >>> times_2sc = {'1': crossing_times['1'], '2': crossing_times['2']}
        >>> n_hat_2sc, V_2sc, sigma_2sc = timing_normal(pos_2sc, times_2sc)

    Notes:
        - **Accuracy**: 4 spacecraft provide the most robust results. 3 spacecraft
          give good results if well-separated. 2 spacecraft provide limited accuracy.

        - **Geometry**: Spacecraft should be well-separated relative to boundary
          thickness for best results. Collinear configurations reduce accuracy.

        - **Assumptions**: Method assumes planar boundary with constant velocity.
          Curved or accelerating boundaries will introduce systematic errors.

        - **Sign Convention**: The normal direction is determined by the SVD solution
          and may point in either direction. Physical interpretation requires
          additional context (e.g., magnetic field rotation).

        - **Uncertainty**: The uncertainty estimate is approximate and based on
          linear error propagation. Actual uncertainties may be larger due to
          systematic effects.

    References:
        - Russell et al. (1983): The ISEE 1 and 2 Plasma Wave Investigation
        - Schwartz (1998): Shock and Discontinuity Normals, Mach Numbers, and
          Related Parameters, in Analysis Methods for Multi-Spacecraft Data
        - Dunlop et al. (2002): Four-point Cluster application of magnetic field
          analysis tools: The Curlometer
    """
    probes = sorted(set(pos_gsm) & set(t_cross))
    if len(probes) < min_sc:
        raise ValueError(f"Need ≥{min_sc} spacecraft; only have {len(probes)}.")
    # Reference spacecraft = first in sorted list
    p0 = probes[0]
    r0 = pos_gsm[p0]
    t0 = t_cross[p0]

    # Build matrix M · x = 0   where x = [n_x, n_y, n_z, V]^T
    # Physics equation: n⃗ · (r⃗ᵢ - r⃗₀) = V(tᵢ - t₀)
    # Rearranged: n⃗ · dr⃗ - V·dt = 0
    # Matrix row: [drₓ, drᵧ, drᵤ, -dt]
    rows: List[np.ndarray] = []
    for p in probes[1:]:
        dr = pos_gsm[p] - r0          # km
        dt = t_cross[p] - t0          # s
        rows.append(np.hstack((dr, -dt)))
    M = np.vstack(rows)               # shape (N-1, 4)

    # SVD → right singular vector of smallest σ gives solution
    U, S, VT = np.linalg.svd(M)
    x = VT[-1]                        # (4,)
    n = x[:3]
    V = x[3]  # No negative sign - already accounted for in matrix
    if normalise:
        n_norm = np.linalg.norm(n)
        if n_norm == 0:
            raise RuntimeError("Degenerate solution: |n|=0")
        n_hat = n / n_norm
        V = V / n_norm  # Scale velocity consistently
    else:
        n_hat = n

    # Uncertainty:  1/√λ for smallest singular value ≈ σ of x
    # Estimate σ_V via linear propagation (approx)
    if len(S) >= 2:
        # smallest singular value
        sigma = S[-1]
        # relative error in x components ~ σ / largest_singular_value
        rel = sigma / S[0] if S[0] > 0 else np.inf
        sigma_V = abs(V) * rel
    else:
        sigma_V = np.nan

    return n_hat, V, sigma_V


# ------------------------------------------------------------
# Align time series by crossing
# ------------------------------------------------------------
def align_time(t_arr: np.ndarray,
               t_cross_probe: float,
               t_cross_ref: float) -> np.ndarray:
    """
    Shift epoch array so that crossing time aligns with reference.

    Parameters
    ----------
    t_arr        : array of epoch seconds (float64)
    t_cross_probe: crossing time of this probe (sec)
    t_cross_ref  : crossing time of reference probe (sec)

    Returns
    -------
    t_shifted : epoch seconds – same length, shifted
    """
    delta = t_cross_probe - t_cross_ref
    return t_arr - delta


# ------------------------------------------------------------
# Build overlay dictionary (after alignment)
# ------------------------------------------------------------
def stack_aligned(vars_by_probe: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
                  t_cross: Dict[str, float],
                  ref_probe: str,
                  var_keys: Iterable[str]
                  ) -> Dict[str, Dict[str, np.ndarray]]:
    """
    For each requested var_key (e.g. 'B_N', 'N_he'), build a dict:
        out[var_key][probe] → aligned array
    Useful for overlay plots.

    vars_by_probe : nested dict {probe: {var_key: (t, data)}}
    t_cross       : crossing times (sec)
    ref_probe     : which probe is the alignment reference
    var_keys      : iterable of keys to extract

    NOTE: assumes each (t,data) already share the same cadence.
    """
    t_ref, _ = vars_by_probe[ref_probe][next(iter(var_keys))]
    out: Dict[str, Dict[str, np.ndarray]] = {k: {} for k in var_keys}

    for p, pdata in vars_by_probe.items():
        for key in var_keys:
            t, d = pdata[key]
            t_shift = align_time(t, t_cross[p], t_cross[ref_probe])
            out[key][p] = np.vstack((t_shift, d)).T  # save as 2-col array

    return out
