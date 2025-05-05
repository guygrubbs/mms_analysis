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
    Solve for boundary normal **n** and phase speed **V** using
    the multi-spacecraft timing method (SVD formulation).

    Parameters
    ----------
    pos_gsm  : dict {probe: r_vec (km GSM)}
    t_cross  : dict {probe: crossing time (sec since epoch)}
    normalise: whether to unit-normalise n̂ (default True)
    min_sc   : minimum spacecraft required (default 2)

    Returns
    -------
    n_hat     : 3-vector (unit) boundary normal
    V_phase   : phase speed (km/s)  (positive along n̂)
    sigma_V   : 1-σ uncertainty in V (km/s) estimated from SVD
    """
    probes = sorted(set(pos_gsm) & set(t_cross))
    if len(probes) < min_sc:
        raise ValueError(f"Need ≥{min_sc} spacecraft; only have {len(probes)}.")
    # Reference spacecraft = first in sorted list
    p0 = probes[0]
    r0 = pos_gsm[p0]
    t0 = t_cross[p0]

    # Build matrix M · x = 0   where x = [n_x, n_y, n_z, V]^T
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
    V = -x[3]
    if normalise:
        n_norm = np.linalg.norm(n)
        if n_norm == 0:
            raise RuntimeError("Degenerate solution: |n|=0")
        n_hat = n / n_norm
        V *= n_norm
    else:
        n_hat = n

    # Uncertainty:  1/√λ for smallest singular value ≈ σ of x
    # Estimate σ_V via linear propagation (approx)
    if len(S) >= 3:
        # smallest non-zero singular
        sigma = S[-1]
        # relative error in x components ~ σ / Σ
        rel = sigma / S.sum()
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
