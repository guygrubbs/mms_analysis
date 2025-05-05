# mms_mp/motion.py
# ------------------------------------------------------------
# Boundary-normal velocity + displacement integration (enhanced)
# ------------------------------------------------------------
# • normal_velocity()  : rotate any vector → N-component
# • integrate_disp()   : cumulative displacement with selectable
#                        scheme ('trap', 'simpson', 'rect')
# • Confidence/uncertainty propagation via σ_v  (optional)
# • Handles quality masks (skips gaps, linear-fills if desired)
# ------------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

try:
    from scipy.integrate import cumulative_trapezoid, simpson
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    from numpy import cumtrapz as cumulative_trapezoid  # fallback trapz only

# ------------------------------------------------------------------
# Dataclass to return results neatly
# ------------------------------------------------------------------
@dataclass
class DispResult:
    t_sec: np.ndarray          # seconds relative to start
    disp_km: np.ndarray        # displacement (km)
    sigma_km: Optional[np.ndarray] = None   # 1-σ uncertainty (km) if provided
    scheme: str = 'trap'


# ------------------------------------------------------------------
# Rotate → V_N
# ------------------------------------------------------------------
def normal_velocity(v_xyz: np.ndarray,
                    R_lmn: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    v_xyz : (N,3)  – velocity in GSM/GSE (km/s)
    R_lmn : (3,3)  – rows = L, M, N  rotation matrix

    Returns
    -------
    vN : 1-D array (km/s)  – N-component of velocity
    """
    return (R_lmn @ v_xyz.T).T[:, 2]   # take N row


# ------------------------------------------------------------------
# Internal: integrate single segment
# ------------------------------------------------------------------
def _segment_integral(t: np.ndarray,
                      v: np.ndarray,
                      scheme: Literal['trap', 'simpson', 'rect']) -> np.ndarray:
    """
    Returns cumulative integral (km) over *valid* segment (no NaNs).
    """
    if scheme == 'rect':
        dt = np.diff(t)
        return np.concatenate([[0.0], np.cumsum(v[:-1] * dt)])
    if scheme == 'simpson' and SCIPY_OK:
        # Simpson needs even number; integrate incremental
        out = np.zeros_like(v)
        for i in range(2, len(v)):
            seg = simpson(v[:i+1], t[:i+1])
            out[i] = seg
        return out
    # default trapezoid
    return cumulative_trapezoid(v, t, initial=0.0)


# ------------------------------------------------------------------
# Public integration wrapper
# ------------------------------------------------------------------
def integrate_disp(t: np.ndarray,
                   vN: np.ndarray,
                   *,
                   sigma_v: Optional[np.ndarray] = None,
                   good_mask: Optional[np.ndarray] = None,
                   scheme: Literal['trap', 'simpson', 'rect'] = 'trap',
                   fill_gaps: bool = True) -> DispResult:
    """
    Integrate normal velocity → displacement.

    Parameters
    ----------
    t        : epoch array (datetime64 or float seconds)
    vN       : normal velocity (km/s)
    sigma_v  : 1-σ uncertainty for vN (same length) [optional]
    good_mask: boolean – True = valid sample.  If None, assume all good.
    scheme   : 'trap'  (cumulative trapezoid – fast, robust)
               'simpson' (requires SciPy, more accurate)
               'rect'  (left-rectangular – rarely needed)
    fill_gaps: If True, linear-interpolate vN across single-sample gaps.

    Returns
    -------
    DispResult dataclass.
    """
    if good_mask is None:
        good_mask = np.ones_like(vN, dtype=bool)

    # Convert times → seconds
    if np.issubdtype(t.dtype, np.datetime64):
        t_sec = (t - t[0]) / np.timedelta64(1, 's')
        t_sec = t_sec.astype(np.float64)
    else:
        t_sec = t - t[0]

    # Make copy for fill
    v_clean = vN.copy()
    if fill_gaps:
        # detect isolated NaNs/bad and linearly interp
        bad = ~good_mask | ~np.isfinite(v_clean)
        v_clean[bad] = np.nan
        v_clean = _lin_fill(v_clean)

    # Segment integration: break where NaNs remain
    disp = np.zeros_like(v_clean)
    if sigma_v is not None:
        sigma_v = np.asarray(sigma_v)
        sig_arr = np.zeros_like(v_clean)

    idx = 0
    while idx < len(v_clean):
        if not np.isfinite(v_clean[idx]):
            idx += 1
            continue
        # find next NaN/bad break
        jdx = idx
        while jdx < len(v_clean) and np.isfinite(v_clean[jdx]):
            jdx += 1
        seg = slice(idx, jdx)
        disp_seg = _segment_integral(t_sec[seg], v_clean[seg], scheme)
        disp[seg] = disp[seg.start-1] if seg.start > 0 else 0.0
        disp[seg] += disp_seg
        if sigma_v is not None:
            # propagate: σ_disp = sqrt( Σ (dt * σ_v)^2 )
            dt = np.diff(t_sec[seg], prepend=t_sec[seg][0])
            sig_seg = np.sqrt(np.cumsum((dt * sigma_v[seg])**2))
            sig_arr[seg] = sig_arr[seg.start-1] if seg.start > 0 else 0.0
            sig_arr[seg] += sig_seg
        idx = jdx

    return DispResult(t_sec, disp, sig_arr if sigma_v is not None else None, scheme)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _lin_fill(arr: np.ndarray) -> np.ndarray:
    """
    Simple linear fill of NaNs in a 1-D array (in-place safe).
    """
    isnan = ~np.isfinite(arr)
    if not isnan.any():
        return arr
    x = np.arange(len(arr))
    arr[isnan] = np.interp(x[isnan], x[~isnan], arr[~isnan])
    return arr
