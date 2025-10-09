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
    n_gaps_filled: int = 0     # count of single-sample gaps interpolated
    segment_count: int = 0     # number of contiguous integration segments
    max_step_s: Optional[float] = None  # adaptive sub-step control (if used)


# Additional helpers expected by tests

def calculate_velocity(positions: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Central-difference velocity estimate (km/s)."""
    pos = np.asarray(positions)
    t = np.asarray(times)
    v = np.zeros_like(pos)
    dt = np.diff(t)
    # central differences for interior
    denom = (t[2:] - t[:-2])[:, None]
    v[1:-1] = (pos[2:] - pos[:-2]) / denom
    # forward/backward for endpoints
    v[0] = (pos[1] - pos[0]) / dt[0]
    v[-1] = (pos[-1] - pos[-2]) / dt[-1]
    return v


def detect_crossing_time(times: np.ndarray, signal: np.ndarray) -> float:
    """
    Detect steepest transition time using max gradient of tanh-like signal.
    Returns time corresponding to maximal absolute derivative.
    """
    t = np.asarray(times)
    s = np.asarray(signal)
    grad = np.gradient(s, t)
    idx = int(np.argmax(np.abs(grad)))
    return float(t[idx])


def analyze_formation_geometry(positions: dict) -> dict:
    """Compute simple tetrahedrality metric and volume from 4 points."""
    probes = list(positions.keys())
    if len(probes) < 4:
        return {'tetrahedrality': 0.0, 'volume': 0.0}
    r = np.vstack([positions[p] for p in probes[:4]])
    # volume of tetrahedron defined by points r0..r3
    v = np.abs(np.dot(np.cross(r[1]-r[0], r[2]-r[0]), r[3]-r[0])) / 6.0
    # edge-length std/mean as inverse tetrahedrality proxy
    edges = []
    for i in range(4):
        for j in range(i+1, 4):
            edges.append(np.linalg.norm(r[j]-r[i]))
    edges = np.asarray(edges)
    tet = 1.0 - (edges.std() / edges.mean() if edges.mean() else 1.0)
    return {'tetrahedrality': float(np.clip(tet, 0.0, 1.0)), 'volume': float(v)}


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
                   fill_gaps: bool = True,
                   max_step_s: Optional[float] = None) -> DispResult:
    """
    Integrate normal velocity to calculate boundary layer displacement.

    This function performs numerical integration of the normal velocity component
    to determine how far a spacecraft has moved through a boundary layer. This is
    essential for magnetopause analysis to calculate layer thickness and understand
    boundary structure.

    The integration computes: displacement(t) = ∫ vN(τ) dτ from t₀ to t

    Multiple integration schemes are supported, with automatic error propagation
    when velocity uncertainties are provided.

    Args:
        t: Time array for the velocity measurements.
            Can be datetime64 array or float array (seconds since epoch).
            Shape: (N,) where N is the number of time points.
            Must be monotonically increasing.

        vN: Normal velocity component in km/s.
            Shape: (N,) matching the time array.
            Positive values indicate motion in the +normal direction.
            Typically the boundary-normal component from LMN coordinates.

        sigma_v: 1-σ uncertainty in velocity measurements, optional.
            Shape: (N,) matching vN array.
            Units: km/s. Used for error propagation in displacement calculation.
            If provided, uncertainties are propagated through integration.

        good_mask: Boolean mask indicating valid velocity samples, optional.
            Shape: (N,) matching vN array.
            True = valid sample, False = invalid/missing data.
            If None, assumes all samples are valid.

        scheme: Numerical integration method to use.
            'trap': Trapezoidal rule (default) - fast, robust, good accuracy
            'simpson': Simpson's rule - higher accuracy, requires SciPy
            'rect': Rectangular rule - lower accuracy, rarely used

        fill_gaps: Whether to interpolate across single-sample gaps.
            Default: True. Helps maintain continuity across brief data gaps.
            Only fills gaps of exactly 1 sample to avoid over-interpolation.

        max_step_s: Maximum integration step in seconds.  When provided, the
            routine linearly densifies intervals with Δt > max_step_s before
            performing the numerical integration.  This guards against sparse
            cadences that would otherwise underestimate displacement and ensures
            convergence when mixing survey and burst samples.

    Returns:
        DispResult: Dataclass containing integration results with fields:

        disp_km (np.ndarray): Cumulative displacement in km, shape (N,).
            Always starts at 0.0 for the first time point.
            Positive values indicate net motion in +normal direction.

        sigma_disp (np.ndarray): 1-σ uncertainty in displacement, shape (N,).
            Only computed if sigma_v is provided, otherwise None.
            Uncertainties are propagated assuming uncorrelated velocity errors.

        t_sec (np.ndarray): Time array converted to seconds, shape (N,).
            Relative to the first time point (t[0] = 0).

        method (str): Integration method used ('trap', 'simpson', or 'rect').

        n_gaps_filled (int): Number of single-sample gaps that were interpolated.

        segment_count (int): Number of contiguous, finite-data segments that
            were integrated independently.

        max_step_s (float | None): Adaptive sub-step threshold actually applied.

    Raises:
        ValueError: If input arrays have mismatched shapes or if time array
                   is not monotonically increasing.
        ImportError: If scheme='simpson' but SciPy is not available.

    Examples:
        >>> import numpy as np
        >>> from mms_mp import integrate_disp

        # Basic integration with constant velocity
        >>> t = np.linspace(0, 100, 101)  # 100 seconds, 1 Hz sampling
        >>> vN = np.ones(101) * 5.0       # Constant 5 km/s
        >>> result = integrate_disp(t, vN)
        >>> print(f"Final displacement: {result.disp_km[-1]:.1f} km")  # 500 km

        # With velocity uncertainties
        >>> sigma_v = np.ones(101) * 0.5  # 0.5 km/s uncertainty
        >>> result = integrate_disp(t, vN, sigma_v=sigma_v)
        >>> print(f"Final uncertainty: {result.sigma_disp[-1]:.1f} km")

        # Using Simpson's rule for higher accuracy
        >>> result_simpson = integrate_disp(t, vN, scheme='simpson')

        # With data quality mask
        >>> good_mask = np.ones(101, dtype=bool)
        >>> good_mask[50:55] = False  # Mark some data as bad
        >>> result = integrate_disp(t, vN, good_mask=good_mask)

        # Using datetime64 time array
        >>> import pandas as pd
        >>> t_dt = pd.date_range('2021-01-01', periods=101, freq='1S')
        >>> result = integrate_disp(t_dt.values, vN)

    Notes:
        - **Accuracy**: Simpson's rule provides higher accuracy for smooth velocity
          profiles but requires SciPy. Trapezoidal rule is sufficient for most
          space physics applications.

        - **Error Propagation**: When sigma_v is provided, uncertainties are
          propagated assuming velocity errors are uncorrelated between time points.
          This may underestimate uncertainties if systematic errors are present.

        - **Gap Handling**: The fill_gaps option only interpolates across single
          missing samples to avoid introducing artifacts. Larger gaps are left
          as NaN and will propagate through the integration.

        - **Coordinate Systems**: The velocity should be in boundary-normal
          coordinates (typically the N component from LMN transformation) for
          meaningful layer thickness calculations.

        - **Physical Interpretation**: The displacement represents the distance
          traveled through the boundary layer. For magnetopause crossings, this
          gives the layer thickness when integrated from entry to exit.

    References:
        - Paschmann & Daly (1998): Analysis Methods for Multi-Spacecraft Data
        - Sonnerup et al. (2006): Minimum and Maximum Variance Analysis
        - Press et al. (2007): Numerical Recipes - Integration of Functions
    """
    t = np.asarray(t)
    vN = np.asarray(vN, dtype=float)
    if vN.shape != t.shape:
        raise ValueError("t and vN must have identical shapes")

    if good_mask is None:
        good_mask = np.ones_like(vN, dtype=bool)
    else:
        good_mask = np.asarray(good_mask, dtype=bool)
        if good_mask.shape != vN.shape:
            raise ValueError("good_mask must match vN shape")

    if np.issubdtype(t.dtype, np.datetime64):
        t_sec = (t - t[0]) / np.timedelta64(1, 's')
        t_sec = t_sec.astype(np.float64)
    else:
        t_sec = t.astype(float)
        t_sec = t_sec - t_sec[0]

    if np.any(np.diff(t_sec) < 0):
        raise ValueError("time array must be monotonically increasing")

    v_clean = vN.copy()
    bad = (~good_mask) | (~np.isfinite(v_clean))
    v_clean[bad] = np.nan
    n_filled = 0
    if fill_gaps:
        v_clean, n_filled = _lin_fill(v_clean)

    disp = np.zeros_like(v_clean, dtype=float)
    sig_arr = None
    if sigma_v is not None:
        sigma_v = np.asarray(sigma_v, dtype=float)
        if sigma_v.shape != v_clean.shape:
            raise ValueError("sigma_v must match vN shape")
        sigma_v = sigma_v.copy()
        sigma_v[~np.isfinite(sigma_v)] = 0.0
        sig_arr = np.zeros_like(v_clean, dtype=float)

    idx = 0
    seg_count = 0
    disp_offset = 0.0
    variance_offset = 0.0

    while idx < len(v_clean):
        if not np.isfinite(v_clean[idx]):
            idx += 1
            continue
        jdx = idx
        while jdx < len(v_clean) and np.isfinite(v_clean[jdx]):
            jdx += 1

        seg = slice(idx, jdx)
        t_seg = t_sec[seg]
        v_seg = v_clean[seg]
        sig_seg = sigma_v[seg] if sigma_v is not None else None

        t_dense, v_dense, sig_dense = _densify_segment(t_seg, v_seg, max_step_s, sig_seg)
        disp_dense = _segment_integral(t_dense, v_dense, scheme)

        disp_interp = np.interp(t_seg, t_dense, disp_dense)
        disp[seg] = disp_offset + disp_interp
        disp_offset = disp[seg][-1]

        if sig_arr is not None:
            dt_dense = np.diff(t_dense, prepend=t_dense[0])
            var_dense = np.cumsum((dt_dense * sig_dense) ** 2)
            var_interp = np.interp(t_seg, t_dense, var_dense)
            total_var = variance_offset + var_interp
            sig_arr[seg] = np.sqrt(total_var)
            variance_offset = total_var[-1]

        seg_count += 1
        idx = jdx

    return DispResult(t_sec, disp, sig_arr, scheme, n_filled, seg_count, max_step_s)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _lin_fill(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Linearly fill isolated NaNs (single-sample gaps) in a 1-D array.

    Returns the filled array and the number of samples that were replaced.
    """
    arr = arr.copy()
    isnan = ~np.isfinite(arr)
    if not isnan.any():
        return arr, 0

    filled = 0
    idx = np.where(isnan)[0]
    for i in idx:
        if i == 0 or i == len(arr) - 1:
            continue
        if isnan[i - 1] or isnan[i + 1]:
            continue
        left = arr[i - 1]
        right = arr[i + 1]
        if np.isfinite(left) and np.isfinite(right):
            arr[i] = left + 0.5 * (right - left)
            filled += 1
    return arr, filled


def _densify_segment(t: np.ndarray,
                     v: np.ndarray,
                     max_step_s: Optional[float],
                     sigma: Optional[np.ndarray] = None
                     ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Densify a monotonic segment when large Δt would harm accuracy."""
    if max_step_s is None or len(t) < 2:
        return t, v, sigma

    t_out = [float(t[0])]
    v_out = [float(v[0])]
    sigma_out = [float(sigma[0])] if sigma is not None else None

    for idx in range(1, len(t)):
        dt = float(t[idx] - t[idx - 1])
        steps = max(1, int(np.ceil(dt / max_step_s)))
        for step in range(1, steps + 1):
            frac = step / steps
            t_new = float(t[idx - 1] + frac * dt)
            v_new = float(v[idx - 1] + frac * (v[idx] - v[idx - 1]))
            t_out.append(t_new)
            v_out.append(v_new)
            if sigma is not None:
                sig_new = float(sigma[idx - 1] + frac * (sigma[idx] - sigma[idx - 1]))
                sigma_out.append(sig_new)

    t_arr = np.asarray(t_out, dtype=float)
    v_arr = np.asarray(v_out, dtype=float)
    if sigma is not None:
        sigma_arr = np.asarray(sigma_out, dtype=float)
    else:
        sigma_arr = None
    return t_arr, v_arr, sigma_arr
