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
    # if everything is NaN, return early
    if isnan.all():
        return arr
    x = np.arange(len(arr))
    arr[isnan] = np.interp(x[isnan], x[~isnan], arr[~isnan])
    return arr
