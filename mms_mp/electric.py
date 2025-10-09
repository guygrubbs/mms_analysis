# mms_mp/electric.py
# ---------------------------------------------------------------------
# E×B drift utilities
# ---------------------------------------------------------------------
# 1. exb_velocity        – (E × B)/|B|²  → km s⁻¹
# 2. exb_velocity_sync   – resample/interp E & B, then E×B
# 3. normal_velocity     – decide which V N to use (bulk / ExB / blend)
# 4. correct_E_for_scpot – very lightweight SC-pot correction
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Convenience unit converter expected by tests
_DEF_E_UNITS = ('V/m', 'mV/m')
_DEF_B_UNITS = ('T', 'nT')

def convert_electric_field_units(E: np.ndarray,
                                 from_unit: str,
                                 to_unit: str) -> np.ndarray:
    """Convert electric field units between 'V/m' and 'mV/m'."""
    if from_unit not in _DEF_E_UNITS or to_unit not in _DEF_E_UNITS:
        raise ValueError("Supported units: 'V/m' or 'mV/m'")
    if from_unit == to_unit:
        return np.asarray(E)
    if from_unit == 'mV/m' and to_unit == 'V/m':
        return np.asarray(E) * 1e-3
    if from_unit == 'V/m' and to_unit == 'mV/m':
        return np.asarray(E) * 1e3
    raise ValueError('Unsupported conversion')


def calculate_exb_drift(E_field: np.ndarray,
                        B_field: np.ndarray,
                        *,
                        unit_E: Literal['V/m', 'mV/m'] = 'mV/m',
                        unit_B: Literal['T', 'nT'] = 'nT') -> np.ndarray:
    """Alias for exb_velocity with scalar/vector convenience."""
    return exb_velocity(np.atleast_2d(E_field), np.atleast_2d(B_field),
                        unit_E=unit_E, unit_B=unit_B).squeeze()


def calculate_convection_field(velocity_km_s: np.ndarray,
                               B_nT: np.ndarray) -> np.ndarray:
    """
    Motional electric field: E = - v × B.
    Inputs: v in km/s, B in nT. Output: E in mV/m.
    """
    v_m_s = np.asarray(velocity_km_s) * 1e3
    B_T = np.asarray(B_nT) * 1e-9
    E_V_m = -np.cross(v_m_s, B_T)
    return E_V_m * 1e3  # → mV/m

from .data_loader import to_dataframe, resample  # helper functions

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
KM_PER_M = 1.0e-3          # m → km

# ------------------------------------------------------------------
# Core: E×B in km s⁻¹
# ------------------------------------------------------------------
def exb_velocity(
    E_xyz: np.ndarray,
    B_xyz: np.ndarray,
    *,
    unit_E: Literal['V/m', 'mV/m'] = 'mV/m',
    unit_B: Literal['T', 'nT'] = 'nT',
    min_b: Optional[float] = None,
    return_quality: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Calculate E×B drift velocity for charged particles in crossed electric and magnetic fields.

    The E×B drift is a fundamental plasma physics phenomenon where charged particles
    drift perpendicular to both electric and magnetic fields with a velocity that is
    independent of particle charge, mass, and energy. This drift is given by:

        v⃗_E×B = (E⃗ × B⃗) / |B⃗|²

    The direction follows the right-hand rule for the cross product E⃗ × B⃗, and the
    magnitude is |E|/|B| when E⃗ ⊥ B⃗.

    Args:
        E_xyz: Electric field vectors in any Cartesian coordinate system.
            Shape: (N, 3) where N is the number of time points or spatial locations.
            Each row contains [Ex, Ey, Ez] components.
            Units: Specified by unit_E parameter (V/m or mV/m).

        B_xyz: Magnetic field vectors in the same coordinate system as E_xyz.
            Shape: (N, 3) matching E_xyz dimensions.
            Each row contains [Bx, By, Bz] components.
            Units: Specified by unit_B parameter (T or nT).

        unit_E: Units of the electric field input.
            'V/m': Volts per meter (SI units)
            'mV/m': Millivolts per meter (common in space physics)
            Default: 'mV/m'

        unit_B: Units of the magnetic field input.
            'T': Tesla (SI units)
            'nT': Nanotesla (common in space physics)
            Default: 'nT'

    Returns:
        np.ndarray: E×B drift velocity vectors, shape (N, 3).
            Each row contains [vx, vy, vz] components in km/s.
            Direction: Perpendicular to both E⃗ and B⃗ following right-hand rule.
            Magnitude: |E|/|B| when fields are perpendicular.

    Raises:
        ValueError: If unit_E is not 'V/m' or 'mV/m', or if unit_B is not 'T' or 'nT'.

    Examples:
        >>> import numpy as np
        >>> from mms_mp import exb_velocity

        # Simple case: E in +X, B in +Z
        >>> E = np.array([[1.0, 0.0, 0.0]])  # 1 mV/m in X direction
        >>> B = np.array([[0.0, 0.0, 1.0]])  # 1 nT in Z direction
        >>> v_exb = exb_velocity(E, B)
        >>> print(f"E×B velocity: {v_exb[0]} km/s")  # [0, -1000, 0]

        # Multiple time points
        >>> n_points = 100
        >>> E_time = np.column_stack([
        ...     np.ones(n_points),           # Constant Ex = 1 mV/m
        ...     0.5 * np.sin(np.linspace(0, 2*np.pi, n_points)),  # Varying Ey
        ...     np.zeros(n_points)           # Ez = 0
        ... ])
        >>> B_time = np.column_stack([
        ...     np.zeros(n_points),          # Bx = 0
        ...     np.zeros(n_points),          # By = 0
        ...     np.ones(n_points) * 50       # Constant Bz = 50 nT
        ... ])
        >>> v_exb_time = exb_velocity(E_time, B_time, unit_E='mV/m', unit_B='nT')

        # Using SI units
        >>> E_SI = np.array([[0.001, 0.0, 0.0]])  # 1 mV/m = 0.001 V/m
        >>> B_SI = np.array([[0.0, 0.0, 1e-9]])   # 1 nT = 1e-9 T
        >>> v_exb_SI = exb_velocity(E_SI, B_SI, unit_E='V/m', unit_B='T')

    Notes:
        - **Physics**: The E×B drift is independent of particle species, making it
          a bulk plasma motion. All particles (electrons, protons, heavy ions) drift
          with the same velocity.

        - **Direction**: The drift direction follows the right-hand rule for E⃗ × B⃗.
          For E⃗ pointing East and B⃗ pointing North, particles drift downward.

        - **Magnitude**: When E⃗ ⊥ B⃗, the drift speed is simply |E|/|B|.
          For typical magnetospheric values (E ~ 1 mV/m, B ~ 50 nT),
          this gives v ~ 20 km/s.

        - **Coordinate Systems**: Input vectors should be in the same coordinate
          system. Common choices include GSM, GSE, or LMN coordinates.

        - **Numerical Stability**: The calculation is stable for typical space
          physics field magnitudes. Very weak magnetic fields (|B| << 1 nT) may
          lead to numerical issues.

    References:
        - Chen, F. F. (2016): Introduction to Plasma Physics and Controlled Fusion
        - Baumjohann, W. & Treumann, R. A. (1996): Basic Space Plasma Physics
        - Kivelson, M. G. & Russell, C. T. (1995): Introduction to Space Physics
    """
    # --- unit conversions ------------------------------------------
    if unit_E == 'mV/m':
        E = np.asarray(E_xyz, dtype=float) * 1.0e-3  # → V/m
    elif unit_E == 'V/m':
        E = np.asarray(E_xyz, dtype=float)
    else:
        raise ValueError("unit_E must be 'V/m' or 'mV/m'")

    if unit_B == 'nT':
        B = np.asarray(B_xyz, dtype=float) * 1.0e-9  # → T
        min_b_T = None if min_b is None else float(min_b) * 1.0e-9
    elif unit_B == 'T':
        B = np.asarray(B_xyz, dtype=float)
        min_b_T = None if min_b is None else float(min_b)
    else:
        raise ValueError("unit_B must be 'T' or 'nT'")

    # Normalize to 2-D arrays for vectorised handling
    E = np.atleast_2d(E)
    B = np.atleast_2d(B)
    if E.shape != B.shape:
        raise ValueError("E and B arrays must share the same shape")

    quality = np.isfinite(E).all(axis=1) & np.isfinite(B).all(axis=1)

    # Compute |B| and guard against unrealistically weak fields
    B_mag = np.linalg.norm(B, axis=1)
    if min_b_T is not None:
        quality &= B_mag >= min_b_T

    denom = B_mag**2
    denom = np.where(denom == 0.0, np.nan, denom)

    v_mps = np.full_like(E, np.nan, dtype=float)
    valid_idx = np.where(quality)[0]
    if valid_idx.size:
        cross = np.cross(E[valid_idx], B[valid_idx])
        v_mps[valid_idx] = cross / denom[valid_idx, None]

    v_km_s = v_mps * KM_PER_M
    if v_km_s.shape[0] == 1:
        v_km_s = v_km_s[0]
        quality = np.asarray(quality[0])

    if return_quality:
        return v_km_s, quality
    return v_km_s


# ------------------------------------------------------------------
# E & B merger / resample helper
# ------------------------------------------------------------------
def exb_velocity_sync(t_E: np.ndarray, E_xyz: np.ndarray,
                      t_B: np.ndarray, B_xyz: np.ndarray,
                      *,
                      cadence: str = '250ms'
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample E and B to a common regular clock, then compute v_exb.

    Returns
    -------
    t_common : np.ndarray[datetime64[ns]]
    v_exb    : (N, 3) array  [km s⁻¹]
    """
    if len(t_E) == 0 or len(t_B) == 0:                 # ▲ NEW
        return np.empty(0, dtype='datetime64[ns]'), np.empty((0, 3))

    dfE = to_dataframe(t_E, E_xyz[:, :3], ['Ex', 'Ey', 'Ez'])
    dfB = to_dataframe(t_B, B_xyz[:, :3], ['Bx', 'By', 'Bz'])

    dfE_r = resample(dfE, cadence=cadence, method='nearest')
    dfB_r = resample(dfB, cadence=cadence, method='nearest')

    # use an **inner join** to avoid the concat/union-index problem
    dfE_r.index.name = dfB_r.index.name = "utc"      # <── add this line
    df = dfE_r.join(dfB_r, how='inner')

    E_arr = df[['Ex', 'Ey', 'Ez']].values
    B_arr = df[['Bx', 'By', 'Bz']].values

    v_exb   = exb_velocity(E_arr, B_arr)          # km s⁻¹
    t_index = df.index.values.astype('datetime64[ns]')
    return t_index, v_exb


# ------------------------------------------------------------------
# Simple spacecraft-potential correction (optional)
# ------------------------------------------------------------------
def correct_E_for_scpot(E_raw: np.ndarray,
                        sc_pot: np.ndarray,
                        *,
                        gain: float = 0.01  # mV m⁻¹ per Volt
                        ) -> np.ndarray:
    """
    Very crude correction: subtract gain·V_sc from each E component.
    """
    correction = gain * sc_pot[:, None] * 1.0e-3   # → V/m
    return E_raw - correction


# ------------------------------------------------------------------
# Normal-velocity selector / blender
# ------------------------------------------------------------------
@dataclass
class NormalVelocityBlendResult:
    """Container describing the outcome of :func:`normal_velocity`."""

    vn: np.ndarray
    source: np.ndarray
    exb_valid: np.ndarray
    bulk_valid: np.ndarray


def normal_velocity(
    v_bulk_lmn: np.ndarray,
    v_exb_lmn: np.ndarray,
    *,
    strategy: Literal['prefer_exb', 'prefer_bulk', 'average'] = 'prefer_exb',
    exb_quality: Optional[np.ndarray] = None,
    b_mag_nT: Optional[np.ndarray] = None,
    min_b_nT: float = 0.5,
    return_metadata: bool = False,
) -> Union[np.ndarray, NormalVelocityBlendResult]:
    """
    Combine the L-M-N vectors and output **V_N** (1-D).

    • If one source is entirely NaN the other is used automatically.    ▲ CHG
      This prevents later crashes when optional data are absent.
    """
    vN_b = np.atleast_1d(np.asarray(v_bulk_lmn)[..., -1])
    vN_e = np.atleast_1d(np.asarray(v_exb_lmn)[..., -1])

    if vN_b.size == 0 and vN_e.size:
        vN_b = np.full_like(vN_e, np.nan)
    if vN_e.size == 0 and vN_b.size:
        vN_e = np.full_like(vN_b, np.nan)

    bulk_valid = np.isfinite(vN_b)

    if exb_quality is None:
        exb_valid = np.isfinite(vN_e)
    else:
        exb_valid = np.asarray(exb_quality, dtype=bool)
        if exb_valid.shape != vN_e.shape:
            exb_valid = np.broadcast_to(exb_valid, vN_e.shape)

    if b_mag_nT is not None:
        b_arr = np.asarray(b_mag_nT, dtype=float)
        if b_arr.shape != vN_e.shape:
            b_arr = np.broadcast_to(b_arr, vN_e.shape)
        exb_valid &= np.isfinite(b_arr) & (b_arr >= float(min_b_nT))

    out = np.full_like(vN_b, np.nan, dtype=float)
    source = np.full(vN_b.shape, 'none', dtype=object)

    if strategy == 'average':
        weights = np.stack([bulk_valid.astype(float), exb_valid.astype(float)])
        values = np.stack([vN_b, vN_e])
        weight_sum = weights.sum(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            avg = np.nansum(values * weights, axis=0) / weight_sum
        out = np.where(weight_sum > 0, avg, np.nan)
        both_valid = bulk_valid & exb_valid
        source[both_valid] = 'average'
        source[~both_valid & exb_valid] = 'exb'
        source[~both_valid & bulk_valid] = 'bulk'
    elif strategy == 'prefer_exb':
        use_exb = exb_valid
        out[use_exb] = vN_e[use_exb]
        source[use_exb] = 'exb'
        fallback = ~use_exb & bulk_valid
        out[fallback] = vN_b[fallback]
        source[fallback] = 'bulk'
    elif strategy == 'prefer_bulk':
        use_bulk = bulk_valid
        out[use_bulk] = vN_b[use_bulk]
        source[use_bulk] = 'bulk'
        fallback = ~use_bulk & exb_valid
        out[fallback] = vN_e[fallback]
        source[fallback] = 'exb'
    else:
        raise ValueError(f'unknown strategy {strategy}')

    if not np.isfinite(out).any() and bulk_valid.any():
        out = np.where(bulk_valid, vN_b, out)
        source = np.where(bulk_valid, 'bulk', source)

    if return_metadata:
        return NormalVelocityBlendResult(out, source, exb_valid, bulk_valid)
    return out
