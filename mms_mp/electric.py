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

import numpy as np
import pandas as pd
from typing import Tuple, Literal, Optional

from .data_loader import to_dataframe, resample  # helper functions

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
KM_PER_M = 1.0e-3          # m → km

# ------------------------------------------------------------------
# Core: E×B in km s⁻¹
# ------------------------------------------------------------------
def exb_velocity(E_xyz: np.ndarray,
                 B_xyz: np.ndarray,
                 *,
                 unit_E: Literal['V/m', 'mV/m'] = 'mV/m',
                 unit_B: Literal['T', 'nT']     = 'nT') -> np.ndarray:
    """
    v_exb = (E × B) / |B|²    [km s⁻¹]

    Parameters
    ----------
    E_xyz : (N, 3)  electric-field vector
    B_xyz : (N, 3)  magnetic-field vector
    unit_E : 'V/m'  or 'mV/m'
    unit_B : 'T'    or 'nT'
    """
    # --- unit conversions ------------------------------------------
    if unit_E == 'mV/m':
        E = E_xyz * 1.0e-3          # → V/m
    elif unit_E == 'V/m':
        E = E_xyz
    else:
        raise ValueError("unit_E must be 'V/m' or 'mV/m'")

    if unit_B == 'nT':
        B = B_xyz * 1.0e-9          # → T
    elif unit_B == 'T':
        B = B_xyz
    else:
        raise ValueError("unit_B must be 'T' or 'nT'")

    # --- E × B -----------------------------------------------------
    v_mps  = np.cross(E, B) / np.sum(B**2, axis=1, keepdims=True)
    return v_mps * KM_PER_M          # → km s⁻¹


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
def normal_velocity(v_bulk_lmn: np.ndarray,
                    v_exb_lmn : np.ndarray,
                    *,
                    strategy: Literal['prefer_exb',
                                      'prefer_bulk',
                                      'average'] = 'prefer_exb'
                    ) -> np.ndarray:
    """
    Combine the L-M-N vectors and output **V_N** (1-D).

    • If one source is entirely NaN the other is used automatically.    ▲ CHG
      This prevents later crashes when optional data are absent.
    """
    vN_b = np.atleast_1d(v_bulk_lmn[..., -1])
    vN_e = np.atleast_1d(v_exb_lmn[...,  -1])

    # ensure shapes match
    if vN_b.size == 0 and vN_e.size:
        vN_b = np.full_like(vN_e, np.nan)
    if vN_e.size == 0 and vN_b.size:
        vN_e = np.full_like(vN_b, np.nan)

    # ▲ CHG — automatic fallback if chosen source is all-NaN
    if strategy == 'average':
        out = np.nanmean(np.stack([vN_b, vN_e]), axis=0)
    elif strategy == 'prefer_exb':
        out = np.where(np.isfinite(vN_e), vN_e, vN_b)
    elif strategy == 'prefer_bulk':
        out = np.where(np.isfinite(vN_b), vN_b, vN_e)
    else:
        raise ValueError(f'unknown strategy {strategy}')

    # if both arrays were completely NaN, fall back to bulk
    if not np.isfinite(out).any():
        out = vN_b

    return out
