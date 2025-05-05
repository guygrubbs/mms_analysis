# mms_mp/electric.py
# ------------------------------------------------------------
# E×B drift utilities  – NEW MODULE
# ------------------------------------------------------------
# Features
# --------
# 1. `exb_velocity`      – compute (E × B) / |B|²  → km s⁻¹
# 2. `exb_velocity_sync` – resample / interp E & B onto common clock
# 3. `normal_velocity`   – extract V_N from either ExB or bulk-ion,
#                          switch / blend strategies
# 4. Optional spacecraft-potential correction (very lightweight)
# ------------------------------------------------------------
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Literal, Optional

from .data_loader import to_dataframe, resample   # reuse helper

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
MU0 = 4 * np.pi * 1e-7  # (unused for now)
KM_PER_M = 1.0e-3

# ------------------------------------------------------------------
# Core: E×B in km/s
# ------------------------------------------------------------------
def exb_velocity(E_xyz: np.ndarray,
                 B_xyz: np.ndarray,
                 *,
                 unit_E: Literal['V/m', 'mV/m'] = 'mV/m',
                 unit_B: Literal['T', 'nT'] = 'nT') -> np.ndarray:
    """
    Compute v = (E × B) / |B|²  (m/s)  and return in km/s.

    Parameters
    ----------
    E_xyz : (N,3) array – electric field vector
    B_xyz : (N,3) array – magnetic field vector
    unit_E : unit of E_xyz ('V/m' or 'mV/m')
    unit_B : unit of B_xyz ('T'   or 'nT')
    """
    # --- unit conversions ---
    if unit_E == 'mV/m':
        E = E_xyz * 1.0e-3  # → V/m
    elif unit_E == 'V/m':
        E = E_xyz
    else:
        raise ValueError("unit_E must be 'V/m' or 'mV/m'")

    if unit_B == 'nT':
        B = B_xyz * 1.0e-9  # → Tesla
    elif unit_B == 'T':
        B = B_xyz
    else:
        raise ValueError("unit_B must be 'T' or 'nT'")

    # --- cross & magnitude ---
    v_mps = np.cross(E, B) / np.sum(B**2, axis=1, keepdims=True)
    v_kmps = v_mps * KM_PER_M
    return v_kmps


# ------------------------------------------------------------------
# E & B merger / resample helper
# ------------------------------------------------------------------
def exb_velocity_sync(t_E: np.ndarray, E_xyz: np.ndarray,
                      t_B: np.ndarray, B_xyz: np.ndarray,
                      cadence: str = '250ms') -> Tuple[np.ndarray, np.ndarray]:
    """
    *Resample* E & B onto common cadence via pandas, then compute v_exb.

    Returns (t_common, v_kmps  (N,3)).
    """
    # Build DataFrames
    dfE = to_dataframe(t_E, E_xyz[:, :3], ['Ex', 'Ey', 'Ez'])
    dfB = to_dataframe(t_B, B_xyz[:, :3], ['Bx', 'By', 'Bz'])

    dfE = dfE[~dfE.index.isna()]
    dfB = dfB[~dfB.index.isna()]

    # Ensure index uniqueness before resampling
    for df in (dfE, dfB):
        if not df.index.is_unique:
            df.drop_duplicates(inplace=True)                # remove duplicate rows
            df = df.loc[~df.index.duplicated(keep='first')] # remove any still-duplicate index stamps

    dfE_r = resample(dfE, cadence=cadence, method='nearest')
    dfB_r = resample(dfB, cadence=cadence, method='nearest')

    df = pd.concat([dfE_r, dfB_r], axis=1).dropna()
    E_arr = df[['Ex', 'Ey', 'Ez']].values
    B_arr = df[['Bx', 'By', 'Bz']].values

    v_exb = exb_velocity(E_arr, B_arr)  # km/s
    t_common = df.index.values.astype('datetime64[ns]')
    return t_common, v_exb


# ------------------------------------------------------------------
# Simple spacecraft-potential correction
# ------------------------------------------------------------------
def correct_E_for_scpot(E_raw: np.ndarray,
                        sc_pot: np.ndarray,
                        *,
                        gain: float = 1.0) -> np.ndarray:
    """
    Very crude correction: subtract scalar × scp from each E component.
    For MMS, typical factor ~ (dV/dL) ≈ 0.01 mV/m per Volt of sc pot.
    """
    correction = gain * sc_pot[:, None] * 1.0e-3   # V/m if gain given in mV/m per Volt
    return E_raw - correction


# ------------------------------------------------------------------
# Normal-velocity selector / blender
# ------------------------------------------------------------------
def normal_velocity(v_bulk_lmn: np.ndarray,
                    v_exb_lmn: np.ndarray,
                    *,
                    strategy: Literal['prefer_exb',
                                      'prefer_bulk',
                                      'average',
                                      'he_plus_only'] = 'prefer_exb') -> np.ndarray:
    """
    Blend He⁺ bulk and E×B drift **normal component**.

    Handles edge cases when one array is empty or all-NaN.
    `strategy` chooses which source wins when both are finite.
        'prefer_exb'  (default)
        'prefer_bulk'
        'average'      (mean of available)
    """
    # --- extract normal component & ensure 1-D -----------------------
    vN_b = np.atleast_1d(v_bulk_lmn[..., -1])
    vN_e = np.atleast_1d(v_exb_lmn[..., -1])

    # guard: if one source is length-0, replicate the other’s shape
    if vN_b.size == 0 and vN_e.size:
        vN_b = np.full_like(vN_e, np.nan)
    if vN_e.size == 0 and vN_b.size:
        vN_e = np.full_like(vN_b, np.nan)

    # now both arrays have matching shape
    if strategy == 'average':
        out = np.nanmean(np.stack([vN_b, vN_e]), axis=0)
    elif strategy == 'prefer_exb':
        out = np.where(np.isfinite(vN_e), vN_e, vN_b)
    elif strategy == 'prefer_bulk':
        out = np.where(np.isfinite(vN_b), vN_b, vN_e)
    else:
        raise ValueError(f'unknown strategy {strategy}')

    return out
