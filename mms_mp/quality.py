# mms_mp/quality.py
# ------------------------------------------------------------
# Automated sample-quality handling  – NEW MODULE
# ------------------------------------------------------------
# MMS Level-2 CDFs include “quality” or “status” bit-fields
# for several instruments.  Systematically masking / fixing
# them here keeps the science steps cleaner.
#
# Supported today
# ---------------
# • FPI-DIS / DES   (ions / electrons)
#     var: *_quality_flag
#     0 = good, 1 = suspect, 2 = bad
#
# • HPCA
#     var: *_status_flag
#     bit-decoded per HPCA guide;   we treat any non-zero as bad.
#
# • Generic boolean mask helpers
# ------------------------------------------------------------
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Iterable

# ------------------------------------------------------------------
# FPI Quality
# ------------------------------------------------------------------
def fpi_good_mask(qflag: np.ndarray,
                  *, accept_levels: Iterable[int] = (0,)) -> np.ndarray:
    """
    Parameters
    ----------
    qflag : 1-D array of uint8 (FPI quality flag per CDF docs)
    accept_levels :  tuple/list of integers deemed "usable"
                     default (0,)  == only perfect data

    Returns  – boolean mask (True = sample accepted)
    """
    return np.isin(qflag, list(accept_levels))


# ------------------------------------------------------------------
# HPCA Status
# ------------------------------------------------------------------
def hpca_good_mask(status_flag: np.ndarray,
                   *, accept_zero_only: bool = True) -> np.ndarray:
    """
    HPCA status bit-field (uint32).
    If accept_zero_only is True, any non-zero bit is bad.
    """
    if accept_zero_only:
        return status_flag == 0
    # else: could add bit-level nuance later
    return np.ones_like(status_flag, dtype=bool)


# ------------------------------------------------------------------
# Apply mask to data arrays
# ------------------------------------------------------------------
def apply_mask(data: np.ndarray,
               mask: np.ndarray,
               *,
               fill_value: float = np.nan) -> np.ndarray:
    """
    Set data[~mask] = fill_value.  Works for N×… arrays.
    """
    masked = data.copy()
    if masked.ndim == 1:
        masked[~mask] = fill_value
    else:
        masked[~mask, ...] = fill_value
    return masked


# ------------------------------------------------------------------
# Combine multiple masks (logical AND)
# ------------------------------------------------------------------
def combine_masks(*masks: np.ndarray) -> np.ndarray:
    """
    ANDs an arbitrary number of boolean masks.
    Shape checking left to caller.
    """
    if not masks:
        raise ValueError("No masks given")
    combo = masks[0].copy()
    for m in masks[1:]:
        combo &= m
    return combo


# ------------------------------------------------------------------
# Gap-fill tiny holes (optional)
# ------------------------------------------------------------------
def patch_small_gaps(mask: np.ndarray,
                     max_gap: int = 3) -> np.ndarray:
    """
    Turn short sequences of False (bad) shorter than max_gap
    into True (good) to ease plotting / interpolation.
    """
    bad = ~mask
    # find sequences
    idx = np.where(bad)[0]
    if idx.size == 0:
        return mask
    gaps = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)

    patched = mask.copy()
    for g in gaps:
        if 0 < len(g) <= max_gap:
            patched[g] = True
    return patched


# ------------------------------------------------------------------
# Example convenience wrapper
# ------------------------------------------------------------------
def build_quality_masks(event: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
                        probe: str = '1') -> Dict[str, np.ndarray]:
    """
    From the event dict (output of data_loader.load_event),
    build standard masks for key instruments on a per-probe basis.
    """
    q = {}

    # --- FPI ---
    try:
        t_q, fpi_flag = event[probe]['fpi_quality']  # must be loaded separately
        q['FPI'] = fpi_good_mask(fpi_flag)
    except KeyError:
        pass

    # --- HPCA ---
    try:
        t_s, hpca_stat = event[probe]['hpca_status']
        q['HPCA'] = hpca_good_mask(hpca_stat)
    except KeyError:
        pass

    return q
