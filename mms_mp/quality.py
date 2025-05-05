# mms_mp/quality.py
# ------------------------------------------------------------
# Automated sample-quality handling   (May-2025 refresh)
# ------------------------------------------------------------
# ‣ Adds DES (electron) quality masks, matching the MMS4 anomaly logic.
# ‣ Provides `resample_mask()` so bit-field masks can be put on the
#   same uniform grid as science data (wraps mms_mp.resample.resample).
# ‣ `build_quality_masks()` now returns masks for FPI-ion (DIS),
#   FPI-electron (DES), and HPCA when they exist in the event dict
#   produced by data_loader.load_event().
# ------------------------------------------------------------
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Iterable, Optional
# local resample helper
try:
    from ..resample import resample as _resample  # inside package
except ValueError:  # running as script
    from resample import resample as _resample


# ------------------------------------------------------------------
# Generic flag-mask helpers
# ------------------------------------------------------------------
def _flag_mask(flag: np.ndarray,
               accept_levels: Iterable[int] = (0,)) -> np.ndarray:
    """Return True for samples whose flag is in accept_levels."""
    return np.isin(flag, list(accept_levels))


# ------------------------------------------------------------------
# DIS / DES quality masks
# ------------------------------------------------------------------
def dis_good_mask(flag: np.ndarray,
                  accept_levels: Iterable[int] = (0,)) -> np.ndarray:
    """Ion spectrometer (DIS) – same convention for burst/fast."""
    return _flag_mask(flag, accept_levels)


def des_good_mask(flag: np.ndarray,
                  accept_levels: Iterable[int] = (0, 1)) -> np.ndarray:
    """
    Electron spectrometer (DES) – by default keep level-1 (“suspect”)
    samples because MMS4 post-2018 often has no level-0 burst data.
    """
    return _flag_mask(flag, accept_levels)


# ------------------------------------------------------------------
# HPCA status mask
# ------------------------------------------------------------------
def hpca_good_mask(status: np.ndarray, accept_zero_only: bool = True) -> np.ndarray:
    """Treat any non-zero status bit as bad by default."""
    return status == 0 if accept_zero_only else np.ones_like(status, bool)


# ------------------------------------------------------------------
# Mask application / utilities
# ------------------------------------------------------------------
def apply_mask(data: np.ndarray,
               mask: np.ndarray,
               fill_value: float = np.nan) -> np.ndarray:
    out = data.copy()
    out[~mask] = fill_value
    return out


def combine_masks(*masks: np.ndarray) -> np.ndarray:
    if not masks:
        raise ValueError("No masks given")
    combo = masks[0].copy()
    for m in masks[1:]:
        combo &= m
    return combo


def patch_small_gaps(mask: np.ndarray, max_gap: int = 3) -> np.ndarray:
    bad = ~mask
    idx = np.where(bad)[0]
    if idx.size == 0:
        return mask
    gaps = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    fixed = mask.copy()
    for g in gaps:
        if 0 < len(g) <= max_gap:
            fixed[g] = True
    return fixed


def resample_mask(t_orig: np.ndarray,
                  mask: np.ndarray,
                  t_target: np.ndarray,
                  method: str = 'nearest') -> np.ndarray:
    """
    Resample a boolean quality mask onto *t_target* grid using
    the toolkit’s resample helper.
    """
    _, mask_rs, _ = _resample(t_orig, mask.astype(float),
                              cadence='custom', method=method,
                              t_new=t_target)  # custom cadence path
    return np.isfinite(mask_rs) & (mask_rs > 0.5)


# ------------------------------------------------------------------
# Build masks from event dict
# ------------------------------------------------------------------
def build_quality_masks(event: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
                        probe: str = '1') -> Dict[str, np.ndarray]:
    """
    Collect standard masks for DIS (ions), DES (electrons), and HPCA.
    The event dict must come from data_loader.load_event (which loads
    the _quality_flag variables automatically when present).
    """
    q: Dict[str, np.ndarray] = {}

    # FPI ion (DIS)
    key = f'mms{probe}'
    dis_flag_name = f'{key}_dis_quality_flag'
    if dis_flag_name in event[probe]:
        _, flag_dis = event[probe][dis_flag_name]
        q['DIS'] = dis_good_mask(flag_dis)

    # FPI electron (DES)
    des_flag_name = f'{key}_des_quality_flag'
    if des_flag_name in event[probe]:
        _, flag_des = event[probe][des_flag_name]
        q['DES'] = des_good_mask(flag_des)

    # HPCA
    hpca_stat_name = f'{key}_hpca_status_flag'
    if hpca_stat_name in event[probe]:
        _, stat_hpca = event[probe][hpca_stat_name]
        q['HPCA'] = hpca_good_mask(stat_hpca)

    return q
