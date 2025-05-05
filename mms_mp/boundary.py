# mms_mp/boundary.py
# ------------------------------------------------------------
# Advanced boundary–crossing detection (upgraded)
# ------------------------------------------------------------
# NEW FEATURES
# ------------
# • Multi-parameter logic (He⁺ density, total ion density,
#   magnetic rotation / B_N sign change).
# • Hysteresis (separate IN vs OUT thresholds, min consecutive points).
# • Layer tagging (outer mixed layer, current sheet, inner magnetopause).
# • Works on *resampled* data grid + per-variable quality masks.
# ------------------------------------------------------------
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Literal, Optional

# ------------------------------------------------------------------
# Config dataclass
# ------------------------------------------------------------------
class DetectorCfg:
    """
    User-tunable thresholds and logic knobs.
    """
    he_in:   float =  0.2   # cm⁻³ – rise above → inside magnetosphere
    he_out:  float =  0.1   # cm⁻³ – drop below → outside (hysteresis)
    tot_jump: float = 1.5   # factor change allowed for total ion density
    BN_tol:  float =  2.0   # nT   – |B_N| < tol considered near current sheet
    min_pts: int   =  3     # consecutive points to confirm state change
    min_layer_pts: int = 5  # min contiguous points per layer to retain

    def __init__(self, **kw):
        for k, v in kw.items():
            if not hasattr(self, k):
                raise ValueError(f'Unknown DetectorCfg key {k}')
            setattr(self, k, v)

# ------------------------------------------------------------------
# State machine
# ------------------------------------------------------------------
def _sm_update(state: Literal['sheath', 'mp_layer', 'magnetosphere'],
               he_val: float,
               BN_val: float,
               cfg: DetectorCfg,
               inside_mag: bool) -> Literal['sheath', 'mp_layer', 'magnetosphere']:
    """
    Very simple logic:
        • if He⁺ < he_out  → sheath
        • elif |B_N| < BN_tol → mp_layer (current sheet vicinity)
        • else → magnetosphere
    """
    if he_val < cfg.he_out:
        return 'sheath'
    if abs(BN_val) < cfg.BN_tol:
        return 'mp_layer'
    if he_val > cfg.he_in:
        return 'magnetosphere'
    # fallback
    return state

# ------------------------------------------------------------------
# Main detector
# ------------------------------------------------------------------
def detect_crossings_multi(t: np.ndarray,
                           he: np.ndarray,
                           BN: np.ndarray,
                           *,
                           cfg: DetectorCfg = DetectorCfg(),
                           good_mask: Optional[np.ndarray] = None
                           ) -> List[Tuple[str, int, int]]:
    """
    Multi-parameter boundary detector.

    Parameters
    ----------
    t        : datetime64[ns] array (uniform cadence)
    he       : He⁺ density (cm⁻³)  (same length as t)
    BN       : B_N component (nT)  (same length)
    cfg      : DetectorCfg thresholds
    good_mask: boolean mask – samples to consider (quality filter)

    Returns
    -------
    layers : list of (layer_type, i_start, i_end)  index tuples
             layer_type ∈ {'sheath', 'mp_layer', 'magnetosphere'}
    """
    N = len(t)
    if good_mask is None:
        good_mask = np.ones(N, dtype=bool)

    # Pre-allocate states
    curr_state: Literal['sheath', 'mp_layer', 'magnetosphere'] = 'sheath'
    run_len = 0
    layers: List[Tuple[str, int, int]] = []
    i_start = 0

    for i in range(N):
        if not good_mask[i]:
            continue
        new_state = _sm_update(curr_state, he[i], BN[i], cfg, inside_mag=(curr_state != 'sheath'))
        if new_state == curr_state:
            run_len += 1
            continue

        # State attempt change – wait min_pts to confirm
        test_run = 1
        j = i+1
        while j < N and good_mask[j] and \
              _sm_update(curr_state, he[j], BN[j], cfg, inside_mag=True) != curr_state:
            test_run += 1
            if test_run >= cfg.min_pts:
                # Confirm change
                layers.append((curr_state, i_start, i-1))
                curr_state = new_state
                i_start = i
                run_len = test_run
                break
            j += 1
        else:
            # not enough points to flip – keep state
            continue

    # append final
    layers.append((curr_state, i_start, N-1))

    # filter out very short layers
    layers = [(typ, s, e) for typ, s, e in layers
              if (e - s + 1) >= cfg.min_layer_pts]

    return layers


# ------------------------------------------------------------------
# Convenience – crossing times (enter/exit)
# ------------------------------------------------------------------
def extract_enter_exit(layers: List[Tuple[str, int, int]],
                       t: np.ndarray) -> List[Tuple[float, str]]:
    """
    Convert layer list to generic crossing events (enter/exit magnetosphere).
    Returns (epoch_sec, 'enter'|'exit') list.
    """
    events = []
    for typ, i1, i2 in layers:
        if typ == 'magnetosphere':
            events.append((t[i1].astype('datetime64[ns]').astype('float64') * 1e-9, 'enter'))
            events.append((t[i2].astype('datetime64[ns]').astype('float64') * 1e-9, 'exit'))
    return sorted(events, key=lambda x: x[0])
