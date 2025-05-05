# mms_mp/resample.py
# ------------------------------------------------------------
# Advanced resampling & time-base utilities  – NEW MODULE
# ------------------------------------------------------------
# Why?
# ----
# • MMS instruments run at wildly different rates (FGM 128 Hz,
#   FPI-DIS 150 ms, HPCA 4 s, EDP 8 kHz burst).  A single helper
#   that merges *heterogeneous* vectors onto a uniform clock
#   keeps the rest of the toolkit tidy.
# • Offers three algorithms:
#       'nearest'  – fast; good for inspection
#       'linear'   – pandas/NumPy interp1d
#       'fft'      – uniform-FFT resample (SciPy) for spectra
# • Returns a “quality” mask so later code can ignore pads / gaps.
# ------------------------------------------------------------
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Literal, Union

try:
    from scipy.signal import resample as fft_resample
except ImportError:
    fft_resample = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _to_datetime64_ns(t: np.ndarray) -> np.ndarray:
    """
    Convert TT2000‐like int64 or already datetime64 to datetime64[ns].
    Assumes any 1-ns resolution int64 that is > ~1e17 represents
    TT2000 (ns since 2000-01-01 12:00:00).
    """
    if np.issubdtype(t.dtype, np.datetime64):
        return t.astype('datetime64[ns]')

    # Heuristic: TT2000 numbers are ~10^18
    if t.max() > 1e17:
        epoch2000 = np.datetime64('2000-01-01T12:00:00')
        return epoch2000 + t.astype('timedelta64[ns]')
    # Otherwise treat as seconds since 1970
    epoch1970 = np.datetime64('1970-01-01T00:00:00')
    return epoch1970 + (t.astype('float64') * 1e9).astype('timedelta64[ns]')


def _make_time_grid(start: np.datetime64,
                    end: np.datetime64,
                    cadence: str) -> np.ndarray:
    """
    Return numpy datetime64[ns] equally spaced by pandas offset string
    e.g. '10ms', '1s', '250ms'.
    """
    dt_index = pd.date_range(start=start, end=end, freq=cadence)
    return dt_index.values.astype('datetime64[ns]')


# ------------------------------------------------------------------
# Public: single-array resample
# ------------------------------------------------------------------
def resample(t_orig: np.ndarray,
             data_orig: np.ndarray,
             cadence: str = '1s',
             method: Literal['nearest', 'linear', 'fft'] = 'nearest'
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    t_orig : epoch array  (TT2000 int64, datetime64, or seconds)
    data_orig : (N, …) array
    cadence : pandas offset, e.g. '250ms', '1s'
    method : 'nearest', 'linear', or 'fft'

    Returns
    -------
    t_new : datetime64[ns] uniform grid
    data_new : resampled array
    good_mask : bool array (True = original datum existed nearby)
    """
    t_dt = _to_datetime64_ns(t_orig)
    start, end = t_dt.min(), t_dt.max()
    t_new = _make_time_grid(start, end, cadence)

    # Flatten time axis for interpolation convenience
    if method == 'nearest':
        df = pd.DataFrame(data_orig, index=t_dt)
        data_new = df.reindex(t_new, method='nearest').values
        good = ~np.isnan(data_new).any(axis=-1) if data_new.ndim > 1 else ~np.isnan(data_new)

    elif method == 'linear':
        df = pd.DataFrame(data_orig, index=t_dt)
        data_new = df.reindex(t_new).interpolate(limit_direction='both').values
        # nearest-neighbour to build quality mask (within half-cadence)
        dt_half = (pd.to_timedelta(cadence) / 2).to_timedelta64()
        idx_near = df.index.searchsorted(t_new)
        good = np.abs(df.index[idx_near] - t_new) <= dt_half

    elif method == 'fft':
        if fft_resample is None:
            raise RuntimeError("SciPy not installed → cannot use method='fft'")
        # convert to seconds float grid
        secs = (t_dt - start) / np.timedelta64(1, 's')
        secs_grid = (t_new - start) / np.timedelta64(1, 's')
        n_out = len(secs_grid)
        if data_orig.ndim == 1:
            y = fft_resample(data_orig, n_out)
        else:
            y = np.vstack([fft_resample(data_orig[:, i], n_out)
                           for i in range(data_orig.shape[1])]).T
        data_new = y
        good = np.ones(n_out, dtype=bool)  # assume FFT creates full series
    else:
        raise ValueError(method)

    return t_new, data_new, good


# ------------------------------------------------------------------
# Public: multi-dict merge
# ------------------------------------------------------------------
def merge_vars(vars_in: Dict[str, Tuple[np.ndarray, np.ndarray]],
               cadence: str = '250ms',
               method: Literal['nearest', 'linear', 'fft'] = 'nearest'
               ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Convenience wrapper:
        • Each input entry:  var_name → (t, data)
        • Resamples *all* to common grid
        • Returns t_common, dict(var→data_resampled), dict(var→good_mask)
    """
    # Use the widest start–end from inputs
    starts = [_to_datetime64_ns(t).min() for t, _ in vars_in.values()]
    ends   = [_to_datetime64_ns(t).max() for t, _ in vars_in.values()]
    t_new = _make_time_grid(min(starts), max(ends), cadence)

    out_data: Dict[str, np.ndarray] = {}
    out_good: Dict[str, np.ndarray] = {}

    for var, (t, d) in vars_in.items():
        # Reuse single-series resample but anchored to global grid
        t_local, d_res, g_mask = resample(t, d, cadence, method)
        # Align to global grid (pad NaN where resample shorter)
        df = pd.DataFrame(d_res, index=t_local)
        df_global = pd.DataFrame(index=t_new)
        merged = df_global.join(df, how='left')
        # Adopt merged arrays
        out_data[var] = merged.values
        g_pad = np.zeros(len(t_new), dtype=bool)
        g_pad[merged.notna().all(axis=1).values] = True
        out_good[var] = g_pad & g_mask  # good only if original good too

    return t_new, out_data, out_good
