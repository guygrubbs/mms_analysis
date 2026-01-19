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
    """Resample a 1-D or 2-D time series onto a regular time grid.

    Parameters
    ----------
    t_orig
        Original sample times.  May be ``datetime64`` values, TT2000-style
        int64 nanoseconds since 2000-01-01, or Unix-like floats in seconds.
        Values are converted internally to ``datetime64[ns]`` while
        preserving ordering.

    data_orig
        Data values at times ``t_orig``.  Shape ``(N,)`` or ``(N, M)`` where N
        is the number of samples and M the number of components (e.g. 3 for a
        vector quantity such as B or V).

    cadence
        Target sampling interval specified as a pandas offset string (for
        example ``'250ms'``, ``'1s'``).  The resampled time grid spans the full
        range of ``t_orig``.

    method
        Interpolation method used to fill the regular grid:

        - ``'nearest'``  – piecewise-constant nearest-neighbour (robust to
          gaps; default).
        - ``'linear'``   – first-order interpolation in time using pandas.
        - ``'fft'``      – frequency-domain resampling (requires SciPy) for
          uniformly sampled inputs.

    Returns
    -------
    t_new : ndarray of datetime64[ns]
        Regular time grid covering ``[t_orig.min(), t_orig.max()]`` at the
        specified cadence.

    data_new : ndarray
        Resampled data with shape ``(len(t_new),)`` or ``(len(t_new), M)``
        matching the input dimensionality.

    good_mask : ndarray of bool
        Boolean array with shape ``(len(t_new),)`` marking samples where an
        original datum existed “near enough” to the grid point.  For
        ``method='nearest'`` this is effectively ``~isnan(data_new)``.

    Notes
    -----
    This helper is used to place heterogeneous MMS instruments (FGM, FPI,
    HPCA, EDP) on a common clock before computing derived quantities such as
    VN, DN, and spectrograms.  Units and coordinate systems of ``data_orig``
    are preserved; only the time axis is modified.
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
# Convenience: interpolate to a specific time array
# ------------------------------------------------------------------
def interpolate_to_time(t_src: np.ndarray,
                        y_src: np.ndarray,
                        t_target: np.ndarray,
                        *, kind: str = 'linear') -> np.ndarray:
    """Interpolate a 1-D signal from ``t_src`` onto a new time grid.

    Parameters
    ----------
    t_src
        Source time samples as a 1-D array (numeric or datetime-like).  Values
        are converted to float seconds internally.

    y_src
        Source data values, 1-D or 2-D.  For 2-D input the interpolation is
        applied column-wise.

    t_target
        Target time samples onto which ``y_src`` should be interpolated.  Must
        be sortable and of compatible type with ``t_src``.

    kind
        Currently kept for API compatibility; interpolation is always linear
        via :func:`numpy.interp`.

    Returns
    -------
    ndarray
        Interpolated values at ``t_target`` with the same trailing shape as
        ``y_src``.
    """
    # Ensure 1-D arrays
    t_src = np.asarray(t_src).astype(float)
    t_target = np.asarray(t_target).astype(float)
    y_src = np.asarray(y_src)
    if y_src.ndim == 1:
        return np.interp(t_target, t_src, y_src)
    else:
        # Column-wise interpolation
        out = np.vstack([np.interp(t_target, t_src, y_src[:, i])
                         for i in range(y_src.shape[1])]).T
        return out


# ------------------------------------------------------------------
# Public: multi-dict merge
# ------------------------------------------------------------------
def merge_vars(vars_in: Dict[str, Tuple[np.ndarray, np.ndarray]],
               cadence: str = '250ms',
               method: Literal['nearest', 'linear', 'fft'] = 'nearest'
               ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Resample multiple ``(t, data)`` series to a shared regular time grid.

    Parameters
    ----------
    vars_in
        Mapping ``name -> (t, data)`` where each pair is suitable for
        :func:`resample`.  Time units and coordinate systems are not modified;
        only the time axis is harmonised.

    cadence, method
        Passed through to :func:`resample` and interpreted identically for each
        variable.

    Returns
    -------
    t_common : ndarray of datetime64[ns]
        Common regular time grid covering the union of all input intervals.

    data : dict[str, ndarray]
        Dictionary mapping each input name to its resampled data array on
        ``t_common``.

    masks : dict[str, ndarray]
        Dictionary mapping each input name to a boolean mask indicating where
        that variable is considered valid on ``t_common`` (both present after
        join *and* marked good by the single-series resampler).

    Notes
    -----
    This utility underpins many higher-level operations in :mod:`mms_mp`,
    including VN blending (synchronising bulk and E×B velocities) and joint
    spectrogram plotting.
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
        # Save data array
        out_data[var] = merged.values

        # 1) mask saying “sample exists after join”
        exists_mask = merged.notna().all(axis=1).values

        # 2) re-index original good_mask to global grid
        g_mask_full = (
            pd.Series(g_mask.astype(bool), index=t_local)
              .reindex(t_new, fill_value=False)
              .values
        )

        # 3) final quality: sample exists **and** original was good
        out_good[var] = exists_mask & g_mask_full

    return t_new, out_data, out_good
