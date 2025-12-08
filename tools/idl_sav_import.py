"""
IDL .sav importer utilities for MMS magnetopause comparison.

Requires: SciPy (scipy.io.readsav)

This module provides:
- load_idl_sav(path): returns a dict with keys
  'trange_full', 'trange_lmn', 'Lhat','Mhat','Nhat',
  'vi_lmn' -> { '1'..'4': { 't': np.ndarray[float seconds], 'vlmn': np.ndarray[N,3] } }, and
  'b_lmn'  -> { '1'..'4': { 't': np.ndarray[float seconds], 'blmn': np.ndarray[N,3] } }

- extract_vn_series(sav_data): returns { probe: (t, v_n) }

Notes:
- Time arrays are returned as float seconds since Unix epoch.
- If SciPy is missing, ImportError is raised with a clear message.
"""
from __future__ import annotations
from typing import Dict, Any

import numpy as np

try:
    from scipy.io import readsav  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "scipy is required to read IDL .sav files. Please install SciPy (compatible with NumPy<2)."
    ) from e


def _coerce_array(x) -> np.ndarray:
    """Best-effort conversion of IDL/SciPy object arrays into plain numpy arrays.
    Flattens object arrays by concatenation of their elements when necessary.
    """
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            parts = []
            for e in x.ravel():
                try:
                    parts.append(np.asarray(e))
                except Exception:
                    pass
            if not parts:
                return np.array([])
            if len(parts) == 1:
                return np.asarray(parts[0])
            return np.concatenate([p.ravel() for p in parts if p is not None])
        return x
    try:
        return np.asarray(x)
    except Exception:
        return np.array([])


def _to_float_seconds(x) -> np.ndarray:
    """Convert IDL time arrays to float seconds since epoch with robust fallback.
    Tries numeric casting; if that fails, falls back to per-element float extraction.
    """
    arr = _coerce_array(x)
    # Fast path for numeric arrays
    try:
        if np.issubdtype(arr.dtype, np.number):
            return arr.astype(float, copy=False).ravel()
    except Exception:
        pass
    # Slow path: extract first scalar from each element
    out = []
    it = arr.ravel() if isinstance(arr, np.ndarray) else [arr]
    for e in it:
        try:
            ae = np.asarray(e)
            if ae.size == 0:
                continue
            out.append(float(ae.reshape(-1)[0]))
        except Exception:
            try:
                out.append(float(e))
            except Exception:
                continue
    return np.array(out, dtype=float)


def load_idl_sav(path: str) -> Dict[str, Any]:
    d = readsav(path, python_dict=True, verbose=False)
    # Possible key variants (robust): lowercase in SciPy dict
    keys = {k.lower(): k for k in d.keys()}

    def _get(name: str):
        k = name.lower()
        if k not in keys:
            return None
        return d[keys[k]]

    # Time ranges – can be doubles (seconds since epoch)
    trange_full = _get('TRANGE_FULL')
    if trange_full is not None:
        trange_full = _coerce_array(trange_full).astype(float, copy=False)

    # Per-probe LMN vectors and LMN trange
    lmn_per_probe: Dict[str, Dict[str, np.ndarray]] = {}
    trange_lmn_per_probe: Dict[str, np.ndarray] = {}
    for probe in ['1','2','3','4']:
        Lhat = _get(f'lhat{probe}')
        Mhat = _get(f'mhat{probe}')
        Nhat = _get(f'nhat{probe}')
        if Lhat is not None and Mhat is not None and Nhat is not None:
            Lhat = _coerce_array(Lhat).astype(float, copy=False).ravel()
            Mhat = _coerce_array(Mhat).astype(float, copy=False).ravel()
            Nhat = _coerce_array(Nhat).astype(float, copy=False).ravel()
            lmn_per_probe[probe] = {'L': Lhat, 'M': Mhat, 'N': Nhat}
        trp = _get(f'trange_lmn{probe}')
        if trp is not None:
            trange_lmn_per_probe[probe] = _coerce_array(trp).astype(float, copy=False)

    # Ion velocity structures per spacecraft (VI_LMN#)
    vi_lmn: Dict[str, Dict[str, np.ndarray]] = {}
    for probe in ['1','2','3','4']:
        key = f'VI_LMN{probe}'
        vdat = _get(key)
        if vdat is None:
            continue
        # SciPy reads IDL structs as numpy.void or record array; handle both
        t = getattr(vdat, 'x', None)
        y = getattr(vdat, 'y', None)
        if t is None:
            try:
                t = vdat['x']
                y = vdat['y']
            except Exception:
                pass
        t_arr = _coerce_array(t)
        # If still multi-d object array, flatten
        if isinstance(t_arr, np.ndarray) and t_arr.dtype == object:
            t_flat_parts = []
            for e in t_arr.ravel():
                try:
                    t_flat_parts.append(np.asarray(e).ravel())
                except Exception:
                    pass
            t_arr = np.concatenate(t_flat_parts) if t_flat_parts else np.array([])
        t_sec = t_arr.astype(float, copy=False)

        vlmn_arr = _coerce_array(y)
        # Handle possible extra wrapper dimension and (3,N) layout
        vlmn_arr = np.asarray(vlmn_arr, dtype=float)
        # Unwrap if array is length-1 enclosing (e.g., shape (1,) with element (3,N))
        if vlmn_arr.dtype == object and vlmn_arr.size == 1:
            vlmn_arr = np.asarray(vlmn_arr.item(), dtype=float)
        if vlmn_arr.ndim == 3 and 1 in vlmn_arr.shape:
            vlmn_arr = np.squeeze(vlmn_arr)
        if vlmn_arr.ndim == 2 and vlmn_arr.shape[0] == 3:
            vlmn_arr = vlmn_arr.T  # (N,3)
        if vlmn_arr.ndim == 1 and vlmn_arr.size % 3 == 0:
            vlmn_arr = vlmn_arr.reshape(-1, 3)

        vi_lmn[probe] = {'t': t_sec, 'vlmn': vlmn_arr}

    # Magnetic field in LMN per spacecraft (B_LMN#), if present
    b_lmn: Dict[str, Dict[str, np.ndarray]] = {}
    for probe in ['1','2','3','4']:
        key = f'B_LMN{probe}'
        bdat = _get(key)
        if bdat is None:
            continue
        arr = _coerce_array(bdat)
        # Typical layout is recarray with shape (1,) where element has (t, B_lmn)
        rec = arr[0] if isinstance(arr, np.ndarray) and arr.size else bdat
        t_raw = getattr(rec, 'x', None)
        y_raw = getattr(rec, 'y', None)
        if t_raw is None or y_raw is None:
            try:
                t_raw = rec[0]
                y_raw = rec[1]
            except Exception:
                continue
        t_sec = _to_float_seconds(t_raw)
        blmn_arr = _coerce_array(y_raw)
        blmn_arr = np.asarray(blmn_arr, dtype=float)
        if blmn_arr.ndim == 3 and 1 in blmn_arr.shape:
            blmn_arr = np.squeeze(blmn_arr)
        if blmn_arr.ndim == 2 and blmn_arr.shape[0] == 3:
            blmn_arr = blmn_arr.T  # (N,3)
        if blmn_arr.ndim == 1 and blmn_arr.size % 3 == 0:
            blmn_arr = blmn_arr.reshape(-1, 3)

	    	# NOTE: For the 2019-01-27 event file
	    	#   "mp_lmn_systems_20190127_1215-1255_mp-ver2b.sav",
	    	# diagnostics against CDF-rotated B show that the native B_LMN arrays
	    	# are stored with columns ordered as (B_N, B_L, B_M), not (B_L, B_M, B_N).
	    	#
	    	# We deliberately keep this "raw" ordering here and handle any
	    	# component re-mapping in downstream diagnostics (e.g.,
	    	# examples/diagnostic_sav_vs_mmsmp_20190127.py) so that this loader
	    	# remains a thin, mostly generic wrapper around the .sav structure.
        b_lmn[probe] = {'t': t_sec, 'blmn': blmn_arr}

    return {
        'trange_full': trange_full,
        'lmn': lmn_per_probe,
        'trange_lmn_per_probe': trange_lmn_per_probe,
        # Back-compat single LMN keys (likely absent in this .sav)
        'Lhat': None,
        'Mhat': None,
        'Nhat': None,
        'vi_lmn': vi_lmn,
        'b_lmn': b_lmn,
    }


def extract_vn_series(sav: Dict[str, Any]) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    out: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for probe, obj in sav.get('vi_lmn', {}).items():
        t = obj['t']
        y = obj['vlmn']
        if y.ndim == 2:
            if y.shape[1] >= 3:  # (N,3)
                vn = y[:, 2]
            elif y.shape[0] == 3:  # (3,N)
                vn = y[2, :]
            else:
                vn = np.full(y.shape[0], np.nan)
        else:
            vn = np.full(len(y), np.nan)
        out[probe] = (t, vn)
    return out

