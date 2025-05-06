# mms_mp/data_loader.py  – 2025-05-06  “fast-only default” edition
# ==========================================================================
# • By default **only the `fast` cadence** is downloaded for FGM/FPI/HPCA.
# • Optional cadences can be enabled instrument-by-instrument:
#       include_brst=True   include_srvy=True   include_slow=True
# • Everything else (sanity tests, trimming, NaN placeholders) unchanged.
# ==========================================================================

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import fnmatch, warnings
import numpy as np
import pandas as pd

from pyspedas.projects import mms
from pyspedas import get_data
from pytplot import data_quants

# ═════════════════════════════ helpers ════════════════════════════════════
def _dl_kwargs(download_only: bool) -> dict:
    return dict(notplot=download_only)


def _load(instr: str, rates: List[str], *, trange, probe,
          download_only: bool = False, **extra):
    """Try each cadence in *rates* until one succeeds (silent failures)."""
    fn = getattr(mms, f"mms_load_{instr}")
    for rate in rates:
        try:
            fn(trange=trange, probe=probe, data_rate=rate,
               **extra, **_dl_kwargs(download_only))
        except Exception:
            continue


def _load_state(trange, probe, *, download_only=False):
    kw = dict(trange=trange, probe=probe, datatypes='pos', level='def')
    if download_only:
        kw['downloadonly'] = True
    return mms.mms_load_state(**kw)

# ───────────────────────── variable discovery ────────────────────────────
def _is_valid(varname: str, expect_cols: Optional[int] = None) -> bool:
    try:
        t, d = get_data(varname)
    except Exception:
        return False
    if t is None or d is None or len(t) == 0:
        return False
    if expect_cols and d.ndim == 2 and d.shape[1] != expect_cols:
        return False
    if not np.isfinite(d).any() or np.nanmin(np.abs(d)) > 9e30:
        return False
    return True


def _first_valid_var(patterns: List[str], expect_cols: Optional[int] = None):
    for pat in patterns:
        if pat in data_quants and _is_valid(pat, expect_cols):
            return pat
        for hit in fnmatch.filter(data_quants.keys(), pat):
            if _is_valid(hit, expect_cols):
                return hit
    return None

# ───────────────────────── misc helpers ───────────────────────────
def _trim_to_match(time: np.ndarray, data: np.ndarray):
    if len(time) == data.shape[0]:
        return time, data
    n = min(len(time), data.shape[0])
    warnings.warn(
        f'[data_loader] trimming time={len(time)} / data={data.shape[0]} → {n}')
    return time[:n], data[:n]


def _tt2000_to_datetime64_ns(arr: np.ndarray) -> np.ndarray:
    epoch2000 = np.datetime64('2000-01-01T12:00:00')
    work = arr.astype('float64', copy=False)
    bad = ~np.isfinite(work) | (np.abs(work) > 9e30)
    work[bad] = 0.0
    out = epoch2000 + work.astype('int64', copy=False).astype('timedelta64[ns]')
    out[bad] = np.datetime64('NaT')
    return out


_tp = get_data  # shorthand

# ═════════════════════════════ main loader ════════════════════════════════
def load_event(
        trange: List[str],
        probes: List[str] = ('1', '2', '3', '4'),
        *,
        data_rate_fgm: str = 'fast',
        data_rate_fpi: str = 'fast',
        data_rate_hpca: str = 'fast',
        include_brst: bool = False,
        include_srvy: bool = False,
        include_slow: bool = False,
        include_edp: bool = False,
        include_ephem: bool = True,
        download_only: bool = False
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    By default only the 'fast' cadence is used.
    Set include_brst / include_srvy / include_slow to True to add extras.
    """

    # ----- 1) decide which cadences to fetch --------------------------------
    def _cadence_list(preferred: str) -> List[str]:
        extra = []
        if include_brst and 'brst' not in extra and preferred != 'brst':
            extra.append('brst')
        if include_srvy and 'srvy' not in extra and preferred != 'srvy':
            extra.append('srvy')
        if include_slow and 'slow' not in extra and preferred != 'slow':
            extra.append('slow')
        return [preferred] + extra

    fgm_rates  = _cadence_list(data_rate_fgm)
    fpi_rates  = _cadence_list(data_rate_fpi)
    hpca_rates = _cadence_list(data_rate_hpca)

    # ----- 2) download phase -------------------------------------------------
    _load('fgm',  fgm_rates,  trange=trange, probe=probes,
          level='l2', get_support_data=True, time_clip=True,
          download_only=download_only)
    _load('fpi',  fpi_rates,  trange=trange, probe=probes,
          level='l2', datatype='dis-moms', time_clip=True,
          download_only=download_only)
    _load('fpi',  fpi_rates,  trange=trange, probe=probes,
          level='l2', datatype='des-moms', time_clip=True,
          download_only=download_only)
    _load('hpca', hpca_rates, trange=trange, probe=probes,
          level='l2', datatype='moments', time_clip=True,
          download_only=download_only)

    if include_edp:
        _load('edp', _cadence_list('fast'), trange=trange, probe=probes,
              level='l2', datatype='dce', time_clip=True,
              download_only=download_only)
        _load('edp', _cadence_list('fast'), trange=trange, probe=probes,
              level='l2', datatype='scpot', time_clip=True,
              download_only=download_only)

    if include_ephem:
        for pr in probes:
            _load_state(trange, pr, download_only=download_only)

    if download_only:
        return {p: {} for p in probes}

    # ----- 3) harvest variables ---------------------------------------------
    evt: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {p: {} for p in probes}
    for p in probes:
        key = f'mms{p}'

        # mandatory ions
        ion_nd = _first_valid_var([f'{key}_dis_numberdensity_*'])
        if ion_nd is None:
            raise RuntimeError(f'MMS{p}: ion density not found — aborting')
        suff = ion_nd.split('_')[-1]
        evt[p]['N_tot']   = _tp(ion_nd)
        evt[p]['V_i_gse'] = _tp(f'{key}_dis_bulkv_gse_{suff}')

        # --- B-field (vector, 3-column preferred) ---------------------------
        # try exact 3-column first …
        B_var = _first_valid_var([f'{key}_fgm_b_gsm_*'], expect_cols=3)

        if B_var is None:
            # fall back to *any* b_gsm_* – often 4-columns (|B| in col-3)
            B_var = _first_valid_var([f'{key}_fgm_b_gsm_*'])

            if B_var is not None:
                tB, dB = _tp(B_var)
                # keep only the first 3 components if there are ≥3
                if dB.ndim == 2 and dB.shape[1] >= 3:
                    dB = dB[:, :3]
                evt[p]['B_gsm'] = _trim_to_match(tB, dB)

            else:
                # nothing at all found → fabricate NaN placeholder so code runs
                warnings.warn(f'[WARN] B-field absent for MMS{p}')
                t_stub = evt[p]['N_tot'][0]
                evt[p]['B_gsm'] = (
                    t_stub,
                    np.full((len(t_stub), 3), np.nan)
                )
        else:
            evt[p]['B_gsm'] = _tp(B_var)

        # --------------------------------------------------------------------
        # placeholders that depend on B_gsm’s shape (now guaranteed to exist)
        t_stub          = evt[p]['B_gsm'][0]
        nan_like_vec    = lambda: np.full_like(evt[p]['B_gsm'][1], np.nan)
        nan_like_scalar = lambda: np.full_like(evt[p]['N_tot'][1], np.nan)

        # --- electrons (DES) -------------------------------------------------
        des_nd = _first_valid_var([f'{key}_des_numberdensity_*'])
        if des_nd:
            suff = des_nd.split('_')[-1]
            evt[p]['N_e']     = _tp(des_nd)
            evt[p]['V_e_gse'] = _tp(f'{key}_des_bulkv_gse_{suff}')
        else:
            warnings.warn(f'[WARN] DES moments absent for MMS{p}')
            evt[p]['N_e']     = (t_stub, nan_like_scalar())   # ← fixed
            evt[p]['V_e_gse'] = (t_stub, nan_like_vec())      # ← already OK

        # He+
        he_nd = _first_valid_var([f'{key}_hpca_*heplus*number_density*'])
        if he_nd:
            evt[p]['N_he']     = _tp(he_nd)
            evt[p]['V_he_gsm'] = _tp(he_nd.replace('number_density', 'ion_bulk_velocity'))
        else:
            warnings.warn(f'[WARN] HPCA He⁺ moments absent for MMS{p}')
            evt[p]['N_he']     = (t_stub, nan_like_scalar())
            evt[p]['V_he_gsm'] = (t_stub, nan_like_vec())

        # EDP
        if include_edp:
            dce_v = _first_valid_var([f'{key}_edp_dce_*_l2'])
            evt[p]['E_gse'] = _tp(dce_v) if dce_v else (t_stub, nan_like_vec())
            pot_v = _first_valid_var([f'{key}_edp_scpot_*_l2'])
            evt[p]['SC_pot'] = _tp(pot_v) if pot_v else (t_stub, nan_like_scalar())

        # ephemeris
        if include_ephem:
            pos_v = _first_valid_var([
                f'{key}_defeph_pos',
                f'{key}_mec_r_gse',
                f'{key}_state_pos_gsm',
                f'{key}_orbatt_r_gsm'
            ], expect_cols=3)
            evt[p]['POS_gsm'] = _tp(pos_v) if pos_v else (t_stub, np.full((len(t_stub), 3), np.nan))

        # final length sanity
        for k, (t_arr, d_arr) in evt[p].items():
            evt[p][k] = _trim_to_match(t_arr, d_arr)

    return evt

# ═══════════════════════ DataFrame + resample utils ═══════════════════════
def to_dataframe(time: np.ndarray, data: np.ndarray, cols: List[str]) -> pd.DataFrame:
    is_num = (
        np.issubdtype(time.dtype, np.integer)  or
        np.issubdtype(time.dtype, np.unsignedinteger) or
        np.issubdtype(time.dtype, np.floating)
    )
    idx = _tt2000_to_datetime64_ns(time) if is_num else time.astype('datetime64[ns]')
    df = pd.DataFrame(data, index=pd.DatetimeIndex(idx, name=None), columns=cols)
    df = df.loc[~df.index.isna()]
    return df.loc[~df.index.duplicated(keep='first')]


def resample(df: pd.DataFrame, cadence: str = '250ms', method: str = 'nearest') -> pd.DataFrame:
    return df.resample(cadence).nearest() if method == 'nearest' else df.resample(cadence).mean()
