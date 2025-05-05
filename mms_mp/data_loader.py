"""
data_loader.py  ·  MMS event data access helpers
================================================

Key additions
-------------
1.  Instrument-specific convenience loaders (FGM, FPI, HPCA, EDP, EPHEM).
2.  Optional *download-only* and local-cache support.
3.  Ability to return **pandas.DataFrame** objects for quick analysis.
4.  Simple, uniform **resample** helper to put disparate variables on a common cadence.
5.  Graceful handling of missing variables (raises ValueError with list of missing keys).
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd

from pyspedas.projects import mms
from pytplot import tplot_names
from pyspedas import get_data, tplot, store_data
from pytplot import data_quants

# -----------------------------------------------------------------------------
# Low-level wrappers around pySPEDAS mission loaders
# -----------------------------------------------------------------------------
def _download_kwargs(download_only: bool) -> dict:
    return dict(downloadonly=download_only, notplot=download_only)

def load_fgm(trange: List[str], probe: List[str], *,
             data_rate: str = 'brst', level: str = 'l2',
             download_only: bool = False):
    return mms.mms_load_fgm(trange=trange, probe=probe, data_rate=data_rate,
                            level=level, get_support_data=True, time_clip=True,
                            **_download_kwargs(download_only))

def load_fpi(trange: List[str], probe: List[str], *,
             data_rate: str = 'brst', level: str = 'l2',
             species: str = 'dis', moment_type: str = 'moms',
             download_only: bool = False):
    return mms.mms_load_fpi(trange=trange, probe=probe,
                            data_rate=data_rate, level=level,
                            datatype=f'{species}-{moment_type}',
                            time_clip=True, **_download_kwargs(download_only))

def load_hpca(trange: List[str], probe: List[str], *,
              data_rate: str = 'brst', level: str = 'l2',
              moment_type: str = 'moments',
              download_only: bool = False):
    return mms.mms_load_hpca(trange=trange, probe=probe,
                             data_rate=data_rate, level=level,
                             datatype=moment_type, time_clip=True,
                             **_download_kwargs(download_only))

def load_edp(trange: List[str], probe: List[str], *,
             data_rate: str = 'brst', level: str = 'l2',
             datatype: str = 'dce',   # electric field (dce) or scpot
             download_only: bool = False):
    return mms.mms_load_edp(trange=trange, probe=probe,
                            data_rate=data_rate, level=level,
                            datatype=datatype, time_clip=True,
                            **_download_kwargs(download_only))

def load_ephemeris(trange: List[str], probe: List[str], *,
                   download_only: bool = False):
    """Spacecraft position (GSM) & attitude."""
    return mms.mms_load_pos(trange=trange, probe=probe, time_clip=True,
                            **_download_kwargs(download_only))

# -----------------------------------------------------------------------------
# High-level “event” loader
# -----------------------------------------------------------------------------
def load_event(trange: List[str],
               probes: List[str] = ('1', '2', '3', '4'),
               *,
               data_rate_fgm: str = 'brst',
               data_rate_fpi: str = 'brst',
               data_rate_hpca: str = 'brst',
               include_edp: bool = False,
               include_ephem: bool = True,
               download_only: bool = False) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Returns
    -------
    event : dict
        event[probe][var_key] = (time_array, data_array)
        var_key naming scheme:
            B_gsm, N_tot, V_i_gse,
            N_he, V_he_gsm,
            E_gse, SC_pot      (if EDP requested)
            POS_gsm            (if ephemeris requested)
    """
    event: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {p: {} for p in probes}

    # ---- Call instrument loaders ----
    load_fgm(trange, probes, data_rate=data_rate_fgm, download_only=download_only)
    load_fpi(trange, probes, data_rate=data_rate_fpi, download_only=download_only)
    load_hpca(trange, probes, data_rate=data_rate_hpca, download_only=download_only)
    if include_edp:
        load_edp(trange, probes, datatype='dce', download_only=download_only)
        load_edp(trange, probes, datatype='scpot', download_only=download_only)
    if include_ephem:
        load_ephemeris(trange, probes, download_only=download_only)

    if download_only:
        return event  # paths downloaded, nothing else to do

    # ---- Collect variables into dict ----
    for p in probes:
        key = f'mms{p}'
        pairs = {
            'B_gsm':           f'{key}_fgm_b_gsm_{data_rate_fgm}_l2',
            'N_tot':           f'{key}_dis_numberdensity_{data_rate_fpi}',
            'V_i_gse':         f'{key}_dis_bulkv_gse_{data_rate_fpi}',
            'N_he':            f'{key}_hpca_heplus_number_density',
            'V_he_gsm':        f'{key}_hpca_heplus_ion_bulk_velocity',
        }
        if include_edp:
            pairs.update({
                'E_gse':        f'{key}_edp_dce_gse_{data_rate_fgm}_l2',  # naming nuance
                'SC_pot':       f'{key}_edp_scpot_brst_l2',
            })
        if include_ephem:
            pairs.update({
                'POS_gsm':      f'{key}_pos_gsm',  # from mms_load_pos
            })

        # Pull from tplot storage
        for new_name, tp_name in pairs.items():
            if tp_name not in tplot_names():
                raise ValueError(f'Variable missing: {tp_name}')
            t_arr, d_arr = get_data(tp_name)
            event[p][new_name] = (t_arr, d_arr)

    return event

# -----------------------------------------------------------------------------
# Helpers – DataFrame conversion & resampling
# -----------------------------------------------------------------------------
def to_dataframe(time: np.ndarray,
                 data: np.ndarray,
                 columns: List[str]) -> pd.DataFrame:
    """
    Build a pandas DataFrame with UTC datetime index (ns resolution).
    """
    # Convert TT2000 (ns since 2000-01-01) → pandas datetime64[ns]
    if np.issubdtype(time.dtype, np.int64) and time.dtype.itemsize == 8:
        # simple TT2000 converter (accurate to ~1µs)
        epoch2000 = np.datetime64('2000-01-01T12:00:00')  # noon TT2000 start
        dt64 = epoch2000 + time.astype('timedelta64[ns]')
    else:
        dt64 = time.astype('datetime64[ns]')
    return pd.DataFrame(data, index=dt64, columns=columns)

def resample(df: pd.DataFrame,
             cadence: str = '250ms',
             method: str = 'nearest') -> pd.DataFrame:
    """
    Resample DataFrame to regular cadence using pandas.
    """
    if method == 'nearest':
        return df.resample(cadence).nearest()
    elif method == 'mean':
        return df.resample(cadence).mean()
    else:
        raise ValueError(f'Unknown resample method: {method}')
