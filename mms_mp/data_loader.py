# mms_mp/data_loader.py  – 2025-05-06  “fast-only default” edition
# ==========================================================================
# • By default **only the `fast` cadence** is downloaded for FGM/FPI/HPCA.
# • Optional cadences can be enabled instrument-by-instrument:
#       include_brst=True   include_srvy=True   include_slow=True
# • Everything else (sanity tests, trimming, NaN placeholders) unchanged.
# ==========================================================================

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import fnmatch, warnings, time
from datetime import datetime
import numpy as np
import pandas as pd

from pyspedas.projects import mms
from pyspedas import get_data
from pytplot import data_quants

# ═════════════════════════════ helpers ════════════════════════════════════
def _dl_kwargs(download_only: bool) -> dict:
    return dict(notplot=download_only)


def _load(instr: str, rates: List[str], *, trange, probe,
          download_only: bool = False, status: Optional[dict] = None,
          max_attempts: int = 3, backoff: float = 1.6, base_sleep: float = 0.05,
          **extra):
    """Try each cadence in *rates* with limited retries and light back-off."""
    fn = getattr(mms, f"mms_load_{instr}")
    last_error: Exception | None = None
    tried: list[dict] = []
    for rate in rates:
        success = False
        for attempt in range(1, max_attempts + 1):
            try:
                fn(trange=trange, probe=probe, data_rate=rate,
                   **extra, **_dl_kwargs(download_only))
                success = True
                tried.append({'rate': rate, 'attempt': attempt, 'error': None})
                break
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                tried.append({'rate': rate, 'attempt': attempt, 'error': str(exc)})
                if attempt < max_attempts:
                    sleep_for = min(base_sleep * (backoff ** (attempt - 1)), 2.0)
                    time.sleep(sleep_for)
        if success:
            if status is not None:
                status.update({
                    'used_rate': rate,
                    'success': True,
                    'attempts': tried.copy(),
                })
            return rate
    if status is not None:
        status.update({
            'used_rate': None,
            'success': False,
            'attempts': tried,
            'error': str(last_error) if last_error else None,
        })
    return None


def _load_state(trange, probe, *, download_only=False, mec_cache: dict | None = None):
    """
    Load spacecraft ephemeris data - prioritizes MEC over other sources

    MEC (Magnetic Electron and Ion Characteristics) provides the most accurate
    spacecraft positions and velocities and should be used as the authoritative
    source for all spacecraft ordering and coordinate transformations.
    If mec_cache is provided, store POS/VEL arrays immediately to avoid
    later discovery issues in the harvest step.
    """
    try:
        # Primary: Load MEC ephemeris data (most accurate)
        kw_mec = dict(
            trange=trange,
            probe=probe,
            data_rate='srvy',
            level='l2',
            datatype='epht89q',
            time_clip=True
        )
        if download_only:
            kw_mec['downloadonly'] = True

        result = mms.mms_load_mec(**kw_mec)

        # Verify that MEC variables were actually loaded
        from pytplot import data_quants
        from pyspedas import get_data as _gd
        mec_vars_loaded = [var for var in data_quants.keys()
                          if f'mms{probe}_mec' in var and ('r_' in var or 'v_' in var)]

        if len(mec_vars_loaded) > 0:
            print(f"✅ MEC ephemeris loaded for MMS{probe}: {len(mec_vars_loaded)} variables")
            print(f"   Variables: {mec_vars_loaded}")
            # Optionally cache POS/VEL now
            if mec_cache is not None:
                key = f'mms{probe}'
                # Prefer GSM, fallback GSE
                pos_name = key + '_mec_r_gsm'
                if pos_name not in data_quants:
                    pos_name = key + '_mec_r_gse' if key + '_mec_r_gse' in data_quants else None
                vel_name = key + '_mec_v_gsm'
                if vel_name not in data_quants:
                    vel_name = key + '_mec_v_gse' if key + '_mec_v_gse' in data_quants else None
                pos_tuple = _gd(pos_name) if pos_name else (None, None)
                vel_tuple = _gd(vel_name) if vel_name else (None, None)
                mec_cache[str(probe)] = {
                    'pos': pos_tuple,
                    'vel': vel_tuple,
                    'pos_name': pos_name,
                    'vel_name': vel_name,
                    'source': 'mec',
                }
            return result
        else:
            print(f"⚠️ MEC loading returned success but no variables found for MMS{probe}")
            print(f"   Available variables: {[v for v in data_quants.keys() if f'mms{probe}' in v]}")
            raise Exception("No MEC variables loaded")

    except Exception as e:
        print(f"Warning: MEC ephemeris loading failed for MMS{probe}: {e}")
        print("Falling back to definitive ephemeris...")

        # Fallback: Try definitive ephemeris
        try:
            kw = dict(trange=trange, probe=probe, datatypes='pos', level='def')
            if download_only:
                kw['downloadonly'] = True
            result = mms.mms_load_state(**kw)
            if mec_cache is not None:
                mec_cache.setdefault(str(probe), {})['source'] = 'definitive'
            return result
        except Exception as e2:
            print(f"Warning: Definitive ephemeris also failed for MMS{probe}: {e2}")
            if mec_cache is not None:
                mec_cache.setdefault(str(probe), {})['source'] = 'unavailable'
            return None

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
        if _is_valid(pat, expect_cols):
            return pat
        if pat in data_quants and _is_valid(pat, expect_cols):
            return pat
        for hit in fnmatch.filter(data_quants.keys(), pat):
            if _is_valid(hit, expect_cols):
                return hit
    return None


# Prefer variables by rate order, e.g., try '*_brst*' then '*_fast*' then '*_srvy*'
def _first_valid_var_by_rate(base: str, rates: List[str], expect_cols: Optional[int] = None):
    patterns = [f"{base}_{r}*" for r in rates] + [f"{base}_*"]
    for pat in patterns:
        hit = _first_valid_var([pat], expect_cols=expect_cols)
        if hit:
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
    """Convert numeric time arrays to ``datetime64[ns]``.

    Historically this helper assumed **TT2000 nanoseconds since
    2000-01-01T12:00:00**. In practice, pySPEDAS / pytplot more commonly
    provide **Unix seconds since 1970-01-01T00:00:00** (as ``float64``)
    for MMS variables such as FPI distributions.

    To robustly handle both conventions we inspect the magnitude of the
    finite values in *arr*:

    - ``|t| < 1e10``  → interpret as **seconds since 1970-01-01**.
      (1e10 seconds ≈ 317 years, safely above the Unix epoch range we
      ever expect.)
    - otherwise       → interpret as **nanoseconds since 2000-01-01T12:00:00**
      (TT2000-style). Year-2019 TT2000 values are O(1e17), so they
      clearly fall in this branch.
    """

    work = np.asarray(arr, dtype='float64')

    # Mask obviously invalid entries.
    bad = ~np.isfinite(work) | (np.abs(work) > 9e30)
    if np.all(bad):
        return np.full(work.shape, np.datetime64('NaT'), dtype='datetime64[ns]')

    finite = work[~bad]
    max_abs = float(np.nanmax(np.abs(finite)))

    if max_abs < 1e10:
        # Treat as Unix seconds since 1970-01-01T00:00:00.
        origin = np.datetime64('1970-01-01T00:00:00')
        scale = 1e9  # seconds → nanoseconds
    else:
        # Treat as TT2000-style nanoseconds since 2000-01-01T12:00:00.
        origin = np.datetime64('2000-01-01T12:00:00')
        scale = 1.0  # already in nanoseconds

    ints = np.zeros_like(work, dtype='int64')
    good = ~bad
    ints[good] = np.round(work[good] * scale).astype('int64')

    out = origin + ints.astype('timedelta64[ns]')
    out[bad] = np.datetime64('NaT')
    return out


def _tp(varname: str):
    """Wrapper so tests that monkeypatch get_data also affect harvest calls."""
    return get_data(varname)


def _parse_trange(trange: List[str]) -> tuple[np.datetime64, np.datetime64]:
    def _norm(ts: str) -> np.datetime64:
        work = ts.replace('/', 'T')
        if len(work) == 16:  # YYYY-MM-DDTHH:MM
            work = work + ':00'
        return np.datetime64(work, 'ns')

    start = _norm(trange[0])
    end = _norm(trange[1])
    if end < start:
        raise ValueError('trange end precedes start')
    return start, end


def _fmt_dt(dt: np.datetime64) -> str:
    return np.datetime_as_string(dt.astype('datetime64[ns]'), unit='s')


def _to_datetime64_any(times: np.ndarray) -> np.ndarray:
    arr = np.asarray(times)
    if arr.size == 0:
        return arr.astype('datetime64[ns]')
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype('datetime64[ns]')
    if np.issubdtype(arr.dtype, np.number):
        return _tt2000_to_datetime64_ns(arr)
    sample = arr[0]
    if isinstance(sample, datetime):
        return arr.astype('datetime64[ns]')
    try:
        return np.array([np.datetime64(str(v).replace('/', 'T')) for v in arr], dtype='datetime64[ns]')
    except Exception:
        return arr.astype('datetime64[ns]')


def _add_warning(meta: dict, message: str) -> None:
    if message not in meta.setdefault('warnings', []):
        meta['warnings'].append(message)


def _resample_array(source_t: np.ndarray,
                    source_y: np.ndarray,
                    target_t: np.ndarray) -> np.ndarray:
    """Resample *source_y* defined at *source_t* onto *target_t* using linear interpolation."""
    if source_t is None or source_y is None:
        return np.full((len(target_t),) if np.ndim(source_y) != 2 else (len(target_t), source_y.shape[1]), np.nan)

    src_time = _to_datetime64_any(np.asarray(source_t))
    tgt_time = _to_datetime64_any(np.asarray(target_t))
    if src_time.size == 0 or tgt_time.size == 0:
        return np.full((len(target_t),) if np.ndim(source_y) != 2 else (len(target_t), source_y.shape[1]), np.nan)

    src = src_time.astype('datetime64[ns]').astype('int64').astype('float64')
    tgt = tgt_time.astype('datetime64[ns]').astype('int64').astype('float64')

    order = np.argsort(src)
    src = src[order]
    data = np.asarray(source_y)
    data = data[order]

    if data.ndim == 1:
        good = np.isfinite(data)
        if np.count_nonzero(good) < 2:
            return np.full(tgt.shape, np.nan)
        return np.interp(tgt, src[good], data[good], left=np.nan, right=np.nan)

    # Vector case – interpolate column-by-column
    out = np.empty((tgt.size, data.shape[1]))
    for idx in range(data.shape[1]):
        col = data[:, idx]
        good = np.isfinite(col)
        if np.count_nonzero(good) < 2:
            out[:, idx] = np.nan
        else:
            out[:, idx] = np.interp(tgt, src[good], col[good], left=np.nan, right=np.nan)
    return out


def _reconstruct_from_neighbors(evt: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
                                probes: List[str],
                                probe: str,
                                var: str,
                                *,
                                meta: dict) -> bool:
    """Attempt to reconstruct *var* for *probe* by averaging other probes."""
    donors: list[str] = []
    target_t = evt[probe]['B_gsm'][0]
    resampled: list[np.ndarray] = []
    for other in probes:
        if other == probe:
            continue
        if var not in evt[other]:
            continue
        t_src, data_src = evt[other][var]
        if data_src is None:
            continue
        arr = np.asarray(data_src)
        if arr.size == 0 or not np.isfinite(arr).any():
            continue
        donors.append(other)
        resampled.append(_resample_array(t_src, data_src, target_t))

    if not donors:
        return False

    stack = np.stack(resampled, axis=0)
    recon = np.nanmean(stack, axis=0)
    if not np.isfinite(recon).any():
        return False

    evt[probe][var] = (target_t, recon)
    donor_labels = ','.join(f'MMS{d}' for d in donors)
    meta['sources'][probe][var] = f'reconstructed-average({donor_labels})'
    _add_warning(meta, f'MMS{probe} {var} reconstructed from probes {donor_labels}')
    return True

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
        include_hpca: bool = True,
        include_edp: bool = False,
        include_ephem: bool = True,
        download_only: bool = False
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    By default only the 'fast' cadence is used.
    Set include_brst / include_srvy / include_slow to True to add extras.
    """

    start_ns, end_ns = _parse_trange(trange)
    probes = [str(p) for p in probes]

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
    # Define a preferred order for rate selection patterns used in harvesting
    prefer_rates = []
    if include_brst:
        prefer_rates.append('brst')
    if data_rate_fpi not in prefer_rates:
        prefer_rates.append(data_rate_fpi)
    if include_srvy and 'srvy' not in prefer_rates:
        prefer_rates.append('srvy')

    meta: dict = {
        'requested_trange': [_fmt_dt(start_ns), _fmt_dt(end_ns)],
        'probes': probes.copy(),
        'cadence_preferences': {
            'fgm': fgm_rates,
            'fpi': fpi_rates,
            'hpca': hpca_rates if include_hpca else [],
            'edp': _cadence_list('fast') if include_edp else [],
        },
        'options': {
            'include_brst': include_brst,
            'include_srvy': include_srvy,
            'include_slow': include_slow,
            'include_hpca': include_hpca,
            'include_edp': include_edp,
            'include_ephem': include_ephem,
            'download_only': download_only,
        },
        'download_summary': {},
        'ephemeris_sources': {},
        'sources': {p: {} for p in probes},
        'time_coverage': {p: {} for p in probes},
        'warnings': [],
    }

    # ----- 2) download phase -------------------------------------------------
    # For FGM, avoid get_support_data to ensure B variables are created reliably
    meta['download_summary']['fgm'] = {'rates_requested': fgm_rates}
    _load('fgm',  fgm_rates,  trange=trange, probe=probes,
          level='l2', get_support_data=False, time_clip=True,
          download_only=download_only,
          status=meta['download_summary']['fgm'])

    meta['download_summary']['fpi_dis'] = {'rates_requested': fpi_rates}
    _load('fpi',  fpi_rates,  trange=trange, probe=probes,
          level='l2', datatype='dis-moms', time_clip=True,
          download_only=download_only,
          status=meta['download_summary']['fpi_dis'])

    meta['download_summary']['fpi_des'] = {'rates_requested': fpi_rates}
    _load('fpi',  fpi_rates,  trange=trange, probe=probes,
          level='l2', datatype='des-moms', time_clip=True,
          download_only=download_only,
          status=meta['download_summary']['fpi_des'])

    if include_hpca:
        meta['download_summary']['hpca'] = {'rates_requested': hpca_rates}
        _load('hpca', hpca_rates, trange=trange, probe=probes,
              level='l2', datatype='moments', time_clip=True,
              download_only=download_only,
              status=meta['download_summary']['hpca'])
    else:
        meta['download_summary']['hpca'] = {'skipped': True, 'rates_requested': []}

    if include_edp:
        edp_rates = _cadence_list('fast')
        meta['download_summary']['edp_dce'] = {'rates_requested': edp_rates}
        _load('edp', edp_rates, trange=trange, probe=probes,
              level='l2', datatype='dce', time_clip=True,
              download_only=download_only,
              status=meta['download_summary']['edp_dce'])
        meta['download_summary']['edp_scpot'] = {'rates_requested': edp_rates}
        _load('edp', edp_rates, trange=trange, probe=probes,
              level='l2', datatype='scpot', time_clip=True,
              download_only=download_only,
              status=meta['download_summary']['edp_scpot'])

    mec_cache: dict[str, dict] = {}
    if include_ephem:
        for pr in probes:
            _load_state(trange, pr, download_only=download_only, mec_cache=mec_cache)
            meta['ephemeris_sources'][pr] = mec_cache.get(pr, {}).get('source', 'unknown')
    else:
        for pr in probes:
            meta['ephemeris_sources'][pr] = 'skipped'

    if download_only:
        out = {p: {} for p in probes}
        out['__meta__'] = meta
        return out

    # ----- 3) harvest variables ---------------------------------------------
    evt: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {p: {} for p in probes}
    reconstruct_queue: Dict[str, List[Tuple[str, bool]]] = {p: [] for p in probes}

    fpi_recovery_attempted = {'dis': False, 'des': False}
    hpca_recovery_attempted = False

    def _ensure_fpi_level(species: str) -> None:
        if fpi_recovery_attempted[species]:
            return
        status_key = f'fpi_{species}_ql'
        meta['download_summary'][status_key] = {'rates_requested': fpi_rates, 'level': 'ql'}
        _load('fpi', fpi_rates, trange=trange, probe=probes,
              level='ql', datatype=f'{species}-moms', time_clip=True,
              download_only=download_only,
              status=meta['download_summary'][status_key])
        fpi_recovery_attempted[species] = True

    def _ensure_hpca_level() -> None:
        nonlocal hpca_recovery_attempted
        if hpca_recovery_attempted or not include_hpca:
            return
        status_key = 'hpca_ql'
        meta['download_summary'][status_key] = {'rates_requested': hpca_rates, 'level': 'ql'}
        _load('hpca', hpca_rates, trange=trange, probe=probes,
              level='ql', datatype='moments', time_clip=True,
              download_only=download_only,
              status=meta['download_summary'][status_key])
        hpca_recovery_attempted = True

    for p in probes:
        key = f'mms{p}'

        # --- Magnetic field (required) -----------------------------------
        B_var = _first_valid_var([f'{key}_fgm_b_gsm_*'], expect_cols=3)
        if B_var is None:
            B_var = _first_valid_var([f'{key}_fgm_b_gsm_*'])
        if B_var is None:
            B_var = _first_valid_var([f'{key}_fgm_b_gse_*'], expect_cols=3)
            if B_var is None:
                B_var = _first_valid_var([f'{key}_fgm_b_gse_*'])
        if B_var is None:
            raise RuntimeError(f'MMS{p}: magnetic field data unavailable for requested interval {trange}')

        tB, dB = _tp(B_var)
        if dB.ndim == 2 and dB.shape[1] >= 3:
            dB = dB[:, :3]
        evt[p]['B_gsm'] = _trim_to_match(tB, dB)
        meta['sources'][p]['B_gsm'] = B_var

        t_stub = evt[p]['B_gsm'][0]

        def nan_like_vec() -> np.ndarray:
            return np.full((len(t_stub), 3), np.nan)

        def nan_like_scalar() -> np.ndarray:
            return np.full(len(t_stub), np.nan)

        # --- Ion moments (FPI DIS) ---------------------------------------
        ion_nd = _first_valid_var_by_rate(f'{key}_dis_numberdensity', prefer_rates)
        if ion_nd is None:
            _ensure_fpi_level('dis')
            ion_nd = _first_valid_var_by_rate(f'{key}_dis_numberdensity', prefer_rates)

        if ion_nd is not None:
            evt[p]['N_tot'] = _trim_to_match(*_tp(ion_nd))
            meta['sources'][p]['N_tot'] = ion_nd
        else:
            reconstruct_queue[p].append(('N_tot', False))
            evt[p]['N_tot'] = (t_stub, nan_like_scalar())
            meta['sources'][p]['N_tot'] = 'pending-reconstruction'

        vel_candidates = []
        if ion_nd:
            suff = ion_nd.split('_')[-1]
            vel_candidates.append(f'{key}_dis_bulkv_gse_{suff}')
        vel_var = None
        for cand in vel_candidates:
            vel_var = _first_valid_var([cand])
            if vel_var:
                break
        if vel_var is None:
            vel_var = _first_valid_var_by_rate(f'{key}_dis_bulkv_gse', prefer_rates)
        if vel_var is None:
            _ensure_fpi_level('dis')
            vel_var = _first_valid_var_by_rate(f'{key}_dis_bulkv_gse', prefer_rates)
        if vel_var:
            evt[p]['V_i_gse'] = _trim_to_match(*_tp(vel_var))
            meta['sources'][p]['V_i_gse'] = vel_var
        else:
            reconstruct_queue[p].append(('V_i_gse', True))
            evt[p]['V_i_gse'] = (t_stub, nan_like_vec())
            meta['sources'][p]['V_i_gse'] = 'pending-reconstruction'

        # --- Electron moments (FPI DES) ----------------------------------
        des_nd = _first_valid_var_by_rate(f'{key}_des_numberdensity', prefer_rates)
        if des_nd is None:
            _ensure_fpi_level('des')
            des_nd = _first_valid_var_by_rate(f'{key}_des_numberdensity', prefer_rates)

        if des_nd:
            evt[p]['N_e'] = _trim_to_match(*_tp(des_nd))
            meta['sources'][p]['N_e'] = des_nd
        else:
            reconstruct_queue[p].append(('N_e', False))
            evt[p]['N_e'] = (t_stub, nan_like_scalar())
            meta['sources'][p]['N_e'] = 'pending-reconstruction'

        des_vel = None
        if des_nd:
            suff = des_nd.split('_')[-1]
            des_vel = _first_valid_var([f'{key}_des_bulkv_gse_{suff}'])
        if des_vel is None:
            des_vel = _first_valid_var_by_rate(f'{key}_des_bulkv_gse', prefer_rates)
        if des_vel is None:
            _ensure_fpi_level('des')
            des_vel = _first_valid_var_by_rate(f'{key}_des_bulkv_gse', prefer_rates)
        if des_vel:
            evt[p]['V_e_gse'] = _trim_to_match(*_tp(des_vel))
            meta['sources'][p]['V_e_gse'] = des_vel
        else:
            reconstruct_queue[p].append(('V_e_gse', True))
            evt[p]['V_e_gse'] = (t_stub, nan_like_vec())
            meta['sources'][p]['V_e_gse'] = 'pending-reconstruction'

        # --- HPCA He+ moments --------------------------------------------
        if include_hpca:
            he_nd = _first_valid_var([f'{key}_hpca_*heplus*number_density*'])
            if he_nd is None:
                _ensure_hpca_level()
                he_nd = _first_valid_var([f'{key}_hpca_*heplus*number_density*'])
            if he_nd:
                evt[p]['N_he'] = _trim_to_match(*_tp(he_nd))
                meta['sources'][p]['N_he'] = he_nd
                he_vel = he_nd.replace('number_density', 'ion_bulk_velocity')
                evt[p]['V_he_gsm'] = _trim_to_match(*_tp(he_vel))
                meta['sources'][p]['V_he_gsm'] = he_vel
            else:
                reconstruct_queue[p].append(('N_he', False))
                reconstruct_queue[p].append(('V_he_gsm', True))
                evt[p]['N_he'] = (t_stub, nan_like_scalar())
                evt[p]['V_he_gsm'] = (t_stub, nan_like_vec())
                meta['sources'][p]['N_he'] = 'pending-reconstruction'
                meta['sources'][p]['V_he_gsm'] = 'pending-reconstruction'
        else:
            evt[p]['N_he'] = (t_stub, nan_like_scalar())
            evt[p]['V_he_gsm'] = (t_stub, nan_like_vec())
            meta['sources'][p]['N_he'] = 'skipped'
            meta['sources'][p]['V_he_gsm'] = 'skipped'

        # --- Electric field / spacecraft potential -----------------------
        if include_edp:
            dce_v = _first_valid_var([f'{key}_edp_dce_*_l2'])
            if dce_v:
                evt[p]['E_gse'] = _trim_to_match(*_tp(dce_v))
                meta['sources'][p]['E_gse'] = dce_v
            else:
                evt[p]['E_gse'] = (t_stub, nan_like_vec())
                meta['sources'][p]['E_gse'] = None
            pot_v = _first_valid_var([f'{key}_edp_scpot_*_l2'])
            if pot_v:
                evt[p]['SC_pot'] = _trim_to_match(*_tp(pot_v))
                meta['sources'][p]['SC_pot'] = pot_v
            else:
                evt[p]['SC_pot'] = (t_stub, nan_like_scalar())
                meta['sources'][p]['SC_pot'] = None

        # --- Ephemeris ---------------------------------------------------
        if include_ephem:
            cached = mec_cache
            pos_tuple = cached.get(p, {}).get('pos') if cached else None
            vel_tuple = cached.get(p, {}).get('vel') if cached else None

            if pos_tuple and pos_tuple[0] is not None and pos_tuple[1] is not None:
                times, pos_data = pos_tuple
                if np.nanmax(np.abs(pos_data)) < 100:
                    pos_data = pos_data * 6371.0
                evt[p]['POS_gsm'] = (times, pos_data)
                meta['sources'][p]['POS_gsm'] = cached.get(p, {}).get('pos_name')
            if vel_tuple and vel_tuple[0] is not None and vel_tuple[1] is not None:
                evt[p]['VEL_gsm'] = vel_tuple
                meta['sources'][p]['VEL_gsm'] = cached.get(p, {}).get('vel_name')

            if 'POS_gsm' not in evt[p] or 'VEL_gsm' not in evt[p]:
                pos_patterns = [
                    f'{key}_mec_r_gsm',
                    f'{key}_mec_r_gse',
                    f'{key}_mec_pos_gsm',
                    f'{key}_mec_pos_gse',
                    f'{key}_defeph_pos',
                    f'{key}_state_pos_gsm',
                    f'{key}_orbatt_r_gsm'
                ]
                vel_patterns = [
                    f'{key}_mec_v_gsm',
                    f'{key}_mec_v_gse',
                    f'{key}_mec_vel_gsm',
                    f'{key}_mec_vel_gse',
                    f'{key}_defeph_vel',
                    f'{key}_state_vel_gsm'
                ]

                pos_v = _first_valid_var(pos_patterns, expect_cols=3)
                vel_v = _first_valid_var(vel_patterns, expect_cols=3)

                if 'POS_gsm' not in evt[p]:
                    if pos_v:
                        times, pos_data = _tp(pos_v)
                        if np.nanmax(np.abs(pos_data)) < 100:
                            pos_data = pos_data * 6371.0
                        evt[p]['POS_gsm'] = (times, pos_data)
                        meta['sources'][p]['POS_gsm'] = pos_v
                    else:
                        raise RuntimeError(f'MMS{p}: ephemeris position data unavailable for interval {trange}')

                if 'VEL_gsm' not in evt[p]:
                    if vel_v:
                        evt[p]['VEL_gsm'] = _tp(vel_v)
                        meta['sources'][p]['VEL_gsm'] = vel_v
                    else:
                        raise RuntimeError(f'MMS{p}: ephemeris velocity data unavailable for interval {trange}')

        # Quality flags (store under canonical keys)
        dis_flag = _first_valid_var_by_rate(f'{key}_dis_quality_flag', prefer_rates)
        if dis_flag:
            evt[p][f'{key}_dis_quality_flag'] = _trim_to_match(*_tp(dis_flag))
            meta['sources'][p][f'{key}_dis_quality_flag'] = dis_flag
        des_flag = _first_valid_var_by_rate(f'{key}_des_quality_flag', prefer_rates)
        if des_flag:
            evt[p][f'{key}_des_quality_flag'] = _trim_to_match(*_tp(des_flag))
            meta['sources'][p][f'{key}_des_quality_flag'] = des_flag
        if include_hpca:
            hpca_flag = _first_valid_var([f'{key}_hpca_*status_flag*'])
            if hpca_flag:
                evt[p][f'{key}_hpca_status_flag'] = _trim_to_match(*_tp(hpca_flag))
                meta['sources'][p][f'{key}_hpca_status_flag'] = hpca_flag

    # --- Reconstruction for probes lacking direct measurements -----------
    for probe, tasks in reconstruct_queue.items():
        for var, _is_vector in tasks:
            success = _reconstruct_from_neighbors(evt, probes, probe, var, meta=meta)
            if success:
                continue
            # Quasi-neutral fallback between electrons and ions
            if var == 'N_tot' and 'N_e' in evt[probe] and np.isfinite(np.asarray(evt[probe]['N_e'][1])).any():
                evt[probe]['N_tot'] = evt[probe]['N_e']
                meta['sources'][probe]['N_tot'] = meta['sources'][probe].get('N_e', 'reconstructed-from-electrons')
                _add_warning(meta, f'MMS{probe} N_tot reconstructed from electron density (quasi-neutrality)')
                continue
            if var == 'N_e' and 'N_tot' in evt[probe] and np.isfinite(np.asarray(evt[probe]['N_tot'][1])).any():
                evt[probe]['N_e'] = evt[probe]['N_tot']
                meta['sources'][probe]['N_e'] = meta['sources'][probe].get('N_tot', 'reconstructed-from-ions')
                _add_warning(meta, f'MMS{probe} N_e reconstructed from ion density (quasi-neutrality)')
                continue
            # For some intervals (including the 2019-01-27 event), electron bulk
            # velocity V_e_gse may be entirely unavailable. In that case we keep
            # the NaN stub but do not fail the whole load; scripts that rely on
            # electrons can check for NaNs explicitly.
            if var == 'V_e_gse':
                meta['sources'][probe]['V_e_gse'] = 'unavailable-no-reconstruction'
                _add_warning(meta, f'MMS{probe}: V_e_gse unavailable; leaving NaNs for interval {trange}')
                continue
            raise RuntimeError(f'MMS{probe}: unable to reconstruct {var} – check data availability for {trange}')

    # --- Final sanity checks & coverage metadata -------------------------
    essential_vars = ['N_tot', 'V_i_gse', 'N_e', 'B_gsm']
    for p in probes:
        for var in essential_vars:
            if var not in evt[p]:
                raise RuntimeError(f'MMS{p}: missing essential variable {var} after reconstruction')
            arr = np.asarray(evt[p][var][1])
            if arr.size == 0 or not np.isfinite(arr).any():
                raise RuntimeError(f'MMS{p}: {var} contains no finite data for interval {trange}')
        if include_hpca:
            if 'N_he' not in evt[p] or not np.isfinite(np.asarray(evt[p]['N_he'][1])).any():
                raise RuntimeError(f'MMS{p}: He⁺ density unavailable – HPCA reconstruction failed')

    tol = np.timedelta64(1, 's')
    for p in probes:
        for k, (t_arr, d_arr) in list(evt[p].items()):
            evt[p][k] = _trim_to_match(t_arr, d_arr)
            t_vals = evt[p][k][0]
            if t_vals is None:
                continue
            try:
                dt_arr = _to_datetime64_any(t_vals)
            except Exception:
                continue
            if dt_arr.size == 0:
                continue
            start_actual = dt_arr.min()
            end_actual = dt_arr.max()
            meta['time_coverage'][p][k] = {
                'start': _fmt_dt(start_actual),
                'end': _fmt_dt(end_actual),
            }
            if k == 'B_gsm':
                if start_actual > start_ns + tol:
                    delta = (start_actual - start_ns) / np.timedelta64(1, 's')
                    _add_warning(meta, f'MMS{p} magnetic field starts {delta:.1f}s after requested window')
                if start_actual < start_ns - tol:
                    delta = (start_ns - start_actual) / np.timedelta64(1, 's')
                    _add_warning(meta, f'MMS{p} magnetic field starts {delta:.1f}s before requested window')
                if end_actual < end_ns - tol:
                    delta = (end_ns - end_actual) / np.timedelta64(1, 's')
                    _add_warning(meta, f'MMS{p} magnetic field ends {delta:.1f}s before requested window finishes')
                if end_actual > end_ns + tol:
                    delta = (end_actual - end_ns) / np.timedelta64(1, 's')
                    _add_warning(meta, f'MMS{p} magnetic field extends {delta:.1f}s beyond requested window')

    evt['__meta__'] = meta
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


# ════════════════════════════════════════════════════════════════════════════
# Forced spectrogram loading helpers (FPI)
# These utilities aggressively attempt to ensure that FPI spectrogram products
# (electron DES and ion DIS) are present in pytplot's registry by trying the
# dedicated spectr products first and then falling back to distributions.
# They also report which source/rate ended up available so callers can annotate
# plots accordingly.
# ════════════════════════════════════════════════════════════════════════════

def _first_omni_by_rate(base: str, rates: list[str]):
    """Return the first omni spectrogram variable.
    Tries by preferred rate; if not found, falls back to any omni match.
    base example: f"mms1_des" or f"mms3_dis".
    """
    import fnmatch as _fn

    def _is_2d_spectrogram(var_name: str, require_many_bins: bool = False) -> bool:
        """Heuristic check for a 2-D (time, energy) omni-type variable.

        Uses pytplot.data_quants metadata only (no get_data calls) so it is
        robust to pyspedas API changes where get_data may return more than two
        values. We simply require ndim == 2 and, optionally, at least ~8 energy
        bins to avoid mistaking tiny arrays for spectra.
        """

        qa = data_quants.get(var_name)
        if qa is None:
            return False
        try:
            ndim = getattr(qa, "ndim", None)
            shape = getattr(qa, "shape", None)
            if ndim != 2 or shape is None:
                return False
            if require_many_bins and shape[1] < 8:
                return False
            return True
        except Exception:
            return False

    # 1) Try explicit rate-suffixed names (most common)
    for rate in (rates or []):
        for pat in [f"{base}_energyspectr_omni_{rate}*", f"{base}_energyspectr_{rate}_omni*"]:
            for v in _fn.filter(data_quants.keys(), pat):
                if _is_2d_spectrogram(v):
                    return v

    # 2) Any omni without explicit rate
    for pat in [f"{base}_energyspectr_omni*", f"{base}*energyspectr*omni*"]:
        for v in _fn.filter(data_quants.keys(), pat):
            if _is_2d_spectrogram(v):
                return v

    # 3) Some datasets store omni as numberflux or differential_flux; accept any 2D spectra
    for pat in [f"{base}*omni*", f"{base}*flux*"]:
        for v in _fn.filter(data_quants.keys(), pat):
            if _is_2d_spectrogram(v, require_many_bins=True):
                return v
    return None


def _first_flux3d_by_rate(base: str, rates: list[str]):
    """Return first matching 3D (t, angle, energy) or (t, energy, angle) by preferred rate.
    Tries pitch-angle distribution patterns; falls back to any 3D under base.
    """
    import fnmatch as _fn
    # 1) Pitch-angle distribution patterns by rate
    for rate in (rates or []):
        for pat in [f"{base}_pitchangdist_*_{rate}*", f"{base}_*pitch*_{rate}*"]:
            for v in _fn.filter(data_quants.keys(), pat):
                try:
                    qa = data_quants.get(v)
                    if qa is not None and hasattr(qa, 'ndim') and qa.ndim == 3:
                        return v
                except Exception:
                    continue
    # 2) Any 3D var under base
    for v in list(data_quants.keys()):
        if not v.startswith(base):
            continue
        try:
            qa = data_quants.get(v)
            if qa is not None and hasattr(qa, 'ndim') and qa.ndim == 3:
                return v
        except Exception:
            continue
    return None

def _first_flux4d_by_rate(base: str, rates: list[str]):
    """Return first matching 4D distribution/spectrogram variable by preferred rate.
    Tries both energyspectr and dist patterns; falls back to any 4D under base.
    """
    import fnmatch as _fn
    # 1) Rate-specific patterns for energyspectr and dist
    for rate in rates:
        for pat in [
            f"{base}_energyspectr_*_{rate}*",
            f"{base}_energyspectr_{rate}*",
            f"{base}_dist_{rate}*",
            f"{base}_*dist*_{rate}*",
        ]:
            for v in _fn.filter(data_quants.keys(), pat):
                try:
                    qa = data_quants.get(v)
                    if qa is not None and hasattr(qa, 'ndim') and qa.ndim == 4:
                        return v
                except Exception:
                    continue
    # 2) Any 4D variable under base
    for v in list(data_quants.keys()):
        if not v.startswith(base):
            continue
        try:
            qa = data_quants.get(v)
            if qa is not None and hasattr(qa, 'ndim') and qa.ndim == 4:
                return v
        except Exception:
            continue
    return None


def force_load_fpi_spectrogram(
    trange: list[str],
    probe: str,
    *,
    species: str,                      # 'des' (electrons) or 'dis' (ions)
    rates: list[str] | None = None,
    time_clip: bool = True,
    verbose: bool = True,
) -> dict:
    """Force-load FPI spectrograms for one species and one probe.

    Returns a dict with keys:
      - omni_var: 2D omni spectrogram variable name (if available)
      - energy_var: energy axis variable name (if available)
      - flux4d_var: 4D distribution/spectrogram variable name (if available)
      - used_rate: rate string associated with the found product (best effort)
      - source: one of {'omni', '4D', 'loaded-dist', 'loaded-spectr', None}
    """
    if rates is None:
        rates = ['brst', 'fast', 'srvy']

    # Prefer omni discovery order: fast -> srvy -> brst (if present)
    omni_rates = [r for r in ['fast', 'srvy', 'brst'] if r in rates] + [r for r in rates if r not in ('fast','srvy','brst')]
    if verbose:
        print(f"[force_spectr] omni search order: {omni_rates}; dist search order: {rates}")

    base = f"mms{probe}_{species}"

    # 1) Check if an omni spectrogram is already present (preferred)
    omni = _first_omni_by_rate(base, omni_rates)
    used_rate = None
    source = None
    if omni:
        # Try to infer the rate from the variable name suffix
        for r in omni_rates:
            if f"_{r}_" in omni or omni.endswith(f"_{r}"):
                used_rate = r
                break
        source = 'omni'
        if verbose:
            print(f"[force_spectr] Found preexisting omni: {omni} (rate={used_rate})")
    else:
        # 2) Try to load dedicated spectr product (fast/srvy before brst). Stop at first success producing an omni.
        for r_try in omni_rates:
            try:
                # Load without varformat filters to avoid excluding valid variables
                mms.fpi(
                    trange=trange,
                    probe=probe,
                    data_rate=r_try,
                    datatype=f"{species}-spectr",
                    level='l2',
                    time_clip=time_clip,
                )
                if verbose:
                    print(f"[force_spectr] Loaded {species}-spectr at rate={r_try}; searching for omni under {base}*")
            except Exception:
                continue
            omni = _first_omni_by_rate(base, omni_rates)
            if omni:
                source = 'loaded-spectr'
                used_rate = r_try
                if verbose:
                    print(f"[force_spectr] Found omni after load: {omni} (rate={used_rate})")
                break

    # 3) If omni still missing, try loading distributions which may enable derived
    flux4d = None
    flux3d = None
    if not omni:
        for r_try in rates:
            try:
                # Load distributions without varformat filters first
                mms.fpi(
                    trange=trange,
                    probe=probe,
                    data_rate=r_try,
                    datatype=f"{species}-dist",
                    level='l2',
                    time_clip=time_clip,
                )
                source = 'loaded-dist'
            except Exception:
                continue
            # Debug: list keys for this base
            base_keys = [k for k in data_quants.keys() if k.startswith(base)]
            print(f"[force_spectr] After loading {species}-dist {r_try}, found {len(base_keys)} keys for {base} (e.g., {base_keys[:5]})")
            # Look for any 4D or 3D flux variable
            found4 = _first_flux4d_by_rate(base, [r_try])
            found3 = _first_flux3d_by_rate(base, [r_try])
            # Fallback: known dist name
            if not found4:
                cand = f"{base}_dist_{r_try}"
                qa = data_quants.get(cand)
                if qa is not None and hasattr(qa, 'ndim') and qa.ndim == 4:
                    found4 = cand
            # Set if found
            flux4d = found4 or flux4d
            flux3d = found3 or flux3d
            print(f"[force_spectr] {base} rate={r_try} flux4d={flux4d} flux3d={flux3d}")
            if (flux4d or flux3d) and used_rate is None:
                used_rate = r_try
                break
    energy_var = None

    # 4a) Prefer explicit energy variable first
    if energy_var is None:
        import fnmatch as _fn
        for pat in [f"{base}_energy_*", f"{base}_energy", f"{base}_*_energy*", f"{base}_logenergy_*"]:
            for v in _fn.filter(data_quants.keys(), pat):
                try:
                    _, e = get_data(v)
                    if e is None:
                        continue
                    if (np.ndim(e) == 1 and np.size(e) > 4) or (np.ndim(e) == 2 and min(e.shape) > 4):
                        energy_var = v
                        if verbose:
                            print(f"[force_spectr] Using explicit energy variable: {energy_var}")
                        break
                except Exception:
                    continue
            if energy_var:
                break

    # 4b) If still missing, attempt to infer energy axis from coords of flux vars
    if energy_var is None:
        cand_var = flux4d or flux3d or omni
        qa = data_quants.get(cand_var) if cand_var else None
        if qa is not None and hasattr(qa, 'coords'):
            # Prefer common coordinate names
            for cname in ('energy','e_energy','v','v1','v2','w','spec_bins'):
                coord = qa.coords.get(cname) if hasattr(qa.coords, 'get') else qa.coords[cname] if cname in qa.coords else None
                if coord is not None:
                    # Register a synthetic 1D energy array if possible
                    try:
                        arr = getattr(coord, 'values', None)
                        if arr is None:
                            arr = np.asarray(coord)
                        # Reduce 2D coord to 1D if necessary
                        if arr.ndim == 2:
                            arr = arr[0] if arr.shape[0] >= arr.shape[1] else arr[:,0]
                        if arr.ndim == 1 and arr.size > 4:
                            energy_var = f"{base}_energy_inferred"
                            # Store into data_quants so callers can get_data it
                            import xarray as xr
                            data_quants[energy_var] = xr.DataArray(arr, dims=(f"{cname}_dim",))
                            if verbose:
                                print(f"[force_spectr] Inferred energy axis from {cand_var} coord '{cname}' -> {energy_var}")
                            break
                    except Exception as ex:
                        if verbose:
                            print(f"[force_spectr] Failed inferring energy from coord {cname}: {ex}")
                        continue


    # 4) Locate the energy axis variable if present (try more patterns) unless already inferred
    if energy_var is None:
        import fnmatch as _fn
        energy_patterns = [
            f"{base}_energy_*",
            f"{base}_logenergy_*",
            f"{base}_energy",
            f"{base}_*_energy*",
            f"{base}_*eenergy*",
        ]
        for pat in energy_patterns:
            for v in _fn.filter(data_quants.keys(), pat):
                try:
                    _, e = get_data(v)
                    if e is None:
                        continue
                    # Accept 1D or 2D energy tables; reduce to 1D later at plot time
                    if (np.ndim(e) == 1 and np.size(e) > 4) or (np.ndim(e) == 2 and min(e.shape) > 4):
                        energy_var = v
                        break
                except Exception:
                    continue
            if energy_var:
                break

    # 5) If still missing, attempt QL level as last resort
    if not omni and not (flux4d or flux3d):
        try_levels = ['ql']
        for lvl in try_levels:
            # Try spectr at QL
            for r_try in omni_rates:
                try:
                    mms.fpi(trange=trange, probe=probe, data_rate=r_try, datatype=f"{species}-spectr", level=lvl, time_clip=time_clip)
                except Exception:
                    continue
                omni = _first_omni_by_rate(base, omni_rates)
                if omni:
                    source = f'{lvl}-spectr'
                    used_rate = r_try
                    break
            # Try dist at QL
            if not omni:
                for r_try in rates:
                    try:
                        mms.fpi(trange=trange, probe=probe, data_rate=r_try, datatype=f"{species}-dist", level=lvl, time_clip=time_clip)
                    except Exception:
                        continue
                    found4 = _first_flux4d_by_rate(base, [r_try])
                    found3 = _first_flux3d_by_rate(base, [r_try])
                    flux4d = found4 or flux4d
                    flux3d = found3 or flux3d
                    if (flux4d or flux3d) and used_rate is None:
                        used_rate = r_try
                        source = f'{lvl}-dist'
                        break
            # Attempt to locate energy axis after QL attempt
            if energy_var is None:
                # reuse inference path
                cand_var = flux4d or flux3d or omni
                qa = data_quants.get(cand_var) if cand_var else None
                if qa is not None and hasattr(qa, 'coords'):
                    for cname in ('energy','e_energy','v','v1','v2','w','spec_bins'):
                        coord = qa.coords.get(cname) if hasattr(qa.coords, 'get') else qa.coords[cname] if cname in qa.coords else None
                        try:
                            arr = np.array(coord)
                            if arr is not None and arr.size > 4:
                                energy_var = f"{base}_energy_inferred"
                                import xarray as xr
                                data_quants[energy_var] = xr.DataArray(arr, dims=(f"{cname}_dim",))
                                break
                        except Exception:
                            continue
            if omni or flux4d or flux3d:
                break

    # Final safety net: if we know we've loaded distribution products but still
    # don't have a 4-D flux variable, fall back to the canonical dist name.
    #
    # This specifically protects cases where _first_flux4d_by_rate() or the
    # earlier fallback failed to latch onto e.g. ``mms1_dis_dist_fast`` or
    # ``mms2_des_dist_fast`` even though the underlying DataArray is present
    # and 4-D.  For standard MMS FPI L2 ``*-dist`` files these names are
    # stable, so using them here is both safe and more robust across events.
    if (source and 'dist' in str(source)) and flux4d is None:
        for r_try in (rates or []):
            cand = f"{base}_dist_{r_try}"
            qa = data_quants.get(cand)
            if qa is None or not hasattr(qa, 'ndim'):
                continue
            try:
                if qa.ndim == 4:
                    flux4d = cand
                    if used_rate is None:
                        used_rate = r_try
                    if verbose:
                        print(f"[force_spectr] Using canonical dist var as flux4d: {flux4d} (rate={used_rate})")
                    break
            except Exception:
                continue

    if verbose:
        print(f"[force_spectr] MMS{probe} {species}: omni={bool(omni)} flux4d={bool(flux4d)} rate={used_rate} source={source}")

    return {
        'omni_var': omni,
        'energy_var': energy_var,
        'flux4d_var': flux4d,
        'flux3d_var': flux3d,
        'used_rate': used_rate,
        'source': source,
    }


def force_load_all_plasma_spectrometers(
    trange: list[str],
    probes: list[str] = ('1','2','3','4'),
    *,
    rates: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """Force-load DES and DIS spectrograms for all requested probes.

    Returns nested dict: {probe: {'des': info, 'dis': info}}
    """
    results: dict = {}
    for p in probes:
        results[p] = {
            'des': force_load_fpi_spectrogram(trange, p, species='des', rates=rates, verbose=verbose),
            'dis': force_load_fpi_spectrogram(trange, p, species='dis', rates=rates, verbose=verbose),
        }
    return results
