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
                mec_cache[str(probe)] = {'pos': pos_tuple, 'vel': vel_tuple}
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
            return mms.mms_load_state(**kw)
        except Exception as e2:
            print(f"Warning: Definitive ephemeris also failed for MMS{probe}: {e2}")
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
    # Define a preferred order for rate selection patterns used in harvesting
    prefer_rates = []
    if include_brst:
        prefer_rates.append('brst')
    if data_rate_fpi not in prefer_rates:
        prefer_rates.append(data_rate_fpi)
    if include_srvy and 'srvy' not in prefer_rates:
        prefer_rates.append('srvy')

    # ----- 2) download phase -------------------------------------------------
    # For FGM, avoid get_support_data to ensure B variables are created reliably
    _load('fgm',  fgm_rates,  trange=trange, probe=probes,
          level='l2', get_support_data=False, time_clip=True,
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

    mec_cache: dict[str, dict] = {}
    if include_ephem:
        for pr in probes:
            _load_state(trange, pr, download_only=download_only, mec_cache=mec_cache)

    if download_only:
        return {p: {} for p in probes}

    # ----- 3) harvest variables ---------------------------------------------
    evt: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {p: {} for p in probes}
    for p in probes:
        key = f'mms{p}'

        # mandatory ions
        # rate-preferential DIS (ions) density
        ion_nd = _first_valid_var_by_rate(f'{key}_dis_numberdensity', prefer_rates)
        if ion_nd is None:
            # Graceful fallback: fabricate empty placeholders instead of aborting
            warnings.warn(f'[WARN] Ion density not found for MMS{p} – using placeholders')
            # Use B_gsm time stub shortly after to size placeholders consistently
            # Temporarily store a flag; we'll replace with real t_stub when available
            evt[p]['N_tot']   = (None, None)
            evt[p]['V_i_gse'] = (None, None)
        else:
            suff = ion_nd.split('_')[-1]
            evt[p]['N_tot']   = _tp(ion_nd)
            evt[p]['V_i_gse'] = _tp(f'{key}_dis_bulkv_gse_{suff}')

        # --- B-field (vector, 3-column preferred) ---------------------------
        # try GSM exact 3-column first …
        B_var = _first_valid_var([f'{key}_fgm_b_gsm_*'], expect_cols=3)

        if B_var is None:
            # fall back to *any* b_gsm_* – often 4-columns (|B| in col-3)
            B_var = _first_valid_var([f'{key}_fgm_b_gsm_*'])

        if B_var is None:
            # try GSE as a fallback
            B_var = _first_valid_var([f'{key}_fgm_b_gse_*'], expect_cols=3)
            if B_var is None:
                B_var = _first_valid_var([f'{key}_fgm_b_gse_*'])

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
            if t_stub is None:
                t_stub = np.linspace(0.0, 1.0, 10)
            evt[p]['B_gsm'] = (
                t_stub,
                np.full((len(t_stub), 3), np.nan)
            )

        # --------------------------------------------------------------------
        # placeholders that depend on B_gsm’s shape (now guaranteed to exist)
        t_stub = evt[p]['B_gsm'][0]
        # Build placeholders strictly from t_stub to avoid None-based shapes
        def nan_like_vec():
            return np.full((len(t_stub), 3), np.nan)
        def nan_like_scalar():
            return np.full(len(t_stub), np.nan)

        # --- electrons (DES) -------------------------------------------------
        # rate-preferential DES (electrons) density
        des_nd = _first_valid_var_by_rate(f'{key}_des_numberdensity', prefer_rates)
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

        # ephemeris - prioritize MEC as authoritative source
        if include_ephem:
            # First, use cached MEC POS/VEL from the load phase if present
            cached = mec_cache  # captured from outer scope
            pos_tuple = cached.get(p, {}).get('pos') if cached else None
            vel_tuple = cached.get(p, {}).get('vel') if cached else None

            if pos_tuple and pos_tuple[0] is not None and pos_tuple[1] is not None:
                times, pos_data = pos_tuple
                if np.nanmax(np.abs(pos_data)) < 100:  # Likely in Earth radii
                    pos_data = pos_data * 6371.0
                evt[p]['POS_gsm'] = (times, pos_data)
            if vel_tuple and vel_tuple[0] is not None and vel_tuple[1] is not None:
                evt[p]['VEL_gsm'] = vel_tuple

            # If cache did not provide, fall back to pattern search
            if 'POS_gsm' not in evt[p] or 'VEL_gsm' not in evt[p]:
                # Priority order: MEC (most accurate) -> definitive -> state -> orbit attitude
                pos_patterns = [
                    f'{key}_mec_r_gsm',      # MEC GSM position (primary)
                    f'{key}_mec_r_gse',      # MEC GSE position (backup)
                    f'{key}_mec_pos_gsm',    # Alternative MEC naming
                    f'{key}_mec_pos_gse',    # Alternative MEC naming
                    f'{key}_defeph_pos',     # Definitive ephemeris
                    f'{key}_state_pos_gsm',  # State position
                    f'{key}_orbatt_r_gsm'    # Orbit attitude
                ]

                print(f"   🔍 Searching for position variables for MMS{p}:")
                print(f"      Patterns: {pos_patterns}")
                available_vars = [v for v in data_quants.keys() if key in v]
                print(f"      Available {key} variables: {available_vars}")

                # Check specifically for MEC variables
                mec_vars = [v for v in data_quants.keys() if f'{key}_mec' in v]
                print(f"      MEC variables in data_quants: {mec_vars}")

                pos_v = _first_valid_var(pos_patterns, expect_cols=3)
                print(f"      Found position variable: {pos_v}")

                # If standard search failed, try direct MEC access
                if pos_v is None:
                    direct_mec_pos = f'{key}_mec_r_gsm'
                    if direct_mec_pos in data_quants:
                        pos_v = direct_mec_pos
                        print(f"      ✅ Found MEC position via direct access: {pos_v}")

                # Also try to get velocity from MEC (for formation analysis)
                vel_patterns = [
                    f'{key}_mec_v_gsm',      # MEC GSM velocity (primary)
                    f'{key}_mec_v_gse',      # MEC GSE velocity (backup)
                    f'{key}_mec_vel_gsm',    # Alternative MEC naming
                    f'{key}_mec_vel_gse',    # Alternative MEC naming
                    f'{key}_defeph_vel',     # Definitive velocity
                    f'{key}_state_vel_gsm'   # State velocity
                ]

                print(f"   🔍 Searching for velocity variables for MMS{p}:")
                print(f"      Patterns: {vel_patterns}")

                vel_v = _first_valid_var(vel_patterns, expect_cols=3)
                print(f"      Found velocity variable: {vel_v}")

                if 'POS_gsm' not in evt[p]:
                    if pos_v:
                        times, pos_data = _tp(pos_v)
                        print(f"   📍 Found position data for MMS{p}: {pos_v}")
                        if np.nanmax(np.abs(pos_data)) < 100:
                            pos_data = pos_data * 6371.0
                        evt[p]['POS_gsm'] = (times, pos_data)
                    else:
                        print(f"   ⚠️ No position data found for MMS{p} - using NaN placeholder")
                        evt[p]['POS_gsm'] = (t_stub, np.full((len(t_stub), 3), np.nan))

                if 'VEL_gsm' not in evt[p]:
                    if vel_v:
                        times, vel_data = _tp(vel_v)
                        print(f"   🚀 Found velocity data for MMS{p}: {vel_v}")
                        evt[p]['VEL_gsm'] = (times, vel_data)
                    else:
                        print(f"   ⚠️ No velocity data found for MMS{p} - using NaN placeholder")
                        evt[p]['VEL_gsm'] = (t_stub, np.full((len(t_stub), 3), np.nan))

        # Replace placeholders if earlier ion density missing
        if evt[p].get('N_tot', (None,))[0] is None:
            t_stub = evt[p]['B_gsm'][0] if 'B_gsm' in evt[p] else np.linspace(0.0, 1.0, 10)
            evt[p]['N_tot']   = (t_stub, np.full_like(t_stub, np.nan, dtype=float))
            evt[p]['V_i_gse'] = (t_stub, np.full((len(t_stub), 3), np.nan))

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
    # 1) Try explicit rate-suffixed names (most common)
    for rate in rates:
        for pat in [f"{base}_energyspectr_omni_{rate}*", f"{base}_energyspectr_{rate}_omni*"]:
            for v in _fn.filter(data_quants.keys(), pat):
                try:
                    t, d = get_data(v)
                    if t is not None and d is not None and hasattr(d, 'ndim') and d.ndim == 2:
                        return v
                except Exception:
                    continue
    # 2) Any omni without explicit rate
    for pat in [f"{base}_energyspectr_omni*", f"{base}*energyspectr*omni*"]:
        for v in _fn.filter(data_quants.keys(), pat):
            try:
                t, d = get_data(v)
                if t is not None and d is not None and hasattr(d, 'ndim') and d.ndim == 2:
                    return v
            except Exception:
                continue
    # 3) Some datasets store omni as numberflux or differential_flux; accept any 2D spectra
    for pat in [f"{base}*omni*", f"{base}*flux*"]:
        for v in _fn.filter(data_quants.keys(), pat):
            try:
                t, d = get_data(v)
                if t is not None and d is not None and hasattr(d, 'ndim') and d.ndim == 2 and d.shape[1] >= 8:
                    return v
            except Exception:
                continue
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
