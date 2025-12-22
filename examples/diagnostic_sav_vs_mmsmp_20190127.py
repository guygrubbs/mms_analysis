"""
Diagnostic comparison for 2019-01-27 (12:15–12:55): .sav vs mms_mp
- BN: rotate B_gsm using (.sav LMN) vs (hybrid_lmn) and compare
- VN: .sav ViN vs mms_mp V_i_gse·N_sav
- DN: integrate both VNs and compare to published DN CSVs
Outputs saved under results/events_pub/2019-01-27_1215-1255/diagnostics/
Strict local caching only.
"""
from __future__ import annotations
import sys, pathlib, importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mms_mp as mp
from tools.idl_sav_import import load_idl_sav, extract_vn_series

EVENT_DIR = pathlib.Path('results/events_pub/2019-01-27_1215-1255')
OUT = EVENT_DIR / 'diagnostics'
OUT.mkdir(parents=True, exist_ok=True)
TRANGE = ('2019-01-27/12:15:00','2019-01-27/12:55:00')
PROBES = ('1','2','3','4')

# Import the minimal notplot loader from the analysis script (strict local cache)
_spec = importlib.util.spec_from_file_location('an2019', ROOT / 'examples' / 'analyze_20190127_dn_shear.py')
an = importlib.util.module_from_spec(_spec) if _spec else None
if _spec and _spec.loader:
    _spec.loader.exec_module(an)  # type: ignore
else:  # fallback
    raise RuntimeError('Unable to import examples/analyze_20190127_dn_shear.py')

sav = load_idl_sav('mp_lmn_systems_20190127_1215-1255_mp-ver2b.sav')
LMN = sav.get('lmn', {})
B_LMN = sav.get('b_lmn', {})

# Load event (B_gsm, V_i_gse, POS_gsm) at 1s cadence
_evt = an._minimal_event(TRANGE, PROBES)

# Optional physics-driven algorithmic LMN map (CDF-only at runtime)
ALG_LMN_MAP = None
if hasattr(an, '_build_algorithmic_lmn_map'):
    try:
        ALG_LMN_MAP = an._build_algorithmic_lmn_map(_evt)
    except Exception as e:
        print('Warning: failed to construct algorithmic LMN map for diagnostics:', e)

def _ensure_utc(s: pd.Series) -> pd.Series:
    idx = s.index
    try:
        if getattr(idx, 'tz', None) is None:
            s.index = idx.tz_localize('UTC')
        else:
            s.index = idx.tz_convert('UTC')
    except Exception:
        pass
    return s
def _ensure_df_utc(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    try:
        if getattr(idx, 'tz', None) is None:
            df.index = idx.tz_localize('UTC')
        else:
            df.index = idx.tz_convert('UTC')
    except Exception:
        pass
    return df
def _metrics(a: pd.Series, b: pd.Series):
    join = pd.concat([a, b], axis=1).dropna()
    if len(join) == 0:
        return np.nan, np.nan, np.nan
    diff = join.iloc[:,0] - join.iloc[:,1]
    mae = float(np.nanmean(np.abs(diff)))
    rmse = float(np.nanmean(diff**2)**0.5)
    try:
        corr = float(np.corrcoef(join.iloc[:,0].values, join.iloc[:,1].values)[0,1])
    except Exception:
        corr = np.nan
    return mae, rmse, corr

def _find_exceedance_intervals(idx: pd.DatetimeIndex, diff: pd.Series, thr: float, min_duration_s: int = 10):
    out = []
    mask = (np.abs(diff.values) > thr)
    if len(mask) == 0:
        return out
    # find contiguous segments
    start = None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        if (not on or i == len(mask)-1) and start is not None:
            end = i if not on else i
            t0 = idx[start]
            t1 = idx[end]
            dur = (t1 - t0).total_seconds()
            if dur >= min_duration_s:
                maxdiff = float(np.nanmax(np.abs(diff.values[start:end+1])))
                out.append((t0, t1, dur, maxdiff))
            start = None
    return out
def _stats_with_gaps(idx: pd.DatetimeIndex, a: pd.Series, b: pd.Series):
    """Compute basic comparison stats plus simple gap/coverage metrics.

    Both series are first reindexed onto *idx* so that coverage_fraction and
    gap counts are defined with respect to the same regular grid.
    """
    a1 = a.reindex(idx)
    b1 = b.reindex(idx)
    av = a1.values.astype(float)
    bv = b1.values.astype(float)
    valid = np.isfinite(av) & np.isfinite(bv)
    n_total = len(idx)
    n_valid = int(valid.sum())
    if n_total == 0 or n_valid == 0:
        return np.nan, np.nan, np.nan, np.nan, 0, 0, 0.0
    diff = av[valid] - bv[valid]
    mae = float(np.nanmean(np.abs(diff)))
    rmse = float(np.nanmean(diff**2)**0.5)
    try:
        corr = float(np.corrcoef(av[valid], bv[valid])[0, 1])
    except Exception:
        corr = np.nan
    max_abs = float(np.nanmax(np.abs(diff)))
    coverage = float(n_valid) / float(n_total)
    # Count contiguous gaps (runs where either series is missing)
    n_gaps = 0
    in_gap = False
    for is_bad in (~valid):
        if is_bad and not in_gap:
            in_gap = True
            n_gaps += 1
        elif not is_bad:
            in_gap = False
    return mae, rmse, corr, max_abs, n_valid, n_gaps, coverage



def _load_windows_all1243(probe: str):
    seg_path = EVENT_DIR / f"mms{probe}_DN_segments.csv"
    if not seg_path.exists():
        return []
    df = pd.read_csv(seg_path)
    # pick the window with earliest start time (longer window assumed for all_1243)
    try:
        df['t0'] = pd.to_datetime(df['t0'], utc=True)
        df['t1'] = pd.to_datetime(df['t1'], utc=True)
    except Exception:
        return []
    if len(df) == 0:
        return []
    df = df.sort_values('t0')
    first = df.iloc[0]
    return [(first['t0'], first['t1'])]

def _integrate_windowed(vn: pd.Series, grid: pd.DatetimeIndex, windows: list[tuple[pd.Timestamp,pd.Timestamp]]):
    vn1 = vn.reindex(grid).fillna(0.0)
    if not windows:
        # simple cumulative as baseline
        return pd.Series(np.cumsum(vn1.values), index=grid)
    mask = np.zeros(len(grid), dtype=bool)
    for (t0,t1) in windows:
        mask |= ((grid >= t0) & (grid <= t1))
    out = np.zeros(len(grid), dtype=float)
    accum = 0.0
    for i, use in enumerate(mask):
        if use:
            accum += vn1.values[i]
        out[i] = accum
    return pd.Series(out, index=grid)



rows_bn = []
rows_vn = []
rows_dn = []
# Consolidated statistics across BN, VN, DN (per probe and quantity)
all_stats: list[dict[str, object]] = []

bn_exceed: dict[str, list[tuple[pd.Timestamp,pd.Timestamp,float,float]]] = {}
vn_exceed: dict[str, list[tuple[pd.Timestamp,pd.Timestamp,float,float]]] = {}
dn_exceed: dict[str, list[tuple[pd.Timestamp,pd.Timestamp,float,float]]] = {}

for p in PROBES:
    key = str(p)
    if key not in _evt or 'B_gsm' not in _evt[key]:
        continue
    # Build 1 s grid
    tB, B = _evt[key]['B_gsm']
    Bdf = mp.data_loader.to_dataframe(tB, B, cols=['Bx','By','Bz'])
    Bdf = mp.data_loader.resample(Bdf, '1s')
    Bdf = _ensure_df_utc(Bdf)
    # LMN from .sav
    L = np.asarray(LMN[key]['L'], float); M = np.asarray(LMN[key]['M'], float); N = np.asarray(LMN[key]['N'], float)
    R_sav = np.vstack([L, M, N]).T
    # Optional algorithmic LMN (physics-driven, CDF-only)
    R_alg = None
    N_alg = None
    if ALG_LMN_MAP is not None and key in ALG_LMN_MAP:
        L_alg = np.asarray(ALG_LMN_MAP[key]['L'], float)
        M_alg = np.asarray(ALG_LMN_MAP[key]['M'], float)
        N_alg = np.asarray(ALG_LMN_MAP[key]['N'], float)
        R_alg = np.vstack([L_alg, M_alg, N_alg]).T

    # Hybrid LMN computed over the same time interval used in the .sav for LMN
    t_range = sav.get('trange_lmn_per_probe', {}).get(key)
    if t_range is not None and len(t_range) >= 2:
        t0 = pd.to_datetime(float(t_range[0]), unit='s', utc=True)
        t1 = pd.to_datetime(float(t_range[1]), unit='s', utc=True)
        Bwin = Bdf.loc[(Bdf.index>=t0) & (Bdf.index<=t1)]
    else:
        mid = len(Bdf)//2
        i0 = max(0, mid-200); i1 = min(len(Bdf), mid+200)
        Bwin = Bdf.iloc[i0:i1]
    if len(Bwin) < 10:
        # Fallback: use ±200 s around center
        mid = len(Bdf)//2
        i0 = max(0, mid-200); i1 = min(len(Bdf), mid+200)
        Bwin = Bdf.iloc[i0:i1]
    lmn_h = mp.coords.hybrid_lmn(Bwin.values, pos_gsm_km=None, eig_ratio_thresh=2.0)
    R_mms = np.vstack([lmn_h.L, lmn_h.M, lmn_h.N]).T

    # B in .sav LMN frame from CDF (all components), and BN in hybrid / algorithmic LMN
    B_lmn_sav = Bdf.values @ R_sav  # (N,3) → BL, BM, BN in .sav LMN
    BL_sav = pd.Series(B_lmn_sav[:, 0], index=Bdf.index, name="BL_sav")
    BM_sav = pd.Series(B_lmn_sav[:, 1], index=Bdf.index, name="BM_sav")
    BN_sav = pd.Series(B_lmn_sav[:, 2], index=Bdf.index, name='BN_sav')

    BN_mms = pd.Series((Bdf.values @ R_mms)[:, 2], index=Bdf.index, name='BN_mms')
    BN_alg = None
    if R_alg is not None:
        BN_alg = pd.Series((Bdf.values @ R_alg)[:, 2], index=Bdf.index, name='BN_alg')

    bn_grid = BN_sav.index

    # Hybrid vs .sav metrics
    mae_bn, rmse_bn, corr_bn, max_abs_bn, n_valid_bn, n_gaps_bn, cov_bn = _stats_with_gaps(bn_grid, BN_mms, BN_sav)
    diff_bn = (BN_mms - BN_sav).reindex(bn_grid)
    n_exceed = int(np.sum(np.abs(diff_bn.values) > 0.5))
    # intervals where |ΔBN|>0.5 nT for >=10 s (hybrid only, for continuity with earlier reports)
    bn_intervals = _find_exceedance_intervals(bn_grid, diff_bn, 0.5, 10)
    bn_exceed[key] = bn_intervals

    N_mms = lmn_h.N / np.linalg.norm(lmn_h.N)
    N_sav = N / np.linalg.norm(N)
    cosang = float(np.clip(np.abs(np.dot(N_mms, N_sav)), -1.0, 1.0))
    ang_deg = float(np.degrees(np.arccos(cosang)))
    rows_bn.append({
        'probe': key,
        'source': 'hybrid',
        'mae_nT': float(mae_bn),
        'rmse_nT': float(rmse_bn),
        'corr': float(corr_bn),
        'n_exceed_0p5nT': n_exceed,
        'N_angle_diff_deg': ang_deg,
    })
    # Record consolidated stats for BN (hybrid vs .sav LMN)
    all_stats.append({
        'probe': key,
        'quantity': 'BN_hybrid',
        'mae': float(mae_bn),
        'rmse': float(rmse_bn),
        'correlation': float(corr_bn),
        'max_abs_diff': float(max_abs_bn),
        'n_samples': int(n_valid_bn),
        'n_gaps': int(n_gaps_bn),
        'coverage_fraction': float(cov_bn),
    })

    # Algorithmic vs .sav metrics (if available)
    if BN_alg is not None and N_alg is not None:
        mae_bn_a, rmse_bn_a, corr_bn_a, max_abs_bn_a, n_valid_bn_a, n_gaps_bn_a, cov_bn_a = _stats_with_gaps(bn_grid, BN_alg, BN_sav)
        diff_bn_a = (BN_alg - BN_sav).reindex(bn_grid)
        n_exceed_a = int(np.sum(np.abs(diff_bn_a.values) > 0.5))
        N_alg_u = N_alg / np.linalg.norm(N_alg)
        cosang_a = float(np.clip(np.abs(np.dot(N_alg_u, N_sav)), -1.0, 1.0))
        ang_deg_a = float(np.degrees(np.arccos(cosang_a)))
        rows_bn.append({
            'probe': key,
            'source': 'algorithmic',
            'mae_nT': float(mae_bn_a),
            'rmse_nT': float(rmse_bn_a),
            'corr': float(corr_bn_a),
            'n_exceed_0p5nT': n_exceed_a,
            'N_angle_diff_deg': ang_deg_a,
        })
        all_stats.append({
            'probe': key,
            'quantity': 'BN_algorithmic',
            'mae': float(mae_bn_a),
            'rmse': float(rmse_bn_a),
            'correlation': float(corr_bn_a),
            'max_abs_diff': float(max_abs_bn_a),
            'n_samples': int(n_valid_bn_a),
            'n_gaps': int(n_gaps_bn_a),
            'coverage_fraction': float(cov_bn_a),
        })

    # Plot overlay for BN (hybrid and algorithmic vs .sav LMN)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(BN_sav.index, BN_sav.values, label='BN (sav LMN)', lw=1.1)
    ax.plot(BN_mms.index, BN_mms.values, label='BN (hybrid LMN)', lw=1.1, alpha=0.85)
    if BN_alg is not None:
        ax.plot(BN_alg.index, BN_alg.values, label='BN (algorithmic LMN)', lw=1.1, alpha=0.8)
    ax.set_title(f'MMS{key} BN comparison (.sav LMN vs hybrid/algorithmic LMN)')
    ax.set_ylabel('B_N (nT)'); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / f'bn_overlay_mms{key}.png', dpi=220); plt.close(fig)

    # Direct BL/BM/BN comparison (active): .sav B_LMN vs B_gsm rotated by .sav LMN
    # This block uses the verified native ordering of the .sav B_LMN arrays,
    # which store components as (B_N, B_L, B_M) in columns (0, 1, 2).
    if key in B_LMN:
        b_entry = B_LMN[key]
        t_raw = np.asarray(b_entry.get('t', []), float)
        blmn = np.asarray(b_entry.get('blmn', []), float)
        if t_raw.size and blmn.size:
            idx_sav = pd.to_datetime(t_raw, unit='s', utc=True)

            def _extract_component(arr: np.ndarray, idx: int) -> np.ndarray:
                """Extract a single component from possible (N,3), (3,N), or flat layouts."""
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    return arr[:, idx]
                if arr.ndim == 2 and arr.shape[0] == 3:
                    return arr[idx, :]
                flat = arr.reshape(-1)
                start = idx
                step = 3
                return flat[start::step] if flat.size >= (idx + 1) else flat

            # Common 1 s grid over the event TRANGE
            t0_evt = pd.to_datetime(TRANGE[0], utc=True)
            t1_evt = pd.to_datetime(TRANGE[1], utc=True)
            grid_evt = pd.date_range(t0_evt, t1_evt, freq='1s')

            # CDF-rotated B components in .sav LMN on the same grid
            B_cdf_grid = {
                'L': BL_sav.resample('1s').mean().reindex(grid_evt),
                'M': BM_sav.resample('1s').mean().reindex(grid_evt),
                'N': BN_sav.resample('1s').mean().reindex(grid_evt),
            }

            # Map desired (L, M, N) onto the native (B_N, B_L, B_M) column ordering
            # in .sav B_LMN
            for comp_name, idx_comp, ylabel, dlabel in [
                ('L', 1, 'B_L (nT)', 'ΔB_L (nT)'),  # raw col 1 is B_L
                ('M', 2, 'B_M (nT)', 'ΔB_M (nT)'),  # raw col 2 is B_M
                ('N', 0, 'B_N (nT)', 'ΔB_N (nT)'),  # raw col 0 is B_N
            ]:
                vals_sav = _extract_component(blmn, idx_comp)
                if vals_sav.size == 0:
                    continue
                B_sav_direct_raw = pd.Series(
                    vals_sav, index=idx_sav, name=f'B{comp_name}_sav_direct'
                )

                B_cdf_1s = B_cdf_grid[comp_name]
                B_sav_1s = B_sav_direct_raw.resample('1s').mean().reindex(grid_evt)

                mae_bd, rmse_bd, corr_bd, max_bd, n_valid_bd, n_gaps_bd, cov_bd = _stats_with_gaps(
                    grid_evt, B_cdf_1s, B_sav_1s
                )

                # Store consolidated stats for direct B component comparison
                all_stats.append({
                    'probe': key,
                    'quantity': f'B{comp_name}_direct',
                    'mae': float(mae_bd),
                    'rmse': float(rmse_bd),
                    'correlation': float(corr_bd),
                    'max_abs_diff': float(max_bd),
                    'n_samples': int(n_valid_bd),
                    'n_gaps': int(n_gaps_bd),
                    'coverage_fraction': float(cov_bd),
                })

                # Two-panel figure: overlay + difference (ΔB = CDF_rotated − .sav B_LMN)
                fig2, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

                ax_top.plot(
                    grid_evt,
                    B_cdf_1s,
                    label=f'B_{comp_name} (CDF rotated by .sav LMN)',
                    lw=1.0,
                )
                ax_top.plot(
                    grid_evt,
                    B_sav_1s,
                    label=f'B_{comp_name} (.sav B_LMN direct)',
                    lw=1.0,
                    alpha=0.85,
                )
                ax_top.set_ylabel(ylabel)
                ax_top.set_title(
                    f'MMS{key} Direct B_{comp_name} comparison (.sav B_LMN vs CDF)'
                )
                ax_top.grid(True, alpha=0.3)
                ax_top.legend()

                diff_dir = B_cdf_1s - B_sav_1s
                ax_bot.plot(grid_evt, diff_dir, lw=1.0, color='C2')
                ax_bot.axhline(0.0, color='k', ls='--', lw=0.8)
                ax_bot.axhline(0.5, color='r', ls=':', lw=0.8, alpha=0.7)
                ax_bot.axhline(-0.5, color='r', ls=':', lw=0.8, alpha=0.7)
                ax_bot.set_ylabel(dlabel)
                ax_bot.set_xlabel('Time (UTC)')
                ax_bot.grid(True, alpha=0.3)

                fig2.tight_layout()
                fig2.savefig(OUT / f'comparison_b{comp_name.lower()}_direct_mms{key}.png', dpi=220)
                plt.close(fig2)
    # VN: sav vs mms_mp
    if 'V_i_gse' in _evt[key]:
        tV, V = _evt[key]['V_i_gse']
        Vdf = mp.data_loader.to_dataframe(tV, V, cols=['Vx','Vy','Vz'])
        Vdf = mp.data_loader.resample(Vdf, '1s')
        VN_mms = pd.Series((Vdf.values @ R_sav)[:,2], index=Vdf.index, name='VN_mms')
        VN_mms = _ensure_utc(VN_mms)
    else:
        VN_mms = pd.Series(dtype=float)
    # sav ViN
    t_sav, vn_sav = extract_vn_series(sav)[key]
    VN_sav = mp.data_loader.to_dataframe(t_sav, vn_sav, cols=['ViN']).resample('1s').mean()['ViN']
    VN_sav = _ensure_utc(VN_sav)
    # Align and stats on a common 1 s grid over the TRANGE
    t0 = pd.to_datetime(TRANGE[0], utc=True); t1 = pd.to_datetime(TRANGE[1], utc=True)
    grid_v = pd.date_range(t0, t1, freq='1s')
    vn_sav_g = VN_sav.reindex(grid_v, method='nearest')
    vn_mms_g = VN_mms.reindex(grid_v, method='nearest') if len(VN_mms) else pd.Series(index=grid_v, dtype=float)
    vn_join = pd.DataFrame({'ViN': vn_sav_g, 'VN_mms': vn_mms_g}).dropna()
    if len(vn_join):
        dv = vn_join['ViN'] - vn_join['VN_mms']
        mae_vn, rmse_vn, corr_vn, max_vn, n_valid_vn, n_gaps_vn, cov_vn = _stats_with_gaps(
            grid_v, vn_mms_g, vn_sav_g
        )
        rows_vn.append({
            'probe': key,
            'vn_mae_km_s': float(mae_vn),
            'vn_rmse_km_s': float(rmse_vn),
            'vn_corr': float(corr_vn),
        })
        # intervals where |ΔVN|>50 km/s for >=10 s
        vn_intervals = _find_exceedance_intervals(vn_join.index, dv, 50.0, 10)
        vn_exceed[key] = vn_intervals

        # Record consolidated stats for VN
        all_stats.append({
            'probe': key,
            'quantity': 'VN',
            'mae': float(mae_vn),
            'rmse': float(rmse_vn),
            'correlation': float(corr_vn),
            'max_abs_diff': float(max_vn),
            'n_samples': int(n_valid_vn),
            'n_gaps': int(n_gaps_vn),
            'coverage_fraction': float(cov_vn),
        })

        # Plot
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(vn_join.index, vn_join['ViN'].values, label='ViN (.sav)', lw=1.0)
        ax.plot(vn_join.index, vn_join['VN_mms'].values, label='ViN (mms_mp)', lw=1.0, alpha=0.8)
        ax.set_title(f'MMS{key} V_N comparison (.sav vs mms_mp)'); ax.set_ylabel('V_N (km/s)')
        ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
        fig.savefig(OUT / f'vn_overlay_mms{key}.png', dpi=220); plt.close(fig)

    # DN: integrate VNs over the same cold-ion windows (all_1243 heuristic)
    t0 = pd.to_datetime(TRANGE[0], utc=True); t1 = pd.to_datetime(TRANGE[1], utc=True)
    grid = pd.date_range(t0, t1, freq='1s')
    windows = _load_windows_all1243(key)
    dn_sav = _integrate_windowed(VN_sav, grid, windows)
    dn_mms = _integrate_windowed(VN_mms if len(VN_mms) else pd.Series(dtype=float), grid, windows)
    dn_df = pd.DataFrame({'DN_sav_km': dn_sav.values, 'DN_mms_km': dn_mms.values}, index=grid)

    # Primary DN comparison (Option B): mms_mp vs .sav, restricted to cold-ion windows
    if windows:
        win_mask = np.zeros(len(grid), dtype=bool)
        for (t0w, t1w) in windows:
            win_mask |= ((grid >= t0w) & (grid <= t1w))
    else:
        win_mask = np.ones(len(grid), dtype=bool)
    valid_dn = pd.Series(win_mask, index=grid)
    idx_dn = dn_df.index[valid_dn.values]
    dn_sav_win = dn_df.loc[idx_dn, 'DN_sav_km'] if len(idx_dn) else pd.Series(dtype=float)
    dn_mms_win = dn_df.loc[idx_dn, 'DN_mms_km'] if len(idx_dn) else pd.Series(dtype=float)
    mae_dn, rmse_dn, corr_dn, max_dn, n_valid_dn, n_gaps_dn, cov_dn = _stats_with_gaps(
        idx_dn, dn_mms_win, dn_sav_win
    )
    rows_dn.append({
        'probe': key,
        'dn_mae_km': float(mae_dn),
        'dn_rmse_km': float(rmse_dn),
        'dn_corr': float(corr_dn),
    })
    # intervals where |ΔDN|>200 km for >=10 s (within cold-ion windows)
    dn_diff_sav = (dn_df['DN_mms_km'] - dn_df['DN_sav_km']).where(valid_dn)
    dn_intervals = _find_exceedance_intervals(dn_df.index, dn_diff_sav.fillna(0.0), 200.0, 10)
    dn_exceed[key] = dn_intervals

    # Record consolidated stats for DN (within cold-ion windows)
    all_stats.append({
        'probe': key,
        'quantity': 'DN',
        'mae': float(mae_dn),
        'rmse': float(rmse_dn),
        'correlation': float(corr_dn),
        'max_abs_diff': float(max_dn),
        'n_samples': int(n_valid_dn),
        'n_gaps': int(n_gaps_dn),
        'coverage_fraction': float(cov_dn),
    })

    # Secondary check: published DN_mms (all_1243) may have no overlap in 12:15–12:55
    dn_csv = EVENT_DIR / f'dn_mms{key}_all_1243.csv'
    if dn_csv.exists():
        pub = pd.read_csv(dn_csv, index_col=0, parse_dates=True)
        try:
            pub.index = pd.to_datetime(pub.index, utc=True)
        except Exception:
            pass
        pub = pub.reindex(grid)['DN_km']
        dn_df['DN_published_km'] = pub.values
    # Save per-probe DN overlay

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(dn_df.index, dn_df['DN_sav_km'], label='DN (sav)', lw=1.0)
    ax.plot(dn_df.index, dn_df['DN_mms_km'], label='DN (mms_mp)', lw=1.0, alpha=0.8)
    if 'DN_published_km' in dn_df:
        ax.plot(dn_df.index, dn_df['DN_published_km'], label='DN (published all_1243)', lw=1.0, alpha=0.8)
    ax.set_ylabel('DN (km)'); ax.grid(True, alpha=0.3); ax.legend(); ax.set_title(f'MMS{key} DN comparison')
    fig.tight_layout(); fig.savefig(OUT / f'dn_overlay_mms{key}.png', dpi=220); plt.close(fig)

# Save summary CSVs
if rows_bn:
    pd.DataFrame(rows_bn).to_csv(OUT / 'bn_difference_stats.csv', index=False)
if rows_vn:
    pd.DataFrame(rows_vn).to_csv(OUT / 'vn_difference_stats.csv', index=False)
if rows_dn:
    pd.DataFrame(rows_dn).to_csv(OUT / 'dn_difference_stats.csv', index=False)

# Consolidated statistics table across BN (hybrid and direct), VN, and DN
if all_stats:
    pd.DataFrame(all_stats).to_csv(OUT / 'comparison_statistics_consolidated.csv', index=False)

# Write a markdown report with metrics and exceedance intervals
report = OUT / 'diagnostic_comparison.md'
lines = [
    '# Diagnostic Comparison (.sav vs mms_mp): 2019-01-27 12:15–12:55',
    '',
    '- Strict local caching: all inputs loaded from local CDFs only (no re-downloads).',
    '- DN integration uses the same cold-ion windowing as published all_1243 outputs.',
    '',
]
if rows_bn:
    lines.append('## BN differences vs .sav LMN (hybrid and algorithmic)')
    sources = sorted({r.get('source', 'hybrid') for r in rows_bn})
    for src in sources:
        lines.append(f'### Source: {src}')
        for r in rows_bn:
            if r.get('source', 'hybrid') != src:
                continue
            lines.append(
                f"- MMS{r['probe']}: MAE={r['mae_nT']:.3f} nT, RMSE={r['rmse_nT']:.3f} nT, "
                f"corr={r['corr']:.3f}, count(|Δ|>0.5 nT)={r['n_exceed_0p5nT']}, "
                f"N-angle diff≈{r['N_angle_diff_deg']:.1f}°"
            )
    # Exceedance intervals summary (hybrid LMN only)
    lines.append('  Exceedance intervals where |ΔBN|>0.5 nT for ≥10 s (hybrid LMN):')
    for key in PROBES:
        segs = bn_exceed.get(key, [])
        if segs:
            for (t0, t1, dur, maxd) in segs[:5]:  # show up to 5 per probe
                lines.append(
                    f"  - MMS{key}: {t0} → {t1} (dur={int(dur)} s), max |ΔBN|={maxd:.2f} nT"
                )

if rows_vn:
    lines.append('\n## VN differences (.sav ViN vs mms_mp V_i·N_sav)')
    for r in rows_vn:
        lines.append(f"- MMS{r['probe']}: MAE={r['vn_mae_km_s']:.1f} km/s, RMSE={r['vn_rmse_km_s']:.1f} km/s, corr={r['vn_corr']:.3f}")
    lines.append('  Exceedance intervals where |ΔVN|>50 km/s for ≥10 s:')
    for key in PROBES:
        segs = vn_exceed.get(key, [])
        if segs:
            for (t0,t1,dur,maxd) in segs[:5]:
                lines.append(f"  - MMS{key}: {t0} → {t1} (dur={int(dur)} s), max |ΔVN|={maxd:.1f} km/s")

if rows_dn:
    lines.append('\n## DN differences (mms_mp vs published all_1243)')
    for r in rows_dn:
        lines.append(f"- MMS{r['probe']}: MAE={r['dn_mae_km']:.0f} km, RMSE={r['dn_rmse_km']:.0f} km, corr={r['dn_corr']:.3f}")
    lines.append('  Exceedance intervals where |ΔDN|>200 km for ≥10 s:')
    for key in PROBES:
        segs = dn_exceed.get(key, [])
        if segs:
            for (t0,t1,dur,maxd) in segs[:5]:
                lines.append(f"  - MMS{key}: {t0} → {t1} (dur={int(dur)} s), max |ΔDN|={maxd:.0f} km")

lines.append('\nNotes:')
lines.append('- BN comparison isolates coordinate-system differences (two LMN triads applied to the same B_gsm).')
lines.append('- VN comparison uses .sav LMN for rotation to isolate instrument/pipeline differences.')
lines.append('- DN comparisons use the cold-ion windows from mms*_DN_segments.csv to match the published methodology.')
report.write_text('\n'.join(lines), encoding='utf-8')
print(f'Wrote diagnostics to {OUT.resolve()}')

