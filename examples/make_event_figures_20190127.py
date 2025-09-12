"""
Generate event-specific visuals for MMS 2019-01-27 12:15–12:55 using .sav + mms_mp.
Outputs go to results/events/2019-01-27_1215-1255.

Figures:
- Per-probe ViN overlay (IDL .sav vs mms_mp) + MAE, saved as PNG
- Per-probe ViN difference histogram and timing delta histogram (ms), PNG
- 4-panel stack overlay of ViN for MMS1–4, PNG
- DES and DIS spectrograms for MMS1–4 (time × energy, flux), PNG (with QL fallbacks and clear annotations)
- DN distance comparisons on vt_mms# intervals (from the IDL .pro), overlays and CSVs
- Save aligned CSVs (ViN) for reproducibility
"""
from __future__ import annotations
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import re

# Ensure repo root in path for sibling imports
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mms_mp as mp
from tools.idl_sav_import import load_idl_sav, extract_vn_series

# Optional: direct CDF reading without PySPEDAS for this event
try:
    import cdflib as _cdflib
    _HAVE_CDFLIB = True
except Exception:
    _HAVE_CDFLIB = False

EVENT_DIR = pathlib.Path('results/events_pub/2019-01-27_1215-1255')
EVENT_DIR.mkdir(parents=True, exist_ok=True)

TRANGE = ('2019-01-27/12:15:00', '2019-01-27/12:55:00')
PROBES = ('1','2','3','4')


def _load_event():
    # FGM srvy, FPI fast; ephem required
    return mp.load_event(list(TRANGE), probes=list(PROBES), include_ephem=True,
                         data_rate_fgm='srvy', data_rate_fpi='fast')


def _parse_vt_from_idl_pro(path='references/IDL_Code/requested_mp_motion_givenlmn_vion.pro') -> dict:
    txt = pathlib.Path(path).read_text(encoding='utf-8', errors='ignore')
    # Focus on the 2019-01-27 block
    block = re.findall(r"If time_string\(trange_full\[0\]\) eq '2019-01-27/04:00:00' then begin(.*?)endif", txt, flags=re.S)
    intervals = { '1': [], '2': [], '3': [], '4': [] }
    if block:
        b = block[0]
        for sc, tag in [('1','vt_mms1'), ('2','vt_mms2'), ('3','vt_mms3'), ('4','vt_mms4')]:
            m = re.search(tag + r"= time_double\(\[(.*?)\]", b, flags=re.S)
            if not m:
                continue
            arr = m.group(1)
            times = re.findall(r"'([0-9\-/:\.]+)'", arr)
            # Pairs [start, stop] repeating
            for i in range(0, len(times), 2):
                if i+1 < len(times):
                    intervals[sc].append((times[i], times[i+1]))
    return intervals


def _rotate_vi_to_sav_lmn(evt, sav):
    """Return dict probe -> DataFrame with columns [ViN_sav, ViN_mmsmp], UTC index.
    Uses .sav LMN as authoritative (per-probe if provided).
    """
    vn_dict = {}
    vn_sav = extract_vn_series(sav)
    lmn_per_probe = sav.get('lmn', {})
    for p in PROBES:
        key = str(p)
        if 'V_i_gse' not in evt[key] or key not in vn_sav:
            continue
        # Select LMN for this probe; fallback: average across probes if needed
        if key in lmn_per_probe:
            L = np.asarray(lmn_per_probe[key]['L'], float)
            M = np.asarray(lmn_per_probe[key]['M'], float)
            N = np.asarray(lmn_per_probe[key]['N'], float)
        elif lmn_per_probe:
            # average across available LMNs
            L = np.nanmean([v['L'] for v in lmn_per_probe.values()], axis=0)
            M = np.nanmean([v['M'] for v in lmn_per_probe.values()], axis=0)
            N = np.nanmean([v['N'] for v in lmn_per_probe.values()], axis=0)
        else:
            raise RuntimeError('Missing LMN in .sav; cannot proceed')

        t_vi, Vi_gse = evt[key]['V_i_gse']
        Vi_df = mp.data_loader.to_dataframe(t_vi, Vi_gse, cols=['Vx','Vy','Vz'])
        Vi_lmn = Vi_df.values @ np.vstack([L, M, N]).T
        vn_bulk = Vi_lmn[:,2]
        series = pd.Series(vn_bulk, index=Vi_df.index.tz_localize('UTC') if Vi_df.index.tz is None else Vi_df.index.tz_convert('UTC'))

        t_s, vn_s = vn_sav[key]
        sav_idx = pd.to_datetime(t_s, unit='s', utc=True)
        # Compute nearest deltas
        nearest = series.index.searchsorted(sav_idx)
        nearest = np.clip(nearest, 0, len(series)-1)
        nearest_times = series.index[nearest]
        delta_ms = (sav_idx - nearest_times).total_seconds()*1e3
        # Interpolate by nearest
        vn_interp = series.reindex(sav_idx, method='nearest')
        out = pd.DataFrame({'ViN_sav': vn_s, 'ViN_mmsmp': vn_interp.values, 'dt_ms': delta_ms}, index=sav_idx)
        out.index.name = 'time_utc'
        vn_dict[key] = out
    return vn_dict


def _draw_vin_overlay(ax, df: pd.DataFrame, vt_intervals=None, probe:str|None=None, title: str|None=None):
    ax.plot(df.index, df['ViN_sav'], label='IDL .sav ViN', lw=1.2)
    ax.plot(df.index, df['ViN_mmsmp'], label='mms_mp ViN', lw=1.0, alpha=0.9)
    ax.set_ylabel('V_N (km/s)')
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    diff = (df['ViN_mmsmp'] - df['ViN_sav']).dropna()
    mae = float(np.nanmean(np.abs(diff))) if len(diff) else np.nan
    ax.text(0.01, 0.95, f"MAE = {mae:.2f} km/s", transform=ax.transAxes, va='top', ha='left')
    if vt_intervals and probe and probe in vt_intervals:
        for (t0s, t1s) in vt_intervals[probe]:
            t0 = pd.to_datetime(t0s, utc=True)
            t1 = pd.to_datetime(t1s, utc=True)
            if t1 < t0:
                t0, t1 = t1, t0
            ax.axvspan(t0, t1, color='k', alpha=0.08)
    ax.legend(loc='upper right', frameon=False)


def _plot_vin_overlay(df: pd.DataFrame, title: str, out_path: pathlib.Path, vt_intervals=None, probe:str|None=None):
    fig, ax = plt.subplots(figsize=(11, 4))
    _draw_vin_overlay(ax, df, vt_intervals=vt_intervals, probe=probe, title=title)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_diff_hist(df: pd.DataFrame, title: str, out_path: pathlib.Path):
    diff = (df['ViN_mmsmp'] - df['ViN_sav']).dropna()
    if diff.empty:
        diff = pd.Series([0.0])
    fig, ax = plt.subplots(figsize=(5.5,4))
    ax.hist(diff, bins=60, alpha=0.85, color='#4c78a8', edgecolor='white')
    ax.set_xlabel('ViN_mmsmp - ViN_sav (km/s)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_offset_hist(df: pd.DataFrame, title: str, out_path: pathlib.Path):
    dtms = pd.Series(df['dt_ms']).dropna()
    if dtms.empty:
        dtms = pd.Series([0.0])
    fig, ax = plt.subplots(figsize=(5.5,4))
    ax.hist(dtms, bins=60, alpha=0.85, color='#f58518', edgecolor='white')
    ax.set_xlabel('Nearest alignment Δt (ms)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _stack_vin(vn: dict, out_path: pathlib.Path, vt_intervals=None):
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(11,8))
    for i, p in enumerate(PROBES):
        key = str(p)
        df = vn.get(key)
        if df is None: continue
        ax = axes[i]
        ax.plot(df.index, df['ViN_sav'], label='IDL .sav ViN', lw=1.0)
        ax.plot(df.index, df['ViN_mmsmp'], label='mms_mp ViN', lw=0.9, alpha=0.9)
        if vt_intervals and key in vt_intervals:
            for (t0s, t1s) in vt_intervals[key]:
                t0 = pd.to_datetime(t0s, utc=True)
                t1 = pd.to_datetime(t1s, utc=True)
                if t1 < t0:
                    t0, t1 = t1, t0
                ax.axvspan(t0, t1, color='k', alpha=0.06)
        ax.set_ylabel(f'MMS{p}\nV_N (km/s)')
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(loc='upper right', frameon=False)
    axes[-1].set_xlabel('Time (UTC)')
    fig.suptitle('ViN comparison (2019-01-27 12:15–12:55)')
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _spectrograms(evt, out_dir: pathlib.Path):
    """Generate spectrograms via .sav/CDF/PySPEDAS in that order (event-only policy).
    We avoid network fetches and rely on local files; PySPEDAS is used as last resort.
    """
    def _cdf_to_spectrograms(probe: str, species: str):
        # Locate local CDFs under mms_data for the event trange
        base_dir = pathlib.Path('mms_data')
        rate_opts = ['fast', 'srvy']
        for rate in rate_opts:
            # example path: mms_data/fpi_fast/l2/mms1/fpi/mms1_fpi_fast_dis-dist_20190127121500_v*.cdf
            sub = f'fpi_{rate}/l2/mms{probe}/fpi'
            if not (base_dir / sub).exists():
                continue
            for kind in ['spectr', 'dist']:
                glob_pat = f'mms{probe}_fpi_{rate}_{species}-{kind}_20190127*.cdf'
                for cdf_path in (base_dir / sub).glob(glob_pat):
                    try:
                        cdf = _cdflib.CDF(str(cdf_path))
                        # Common variables
                        t = cdf.varget('epoch')
                        # Energy axis can be 'energy' or similar
                        for ename in ('energy', 'e_energy', 'energy_electron', 'energy_ion'):
                            try:
                                e = cdf.varget(ename)
                                if e is not None and np.size(e) > 4:
                                    break
                            except Exception:
                                e = None
                        if e is None:
                            continue
                        if kind == 'spectr':
                            # Omni spectrogram (time, energy)
                            for vname in cdf.cdf_info()['rVariables']:
                                if 'omni' in vname and 'energy' not in vname:
                                    omni = cdf.varget(vname)
                                    if omni is not None and omni.size > e.size:
                                        yield t, e, omni
                        else:
                            # 4D flux — try typical variable names
                            for vname in cdf.cdf_info()['zVariables'] + cdf.cdf_info()['rVariables']:
                                if ('flux' in vname or 'dist' in vname) and 'omni' not in vname:
                                    try:
                                        flux4d = cdf.varget(vname)
                                    except Exception:
                                        flux4d = None
                                    if flux4d is None or np.ndim(flux4d) != 4:
                                        continue
                                    # collapse and return as (time, energy)
                                    try:
                                        dat2d = mp.spectra._collapse_fpi(flux4d, e_len=np.size(e), method='sum')
                                        yield t, e, dat2d
                                    except Exception:
                                        continue
                    except Exception:
                        continue
        return

    # Try CDF first if cdflib is available
    have_any = {'dis': False, 'des': False}
    if _HAVE_CDFLIB:
        for p in PROBES:
            # DIS
            produced = False
            for t, e, dat2d in _cdf_to_spectrograms(p, 'dis'):
                fig = mp.spectra.generic_spectrogram(t, e, dat2d, log10=True, ylabel='E$_i$ (eV)', title='Ion energy flux')
                fig.savefig(out_dir / f'mms{p}_DIS_omni.png', dpi=220)
                plt.close(fig)
                produced = True
                have_any['dis'] = True
                break
            if not produced:
                # DES
                for t, e, dat2d in _cdf_to_spectrograms(p, 'des'):
                    fig = mp.spectra.generic_spectrogram(t, e, dat2d, log10=True, ylabel='E$_e$ (eV)', title='Electron energy flux')
                    fig.savefig(out_dir / f'mms{p}_DES_omni.png', dpi=220)
                    plt.close(fig)
                    have_any['des'] = True
                    break

    # If not produced via CDF, fall back to local PySPEDAS data (no re-downloads)
    if not (have_any['dis'] and have_any['des']):
        info = mp.data_loader.force_load_all_plasma_spectrometers(list(TRANGE), probes=list(PROBES), rates=['fast','srvy'], verbose=True)
        from pyspedas import get_data
        for p in PROBES:
            # DIS
            dis = info[p]['dis']
            if dis['omni_var'] and dis['energy_var']:
                t, omni = get_data(dis['omni_var'])
                _, e = get_data(dis['energy_var'])
                if t is not None and omni is not None and e is not None and np.size(omni) > 0:
                    fig = mp.spectra.generic_spectrogram(t, e, omni, log10=True, ylabel='E$_i$ (eV)', title='Ion energy flux')
                    fig.savefig(out_dir / f'mms{p}_DIS_omni.png', dpi=220)
                    plt.close(fig)
            # DES
            des = info[p]['des']
            if des['omni_var'] and des['energy_var']:
                t, omni = get_data(des['omni_var'])
                _, e = get_data(des['energy_var'])
                if t is not None and omni is not None and e is not None and np.size(omni) > 0:
                    fig = mp.spectra.generic_spectrogram(t, e, omni, log10=True, ylabel='E$_e$ (eV)', title='Electron energy flux')
                    fig.savefig(out_dir / f'mms{p}_DES_omni.png', dpi=220)
                    plt.close(fig)

    # As a last resort, create non-empty placeholders to avoid empty figures
    for p in PROBES:
        for tag in ['DIS','DES']:
            f = out_dir / f'mms{p}_{tag}_omni.png'
            if not f.exists():
                fig, ax = plt.subplots(figsize=(7,3))
                ax.text(0.5, 0.5, f'MMS{p} {tag} spectrogram not available\n(all local fallbacks exhausted)', ha='center', va='center')
                ax.set_axis_off()
                fig.tight_layout()
                fig.savefig(f, dpi=150)
                plt.close(fig)


def _compute_dn(evt, vn_dict: dict, vt_intervals: dict, out_dir: pathlib.Path):
    # Compute DN per vt interval per probe using mms_mp.motion.integrate_disp on ViN_mmsmp
    from mms_mp.motion import integrate_disp
    for p in PROBES:
        key = str(p)
        df = vn_dict.get(key)
        if df is None: continue
        segments = vt_intervals.get(key, [])
        if not segments:
            continue
        # Build a VN series with datetime index
        vn_series = pd.Series(df['ViN_mmsmp'].values, index=df.index)
        dn_rows = []
        for i, (t0s, t1s) in enumerate(segments, start=1):
            t0 = pd.to_datetime(t0s, utc=True)
            t1 = pd.to_datetime(t1s, utc=True)
            if t1 < t0:
                t0, t1 = t1, t0
            seg = vn_series.loc[t0:t1]
            if seg.empty:
                continue
            # Convert to seconds grid and integrate
            times = seg.index.view('int64')/1e9
            vnvals = seg.values
            res = integrate_disp(times, vnvals, scheme='trap')  # DispResult
            dn_km = float(res.disp_km[-1] - res.disp_km[0])
            dn_rows.append({'segment': i, 't0': str(t0), 't1': str(t1), 'DN_km': dn_km})
        if dn_rows:
            dn_df = pd.DataFrame(dn_rows)
            dn_df.to_csv(out_dir / f'mms{p}_DN_segments.csv', index=False)
            # Quick bar chart
            fig, ax = plt.subplots(figsize=(6,3.5))
            ax.bar([r['segment'] for r in dn_rows], [r['DN_km'] for r in dn_rows], color='#54a24b')
            ax.set_xlabel('Segment #')
            ax.set_ylabel('Integrated distance DN (km)')
            ax.set_title(f'MMS{p} DN over vt_mms{p} segments')
            fig.tight_layout()
            fig.savefig(out_dir / f'mms{p}_DN_segments.png', dpi=200)
            plt.close(fig)


def _write_metrics(vn: dict, out_dir: pathlib.Path):
    import json
    metrics = {'probes': {}, 'overall': {}}
    maes = []
    for p in PROBES:
        key = str(p)
        df = vn.get(key)
        if df is None or df.empty:
            continue
        diff = (df['ViN_mmsmp'] - df['ViN_sav']).astype(float)
        mae = float(np.nanmean(np.abs(diff))) if diff.size else np.nan
        rmse = float(np.sqrt(np.nanmean(diff**2))) if diff.size else np.nan
        bias = float(np.nanmean(diff)) if diff.size else np.nan
        # Pearson correlation
        valid = df[['ViN_mmsmp','ViN_sav']].dropna()
        if len(valid) > 1:
            c = np.corrcoef(valid['ViN_mmsmp'], valid['ViN_sav'])[0,1]
            corr = float(c)
        else:
            corr = np.nan
        dt = pd.Series(df.get('dt_ms', pd.Series(dtype=float))).dropna()
        metrics['probes'][key] = {
            'mae_km_s': mae,
            'rmse_km_s': rmse,
            'bias_km_s': bias,
            'corr_pearson': corr,
            'n_points': int(len(df)),
            'dt_ms_mean': float(dt.mean()) if not dt.empty else np.nan,
            'dt_ms_median': float(dt.median()) if not dt.empty else np.nan,
            'dt_ms_p95': float(dt.quantile(0.95)) if not dt.empty else np.nan,
        }
        maes.append(mae)
    if maes:
        metrics['overall']['mae_km_s_mean'] = float(np.nanmean(maes))
        metrics['overall']['mae_km_s_max'] = float(np.nanmax(maes))
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))


def main():
    evt = _load_event()
    sav = load_idl_sav('references/IDL_Code/mp_lmn_systems_20190127_1215_1255_mp_ver2.sav')

    # Rotate Vi → VN using .sav LMN and align to .sav times
    vn = _rotate_vi_to_sav_lmn(evt, sav)

    # Parse vt intervals from the IDL .pro for this event
    vt = _parse_vt_from_idl_pro()

    # Save aligned CSVs and individual overlays/histograms
    for p in PROBES:
        key = str(p)
        df = vn.get(key)
        if df is None: continue
        csv_path = EVENT_DIR / f'vn_probe{key}.csv'
        df.to_csv(csv_path)
        _plot_vin_overlay(df, f'MMS{p} ViN overlay', EVENT_DIR / f'vn_overlay_mms{p}.png', vt_intervals=vt, probe=key)
        _plot_diff_hist(df, f'MMS{p} ViN difference', EVENT_DIR / f'vn_diff_hist_mms{p}.png')
        _plot_offset_hist(df, f'MMS{p} time-offset QA', EVENT_DIR / f'vn_offset_hist_mms{p}.png')

    # 4-panel stack
    _stack_vin(vn, EVENT_DIR / 'vn_overlay_stack.png', vt_intervals=vt)

    # Spectrograms
    _spectrograms(evt, EVENT_DIR)

    # DN integration over IDL vt intervals
    _compute_dn(evt, vn, vt, EVENT_DIR)

    # Metrics JSON
    _write_metrics(vn, EVENT_DIR)

    print(f'Figures and data written to: {EVENT_DIR.resolve()}')


if __name__ == '__main__':
    main()

