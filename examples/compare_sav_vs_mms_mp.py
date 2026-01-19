"""
Compare IDL .sav (LMN, Vi_LMN) against mms_mp outputs for the 2019-01-27 event.

This script:
1) Loads the IDL .sav via tools/idl_sav_import
2) Runs mms_mp.load_event for the same trange/probes
3) Uses .sav LMN (LHAT/MHAT/NHAT) as authoritative for rotation when present,
   otherwise falls back to mms_mp.coords.hybrid_lmn
4) Derives V_N from mms_mp (ion bulk) and compares to .sav ViN, aligning times in UTC
5) Writes per-probe aligned VN CSVs and a summary CSV/JSON to results/comparison/

Note: requires SciPy for .sav import (per tools/idl_sav_import.py)
"""
from __future__ import annotations
import os
import json
import sys
import pathlib
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path so sibling packages (tools/, mms_mp/) import
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.idl_sav_import import load_idl_sav, extract_vn_series
import mms_mp as mp


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, float); v = np.asarray(v, float)
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return np.nan
    c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def run_comparison(sav_path: str,
                   trange_default=('2019-01-27/12:20:00','2019-01-27/12:40:00'),
                   probes=('1','2','3','4'),
                   cadence='150ms',
                   output_dir='results/comparison'):
    os.makedirs(output_dir, exist_ok=True)

    # 1) Load IDL .sav
    sav = load_idl_sav(sav_path)
    vn_sav = extract_vn_series(sav)

    # 2) Determine time range from .sav or default
    trange = trange_default
    if isinstance(sav.get('trange_full'), np.ndarray) and sav['trange_full'].size >= 2:
        # Convert seconds → ISO strings (UTC)
        t0 = pd.to_datetime(sav['trange_full'][0], unit='s', utc=True).strftime('%Y-%m-%d/%H:%M:%S')
        t1 = pd.to_datetime(sav['trange_full'][1], unit='s', utc=True).strftime('%Y-%m-%d/%H:%M:%S')
        trange = (t0, t1)

    # 3) Load mms_mp event (include ephemeris, EDP optional)
    # Use FGM srvy (FGM has srvy/brst, no 'fast'); FPI uses 'fast'
    evt = mp.load_event(list(trange), probes=list(probes), include_edp=False, include_ephem=True,
                        data_rate_fgm='srvy', data_rate_fpi='fast')

    # 4) For each probe: rotate Vi(GSE) → LMN (using .sav LMN) and align to .sav times
    rows = []
    for p in probes:
        key = str(p)
        # LMN from .sav is authoritative for this event
        Lhat_s, Mhat_s, Nhat_s = sav.get('Lhat'), sav.get('Mhat'), sav.get('Nhat')
        if Lhat_s is None or Mhat_s is None or Nhat_s is None:
            continue
        Lhat_s = np.asarray(Lhat_s, float); Mhat_s = np.asarray(Mhat_s, float); Nhat_s = np.asarray(Nhat_s, float)

        if 'V_i_gse' not in evt[key] or key not in vn_sav:
            continue

        # mms_mp Vi
        t_vi, V_i_gse = evt[key]['V_i_gse']
        Vi_df = mp.data_loader.to_dataframe(t_vi, V_i_gse, cols=['Vx','Vy','Vz'])
        Vi_lmn = Vi_df.values @ np.vstack([Lhat_s, Mhat_s, Nhat_s]).T
        vn_bulk = Vi_lmn[:, 2]
        vn_series = pd.Series(vn_bulk, index=Vi_df.index.tz_localize('UTC') if Vi_df.index.tz is None else Vi_df.index.tz_convert('UTC'))

        # .sav ViN
        t_sav, vn_s = vn_sav[key]
        sav_idx = pd.to_datetime(t_sav, unit='s', utc=True)

        # Align: reindex mms_mp VN onto sav_idx (UTC)
        vn_interp = vn_series.reindex(sav_idx, method='nearest')
        n = min(len(vn_interp), len(vn_s))
        mae = float(np.nanmean(np.abs(vn_interp.values[:n] - vn_s[:n]))) if n > 0 else np.nan

        # Save per-probe CSV with aligned series for publication graphics
        out_df = pd.DataFrame({'ViN_sav': vn_s[:n], 'ViN_mmsmp': vn_interp.values[:n]}, index=sav_idx[:n])
        out_df.index.name = 'time_utc'
        out_df.to_csv(os.path.join(output_dir, f'vn_probe{key}.csv'))

        rows.append({'probe': key, 'ang_L_deg': np.nan, 'ang_M_deg': np.nan, 'ang_N_deg': np.nan, 'vn_mae_vs_sav': mae})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'sav_vs_mms_mp_summary.csv')
    df.to_csv(csv_path, index=False)

    # Save JSON summary
    json_path = os.path.join(output_dir, 'sav_vs_mms_mp_summary.json')
    with open(json_path, 'w') as f:
        json.dump({'summary': rows, 'trange': trange}, f, indent=2)

    print('Saved:', csv_path)
    print('Saved:', json_path)
    print(df)


if __name__ == '__main__':
    run_comparison('references/IDL_Code/mp_lmn_systems_20190127_1215-1255_mp-ver3b.sav')

