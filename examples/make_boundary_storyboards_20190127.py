"""
Boundary-focused storyboards and 4-spacecraft overview for 2019-01-27 12:15–12:55.

Per-probe storyboard rows:
- Row 1: DES and DIS spectrogram images (side-by-side) from pub folder
- Row 2: Magnetic field: Bx,By,Bz (GSM) and B_N (LMN) with boundary layers shaded and crossings marked
- Row 3: ViN overlay (.sav vs mms_mp) with MAE and vt shading
- Row 4: DN per-segment bar chart

Also produces a combined overview figure stacking BN and ViN for all 4 spacecraft with boundary crossings.
Outputs go to results/events_pub/2019-01-27_1215-1255.
"""
from __future__ import annotations
import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Ensure repo root on path
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mms_mp as mp
from tools.idl_sav_import import load_idl_sav, extract_vn_series

EVENT_DIR = pathlib.Path('results/events_pub/2019-01-27_1215-1255')
EVENT_DIR.mkdir(parents=True, exist_ok=True)
TRANGE = ('2019-01-27/12:15:00', '2019-01-27/12:55:00')
PROBES = ('1','2','3','4')


def _load_evt_and_lmn():
    evt = mp.load_event(list(TRANGE), probes=list(PROBES), include_ephem=True,
                        data_rate_fgm='srvy', data_rate_fpi='fast')
    sav = load_idl_sav('references/IDL_Code/mp_lmn_systems_20190127_1215_1255_mp_ver2.sav')
    vn_sav = extract_vn_series(sav)
    lmn = sav.get('lmn', {})
    return evt, lmn, vn_sav


def _vt_intervals():
    import re
    txt = pathlib.Path('references/IDL_Code/requested_mp_motion_givenlmn_vion.pro').read_text(encoding='utf-8', errors='ignore')
    block = re.findall(r"If time_string\(trange_full\[0\]\) eq '2019-01-27/04:00:00' then begin(.*?)endif", txt, flags=re.S)
    out = { '1': [], '2': [], '3': [], '4': [] }
    if block:
        b = block[0]
        for sc, tag in [('1','vt_mms1'), ('2','vt_mms2'), ('3','vt_mms3'), ('4','vt_mms4')]:
            m = re.search(tag + r"= time_double\(\[(.*?)\]", b, flags=re.S)
            if not m:
                continue
            arr = m.group(1)
            times = re.findall(r"'([0-9\-/:\.]+)'", arr)
            for i in range(0, len(times), 2):
                if i+1 < len(times):
                    out[sc].append((times[i], times[i+1]))
    return out


def _series(df, name):
    return pd.Series(df[name].values, index=df.index)


def _compute_BN(evt, lmn, cadence='1s'):
    out = {}
    for p in PROBES:
        key = str(p)
        if 'B_gsm' not in evt[key]:
            continue
        tB, B = evt[key]['B_gsm']
        B_df = mp.data_loader.to_dataframe(tB, B, cols=['Bx','By','Bz'])
        B_df = mp.data_loader.resample(B_df, cadence)
        # rotate using .sav LMN for this probe (fallback: average LMN)
        if key in lmn:
            L = np.asarray(lmn[key]['L'], float)
            M = np.asarray(lmn[key]['M'], float)
            N = np.asarray(lmn[key]['N'], float)
        elif lmn:
            L = np.nanmean([v['L'] for v in lmn.values()], axis=0)
            M = np.nanmean([v['M'] for v in lmn.values()], axis=0)
            N = np.nanmean([v['N'] for v in lmn.values()], axis=0)
        else:
            continue
        R = np.vstack([L, M, N]).T
        B_lmn = B_df.values @ R
        BN = pd.Series(B_lmn[:,2], index=B_df.index, name='BN')
        out[key] = {'B_df': B_df, 'BN': BN}
    return out


def _compute_crossings(BN_map, evt, cadence='1s'):
    from mms_mp.boundary import detect_crossings_multi
    crossings = {}
    for p in PROBES:
        key = str(p)
        if key not in BN_map: continue
        BN = BN_map[key]['BN']
        he = pd.Series(np.nan, index=BN.index)
        ni = pd.Series(np.nan, index=BN.index)
        if 'N_he' in evt[key] and evt[key]['N_he'][0] is not None:
            t_he, he_vals = evt[key]['N_he']
            he_df = mp.data_loader.to_dataframe(t_he, he_vals, cols=['He'])
            he_df = mp.data_loader.resample(he_df, cadence)
            he = _series(he_df, 'He').reindex(BN.index, method='nearest')
        if 'N_tot' in evt[key] and evt[key]['N_tot'][0] is not None:
            t_ni, ni_vals = evt[key]['N_tot']
            ni_df = mp.data_loader.to_dataframe(t_ni, ni_vals, cols=['Ni'])
            ni_df = mp.data_loader.resample(ni_df, cadence)
            ni = _series(ni_df, 'Ni').reindex(BN.index, method='nearest')
        # Call detector with both He⁺ and total ion density
        layers = detect_crossings_multi(
            BN.index.values,
            he.values,
            BN.values,
            ni=ni.values,
        )
        crossings[key] = {'layers': layers}
    return crossings


def _draw_spectro_row(fig, gs_row, p):
    # Two axes for DES/DIS images
    ax_des = fig.add_subplot(gs_row[0])
    ax_dis = fig.add_subplot(gs_row[1])
    for ax, tag in [(ax_des,'DES'), (ax_dis,'DIS')]:
        ax.set_axis_off()
        f = EVENT_DIR / f'mms{p}_{tag}_omni.png'
        if f.exists():
            try:
                img = Image.open(f)
                ax.imshow(img)
                ax.set_title(f'MMS{p} {tag} spectrogram')
            except Exception:
                ax.text(0.5, 0.5, f'{tag} image not available', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, f'{tag} image not available', ha='center', va='center')
    return ax_des, ax_dis


def _draw_B_row(ax, B_df, BN, vt, p, layers):
    ax.plot(B_df.index, B_df['Bx'], label='Bx GSM', lw=0.8)
    ax.plot(B_df.index, B_df['By'], label='By GSM', lw=0.8)
    ax.plot(B_df.index, B_df['Bz'], label='Bz GSM', lw=0.8)
    ax2 = ax.twinx()
    ax2.plot(BN.index, BN.values, color='k', lw=1.1, label='B_N (LMN)')
    ax.set_ylabel('B GSM (nT)')
    ax2.set_ylabel('B_N (nT)')
    ax.grid(True, alpha=0.2)
    # vt shading
    if vt and p in vt:
        for (t0s, t1s) in vt[p]:
            t0 = pd.to_datetime(t0s, utc=True)
            t1 = pd.to_datetime(t1s, utc=True)
            if t1 < t0: t0, t1 = t1, t0
            ax.axvspan(t0, t1, color='k', alpha=0.06)
    # crossing markers
    for typ, i1, i2 in layers or []:
        t_start = BN.index[int(i1)] if i1 < len(BN) else None
        t_end   = BN.index[int(i2)] if i2 < len(BN) else None
        if t_start is not None:
            ax.axvline(t_start, color='r' if typ=='magnetosphere' else 'g', ls='--', lw=1.0, alpha=0.6)
        if t_end is not None:
            ax.axvline(t_end, color='r' if typ=='magnetosphere' else 'g', ls='--', lw=1.0, alpha=0.6)
    # legends
    ax.legend(loc='upper left', frameon=False)
    ax2.legend(loc='upper right', frameon=False)


def _load_vn_csv(p):
    f = EVENT_DIR / f'vn_probe{p}.csv'
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df['time_utc'] = pd.to_datetime(df['time_utc'], utc=True)
    df = df.set_index('time_utc')
    return df


def _draw_vin_row(ax, df, vt, p):
    ax.plot(df.index, df['ViN_sav'], label='IDL .sav ViN', lw=1.0)
    ax.plot(df.index, df['ViN_mmsmp'], label='mms_mp ViN', lw=0.9, alpha=0.9)
    ax.set_ylabel('V_N (km/s)')
    ax.grid(True, alpha=0.2)
    diff = (df['ViN_mmsmp'] - df['ViN_sav']).dropna()
    mae = float(np.nanmean(np.abs(diff))) if len(diff) else np.nan
    ax.text(0.01, 0.95, f"MAE={mae:.2f} km/s", transform=ax.transAxes, va='top', ha='left')
    if vt and p in vt:
        for (t0s, t1s) in vt[p]:
            t0 = pd.to_datetime(t0s, utc=True)
            t1 = pd.to_datetime(t1s, utc=True)
            if t1 < t0: t0, t1 = t1, t0
            ax.axvspan(t0, t1, color='k', alpha=0.06)
    ax.legend(loc='upper right', frameon=False)


def _draw_dn_row(ax, p):
    f = EVENT_DIR / f'mms{p}_DN_segments.csv'
    if not f.exists():
        ax.text(0.5, 0.5, f'No DN segments for MMS{p}', ha='center', va='center')
        ax.set_axis_off()
        return
    dn = pd.read_csv(f)
    ax.bar(dn['segment'], dn['DN_km'], color='#54a24b')
    ax.set_xlabel('Segment #')
    ax.set_ylabel('DN (km)')
    ax.grid(True, alpha=0.2)


def make_per_probe_storyboards():
    evt, lmn, vn_sav = _load_evt_and_lmn()
    vt = _vt_intervals()
    BN_map = _compute_BN(evt, lmn, cadence='1s')
    crossings = _compute_crossings(BN_map, evt, cadence='1s')

    for p in PROBES:
        key = str(p)
        if key not in BN_map:
            continue
        B_df = BN_map[key]['B_df']
        BN = BN_map[key]['BN']
        vn_df = _load_vn_csv(key)

        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1.0, 0.8, 0.6])
        _draw_spectro_row(fig, (gs[0,0], gs[0,1]), p)
        axB = fig.add_subplot(gs[1, :])
        _draw_B_row(axB, B_df, BN, vt, key, crossings.get(key, {}).get('layers'))
        axV = fig.add_subplot(gs[2, :])
        if vn_df is not None and not vn_df.empty:
            _draw_vin_row(axV, vn_df, vt, key)
        else:
            axV.text(0.5, 0.5, f'ViN not available', ha='center', va='center')
            axV.set_axis_off()
        axD = fig.add_subplot(gs[3, :])
        _draw_dn_row(axD, key)
        fig.suptitle(f'MMS{p} Boundary Crossing Storyboard (2019-01-27 12:15–12:55)')
        fig.tight_layout(rect=[0,0,1,0.97])
        out = EVENT_DIR / f'boundary_storyboard_mms{p}.png'
        fig.savefig(out, dpi=220)
        plt.close(fig)
        print(f'Wrote {out}')


def make_combined_overview():
    evt, lmn, _ = _load_evt_and_lmn()
    vt = _vt_intervals()
    BN_map = _compute_BN(evt, lmn, cadence='1s')

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for i, p in enumerate(PROBES):
        key = str(p)
        if key not in BN_map:
            continue
        B_df = BN_map[key]['B_df']
        BN = BN_map[key]['BN']
        ax = axes[i]
        ax.plot(B_df.index, B_df['Bx'], lw=0.8, label='Bx')
        ax.plot(B_df.index, B_df['By'], lw=0.8, label='By')
        ax.plot(B_df.index, B_df['Bz'], lw=0.8, label='Bz')
        ax2 = ax.twinx()
        ax2.plot(BN.index, BN.values, color='k', lw=1.1, label='B_N')
        ax.set_ylabel(f'MMS{p}\nB (nT)')
        ax.grid(True, alpha=0.2)
        if vt and key in vt:
            for (t0s, t1s) in vt[key]:
                t0 = pd.to_datetime(t0s, utc=True)
                t1 = pd.to_datetime(t1s, utc=True)
                if t1 < t0: t0, t1 = t1, t0
                ax.axvspan(t0, t1, color='k', alpha=0.05)
        if i == 0:
            ax.legend(loc='upper left', frameon=False)
            ax2.legend(loc='upper right', frameon=False)
    axes[-1].set_xlabel('Time (UTC)')
    fig.suptitle('Boundary Crossing Overview: B (GSM) and B_N (LMN), MMS1–4')
    fig.tight_layout(rect=[0,0,1,0.96])
    out = EVENT_DIR / 'combined_boundary_overview.png'
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f'Wrote {out}')


if __name__ == '__main__':
    make_per_probe_storyboards()
    make_combined_overview()

