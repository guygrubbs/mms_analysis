#!/usr/bin/env python
"""
MMS magnetopause distance-time analysis — 27 Jan 2019

Fixes:
  • dtype-aware to_mpl_dates  (already present)
  • safe plt.scatter call (marker keyword, add s=40)
  • global-speed block now interpolates positions at CS times
"""

import numpy as np
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates
from pyspedas import mms
from pytplot   import get_data

# ───────────────────────────────────────────────────────── helpers ─────────
def try_get(name):
    out = get_data(name)
    return out if out is not None else (None, None)

EPOCH_1970 = mdates.date2num(datetime(1970, 1, 1, tzinfo=timezone.utc))
def to_mpl_dates(x):
    """Convert posix-seconds *or* numpy.datetime64 arrays to Matplotlib floats."""
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.floating):      # tplot posix seconds
        return x / 86400.0 + EPOCH_1970
    if np.issubdtype(x.dtype, 'datetime64'):
        sec = x.astype('datetime64[s]').astype(float)
        return sec / 86400.0 + EPOCH_1970
    raise TypeError("Unsupported time array dtype")

# ────────────────────────────────────────────── data loader ───────────────
def load_mms_data(trange):
    mms.mec(trange=trange, probe=['1','2','3','4'],
            data_rate='srvy', level='l2', notplot=False)
    mms.fgm(trange=trange, probe=['1','2','3','4'],
            data_rate='srvy', level='l2', notplot=False)
    mms.fpi(trange=trange, probe=['1','2','3','4'],
            data_rate='fast', level='l2',
            datatype=['dis-moms', 'des-moms'], notplot=False)

    data = {}
    for p in ['1','2','3','4']:
        sid = f"mms{p}"
        t_pos, pos = try_get(f"{sid}_mec_r_gse")
        t_vi , Vi  = try_get(f"{sid}_dis_bulkv_gse_fast")
        if t_pos is None or t_vi is None:
            raise RuntimeError(f"{sid} essential variables missing")
        data[sid] = {'time_pos':t_pos, 'pos':pos,
                     'time_vi' :t_vi , 'Vi' :Vi}
    return data

# ───────────────────────────────────────────────────── geometry check ─────
def confirm_string_of_pearls(m):
    rt = (m['mms1']['time_pos'][0]+m['mms1']['time_pos'][-1])/2
    p  = {s:np.array([np.interp(rt, m[s]['time_pos'], m[s]['pos'][:,i])
                      for i in range(3)]) for s in m}
    sep = [np.linalg.norm(p['mms1']-p['mms2']),
           np.linalg.norm(p['mms2']-p['mms3']),
           np.linalg.norm(p['mms3']-p['mms4'])]
    print(f"String-of-pearls separations (km): 12={sep[0]:.0f} 23={sep[1]:.0f} 34={sep[2]:.0f}")

# ─────────────────────────────────────────────────── Vn + distances ──────
def compute_vn(m, N):
    N = np.asarray(N)/np.linalg.norm(N)
    return {s:{'time':m[s]['time_vi'], 'vn':m[s]['Vi']@N} for s in m}, N

def integrate_vn(t, vn, a64, b64):
    a = a64.astype('datetime64[s]').astype(float)
    b = b64.astype('datetime64[s]').astype(float)
    mask = (t>=a)&(t<=b)
    return np.sum(0.5*(vn[mask][1:]+vn[mask][:-1])*np.diff(t[mask])) if mask.sum()>1 else 0.

def distance_series(t, vn, ref64):
    dt   = np.diff(t)
    disp = np.concatenate(([0.], np.cumsum(0.5*(vn[:-1]+vn[1:])*dt)))
    ref  = ref64.astype('datetime64[s]').astype(float)
    return disp - np.interp(ref, t, disp)

# ─────────────────────────────────────────────── plotting ────────────────
def plot_distance_time_series(dseries, edges):
    plt.figure(figsize=(10,6))
    cols = dict(mms1='tab:red', mms2='tab:blue',
                mms3='tab:green', mms4='tab:purple')
    marks= {'ion_edge':'^','curr_sheet':'o','electron_edge':'v'}

    for sc,dd in dseries.items():
        t_mpl = to_mpl_dates(dd['time'])
        plt.plot(t_mpl, dd['dist'], color=cols[sc], label=sc.upper())
        for edge,mk in marks.items():
            t_edge_mpl = to_mpl_dates(np.array([edges[sc][edge]]))[0]
            y_edge = np.interp(t_edge_mpl, t_mpl, dd['dist'])
            plt.scatter(t_edge_mpl, y_edge, marker=mk, s=40,
                        color=cols[sc], zorder=5)

    plt.axhline(0, ls='--', c='k', lw=0.8)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xlabel('UT'); plt.ylabel('Distance to MP (km)')
    plt.title('MMS Magnetopause Distance – 27 Jan 2019')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.savefig('MMS_magnetopause_distance_20190127.png', dpi=150)
    plt.show()

# ─────────────────────────────────────────────────── main ────────────────
if __name__ == '__main__':
    trange = ['2019-01-27/12:00:00','2019-01-27/13:00:00']
    m = load_mms_data(trange)
    confirm_string_of_pearls(m)

    normal = [0.98,-0.05,0.18]
    vn,Nhat = compute_vn(m, normal)

    edges = {
        'mms1':{'ion_edge':'2019-01-27T12:23:05','curr_sheet':'2019-01-27T12:23:12','electron_edge':'2019-01-27T12:23:18'},
        'mms2':{'ion_edge':'2019-01-27T12:24:15','curr_sheet':'2019-01-27T12:24:22','electron_edge':'2019-01-27T12:24:29'},
        'mms3':{'ion_edge':'2019-01-27T12:25:25','curr_sheet':'2019-01-27T12:25:33','electron_edge':'2019-01-27T12:25:40'},
        'mms4':{'ion_edge':'2019-01-27T12:26:35','curr_sheet':'2019-01-27T12:26:43','electron_edge':'2019-01-27T12:26:50'}
    }
    # cast to datetime64
    edges = {sc:{k:np.datetime64(v) for k,v in d.items()} for sc,d in edges.items()}

    # thickness
    for sc in edges:
        dx = integrate_vn(vn[sc]['time'], vn[sc]['vn'],
                          edges[sc]['ion_edge'], edges[sc]['electron_edge'])
        print(f"{sc.upper()} layer thickness ≈ {dx:.0f} km")

    # global speed – interpolate pos at each CS epoch
    def pos_at(sc, t64):
        t,r = m[sc]['time_pos'], m[sc]['pos']
        return np.array([np.interp(t64.astype('datetime64[s]').astype(float), t, r[:,i])
                         for i in range(3)])

    p1 = pos_at('mms1', edges['mms1']['curr_sheet'])
    p4 = pos_at('mms4', edges['mms4']['curr_sheet'])
    dt_global = (edges['mms4']['curr_sheet'] - edges['mms1']['curr_sheet'])/np.timedelta64(1,'s')
    Vn_global = np.dot(p4-p1, Nhat)/dt_global
    print(f"\nGlobal MP normal speed ≈ {Vn_global:.1f} km/s")

    # predict MMS2/3 current-sheet times
    for sc in ['mms2','mms3']:
        p_sc = pos_at(sc, edges[sc]['curr_sheet'])
        d_sc = np.dot(p_sc-p1, Nhat)
        t_pred = edges['mms1']['curr_sheet'] + np.timedelta64(int(d_sc/Vn_global*1e3),'ms')
        print(f"{sc.upper()} predicted CS {t_pred}  vs  observed {edges[sc]['curr_sheet']}")

    # build distance series
    dser = {sc:{'time':vn[sc]['time'],
                'dist':distance_series(vn[sc]['time'], vn[sc]['vn'],
                                       edges[sc]['curr_sheet'])}
            for sc in vn}
    plot_distance_time_series(dser, edges)
