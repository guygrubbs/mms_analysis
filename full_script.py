#!/usr/bin/env python
"""
MMS magnetopause motion analysis – 27 Jan 2019, 12:00-13:00 UT
Replicates the IDL workflow in Python (PySPEDAS + Matplotlib).

Author: <your-name>
Date  : 2025-05-06
"""

import numpy as np
from pyspedas import mms
from pytplot import get_data, tplot_names
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -------------------------------------------------------------------------
# small helpers
# -------------------------------------------------------------------------
def try_get(name):
    """Return (t, y) if tplot var exists, else (None, None)."""
    out = get_data(name)
    return out if out is not None else (None, None)

def sec2mpl(seconds):
    """POSIX seconds → Matplotlib date-float."""
    return seconds / 86400.0 + mdates.datestr2num('1970-01-01')

# -------------------------------------------------------------------------
# data loader
# -------------------------------------------------------------------------
def load_mms_data(trange):
    """
    Download MEC, FGM and FPI moment data for MMS1-4.
    Electron bulk-V is optional – analysis doesn’t need it.
    Returns a dict keyed by 'mms1' … 'mms4'.
    """
    # download (keep tplot vars)
    mms.mec(trange=trange, probe=['1','2','3','4'],
            data_rate='srvy', level='l2', notplot=False)
    mms.fgm(trange=trange, probe=['1','2','3','4'],
            data_rate='srvy', level='l2', notplot=False)
    mms.fpi(trange=trange, probe=['1','2','3','4'],
            data_rate='fast', level='l2',
            datatype=['dis-moms', 'des-moms'], notplot=False)

    data = {}
    for sc in ['1','2','3','4']:
        sid = f"mms{sc}"

        # position (GSE, km) – epoch seconds
        t_pos, pos = try_get(f"{sid}_mec_r_gse")
        if t_pos is None:
            raise RuntimeError(f"{sid} MEC variable missing")

        # magnetic field (GSE, nT)
        t_b, B = try_get(f"{sid}_fgm_b_gse_srvy_l2")
        if t_b is None:
            raise RuntimeError(f"{sid} FGM variable missing")

        # ion bulk-V (GSE, km/s) – always present in fast moments
        t_vi, Vi = try_get(f"{sid}_dis_bulkv_gse_fast")
        if t_vi is None:
            raise RuntimeError(f"{sid} ion bulk-V missing")

        # electron bulk-V – may be absent
        t_ve, Ve = try_get(f"{sid}_des_bulkv_gse_fast")

        # densities (optional)
        _, Ni = try_get(f"{sid}_dis_numberdensity_fast")
        _, Ne = try_get(f"{sid}_des_numberdensity_fast")

        data[sid] = {
            'time_pos': t_pos, 'pos': pos,        # km
            'time_b'  : t_b,   'B'  : B,          # nT
            'time_vi' : t_vi,  'Vi' : Vi,         # km/s
            'time_ve' : t_ve,  'Ve' : Ve,         # km/s or None
            'Ni'      : Ni,    'Ne' : Ne
        }
    return data

# -------------------------------------------------------------------------
# geometry check – “string-of-pearls”
# -------------------------------------------------------------------------
def confirm_string_of_pearls(mms_data):
    ref_time = (mms_data['mms1']['time_pos'][0] +
                mms_data['mms1']['time_pos'][-1]) / 2
    ref_pos = {}
    for sc in ['mms1','mms2','mms3','mms4']:
        t = mms_data[sc]['time_pos']; r = mms_data[sc]['pos']
        ref_pos[sc] = np.array([np.interp(ref_time, t, r[:,0]),
                                np.interp(ref_time, t, r[:,1]),
                                np.interp(ref_time, t, r[:,2])])
    rad = {sc: np.linalg.norm(vec)/6371.0 for sc, vec in ref_pos.items()}
    sep12 = np.linalg.norm(ref_pos['mms1'] - ref_pos['mms2'])
    sep23 = np.linalg.norm(ref_pos['mms2'] - ref_pos['mms3'])
    sep34 = np.linalg.norm(ref_pos['mms3'] - ref_pos['mms4'])
    print("\nString-of-pearls check @~12:30 UT")
    for sc, r in rad.items():
        print(f"  {sc.upper()}  {r:.2f} Re")
    print(f"  separations: 12={sep12:.0f} km  23={sep23:.0f} km  34={sep34:.0f} km\n")

# -------------------------------------------------------------------------
# Vn computation
# -------------------------------------------------------------------------
def compute_normal_velocity(mms_data, normal_vector):
    N = np.asarray(normal_vector, dtype=float)
    N /= np.linalg.norm(N)
    Vn = {}
    for sc in ['mms1','mms2','mms3','mms4']:
        vi = mms_data[sc]['Vi']              # shape (N,3)
        Vn[sc] = {'time_sec': mms_data[sc]['time_vi'],   # seconds
                  'Vn'      : vi @ N}        # dot product
    return Vn, N

# -------------------------------------------------------------------------
# integrate Vn between times
# -------------------------------------------------------------------------
def integrate_vn(time_sec, Vn, t_start64, t_end64):
    t0 = t_start64.astype('datetime64[s]').astype(float)
    t1 = t_end64  .astype('datetime64[s]').astype(float)
    mask = (time_sec >= t0) & (time_sec <= t1)
    if mask.sum() < 2:
        return 0.0
    dt = np.diff(time_sec[mask])
    return np.sum(0.5*(Vn[mask][:-1] + Vn[mask][1:]) * dt)

# -------------------------------------------------------------------------
# continuous distance time-series (0 at ref_time)
# -------------------------------------------------------------------------
def distance_series(time_sec, Vn, ref64):
    dt = np.diff(time_sec)
    disp = np.concatenate(([0.0],
                           np.cumsum(0.5*(Vn[:-1] + Vn[1:]) * dt)))
    ref_sec = ref64.astype('datetime64[s]').astype(float)
    ref_disp = np.interp(ref_sec, time_sec, disp)
    return disp - ref_disp   # km

# -------------------------------------------------------------------------
# plotting
# -------------------------------------------------------------------------
def plot_distance(dist_data, boundary_times):
    plt.figure(figsize=(10,6))
    colors  = dict(mms1='tab:red', mms2='tab:blue',
                   mms3='tab:green', mms4='tab:purple')
    markers = {'ion_edge':'^', 'curr_sheet':'o', 'electron_edge':'v'}

    for sc, dd in dist_data.items():
        # x-axis already in POSIX seconds – convert once to Matplotlib dates
        t_mpl = [mdates.epoch2num(t) for t in dd['time_sec']]
        plt.plot(t_mpl, dd['dist'], color=colors[sc], label=sc.upper())
        # annotate edges
        for edge, mk in markers.items():
            te64 = boundary_times[sc][edge]
            te_sec = te64.astype('datetime64[s]').astype(float)
            d_val  = np.interp(te_sec, dd['time_sec'], dd['dist'])
            plt.plot(mdates.epoch2num(te_sec), d_val, mk,
                     color=colors[sc], ms=7)

    plt.axhline(0, ls='--', c='k', lw=0.7)
    plt.title("MMS magnetopause distance – 27 Jan 2019")
    plt.ylabel("Distance to MP (km)")
    plt.xlabel("Universal Time")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.legend()
    plt.tight_layout()
    plt.savefig("MMS_magnetopause_distance_20190127.png", dpi=150)
    plt.show()

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
if __name__ == "__main__":
    trange = ['2019-01-27/12:00:00', '2019-01-27/13:00:00']
    mms_data = load_mms_data(trange)
    confirm_string_of_pearls(mms_data)

    normal_vec = [0.98, -0.05, 0.18]
    Vn_data, N = compute_normal_velocity(mms_data, normal_vec)

    # boundary times (update with precise picks)
    boundary_times = {
        'mms1': {'ion_edge':'2019-01-27T12:23:05', 'curr_sheet':'2019-01-27T12:23:12',
                 'electron_edge':'2019-01-27T12:23:18'},
        'mms2': {'ion_edge':'2019-01-27T12:24:15', 'curr_sheet':'2019-01-27T12:24:22',
                 'electron_edge':'2019-01-27T12:24:29'},
        'mms3': {'ion_edge':'2019-01-27T12:25:25', 'curr_sheet':'2019-01-27T12:25:33',
                 'electron_edge':'2019-01-27T12:25:40'},
        'mms4': {'ion_edge':'2019-01-27T12:26:35', 'curr_sheet':'2019-01-27T12:26:43',
                 'electron_edge':'2019-01-27T12:26:50'}
    }
    for sc, times in boundary_times.items():
        for k in times:
            boundary_times[sc][k] = np.datetime64(times[k])

    # thickness via Vn integration
    for sc in ['mms1','mms2','mms3','mms4']:
        dx = integrate_vn(Vn_data[sc]['time_sec'], Vn_data[sc]['Vn'],
                          boundary_times[sc]['ion_edge'],
                          boundary_times[sc]['electron_edge'])
        print(f"{sc.upper()} boundary thickness ≈ {dx:.0f} km")

    # global Vn from MMS1 vs MMS4 (positions at earliest CS epoch)
    t_cs1 = boundary_times['mms1']['curr_sheet']
    t_cs4 = boundary_times['mms4']['curr_sheet']
    def interp_pos(sc, t64):
        t, r = mms_data[sc]['time_pos'], mms_data[sc]['pos']
        return np.array([np.interp(t64.astype('datetime64[s]').astype(float), t, r[:,i])
                         for i in range(3)])
    pos1 = interp_pos('mms1', t_cs1)
    pos4 = interp_pos('mms4', t_cs1)          # same epoch!
    delta_d = np.dot(pos4 - pos1, N)
    dt_sec  = (t_cs4 - t_cs1) / np.timedelta64(1, 's')
    Vn_global = delta_d / dt_sec
    print(f"\nGlobal MP normal speed ≈ {Vn_global:.1f} km/s")

    # predicted CS for MMS2 & MMS3
    for sc in ['mms2','mms3']:
        pos_sc = interp_pos(sc, t_cs1)
        dd_sc  = np.dot(pos_sc - pos1, N)
        t_pred = t_cs1 + np.timedelta64(int(dd_sc / Vn_global * 1e3), 'ms')
        print(f"{sc.upper()} predicted CS {t_pred}  vs  observed {boundary_times[sc]['curr_sheet']}")

    # build distance series dict
    dist_dict = {}
    for sc in ['mms1','mms2','mms3','mms4']:
        dist_dict[sc] = {
            'time_sec': Vn_data[sc]['time_sec'],
            'dist'    : distance_series(Vn_data[sc]['time_sec'],
                                        Vn_data[sc]['Vn'],
                                        boundary_times[sc]['curr_sheet'])
        }

    plot_distance(dist_dict, boundary_times)
