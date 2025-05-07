#!/usr/bin/env python
"""
MMS magnetopause distance-time analysis – 27 Jan 2019, 12:00-13:00 UT
====================================================================

• Dynamic boundary-normal from single-spacecraft Minimum-Variance Analysis (MVA)
• Flip candidate ranked by ion-density drop to reject spurious zero-crossings
• MMS-4 DES fallback to QL if L2 is missing
"""

# ───────────────────────────── Imports ────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates
from datetime  import datetime, timezone
import warnings

from pyspedas import mms
from pytplot   import get_data

# ───────────────────────────── Helpers ────────────────────────────────
def try_get(name):
    """Return (t, v) even when the tplot variable does not exist."""
    out = get_data(name)
    return out if out is not None else (None, None)

EPOCH_1970 = mdates.date2num(datetime(1970, 1, 1, tzinfo=timezone.utc))
def to_mpl_dates(x):
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.floating):
        return x / 86400.0 + EPOCH_1970
    if np.issubdtype(x.dtype, 'datetime64'):
        return x.astype('datetime64[s]').astype(float) / 86400.0 + EPOCH_1970
    raise TypeError("Unsupported time dtype")

def sec_to_dt64(ts):
    return np.array(ts * 1e9, dtype='datetime64[ns]')

# ─────────────────────────── Data loader ──────────────────────────────
def load_mms_data(trange):
    probes = ['1', '2', '3', '4']
    mms.mec(trange=trange, probe=probes, data_rate='srvy', level='l2', notplot=False)
    mms.fgm(trange=trange, probe=probes, data_rate='srvy', level='l2', notplot=False)

    mms.fpi(trange=trange, probe=probes, data_rate='fast',
            level='l2', datatype=['dis-moms', 'des-moms'], notplot=False)

    if get_data('mms4_des_numberdensity_fast') is None:
        print('MMS-4 DES L2 missing – reloading QL')
        mms.fpi(trange=trange, probe='4', data_rate='fast',
                level='ql', datatype='des-moms', notplot=False)

    data = {}
    for p in probes:
        sid = f"mms{p}"
        t_pos, pos = try_get(f"{sid}_mec_r_gse")
        t_vi , Vi  = try_get(f"{sid}_dis_bulkv_gse_fast")
        if t_pos is None or t_vi is None:
            raise RuntimeError(f"{sid}: essential MEC/FPI variables missing")
        data[sid] = {'time_pos': t_pos, 'pos': pos,
                     'time_vi' : t_vi , 'Vi' : Vi}
    return data

# ───────────── Minimum-variance normal (single spacecraft) ─────────────
def mva_normal(t_B, B, centre_idx, window_sec=180):
    """Return unit normal from MVA over ±window/2 s around centre_idx."""
    w = int(window_sec // 2)
    sl = slice(max(centre_idx - w, 0), min(centre_idx + w, len(B)))
    Bxyz = B[sl, :3]                       # drop |B| column if present
    C = np.cov(Bxyz.T)
    eigvals, eigvecs = np.linalg.eigh(C)   # ascending order
    return eigvecs[:, 0] / np.linalg.norm(eigvecs[:, 0])

# ───────────── Automatic boundary identification ───────────────────────
def detect_boundaries(m, sheath_frac=0.5, window=30, drop_lead=15):
    """
    edges[probe]['ion_edge' | 'curr_sheet' | 'electron_edge'] (datetime64[ns])
    * normal = MVA result (per probe)
    * current sheet  = Bn flip candidate that maximises ion-density drop
    * ion / electron edges from density threshold on Ni / Ne
    """
    edges, skipped = {}, []

    for sid in m:
        t_B,  B   = try_get(f"{sid}_fgm_b_gse_srvy_l2")
        t_Ni, Ni  = try_get(f"{sid}_dis_numberdensity_fast")
        t_Ne, Ne  = try_get(f"{sid}_des_numberdensity_fast")
        if t_B is None or t_Ni is None:
            skipped.append(sid); continue

        # t_common = ion-bulk-velocity timeline
        t_common = m[sid]['time_vi']
        def interp(src_t, src_v):
            if src_v is None: return None
            if src_v.ndim == 1:
                return np.interp(t_common, src_t, src_v)
            return np.vstack([np.interp(t_common, src_t, src_v[:, i])
                              for i in range(src_v.shape[1])]).T

        B_int  = interp(t_B,  B)
        Ni_int = interp(t_Ni, Ni)
        Ne_int = interp(t_Ne, Ne)

        if B_int.shape[1] > 3:                       # remove |B|
            B_int = B_int[:, :3]

        # ---- dynamic normal from MVA centred at mid-interval ----------
        Nhat = mva_normal(t_common, B_int,
                          centre_idx=len(t_common)//2, window_sec=180)

        Bn = B_int @ Nhat
        flips = np.where(np.diff(np.sign(Bn)) != 0)[0]
        if not flips.size:
            skipped.append(sid); continue

        # rank flips by Ni drop across ±drop_lead samples
        def density_drop(idx):
            pre  = Ni_int[max(idx - drop_lead, 0)]
            post = Ni_int[min(idx + drop_lead, len(Ni_int) - 1)]
            return (pre - post) / pre if pre else 0.0

        idx_cs = int(max(flips, key=density_drop) + 1)
        t_cs   = sec_to_dt64(t_common[idx_cs])

        # ---- threshold on density to pick edges -----------------------
        thr = (np.nanmedian(Ni_int[:idx_cs - window])
               if idx_cs > window else np.nanmedian(Ni_int[:idx_cs])) * sheath_frac

        idx_ion = (np.where(Ni_int[:idx_cs] > thr)[0][-1]
                   if np.any(Ni_int[:idx_cs] > thr) else idx_cs)
        t_ion = sec_to_dt64(t_common[idx_ion])

        if Ne_int is not None:
            post = np.where(Ne_int[idx_cs:] < thr)[0]
        else:
            post = np.where(Ni_int[idx_cs:] < thr)[0]
        idx_ele = idx_cs + (post[0] if post.size else 0)
        t_ele = sec_to_dt64(t_common[idx_ele])

        edges[sid] = {'ion_edge': t_ion,
                      'curr_sheet': t_cs,
                      'electron_edge': t_ele,
                      'normal': Nhat}

    if skipped:
        warnings.warn(f"Boundary search skipped probes: {', '.join(skipped)}")
    return edges

# ───────────── Geometry, vn, distance helpers ───────────────────────────
def confirm_string_of_pearls(m):
    rt = (m['mms1']['time_pos'][0] + m['mms1']['time_pos'][-1]) / 2
    pos = {s: np.array([np.interp(rt, m[s]['time_pos'], m[s]['pos'][:, i])
                        for i in range(3)]) for s in m}
    sep = [np.linalg.norm(pos['mms1'] - pos['mms2']),
           np.linalg.norm(pos['mms2'] - pos['mms3']),
           np.linalg.norm(pos['mms3'] - pos['mms4'])]
    print(f"String-of-pearls separations (km): 12={sep[0]:.0f} 23={sep[1]:.0f} 34={sep[2]:.0f}")

def compute_vn(m, N):
    N = np.asarray(N) / np.linalg.norm(N)
    return {s: {'time': m[s]['time_vi'], 'vn': m[s]['Vi'] @ N} for s in m}, N

def integrate_vn(t, vn, a64, b64):
    a = a64.astype('datetime64[s]').astype(float)
    b = b64.astype('datetime64[s]').astype(float)
    mask = (t >= a) & (t <= b)
    return np.sum(0.5 * (vn[mask][1:] + vn[mask][:-1]) * np.diff(t[mask])) if mask.sum() > 1 else 0.0

def distance_series(t, vn, ref64):
    disp = np.concatenate(([0.0], np.cumsum(0.5 * (vn[:-1] + vn[1:]) * np.diff(t))))
    ref  = ref64.astype('datetime64[s]').astype(float)
    return disp - np.interp(ref, t, disp)

# ───────────── Plotting ─────────────────────────────────────────────────
def plot_distance_time_series(dseries, edges):
    plt.figure(figsize=(10, 6))
    cols  = dict(mms1='tab:red', mms2='tab:blue', mms3='tab:green', mms4='tab:purple')
    marks = {'ion_edge': '^', 'curr_sheet': 'o', 'electron_edge': 'v'}

    for sc, dd in dseries.items():
        t_mpl = to_mpl_dates(dd['time'])
        plt.plot(t_mpl, dd['dist'], color=cols[sc], label=sc.upper())
        for edge, mk in marks.items():
            t_edge_mpl = to_mpl_dates(np.array([edges[sc][edge]]))[0]
            y_edge = np.interp(t_edge_mpl, t_mpl, dd['dist'])
            plt.scatter(t_edge_mpl, y_edge, marker=mk, s=40, color=cols[sc], zorder=5)

    plt.axhline(0.0, ls='--', c='k', lw=0.8)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xlabel('UT'); plt.ylabel('Distance to MP (km)')
    plt.title('MMS Magnetopause Distance – 27 Jan 2019')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.savefig('MMS_magnetopause_distance_20190127.png', dpi=150)
    plt.show()

# ───────────── Main ─────────────────────────────────────────────────────
if __name__ == '__main__':
    trange = ['2019-01-27/12:00:00', '2019-01-27/13:00:00']
    m = load_mms_data(trange)
    confirm_string_of_pearls(m)

    # ------------------------------------------------------------------
    edges = detect_boundaries(m)
    print("\nDetected boundary-times (UTC):")
    for sc in edges:
        ie, cs, ee = edges[sc]['ion_edge'], edges[sc]['curr_sheet'], edges[sc]['electron_edge']
        print(f"{sc.upper()}: ion={ie}  CS={cs}  elec={ee}")

    # Use MMS-1 normal for distance / speed work
    Nhat = edges['mms1']['normal']
    vn, Nhat = compute_vn(m, Nhat)

    # ------------------------------------------------------------------
    intervals = {'ion_only': ('ion_edge', 'curr_sheet'),
                 'electron_only': ('curr_sheet', 'electron_edge'),
                 'full': ('ion_edge', 'electron_edge')}

    thickness = {sc: {} for sc in edges}
    for sc in edges:
        for lbl, (t0k, t1k) in intervals.items():
            thickness[sc][lbl] = integrate_vn(
                vn[sc]['time'], vn[sc]['vn'],
                edges[sc][t0k], edges[sc][t1k])

    print("\nBoundary-layer thicknesses (km):")
    for sc, d in thickness.items():
        print(f"{sc.upper()}: "+", ".join(f"{k}={abs(v):.0f}" for k, v in d.items()))

    # ------------------------------------------------------------------
    def pos_at(sc, t64):
        t, r = m[sc]['time_pos'], m[sc]['pos']
        return np.array([np.interp(t64.astype('datetime64[s]').astype(float), t, r[:, i])
                         for i in range(3)])

    p1 = pos_at('mms1', edges['mms1']['curr_sheet'])
    p4 = pos_at('mms4', edges['mms4']['curr_sheet'])
    dt_global = (edges['mms4']['curr_sheet'] - edges['mms1']['curr_sheet']) / np.timedelta64(1, 's')
    Vn_global = np.dot(p4 - p1, Nhat) / dt_global
    print(f"\nGlobal MP normal speed ≈ {Vn_global:.1f} km s⁻¹")

    for sc in ['mms2', 'mms3']:
        p_sc = pos_at(sc, edges[sc]['curr_sheet'])
        d_sc = np.dot(p_sc - p1, Nhat)
        t_pred = edges['mms1']['curr_sheet'] + np.timedelta64(int(d_sc / Vn_global * 1e3), 'ms')
        print(f"{sc.upper()}: predicted CS {t_pred} vs observed {edges[sc]['curr_sheet']}")

    # ------------------------------------------------------------------
    dser = {sc: {'time': vn[sc]['time'],
                 'dist': distance_series(vn[sc]['time'], vn[sc]['vn'],
                                         edges[sc]['curr_sheet'])}
            for sc in vn}
    plot_distance_time_series(dser, edges)
