#!/usr/bin/env python
"""
MMS magnetopause distance-time analysis – 27 Jan 2019, 12:00-13:00 UT
=====================================================================

✓ Fallback to quick-look (QL) DES moments for MMS-4 when L2 is absent  
✓ Boundary normal from *per-candidate* 3-min MVA window  
✓ Flip candidate scored by  (0.6 × rotation angle + 0.4 × ion-density drop)  
  – accepts only flips with  Δθ ≥ 45°  AND  ΔNi/Ni ≥ 0.50  
✓ All legacy functionality (thickness, global speed, plotting) retained
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
    """Return (t, v) even if the tplot variable is missing."""
    out = get_data(name)
    return out if out is not None else (None, None)

EPOCH_1970 = mdates.date2num(datetime(1970, 1, 1, tzinfo=timezone.utc))
def to_mpl_dates(x):
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.floating):
        return x / 86400.0 + EPOCH_1970
    if np.issubdtype(x.dtype, 'datetime64'):
        sec = x.astype('datetime64[s]').astype(float)
        return sec / 86400.0 + EPOCH_1970
    raise TypeError("Unsupported time dtype")

def sec_to_dt64(ts):
    """POSIX seconds → datetime64[ns]."""
    return np.array(ts * 1e9, dtype='datetime64[ns]')

# ─────────────────────────── Data loader ──────────────────────────────
def load_mms_data(trange):
    probes = ['1', '2', '3', '4']
    mms.mec(trange=trange, probe=probes, data_rate='srvy', level='l2', notplot=False)
    mms.fgm(trange=trange, probe=probes, data_rate='srvy', level='l2', notplot=False)

    mms.fpi(trange=trange, probe=probes, data_rate='fast',
            level='l2', datatype=['dis-moms', 'des-moms'], notplot=False)

    # MMS-4 DES fallback
    if get_data('mms4_des_numberdensity_fast') is None:
        print('MMS-4 DES L2 missing – loading QL')
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

# ───────────────── Minimum-variance helper ────────────────────────────
def mva_normal(B_xyz):
    """Return unit normal (min-variance eigenvector) for 3-component B slab."""
    C = np.cov(B_xyz.T)
    evals, evecs = np.linalg.eigh(C)
    return evecs[:, 0] / np.linalg.norm(evecs[:, 0])   # min variance

# ───────────── Automatic boundary identification ───────────────────────
def detect_boundaries(m,
                      win_mva_sec=180,
                      drop_lead=15,
                      rot_thresh=45.0,
                      dens_thresh=0.50,
                      w_rot=0.6):
    """
    edges = {probe: {'ion_edge', 'curr_sheet', 'electron_edge', 'normal'}}
    • For every sign-flip of B·N̂ compute:
        Δθ  = rotation angle,  ΔNi/Ni  = density drop
    • Score = w_rot*(Δθ/180) + (1-w_rot)*(ΔNi/Ni);  pick highest score
    """
    edges, skipped = {}, []

    for sid in m:
        # ------------ fetch variables -------------------------------
        t_B,  B   = try_get(f"{sid}_fgm_b_gse_srvy_l2")
        t_Ni, Ni  = try_get(f"{sid}_dis_numberdensity_fast")
        t_Ne, Ne  = try_get(f"{sid}_des_numberdensity_fast")
        if t_B is None or t_Ni is None:
            skipped.append(sid); continue

        # common 4 s grid (ion bulk-velocity)
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
        if B_int.shape[1] > 3:
            B_int = B_int[:, :3]

        # ------------ locate flip candidates ------------------------
        flips = np.where(np.diff(np.sign(B_int[:, 0])) != 0)[0]   # any sign change in Bx

        best = None
        best_score = -np.inf
        for idx in flips:
            # local MVA ±win/2 around idx
            half = int(win_mva_sec / 2 / 4)                       # 4 s cadence
            s0, s1 = max(idx - half, 0), min(idx + half, len(B_int))
            Nhat = mva_normal(B_int[s0:s1, :])

            Bn = B_int @ Nhat
            if np.sign(Bn[idx-1]) == np.sign(Bn[idx+1]):          # not a true flip
                continue

            # rotation angle of transverse B across flip
            vec_pre  = B_int[max(idx-1, 0)]
            vec_post = B_int[min(idx+1, len(B_int)-1)]
            cosang = np.clip(np.dot(vec_pre, vec_post) /
                             (np.linalg.norm(vec_pre)*np.linalg.norm(vec_post)), -1, 1)
            rot = np.degrees(np.arccos(cosang))

            # ion-density drop
            pre  = Ni_int[max(idx-drop_lead, 0)]
            post = Ni_int[min(idx+drop_lead, len(Ni_int)-1)]
            if pre == 0: continue
            ddrop = (pre - post) / pre

            if rot < rot_thresh or ddrop < dens_thresh:
                continue

            score = w_rot * (rot/180) + (1-w_rot) * ddrop
            if score > best_score:
                best_score = score
                best = {'idx_cs': idx,
                        'Nhat'  : Nhat,
                        'rot'   : rot,
                        'ddrop' : ddrop}

        if best is None:
            warnings.warn(f"{sid}: no flip passed quality filters"); skipped.append(sid); continue

        idx_cs = best['idx_cs']
        Nhat   = best['Nhat']
        t_cs   = sec_to_dt64(t_common[idx_cs])

        # ----- density threshold for ion / electron edges ----------
        sheath_med = np.nanmedian(Ni_int[:idx_cs]) if idx_cs else np.nanmedian(Ni_int)
        thr = sheath_med * 0.5

        idx_ion = np.where(Ni_int[:idx_cs] > thr)[0]
        idx_ion = idx_ion[-1] if idx_ion.size else idx_cs
        t_ion   = sec_to_dt64(t_common[idx_ion])

        if Ne_int is not None:
            post = np.where(Ne_int[idx_cs:] < thr)[0]
        else:
            post = np.where(Ni_int[idx_cs:] < thr)[0]
        idx_ele = idx_cs + (post[0] if post.size else 0)
        t_ele   = sec_to_dt64(t_common[idx_ele])

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
    a, b = (x.astype('datetime64[s]').astype(float) for x in (a64, b64))
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
            plt.scatter(t_edge_mpl, y_edge, marker=mk, s=40,
                        color=cols[sc], zorder=5)

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

    # ---------- detect boundaries ------------------------------------
    edges = detect_boundaries(m)
    print("\nDetected boundary times (UTC):")
    for sc, d in edges.items():
        print(f"{sc.upper()}: ion={d['ion_edge']}  CS={d['curr_sheet']}  elec={d['electron_edge']}")

    # ---------- vn, thickness, speed ---------------------------------
    Nhat = edges['mms1']['normal']          # use MMS-1 normal
    vn, _ = compute_vn(m, Nhat)

    intervals = {'ion_only': ('ion_edge', 'curr_sheet'),
                 'electron_only': ('curr_sheet', 'electron_edge'),
                 'full': ('ion_edge', 'electron_edge')}

    thickness = {sc: {} for sc in edges}
    for sc in edges:
        for lbl, (t0k, t1k) in intervals.items():
            thickness[sc][lbl] = integrate_vn(vn[sc]['time'], vn[sc]['vn'],
                                              edges[sc][t0k], edges[sc][t1k])

    print("\nBoundary-layer thicknesses (km):")
    for sc, d in thickness.items():
        print(f"{sc.upper()}: " + ", ".join(f"{k}={abs(v):.0f}" for k, v in d.items()))

    # ---------- global MP speed --------------------------------------
    def pos_at(sc, t64):
        t, r = m[sc]['time_pos'], m[sc]['pos']
        return np.array([np.interp(t64.astype('datetime64[s]').astype(float), t, r[:, i])
                         for i in range(3)])

    p1 = pos_at('mms1', edges['mms1']['curr_sheet'])
    p4 = pos_at('mms4', edges['mms4']['curr_sheet'])
    dt_glob = (edges['mms4']['curr_sheet'] - edges['mms1']['curr_sheet']) / np.timedelta64(1, 's')
    Vn_glob = np.dot(p4 - p1, Nhat) / dt_glob
    print(f"\nGlobal MP normal speed ≈ {Vn_glob:.1f} km s⁻¹")

    for sc in ['mms2', 'mms3']:
        p_sc = pos_at(sc, edges[sc]['curr_sheet'])
        d_sc = np.dot(p_sc - p1, Nhat)
        t_pred = edges['mms1']['curr_sheet'] + np.timedelta64(int(d_sc / Vn_glob * 1e3), 'ms')
        print(f"{sc.upper()}: predicted CS {t_pred} vs observed {edges[sc]['curr_sheet']}")

    # ---------- distance-time plot -----------------------------------
    dser = {sc: {'time': vn[sc]['time'],
                 'dist': distance_series(vn[sc]['time'], vn[sc]['vn'],
                                         edges[sc]['curr_sheet'])}
            for sc in vn}
    plot_distance_time_series(dser, edges)
