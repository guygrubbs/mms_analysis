#!/usr/bin/env python
"""
MMS multi-species, multi-crossing magnetopause analysis
Event: 27 Jan 2019 12:00–13:00 UT
Author: ChatGPT o3   (May-2025)

Required: pyspedas ≥ 1.4, numpy ≥ 1.20, matplotlib ≥ 3.7
"""

# ────────────── IMPORTS ────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
import warnings
from pyspedas import mms
from pytplot import get_data

# ────────────── SIMPLE HELPERS ─────────────────────────────────────────
def try_get(name):
    out = get_data(name)
    return out if out is not None else (None, None)

EPOCH_1970 = mdates.date2num(datetime(1970, 1, 1, tzinfo=timezone.utc))
def to_mpl(t64):
    x = np.asarray(t64)
    if np.issubdtype(x.dtype, 'datetime64'):
        return x.astype('datetime64[s]').astype(float) / 86400 + EPOCH_1970
    raise TypeError("expected datetime64")

def sec2dt64(sec):
    return np.array(sec * 1e9, dtype='datetime64[ns]')

# ────────────── DATA LOADER ────────────────────────────────────────────
def load(trange):
    probes = ['1', '2', '3', '4']
    # MEC & FGM (always L2)
    mms.mec(trange=trange, probe=probes, data_rate='srvy', level='l2', notplot=False)
    mms.fgm(trange=trange, probe=probes, data_rate='srvy', level='l2', notplot=False)
    # FPI DIS & DES
    mms.fpi(trange=trange, probe=probes, data_rate='fast', level='l2',
            datatype=['dis-moms', 'des-moms'], notplot=False)
    if get_data('mms4_des_numberdensity_fast') is None:
        mms.fpi(trange=trange, probe='4', data_rate='fast', level='ql',
                datatype='des-moms', notplot=False)
    # HPCA heavy ions
    mms.hpca(trange=trange, probe=probes, data_rate='fast', level='l2',
             datatype='heplus-moms', notplot=True)
    mms.hpca(trange=trange, probe=probes, data_rate='fast', level='l2',
             datatype='oplus-moms', notplot=True)

    data = {}
    for p in probes:
        sid = f"mms{p}"
        t_pos, pos = try_get(f"{sid}_mec_r_gse")
        t_vi,  Vi  = try_get(f"{sid}_dis_bulkv_gse_fast")
        if t_pos is None or t_vi is None:
            raise RuntimeError(f"{sid}: essential variables missing")
        data[sid] = {'time_pos': t_pos, 'pos': pos,
                     'time_vi' : t_vi , 'Vi' : Vi}
    return data

# ────────────── MVA NORMAL ─────────────────────────────────────────────
def mva_normal(B_xyz):
    C = np.cov(B_xyz.T)
    _, vecs = np.linalg.eigh(C)
    n = vecs[:, 0]
    return n / np.linalg.norm(n)

# ────────────── SPECIES LIST & ACCESSORS ───────────────────────────────
SPEC = {
    'H+':  ('dis_numberdensity_fast',     0.7),
    'e-':  ('des_numberdensity_fast',     0.7),
    'He+': ('hpca_heplus_numberdensity_fast', 0.5),
    'O+':  ('hpca_oplus_numberdensity_fast',  0.5)
}

def get_species_timeseries(sid, t_common):
    ts = {}
    for sp, (var, _) in SPEC.items():
        t, v = try_get(f"{sid}_{var}")
        ts[sp] = np.interp(t_common, t, v) if t is not None else None
    return ts

# ────────────── FLIP DETECTOR ──────────────────────────────────────────
def detect_all(m, win_mva=180, cadence=4.0,
               rot_thresh=45., dens_thresh=0.5, w_rot=0.6,
               min_sep=30):
    """
    Returns events[sc] = [ {times, drops, rot, class, thickness_km, Nhat}, ... ]
    times: dict(ion_edge, curr_sheet, electron_edge)   (datetime64)
    drops: dict for every species (fractional)
    """
    events, skipped = {}, []
    points_lead = int(60 / cadence)  # 60 s forward/back for drops

    for sid in m:
        t_common = m[sid]['time_vi']
        B_int  = np.vstack([np.interp(t_common, *try_get(f"{sid}_fgm_b_gse_srvy_l2"))[:3].T]).T
        if B_int.shape[1] > 3: B_int = B_int[:, :3]
        S_ts = get_species_timeseries(sid, t_common)

        flips = np.where(np.diff(np.sign(B_int[:, 0])) != 0)[0]
        cand = []

        for idx in flips:
            # local normal
            half = int((win_mva / 2) / cadence)
            sl = slice(max(idx-half, 0), min(idx+half, len(B_int)))
            Nhat = mva_normal(B_int[sl])

            # flip check with this normal
            Bn = B_int @ Nhat
            if np.sign(Bn[idx-1]) == np.sign(Bn[idx+1]): continue

            # rotation
            pre, post = B_int[idx-1], B_int[idx+1]
            rot = np.degrees(np.arccos(
                np.clip(np.dot(pre, post)/(np.linalg.norm(pre)*np.linalg.norm(post)), -1, 1)))

            # drops
            drops = {}
            for sp, _ in SPEC.items():
                arr = S_ts[sp]
                if arr is None: drops[sp] = np.nan; continue
                a, b = arr[max(idx-points_lead, 0)], arr[min(idx+points_lead, len(arr)-1)]
                drops[sp] = (a-b)/a if a>0 else np.nan

            if rot < rot_thresh or (np.nanmax(list(drops.values())) < dens_thresh):
                continue

            score = w_rot*(rot/180) + (1-w_rot)*np.nanmax(list(drops.values()))
            cand.append(dict(idx=idx, N=Nhat, rot=rot, drops=drops, score=score))

        # sort by time, enforce min separation
        cand.sort(key=lambda d: d['idx'])
        pruned=[]
        for c in cand:
            if not pruned or (c['idx'] - pruned[-1]['idx'])*cadence >= min_sep:
                pruned.append(c)

        if not pruned:
            skipped.append(sid); continue

        # build full event entries
        ev_list=[]
        for c in pruned:
            idx = c['idx']; Nhat = c['N']
            Bn = B_int @ Nhat
            # find edges per species H+ baseline
            sheath_med = np.nanmedian(S_ts['H+'][:idx]) if S_ts['H+'] is not None else np.nanmedian(Bn[:idx]**2)**0.5
            thr = sheath_med*0.5
            idx_ion = np.where(S_ts['H+'][:idx] > thr)[0]
            idx_ion = idx_ion[-1] if idx_ion.size else idx
            idx_ele = idx + np.where((S_ts['e-'] if S_ts['e-'] is not None else S_ts['H+'])[idx:] < thr)[0][0]

            times = dict(ion_edge=sec2dt64(t_common[idx_ion]),
                         curr_sheet=sec2dt64(t_common[idx]),
                         electron_edge=sec2dt64(t_common[idx_ele]))

            ev = dict(times=times, rot=c['rot'], drops=c['drops'], Nhat=Nhat)
            ev_list.append(ev)

        events[sid]=ev_list

    if skipped:
        warnings.warn(f"Skipped: {', '.join(skipped)} (no valid flips)")
    return events

# ────────────── TAXONOMY & THICKNESS ───────────────────────────────────
def integrate_vn(t, vn, t0, t1):
    a,b = (x.astype('datetime64[s]').astype(float) for x in (t0,t1))
    msk=(t>=a)&(t<=b)
    return np.sum(0.5*(vn[msk][1:]+vn[msk][:-1])*np.diff(t[msk])) if msk.sum()>1 else 0.

def classify_event(ev, thickness_km):
    d=ev['drops']; h=e=o=he=np.nan
    h=d.get('H+'); e=d.get('e-'); he=d.get('He+'); o=d.get('O+')
    # defaults
    cat='unknown'
    if h>=0.7 and e>=0.7: cat='MP full'
    elif h>=0.5 and (e<0.5 or np.isnan(e)): cat='MP ion-skim'
    elif e>=0.7 and (h<0.3 or np.isnan(h)): cat='EDR'
    elif (o>=0.5 or he>=0.5) and (h<0.4 or np.isnan(h)): cat='plume'
    ev['class']=cat
    ev['thickness_km']=abs(thickness_km)
    ev['cross'] = 'cross' if abs(thickness_km)>=500 else 'skim'
    return ev

# ────────────── PLOT ───────────────────────────────────────────────────
CAT_MARK = {'MP full':'o', 'MP ion-skim':'^', 'EDR':'s', 'plume':'D', 'unknown':'x'}
CAT_CLR  = {'MP full':'tab:red', 'MP ion-skim':'tab:orange',
            'EDR':'tab:blue', 'plume':'tab:green', 'unknown':'k'}

def plot_events(m, events, vn_all):
    plt.figure(figsize=(11,6))
    for sc, evs in events.items():
        for ev in evs:
            t_series = vn_all[sc]['time']
            dist = distance_series(t_series, vn_all[sc]['vn'], ev['times']['curr_sheet'])
            plt.plot(to_mpl(t_series), dist, color='grey', alpha=0.2)

    for sc, evs in events.items():
        for ev in evs:
            tcs = ev['times']['curr_sheet']
            t_mpl = to_mpl(np.array([tcs]))[0]
            plt.scatter(t_mpl, 0, marker=CAT_MARK[ev['class']],
                        color=CAT_CLR[ev['class']], s=70,
                        label=f"{sc.upper()} {ev['class']}" if plt.gca().get_legend_handles_labels()[1].count(f"{sc.upper()} {ev['class']}")==0 else "")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.axhline(0,color='k',lw=.5); plt.grid(True,ls=':')
    plt.xlabel('UT'); plt.title('MMS boundary events 27 Jan 2019')
    plt.legend(fontsize=8); plt.tight_layout()
    plt.show()

# ────────────── MAIN ───────────────────────────────────────────────────
if __name__ == '__main__':
    TR = ['2019-01-27/12:00:00','2019-01-27/13:00:00']
    m   = load(TR)
    evs = detect_all(m)

    # vn with reference normal = earliest MP(full) event
    all_mp_full = [(sid, ev) for sid, lst in evs.items() for ev in lst if ev['class']=='MP full']
    if not all_mp_full:
        raise RuntimeError("No MP full events found – cannot compute speeds")
    earliest_sid, earliest_ev = min(all_mp_full, key=lambda x: x[1]['times']['curr_sheet'])
    refN = earliest_ev['Nhat']
    vn_all, _ = compute_vn(m, refN)

    # thickness + classification
    for sc, lst in evs.items():
        for ev in lst:
            thick = integrate_vn(vn_all[sc]['time'], vn_all[sc]['vn'],
                                 ev['times']['ion_edge'], ev['times']['electron_edge'])
            classify_event(ev, thick)

    # report
    print("\n=== Events ===")
    for sc,lst in evs.items():
        for ev in lst:
            t=ev['times']; print(f"{sc.upper()}  {t['curr_sheet']}  {ev['class']:10s}  "
                                 f"{ev['cross']:5s}  Δθ={ev['rot']:.0f}°  "
                                 + " ".join(f"{sp}:{ev['drops'][sp]:.2f}" for sp in SPEC))

    # global speed from earliest & latest MP(full)
    latest_sid, latest_ev = max(all_mp_full, key=lambda x: x[1]['times']['curr_sheet'])
    def pos_at(sc,t64):
        t_sec,r = m[sc]['time_pos'], m[sc]['pos']
        return np.array([np.interp(t64.astype('datetime64[s]').astype(float), t_sec, r[:,i])
                         for i in range(3)])
    t0,t1 = earliest_ev['times']['curr_sheet'], latest_ev['times']['curr_sheet']
    p0,p1 = pos_at(earliest_sid,t0), pos_at(latest_sid,t1)
    Vn = np.dot(p1-p0, refN) / ((t1-t0)/np.timedelta64(1,'s'))
    print(f"\nGlobal MP Vn ≈ {Vn:.1f} km s⁻¹  ({earliest_sid.upper()} → {latest_sid.upper()})")

    # plot
    plot_events(m, evs, vn_all)
