#!/usr/bin/env python
"""
MMS magnetopause analysis – 27 Jan 2019, 12:00–13:00 UT
  • auto-detect ion edge, current sheet, electron edge for MMS1-4
  • distance–time curves (+ layer thickness)
  • LMN trajectories with cross / skim classification (auto MEC download)
"""

# ───────────────────────── Imports ──────────────────────────
import os, glob, numpy as np, matplotlib.pyplot as plt, matplotlib.dates as mdates
from datetime import datetime, timezone
from pyspedas import mms
from pytplot   import get_data
import spacepy.pycdf as cdf

# ───────────────────────── Parameters ───────────────────────
TRANGE      = ['2019-01-27/12:00:00', '2019-01-27/13:00:00']
NORMAL_VEC  = [0.98, -0.05, 0.18]          # LMN-N (from MVA)
ION_THR_KM  = -1000                        # ion edge ≤ −1 Mm
ELEC_THR_KM =  1000                        # electron edge ≥ +1 Mm
SLOPE_THR   =   10                         # km/s for CS zero-crossing
SKIM_THR    =  10000000                         # km |N| → “skimmer”

# ───────────────────────── helpers ──────────────────────────
EPOCH_1970 = mdates.date2num(datetime(1970, 1, 1, tzinfo=timezone.utc))
def to_mpl(t):
    t = np.asarray(t)
    return (t/86400+EPOCH_1970 if np.issubdtype(t.dtype, np.floating)
            else t.astype('datetime64[s]').astype(float)/86400+EPOCH_1970)

def try_get(name):
    out = get_data(name)
    return out if out is not None else (None, None)

def lmn_basis(Nvec):
    N = np.asarray(Nvec,float); N/=np.linalg.norm(N)
    ref = np.array([0,0,1]) if abs(N[2])<.9 else np.array([1,0,0])
    M = np.cross(N,ref); M/=np.linalg.norm(M); L=np.cross(M,N)
    return L,M,N

def to_lmn(pos,L,M,N): return pos @ np.vstack((L,M,N)).T

def interp_pos(tt,pos,t_dt):
    t0=tt[0]; sec=np.array([(t-t0).total_seconds() for t in tt])
    tgt=(t_dt-t0).total_seconds()
    return np.array([np.interp(tgt,sec,pos[:,i]) for i in range(3)])

# ───────────────────────── load FPI & MEC (PySPEDAS) ────────
def spedas_load(tr):
    mms.mec(trange=tr, probe=[1,2,3,4], data_rate='srvy', level='l2', notplot=False)
    mms.fpi(trange=tr, probe=[1,2,3,4], data_rate='fast', level='l2',
            datatype='dis-moms', notplot=False)
    d={}
    for p in '1234':
        sid=f"mms{p}"
        tpos,pos=try_get(f"{sid}_mec_r_gse"); tvi,Vi=try_get(f"{sid}_dis_bulkv_gse_fast")
        if tpos is None or tvi is None:
            raise RuntimeError(f"{sid} variables missing")
        d[sid]={'time_pos':tpos,'pos':pos,'time_vi':tvi,'Vi':Vi}
    return d

# ───────────────────────── derive Vn & distance ─────────────
def vn_dict(m,N):
    N=np.array(N)/np.linalg.norm(N)
    return {s:{'time':m[s]['time_vi'],'vn':m[s]['Vi']@N} for s in m},N

def cum_dist(time_sec, vn):
    dt=np.diff(time_sec)
    return np.concatenate(([0],np.cumsum(0.5*(vn[:-1]+vn[1:])*dt)))

def dist_series(t, vn, ref64):          # ← this is the function
    dt   = np.diff(t)
    cum  = np.concatenate(([0],
                           np.cumsum(0.5*(vn[:-1] + vn[1:]) * dt)))
    ref  = ref64.astype('datetime64[s]').astype(float)
    return cum - np.interp(ref, t, cum)

# ───────────────────────── auto-detect crossings ─────────────
def detect_edges(time_sec, dist):
    # Current sheet = first robust zero-crossing after 12:10 UT
    t_mid = np.datetime64('2019-01-27T12:10')
    mid_sec=t_mid.astype('datetime64[s]').astype(float)
    idx = np.where(time_sec>=mid_sec)[0]
    cs_i=None
    for i in idx[:-1]:
        if dist[i]*dist[i+1]<0 and abs((dist[i+1]-dist[i])/(time_sec[i+1]-time_sec[i]))>SLOPE_THR:
            cs_i=i; break
    if cs_i is None: return None  # detection failed

    # ion edge: last point before CS where dist <= ION_THR_KM
    ion_i   = np.where(dist[:cs_i]<=ION_THR_KM)[0]
    ion_i   = ion_i[-1] if ion_i.size else cs_i
    # electron edge: first after CS where dist >= ELEC_THR_KM
    elec_i  = np.where(dist[cs_i:]>=ELEC_THR_KM)[0]
    elec_i  = cs_i+elec_i[0] if elec_i.size else cs_i

    return {'ion_edge':time_sec[ion_i],
            'curr_sheet':time_sec[cs_i],
            'electron_edge':time_sec[elec_i]}

# ───────────────────────── MEC helper (auto-download) ───────
def ensure_mec_cdf(date_str,probe):
    y,m,d=date_str.split('-')
    cache=f"./pydata/mms{probe}/mec/srvy/l2/epht89q/{y}/{m}"
    os.makedirs(cache,exist_ok=True)
    patt=os.path.join(cache,f"mms{probe}_mec_srvy_l2_epht89q_{y}{m}{d}_v*.cdf")
    hits=glob.glob(patt)
    if hits: return hits[0]
    print(f"Downloading MEC epht89q for MMS{probe} {date_str} …")
    mms.mec(trange=[f"{date_str} 00:00",f"{date_str} 23:59"],
            probe=[int(probe)], data_rate='srvy', level='l2',
            datatype='epht89q', notplot=False)
    hits=glob.glob(patt); 
    if not hits: raise FileNotFoundError(f"MEC CDF missing for MMS{probe}")
    return hits[0]

def read_mec(path,var):
    with cdf.CDF(path) as f:
        pos=f[var][...]; raw=f['Epoch'][...]
        tt = (cdf.lib.tt2000_to_datetime(raw.astype(np.int64))
              if np.issubdtype(raw.dtype,np.integer)
              else np.asarray(raw,dtype=object))
    return np.asarray(tt),pos

# ───────────────────────── Plot routines ────────────────────
def plot_distance(dser,edges):
    plt.figure(figsize=(10,6))
    colors=dict(mms1='tab:red',mms2='tab:blue',mms3='tab:green',mms4='tab:purple')
    mk   =dict(ion_edge='^',curr_sheet='o',electron_edge='v')
    for sc,d in dser.items():
        tm=to_mpl(d['time']); plt.plot(tm,d['dist'],c=colors[sc],label=sc.upper())
        for key,sym in mk.items():
            te=to_mpl(np.array([edges[sc][key]]))[0]
            plt.scatter(te,np.interp(te,tm,d['dist']),marker=sym,s=40,c=colors[sc])
    plt.axhline(0,c='k',ls='--'); plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xlabel('UT'); plt.ylabel('Distance to MP (km)')
    plt.title('MMS magnetopause distance – auto-detected edges')
    plt.legend(); plt.tight_layout(); plt.gcf().autofmt_xdate()
    plt.savefig('MMS_MP_distance_auto.png',dpi=150); plt.show()

def plot_lmn(paths,skim):
    if not paths: return
    fig=plt.figure(figsize=(9,7)); ax=fig.add_subplot(111,projection='3d')
    cols=dict(mms1='tab:red',mms2='tab:blue',mms3='tab:green',mms4='tab:purple')
    for sc,tr in paths.items():
        ax.plot(tr[:,0],tr[:,1],tr[:,2],lw=1.5,c=cols[sc],label=sc.upper())
        ax.scatter(0,0,0,s=90,marker='o',edgecolors='k',
                   c='orange' if skim[sc] else 'red')
    ax.set_xlabel('ΔL (km)'); ax.set_ylabel('ΔM (km)'); ax.set_zlabel('ΔN (km)')
    ax.set_title('Trajectories in LMN – origin = MMS1 current-sheet')
    allxyz=np.vstack(list(paths.values())); rng=np.ptp(allxyz,axis=0).max()/2
    for f in (ax.set_xlim,ax.set_ylim,ax.set_zlim): f(-rng,rng)
    ax.legend(); plt.tight_layout(); plt.show()

# ───────────────────────── main ─────────────────────────────
if __name__=='__main__':
    # 1) load FPI / MEC via PySPEDAS
    m = spedas_load(TRANGE)
    vn,Nhat = vn_dict(m, NORMAL_VEC)

    # 2) auto-detect boundary times
    edges = {}
    for sc in vn:
        dist = cum_dist(vn[sc]['time'], vn[sc]['vn'])
        e = detect_edges(vn[sc]['time'], dist)
        if e is None:
            raise RuntimeError(f"{sc}: could not find crossings – adjust thresholds")
        edges[sc] = {k:np.datetime64(int(v),'s') for k,v in e.items()}

    # 3) thickness + distance-time plot
    dser={}
    print("\nSC   status  |N|km  thickness km")
    for sc in vn:
        dser[sc]={'time':vn[sc]['time'],
                  'dist':dist_series(vn[sc]['time'],vn[sc]['vn'],edges[sc]['curr_sheet'])}
    plot_distance(dser,edges)

    # 4) LMN trajectories
    L,M,N = lmn_basis(NORMAL_VEC)
    paths, skim = {}, {}
    ref_cs = None   # MMS1 CS origin

    for sc in ['mms1','mms2','mms3','mms4']:
        probe=sc[-1]
        mec=ensure_mec_cdf('2019-01-27',probe)
        tt,pos = read_mec(mec,f'mms{probe}_mec_r_gse')
        pos_lmn=to_lmn(pos,L,M,N)

        cs_dt  = edges[sc]['curr_sheet'].astype(datetime)
        cp_lmn = interp_pos(tt,pos_lmn,cs_dt)   # crossing in LMN

        if sc=='mms1': ref_cs = cp_lmn.copy()
        paths[sc] = pos_lmn - ref_cs
        N_off     = cp_lmn[2]-ref_cs[2]
        skim[sc]  = abs(N_off) < SKIM_THR

        thick = abs(dist_series(vn[sc]['time'],vn[sc]['vn'],edges[sc]['electron_edge']) -
                    dist_series(vn[sc]['time'],vn[sc]['vn'],edges[sc]['ion_edge']))[-1]
        print(f"{sc.upper()} {'skimmer' if skim[sc] else 'crosser':8s} "
              f"{abs(N_off):6.0f} {thick:10.0f}")

    plot_lmn(paths, skim)
