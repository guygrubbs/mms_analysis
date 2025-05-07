#!/usr/bin/env python
"""
MMS multi-species, multi-crossing magnetopause analysis
Event: 27 Jan 2019 12:00–13:00 UT
"""

# ───────────── Imports ────────────────────────────────────────────────
import numpy as np, matplotlib.pyplot as plt, matplotlib.dates as mdates
from datetime import datetime, timezone
import warnings, math
from pyspedas import mms
from pytplot   import get_data

# ───────────── Basic helpers ──────────────────────────────────────────
def try_get(name):
    """Always return (t, v); (None, None) if variable absent."""
    v = get_data(name)
    return (None, None) if v is None else v

def sec2dt64(sec):  # POSIX → datetime64[ns]
    return np.array(sec*1e9, dtype='datetime64[ns]')

EPOCH_1970 = mdates.date2num(datetime(1970,1,1,tzinfo=timezone.utc))
def mpl_time(t64):  # datetime64 → Matplotlib float
    sec = np.asarray(t64).astype('datetime64[s]').astype(float)
    return sec/86400 + EPOCH_1970

# ───────────── Mission loader ─────────────────────────────────────────
def load_data(trange):
    probes = ['1','2','3','4']
    # MEC + FGM (always L2)
    mms.mec(trange=trange, probe=probes, data_rate='srvy', level='l2', notplot=False)
    mms.fgm(trange=trange, probe=probes, data_rate='srvy', level='l2', notplot=False)
    # FPI DIS/DES
    mms.fpi(trange=trange, probe=probes, data_rate='fast',
            level='l2', datatype=['dis-moms','des-moms'], notplot=False)
    if get_data('mms4_des_numberdensity_fast') is None:          # fallback
        print('MMS-4 DES L2 missing – loading QL')
        mms.fpi(trange=trange, probe='4', data_rate='fast',
                level='ql', datatype='des-moms', notplot=False)
    # HPCA – one call per probe; datatype must be “moments” :contentReference[oaicite:2]{index=2}
    for p in probes:
        mms.hpca(trange=trange, probe=p, data_rate='fast',
                 level='l2', datatype='moments', notplot=True)

    out={}
    for p in probes:
        sid=f"mms{p}"
        t_pos,pos = try_get(f"{sid}_mec_r_gse")
        t_vi ,Vi  = try_get(f"{sid}_dis_bulkv_gse_fast")
        if t_pos is None or t_vi is None:
            raise RuntimeError(f"{sid}: MEC or DIS moments missing")
        out[sid]={'time_pos':t_pos,'pos':pos,'time_vi':t_vi,'Vi':Vi}
    return out

# ───────────── Species table ──────────────────────────────────────────
SPEC={
    'H+':('dis_numberdensity_fast',0.7),
    'e-':('des_numberdensity_fast',0.7),
    'He+':('hpca_heplus_numberdensity_fast',0.5),
    'O+':('hpca_oplus_numberdensity_fast',0.5)
}

# ───────────── Interpolation util ─────────────────────────────────────
def interp_1d(src_t,src_v,target):
    return np.interp(target, src_t, src_v) if src_t is not None else None

# ───────────── Local MVA normal ───────────────────────────────────────
def mva_normal(Bxyz):
    C=np.cov(Bxyz.T); _,vecs=np.linalg.eigh(C)
    n=vecs[:,0]; return n/np.linalg.norm(n)

# ───────────── Flip detector ──────────────────────────────────────────
def detect_events(m, win_sec=180, cadence=4, rot_thr=45, drop_thr=0.5,
                  w_rot=0.6, min_sep=30):
    evs={}
    pts_lead=int(60/cadence)

    for sid in m:
        t_com=m[sid]['time_vi']                  # 4-s cadence
        # --- interpolate FGM three components separately  :contentReference[oaicite:3]{index=3}
        tB,B=try_get(f"{sid}_fgm_b_gse_srvy_l2")
        if tB is None:
            warnings.warn(f"{sid}: FGM missing"); continue
        if B.shape[1]>3: B=B[:,:3]
        B_int=np.column_stack([interp_1d(tB,B[:,i],t_com) for i in range(3)])

        # --- species
        S_ts={}
        for sp,(var,_) in SPEC.items():
            t,v=try_get(f"{sid}_{var}")
            S_ts[sp]=interp_1d(t,v,t_com) if v is not None else None

        flips=np.where(np.diff(np.sign(B_int[:,0]))!=0)[0]
        cand=[]
        half=int(win_sec/2/cadence)
        for idx in flips:
            sl=slice(max(idx-half,0), min(idx+half,len(B_int)))
            Nhat=mva_normal(B_int[sl])
            Bn=B_int@Nhat
            if np.sign(Bn[idx-1]) == np.sign(Bn[idx+1]): continue

            # rotation
            vec0,vec1=B_int[idx-1],B_int[idx+1]
            cosang=np.clip(np.dot(vec0,vec1)/(np.linalg.norm(vec0)*np.linalg.norm(vec1)),-1,1)
            rot=np.degrees(np.arccos(cosang))

            # drops
            drops={}
            for sp in SPEC:
                arr=S_ts[sp]; drops[sp]=np.nan
                if arr is not None:
                    pre,post=arr[max(idx-pts_lead,0)], arr[min(idx+pts_lead,len(arr)-1)]
                    if pre>0: drops[sp]=(pre-post)/pre
            try:
                max_drop=np.nanmax(list(drops.values()))
            except ValueError:  # all-NaN
                continue
            if rot<rot_thr or max_drop<drop_thr: continue
            score=w_rot*(rot/180)+(1-w_rot)*max_drop
            cand.append(dict(idx=idx,N=Nhat,rot=rot,drops=drops,score=score))

        # prune by min separation
        cand.sort(key=lambda c:c['idx'])
        pruned=[]
        for c in cand:
            if not pruned or (c['idx']-pruned[-1]['idx'])*cadence >= min_sep:
                pruned.append(c)

        evs[sid]=[]
        for c in pruned:
            idx=c['idx']; Nhat=c['N']
            # threshold from H+ if present else B magnitude surrogate
            ni=S_ts['H+']; base=np.nanmedian(ni[:idx]) if ni is not None else np.nanmedian(np.linalg.norm(B_int[:idx],axis=1))
            thr=0.5*base
            idx_ion=np.where((ni if ni is not None else B_int[:,0])[:idx]>thr)[0]
            idx_ion=idx_ion[-1] if idx_ion.size else idx
            arr_e=S_ts['e-'] if S_ts['e-'] is not None else ni
            idx_ele=idx+np.where(arr_e[idx:]<thr)[0][0] if arr_e is not None else idx
            evs[sid].append(dict(
                times=dict(
                    ion_edge=sec2dt64(t_com[idx_ion]),
                    curr_sheet=sec2dt64(t_com[idx]),
                    electron_edge=sec2dt64(t_com[idx_ele])),
                rot=c['rot'], drops=c['drops'], Nhat=Nhat))
    return evs

# ───────────── Taxonomy & thickness ────────────────────────────────────
def vn_series(m, N):
    N=N/np.linalg.norm(N)
    return {s:{'time':m[s]['time_vi'],'vn':m[s]['Vi']@N} for s in m}

def integrate_vn(t,vn,t0,t1):
    a,b=[x.astype('datetime64[s]').astype(float) for x in (t0,t1)]
    msk=(t>=a)&(t<=b)
    return np.sum(0.5*(vn[msk][1:]+vn[msk][:-1])*np.diff(t[msk])) if msk.sum()>1 else np.nan

def classify(ev,thick):
    d=ev['drops']; h,e,he,o=[d.get(k,np.nan) for k in ('H+','e-','He+','O+')]
    if h>=0.7 and e>=0.7: cat='MP full'
    elif h>=0.5 and (math.isnan(e) or e<0.5): cat='MP ion-skim'
    elif e>=0.7 and (math.isnan(h) or h<0.3): cat='EDR'
    elif (o>=0.5 or he>=0.5) and (math.isnan(h) or h<0.4): cat='plume'
    else: cat='unknown'
    ev.update(class_=cat, thickness_km=abs(thick),
              cross='cross' if abs(thick)>=500 else 'skim')

# ───────────── Plotting ────────────────────────────────────────────────
CAT_M={'MP full':'o','MP ion-skim':'^','EDR':'s','plume':'D','unknown':'x'}
CAT_C={'MP full':'tab:red','MP ion-skim':'tab:orange','EDR':'tab:blue',
       'plume':'tab:green','unknown':'k'}

def plot_evs(evs,vn_all):
    plt.figure(figsize=(11,6))
    # pale traces
    for sc in vn_all:
        t=vn_all[sc]['time']; d=distance_series(t,vn_all[sc]['vn'],t[0])
        plt.plot(mpl_time(t),d,color='grey',alpha=.12)
    # events
    shown=set()
    for sc,lst in evs.items():
        for ev in lst:
            tcs=ev['times']['curr_sheet']; lab=f"{sc.upper()} {ev['class_']}"
            plt.scatter(mpl_time(np.array([tcs]))[0],0,
                        marker=CAT_M[ev['class_']],color=CAT_C[ev['class_']],
                        s=80,label=lab if lab not in shown else "")
            shown.add(lab)
    plt.axhline(0,ls='--',c='k'); plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.title('MMS boundary events 27 Jan 2019'); plt.xlabel('UT')
    plt.legend(fontsize=8); plt.tight_layout(); plt.show()

def distance_series(t,vn,ref64):
    disp=np.concatenate(([0],np.cumsum(0.5*(vn[:-1]+vn[1:])*np.diff(t))))
    ref=ref64.astype('datetime64[s]').astype(float)
    return disp-np.interp(ref,t,disp)

# ───────────── Main ────────────────────────────────────────────────────
if __name__=='__main__':
    TR=['2019-01-27/12:00:00','2019-01-27/13:00:00']
    m=load_data(TR)
    evs=detect_events(m)

    # pick earliest MP(full) for reference normal
    mp_full=[(s,e) for s,l in evs.items() for e in l if e['drops']['H+']>=0.7 and e['drops']['e-']>=0.7]
    if not mp_full: raise RuntimeError('No MP full events found')
    ref_sid,ref_ev=min(mp_full,key=lambda x:x[1]['times']['curr_sheet'])
    Nhat=ref_ev['Nhat']
    vn_all=vn_series(m,Nhat)

    # thickness & classification
    for sc,lst in evs.items():
        for ev in lst:
            thick=integrate_vn(vn_all[sc]['time'],vn_all[sc]['vn'],
                               ev['times']['ion_edge'],ev['times']['electron_edge'])
            classify(ev,thick)

    # report
    print("\nEvent list (UT):")
    for sc,lst in evs.items():
        for ev in lst:
            t=ev['times']['curr_sheet']; print(f"{sc.upper()} {t}  {ev['class_']:10s} "
                                               f"{ev['cross']}  rot={ev['rot']:.0f}°  "
                                               + " ".join(f"{k}:{ev['drops'][k]:.2f}" for k in SPEC))

    # global MP speed earliest→latest MP(full)
    latest_sid, latest_ev=max(mp_full,key=lambda x:x[1]['times']['curr_sheet'])
    def pos(sc,t64):
        tt,rr=m[sc]['time_pos'],m[sc]['pos']
        return np.array([np.interp(t64.astype('datetime64[s]').astype(float),tt,rr[:,i]) for i in range(3)])
    t0,t1=ref_ev['times']['curr_sheet'],latest_ev['times']['curr_sheet']
    Vn=np.dot(pos(latest_sid,t1)-pos(ref_sid,t0),Nhat)/( (t1-t0)/np.timedelta64(1,'s') )
    print(f"\nGlobal MP normal speed ≈ {Vn:.1f} km s⁻¹  ({ref_sid.upper()}→{latest_sid.upper()})")

    plot_evs(evs,vn_all)
