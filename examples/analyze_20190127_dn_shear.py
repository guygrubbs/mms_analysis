"""
Event 2019-01-27 12:15–12:55 — DN prediction, shear/X-line, and BN stacks
Uses two .sav LMN sets:
- ver2b: all MMS (1–4) use ~12:43 UT crossing LMN
- ver3b: mixed — MMS1/2 use ~12:43 UT, MMS3/4 use ~12:30 UT MVA
Outputs to results/events_pub/2019-01-27_1215-1255
"""
from __future__ import annotations
import pathlib, sys, re, math
import numpy as np, pandas as pd, matplotlib.pyplot as plt
ROOT=str(pathlib.Path(__file__).resolve().parents[1]); sys.path.insert(0, ROOT)
import mms_mp as mp
from tools.idl_sav_import import load_idl_sav

EVENT_DIR=pathlib.Path('results/events_pub/2019-01-27_1215-1255'); EVENT_DIR.mkdir(parents=True, exist_ok=True)
TRANGE=('2019-01-27/12:15:00','2019-01-27/12:55:00'); PROBES=('1','2','3','4')
SAVS={
 'all_1243':'mp_lmn_systems_20190127_1215-1255_mp-ver2b.sav',
 'mixed_1230_1243':'mp_lmn_systems_20190127_1215-1255_mp-ver3b.sav'
}

# Cold-ion window inference (Option 2)
# Infer vt-style windows from local DIS moments using strict local cache
# Uses DIS quality flags (accept level 0), ion density (N_tot), and ion bulk speed quantiles
# Windows must be >=30 s contiguous

def infer_cold_ion_windows(evt, speed_q: float = 0.4, density_q: float = 0.4, min_duration_s: int = 30):
    from mms_mp.quality import build_quality_masks
    vt = {p: [] for p in PROBES}
    for p in PROBES:
        key=str(p)
        # Require ion velocity for speed proxy
        if 'V_i_gse' not in evt[key]:
            continue
        tV, V = evt[key]['V_i_gse']
        Vdf = to_df(tV, V, ['Vx','Vy','Vz'])
        if len(Vdf.index) < 2:
            continue
        speed = np.sqrt((Vdf[['Vx','Vy','Vz']].values**2).sum(axis=1))
        speed_s = pd.Series(speed, index=Vdf.index)
        # Optional density (may be reconstructed)
        Nd = None
        if 'N_tot' in evt[key] and evt[key]['N_tot'][0] is not None:
            tN, N = evt[key]['N_tot']
            Ndf = to_df(tN, N, ['Ni'])
            Nd = Ndf['Ni'].reindex(Vdf.index, method='nearest')
        # DIS quality mask (accept level 0); fall back to all True if unavailable
        try:
            masks = build_quality_masks(evt, probe=p)
            qmask = masks.get('DIS', np.ones(len(Vdf), dtype=bool))
            if isinstance(qmask, np.ndarray):
                qmask = pd.Series(qmask, index=Vdf.index)
            else:
                qmask = pd.Series(np.asarray(qmask), index=Vdf.index)
        except Exception:
            qmask = pd.Series(np.ones(len(Vdf), dtype=bool), index=Vdf.index)
        # Thresholds (quantile-based, robust to scale)
        valid = qmask & np.isfinite(speed_s)
        if not valid.any():
            continue
        sp_th = np.nanquantile(speed_s[valid], speed_q)
        mask = (speed_s <= sp_th) & qmask
        if Nd is not None and Nd.notna().any():
            dn_th = np.nanquantile(Nd[qmask & Nd.notna()], density_q)
            mask = mask & (Nd <= dn_th)
        # Extract contiguous windows of >= min_duration_s
        arr = mask.fillna(False).astype(bool).values
        if not arr.any():
            continue
        changes = np.diff(np.r_[False, arr, False].astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        for i0, i1 in zip(starts, ends):
            # Guard against degenerate/duplicate timestamps
            t0 = Vdf.index[i0]
            t1 = Vdf.index[i1-1]
            try:
                dur = (t1 - t0).total_seconds()
            except Exception:
                dur = 0.0
            if dur >= min_duration_s:
                vt[key].append((t0.isoformat(), t1.isoformat()))
    return vt

# Fallback cold-ion windows from .sav VI_LMN if DIS-based inference yields nothing
# Uses |V_lmn| quantile (q) with same min_duration_s rule.
def infer_cold_ion_windows_from_sav(sav, q: float = 0.4, min_duration_s: int = 30):
    vt = {p: [] for p in PROBES}
    vi = sav.get('vi_lmn', {})
    for p in PROBES:
        obj = vi.get(p)
        if not obj:
            continue
        t = np.asarray(obj.get('t', []), dtype=float)
        y = np.asarray(obj.get('vlmn', []), dtype=float)
        if t.size < 2 or y.ndim != 2 or y.shape[0] != t.size:
            continue
        if y.shape[1] >= 3:
            speed = np.sqrt((y[:, :3]**2).sum(axis=1))
        else:
            speed = np.sqrt((y**2).sum(axis=1))
        # Build 1s UTC index
        idx = pd.to_datetime(t, unit='s', utc=True)
        df = pd.DataFrame({'speed': speed}, index=idx)
        df = mp.data_loader.resample(df, '1s')
        sp = df['speed']
        if sp.dropna().empty:
            continue
        thr = np.nanquantile(sp.values, q)
        mask = sp <= thr
        arr = mask.fillna(False).astype(bool).values
        if not arr.any():
            continue
        changes = np.diff(np.r_[False, arr, False].astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        for i0, i1 in zip(starts, ends):
            t0 = df.index[i0]
            t1 = df.index[i1-1]
            dur = (t1 - t0).total_seconds()
            if dur >= min_duration_s:
                vt[p].append((t0.isoformat(), t1.isoformat()))
    return vt

# Helpers
def to_df(t,val,cols):
    df=mp.data_loader.to_dataframe(t,val,cols=cols)
    # Ensure UTC-aware index for robust comparisons
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    return mp.data_loader.resample(df,'1s')

def rotate_series(vec_df:pd.DataFrame,L,M,N):
    R=np.vstack([L,M,N]).T
    LMN=vec_df.values@R
    return pd.DataFrame(LMN, index=vec_df.index, columns=['L','M','N'])

# Build per-probe B_N, V_N and DN in cold-ion windows
# If DIS V_i is degenerate/missing, fall back to .sav VI_LMN to derive VN
def build_timeseries(evt, lmn_map, vt, sav=None):
    BN, VN, DN = {}, {}, {}
    vi_sav = (sav or {}).get('vi_lmn', {}) if isinstance(sav, dict) else {}
    for p in PROBES:
        key=str(p)
        if 'B_gsm' not in evt[key]:
            continue
        L=lmn_map[key]['L']; M=lmn_map[key]['M']; N=lmn_map[key]['N']
        tB,B=evt[key]['B_gsm']; Bdf=to_df(tB,B,['Bx','By','Bz'])
        Blmn=rotate_series(Bdf,L,M,N); BN[key]=Blmn['N']
        # Prefer DIS velocity rotated to VN; fall back to .sav VI_LMN
        Vn_series=None
        if 'V_i_gse' in evt[key]:
            try:
                tV,V=evt[key]['V_i_gse']; Vdf=to_df(tV,V,['Vx','Vy','Vz'])
                Vlmn=rotate_series(Vdf,L,M,N)
                if len(Vdf.index) >= 2 and Vdf.index.min() != Vdf.index.max():
                    Vn_series = Vlmn['N']
            except Exception:
                Vn_series = None
        if Vn_series is None and p in vi_sav:
            obj=vi_sav[p]
            t=np.asarray(obj.get('t', []), dtype=float)
            y=np.asarray(obj.get('vlmn', []), dtype=float)
            if t.size and y.ndim==2 and y.shape[0]==t.size:
                vn = y[:,2] if y.shape[1] >= 3 else (y if y.ndim==1 else np.full(t.size, np.nan))
                idx=pd.to_datetime(t, unit='s', utc=True)
                df=pd.DataFrame({'VN':vn}, index=idx)
                df=mp.data_loader.resample(df, '1s')
                Vn_series = df['VN']
        if Vn_series is None:
            continue
        VN[key]=Vn_series
        # Cold-ion mask: use vt windows as authoritative (no source labels on plots)
        mask=pd.Series(False,index=VN[key].index)
        for (t0s,t1s) in vt.get(key,[]):
            t0=pd.to_datetime(t0s,utc=True); t1=pd.to_datetime(t1s,utc=True)
            if t1<t0: t0,t1=t1,t0
            mask[(mask.index>=t0)&(mask.index<=t1)]=True
        # Integrate DN only where mask True
        from mms_mp.motion import integrate_disp
        series=VN[key].copy(); series[~mask]=np.nan
        # Piecewise integration per contiguous region
        dn=np.full(series.shape, np.nan)
        idx=series.index.view('int64')/1e9
        isn=np.isnan(series.values)
        start=None
        for i in range(len(series)):
            if not isn[i] and start is None: start=i
            end_block=(i==len(series)-1) or (not isn[i] and isn[i+1])
            if start is not None and end_block:
                j=i
                t=idx[start:j+1]; v=series.values[start:j+1]
                try:
                    res=integrate_disp(t,v,scheme='trap')
                    dn[start:j+1]=res.disp_km - res.disp_km[0]
                except Exception:
                    pass
                start=None
        DN[key]=pd.Series(dn,index=series.index,name='DN_km')
    return BN,VN,DN

# Crossing detection and predictions (use |dBN/dt| ≥ 0.4 peaks with ≥30 s separation)
def crossings_and_predictions(BN,VN,evt,lmn_map,label):
    rows=[]; preds=[]
    t0=pd.to_datetime(TRANGE[0], utc=True); t1=pd.to_datetime(TRANGE[1], utc=True)
    # crossings per probe
    for p in PROBES:
        bn=BN.get(p)
        if bn is None or len(bn) < 3:
            continue
        # Restrict to the analysis window
        bn = bn[(bn.index>=t0)&(bn.index<=t1)]
        if bn.empty:
            continue
        # Compute normalized rotation rate |dBN/dt| and gate at ≥ 0.4
        tsec = bn.index.view('int64').astype(float) * 1e-9
        try:
            g = np.gradient(bn.values, tsec, edge_order=2)
        except Exception:
            continue
        g = np.abs(g)
        finite = np.isfinite(g)
        if not finite.any():
            continue
        gmax = np.nanpercentile(g[finite], 95) if finite.any() else np.nan
        gn = g / gmax if (gmax and np.isfinite(gmax) and gmax>0) else g
        # Active segments where gn >= 0.4; pick the local max index per segment
        active = np.isfinite(gn) & (gn >= 0.4)
        if not active.any():
            layers = []
        else:
            changes = np.diff(np.r_[False, active, False].astype(int))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            keep_idx=[]; last_t=None
            for i0,i1 in zip(starts, ends):
                j0=max(0,i0); j1=min(len(gn)-1,i1-1)
                if j1 < j0: continue
                j = j0 + int(np.nanargmax(gn[j0:j1+1]))
                t = bn.index[j]
                if last_t is not None:
                    dt_s = (t - last_t).total_seconds() if hasattr(t - last_t, 'total_seconds') else float((t - last_t) / np.timedelta64(1, 's'))
                    if dt_s < 30:
                        # too close; keep the stronger one
                        if gn[j] > gn[keep_idx[-1]]:
                            keep_idx[-1] = j; last_t = t
                        continue
                keep_idx.append(j); last_t = t
            layers=[('MP_boundary', int(i), None) for i in keep_idx]
        # Positions projected on N for each crossing index
        if f'POS_gsm' not in evt[p]:
            continue
        tpos, pos = evt[p]['POS_gsm']
        posdf=to_df(tpos,pos,['X','Y','Z'])
        for typ,i1,i2 in layers or []:
            for idx in [i1,i2]:
                if idx is None: continue
                idx=int(idx)
                if idx<0 or idx>=len(bn): continue
                tc=bn.index[idx]
                # Robust: guard if tc not exactly in ephemeris index
                if tc not in posdf.index:
                    near = posdf.index.get_indexer([tc], method='nearest')
                    if near.size and 0 <= near[0] < len(posdf.index):
                        tc = posdf.index[near[0]]
                    else:
                        continue
                posN=float((posdf.loc[tc, ['X','Y','Z']].values @ lmn_map[p]['N']))
                rows.append({'set':label,'probe':p,'type':typ,'time_utc':tc.isoformat(), 'rN_km':posN})
    cross_df=pd.DataFrame(rows)
    if cross_df.empty or 'probe' not in cross_df.columns:
        return cross_df, pd.DataFrame([], columns=['set','ref','tgt','type','t_ref','t_act','t_pred','err_s','Vn_ref_km_s'])
    # predictions (pairwise)
    for ref in PROBES:
        for tgt in PROBES:
            if ref==tgt: continue
            ref_rows=cross_df[cross_df['probe']==ref]
            tgt_rows=cross_df[cross_df['probe']==tgt]
            if ref_rows.empty or tgt_rows.empty: continue
            for _,r in ref_rows.iterrows():
                tc=pd.to_datetime(r['time_utc'])
                # estimate Vn_ref near tc (±10s) from VN
                vn=VN.get(ref)
                if vn is None: continue
                win=vn.loc[(vn.index>=tc-np.timedelta64(10,'s'))&(vn.index<=tc+np.timedelta64(10,'s'))]
                if win.dropna().empty: continue
                Vn_ref=float(np.nanmedian(win.values)) # km/s
                # find target actual crossing nearest tc
                q=tgt_rows.copy(); q['t']=pd.to_datetime(q['time_utc'])
                j=int(np.argmin(np.abs(q['t']-tc)))
                t_act=q.iloc[j]['t']
                # delta rN along ref's N
                tpos_ref,_=evt[ref]['POS_gsm']; pref=to_df(tpos_ref,evt[ref]['POS_gsm'][1],['X','Y','Z'])
                tpos_tgt,_=evt[tgt]['POS_gsm']; ptgt=to_df(tpos_tgt,evt[tgt]['POS_gsm'][1],['X','Y','Z'])
                # Nearest position timestamps to tc
                if tc not in pref.index:
                    ii = pref.index.get_indexer([tc], method='nearest')
                    if ii.size and 0 <= ii[0] < len(pref.index): tc = pref.index[ii[0]]
                rN_ref=float((pref.loc[tc, ['X','Y','Z']].values @ lmn_map[ref]['N']))
                rN_tgt=float((ptgt.loc[tc, ['X','Y','Z']].values @ lmn_map[ref]['N']))
                dt_pred_s=(rN_tgt - rN_ref)/(-Vn_ref if Vn_ref!=0 else np.nan)
                t_pred=pd.Timestamp(tc)+pd.to_timedelta(dt_pred_s, unit='s')
                err_s=(t_pred - t_act).total_seconds()
                preds.append({'set':label,'ref':ref,'tgt':tgt,'type':r['type'],'t_ref':tc.isoformat(),'t_act':t_act.isoformat(),'t_pred':t_pred.isoformat(), 'err_s':err_s,'Vn_ref_km_s':Vn_ref})
    pred_df=pd.DataFrame(preds)
    return cross_df, pred_df

# Plotters
def plot_dn(DN,vt,label):
    fig,axs=plt.subplots(4,1,sharex=True,figsize=(12,8))
    for i,p in enumerate(PROBES):
        dn=DN.get(p); ax=axs[i]
        if dn is None: continue
        ax.plot(dn.index,dn.values,lw=1.2,color='k',label=f'MMS{p} DN')
        for (t0s,t1s) in vt.get(p,[]):
            t0=pd.to_datetime(t0s,utc=True); t1=pd.to_datetime(t1s,utc=True)
            if t1<t0: t0,t1=t1,t0
            ax.axvspan(t0,t1,color='k',alpha=0.05)
        ax.set_ylabel(f'MMS{p}\nDN (km)')
        ax.grid(True,alpha=0.25)
    axs[-1].set_xlabel('Time (UTC)'); fig.suptitle(f'DN vs time ({label})'); fig.autofmt_xdate(); fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(EVENT_DIR/f'dn_vs_time_{label}.png',dpi=220); plt.close(fig)

def plot_pred_vs_actual(pred_df,label):
    if pred_df.empty:
        return
    # Scatter of predicted vs actual times → plot errors as bars
    pred_df=pred_df.copy(); pred_df['abs_err']=pred_df['err_s'].abs()
    fig,ax=plt.subplots(figsize=(12,5))
    by=pred_df.groupby(['ref','tgt'])['abs_err'].median().reset_index()
    cats=[f'{r}->{t}' for r,t in zip(by['ref'],by['tgt'])]
    ax.bar(cats, by['abs_err']); ax.set_ylabel('Median |prediction error| (s)'); ax.set_title(f'Predicted vs actual crossing time error ({label})'); ax.grid(True,alpha=0.25)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(EVENT_DIR/f'pred_vs_actual_{label}.png',dpi=220); plt.close(fig)

# Shear angles and 3D formation at specific times
def shear_and_xline(evt,lmn_map,BN,label):
    times=['2019-01-27/12:18:00','2019-01-27/12:25:00','2019-01-27/12:45:00']
    rows=[]
    for ts in times:
        t=pd.to_datetime(ts,utc=True)
        for p in PROBES:
            bn=BN.get(p)
            if bn is None: continue
            # Classify sheath vs magnetosphere around t using BN sign as proxy
            win=bn.loc[(bn.index>=t-np.timedelta64(60,'s'))&(bn.index<=t+np.timedelta64(60,'s'))]
            if win.empty: continue
            tb, B = evt[p]['B_gsm']; Bdf=to_df(tb,B,['Bx','By','Bz']).loc[win.index.min():win.index.max()]
            # Split by BN sign
            ms=Bdf[win>0].values; sh=Bdf[win<0].values
            if len(ms)<3 or len(sh)<3: continue
            B_ms=ms.mean(axis=0); B_sh=sh.mean(axis=0)
            cosang=np.dot(B_ms,B_sh)/(np.linalg.norm(B_ms)*np.linalg.norm(B_sh))
            cosang=np.clip(cosang,-1,1); ang=math.degrees(math.acos(cosang))
            # Uncertainty: std of instantaneous angle within window
            inst=[]
            for i in range(min(len(ms),len(sh))):
                a=np.dot(ms[i%len(ms)], sh[i%len(sh)])/(np.linalg.norm(ms[i%len(ms)])*np.linalg.norm(sh[i%len(sh)])); inst.append(math.degrees(math.acos(np.clip(a,-1,1))))
            unc=np.nanstd(inst)
            rows.append({'set':label,'probe':p,'time_utc':t.isoformat(),'shear_deg':ang,'sigma_deg':unc})
    shear_df=pd.DataFrame(rows)
    # Plot shear evolution
    if not shear_df.empty:
	        fig,ax=plt.subplots(figsize=(10,5))
	        all_t = []
	        for p in PROBES:
	            d=shear_df[shear_df.probe==p]
	            if d.empty:
	                continue
	            t = pd.to_datetime(d['time_utc'])
	            all_t.append(t)
	            ax.plot(t, d['shear_deg'], marker='o', label=f'MMS{p}')
	        # Explicitly restrict the x-axis to the shear-evaluation times (with
	        # a small padding) so the plot does not span the full mission window
	        # when the underlying event data cover multiple years.
	        if all_t:
	            t_concat = pd.concat(all_t)
	            t_min, t_max = t_concat.min(), t_concat.max()
	            pad = pd.Timedelta(minutes=2)
	            ax.set_xlim(t_min - pad, t_max + pad)
	        ax.set_ylabel('Magnetic shear (deg)')
	        ax.set_title(f'Shear angles at key times ({label})')
	        ax.grid(True,alpha=0.3)
	        ax.legend(frameon=False)
	        fig.autofmt_xdate()
	        fig.tight_layout()
	        fig.savefig(EVENT_DIR/f'shear_angles_{label}.png',dpi=220)
	        plt.close(fig)
    # 3D formation vs X-line (use mean M as X-line)
    fig=plt.figure(figsize=(6,5)); ax=fig.add_subplot(111,projection='3d')
    t0=pd.to_datetime('2019-01-27/12:45:00',utc=True)
    Ms=[]; pts=[]; labels=[]
    for p in PROBES:
        try:
            Ms.append(lmn_map[p]['M'])
            tpos,pos=evt[p].get('POS_gsm', (None, None))
            if tpos is None or pos is None:
                continue
            posdf=to_df(tpos,pos,['X','Y','Z'])
            if posdf.empty:
                continue
            # Use the ephemeris sample at or nearest to t0 so the formation
            # is always populated when MEC data exist for the event window.
            if t0 not in posdf.index:
                idx = posdf.index.get_indexer([t0], method='nearest')
                if not (idx.size and 0 <= idx[0] < len(posdf.index)):
                    continue
                t_use = posdf.index[idx[0]]
            else:
                t_use = t0
            pts.append(posdf.loc[t_use,['X','Y','Z']].values); labels.append(f'MMS{p}')
        except Exception:
            continue
    P=None
    if pts:
        P=np.vstack(pts)
        colors=['C0','C1','C2','C3'][:len(P)]
        ax.scatter(P[:,0],P[:,1],P[:,2], c=colors, label='MMS spacecraft')
        for i,lab in enumerate(labels):
            ax.text(P[i,0],P[i,1],P[i,2],lab)
    if Ms:
        M_avg=np.mean(np.vstack(Ms),axis=0); M_avg/=np.linalg.norm(M_avg)
    else:
        M_avg=np.array([0.0,1.0,0.0])
    o=P.mean(axis=0) if P is not None else np.zeros(3); L=3e3*M_avg
    ax.quiver(o[0],o[1],o[2], L[0],L[1],L[2], color='k', linewidth=2, label='Inferred X-line')
    ax.set_title(f'MMS formation & inferred X-line ({label})')
    ax.set_xlabel('X_gsm km'); ax.set_ylabel('Y_gsm km'); ax.set_zlabel('Z_gsm km')
    # Include a legend so viewers can distinguish spacecraft markers from the
    # inferred X-line direction.
    handles,leg_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, leg_labels, loc='best', frameon=False)
    fig.tight_layout(); fig.savefig(EVENT_DIR/f'xline_3d_{label}.png',dpi=220); plt.close(fig)
    return shear_df

# BN stacked visualization
def plot_bn_stack(BN,vt,label):
    fig,axs=plt.subplots(4,1,sharex=True,figsize=(12,8))
    for i,p in enumerate(PROBES):
        bn=BN.get(p); ax=axs[i]
        if bn is None: continue
        ax.plot(bn.index,bn.values,color='k',lw=1.2,label='B_N')
        for (t0s,t1s) in vt.get(p,[]):
            t0=pd.to_datetime(t0s,utc=True); t1=pd.to_datetime(t1s,utc=True)
            if t1<t0: t0,t1=t1,t0
            ax.axvspan(t0,t1,color='k',alpha=0.05)
        ax.set_ylabel(f'MMS{p}\nB_N (nT)'); ax.grid(True,alpha=0.25)
    axs[-1].set_xlabel('Time (UTC)'); fig.suptitle(f'B_N stacked (LMN: {label})'); fig.autofmt_xdate(); fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(EVENT_DIR/f'bn_stacked_{label}.png',dpi=220); plt.close(fig)

# Minimal event loader avoiding electron requirements
# Uses PySPEDAS notplot=True to bypass pytplot registration issues and strictly read local CDFs
# Returns arrays compatible with downstream to_df()/rotate_series logic
def _minimal_event(trange, probes):
    from pyspedas.projects import mms
    evt = {p: {} for p in probes}
    t0, t1 = trange
    for p in probes:
        key = f'mms{p}'
        # FGM (B in GSM preferred; fallback to GSE)
        try:
            out_fgm = mms.mms_load_fgm(trange=[t0, t1], probe=p, data_rate='srvy', level='l2', time_clip=True, notplot=True)
            bname = f'{key}_fgm_b_gsm_srvy_l2'
            if bname not in out_fgm:
                bname = f'{key}_fgm_b_gse_srvy_l2' if f'{key}_fgm_b_gse_srvy_l2' in out_fgm else None
            if bname is not None:
                yb = out_fgm.get(bname, {})
                tb_src = yb.get('x') if isinstance(yb, dict) and 'x' in yb else out_fgm.get('Epoch', {}).get('y') if isinstance(out_fgm.get('Epoch', {}), dict) else None
                if tb_src is not None:
                    t_b = np.asarray(tb_src)
                    y_b = np.asarray(yb.get('y')) if isinstance(yb, dict) else None
                    if y_b is not None:
                        # Expect (N,4): Bx,By,Bz,|B|; take first 3 columns if present
                        if y_b.ndim == 2 and y_b.shape[1] >= 3:
                            B = y_b[:, :3]
                        else:
                            B = y_b
                        evt[p]['B_gsm'] = (t_b, B)
        except Exception:
            pass
        # FPI DIS moments (ion bulk velocity in GSE for VN derivation + number density)
        try:
            out_dis = mms.mms_load_fpi(trange=[t0, t1], probe=p, data_rate='fast', level='l2', datatype='dis-moms', time_clip=True, notplot=True)
            vname = f'{key}_dis_bulkv_gse_fast'
            if vname in out_dis:
                vd = out_dis.get(vname, {})
                t_v = np.asarray(vd.get('x')) if isinstance(vd, dict) and 'x' in vd else np.asarray(out_dis.get('Epoch', {}).get('y')) if isinstance(out_dis.get('Epoch', {}), dict) else None
                V = np.asarray(vd.get('y')) if isinstance(vd, dict) else None
                # Expect (N,3)
                if t_v is not None and V is not None:
                    if V.ndim == 2 and V.shape[1] >= 3:
                        V = V[:, :3]
                    evt[p]['V_i_gse'] = (t_v, V)
            nname = f'{key}_dis_numberdensity_fast'
            if nname in out_dis:
                nd = out_dis.get(nname, {})
                t_n = np.asarray(nd.get('x')) if isinstance(nd, dict) and 'x' in nd else np.asarray(out_dis.get('Epoch', {}).get('y')) if isinstance(out_dis.get('Epoch', {}), dict) else None
                N = np.asarray(nd.get('y')).reshape(-1, 1) if isinstance(nd, dict) and 'y' in nd else None
                if t_n is not None and N is not None:
                    evt[p]['N_tot'] = (t_n, N)
        except Exception:
            pass
        # Ephemeris (GSM preferred; fallback to GSE); convert Earth radii → km if needed
        try:
            out_mec = mms.mms_load_mec(trange=[t0, t1], probe=p, data_rate='srvy', level='l2', datatype='epht89q', time_clip=True, notplot=True)
            pos_key = f'{key}_mec_r_gsm'
            if pos_key not in out_mec:
                pos_key = f'{key}_mec_r_gse' if f'{key}_mec_r_gse' in out_mec else None
            if pos_key is not None:
                md = out_mec.get(pos_key, {})
                t_pos = np.asarray(md.get('x')) if isinstance(md, dict) and 'x' in md else np.asarray(out_mec.get('Epoch', {}).get('y')) if isinstance(out_mec.get('Epoch', {}), dict) else None
                POS = np.asarray(md.get('y')) if isinstance(md, dict) else None
                if t_pos is not None and POS is not None:
                    if np.nanmax(np.abs(POS)) < 100:
                        POS = POS * 6371.0  # Re→km
                    evt[p]['POS_gsm'] = (t_pos, POS)
        except Exception:
            pass
    return evt


def _run_analysis_for_lmn_set(evt, vt, label, lmn_map, sav=None, summary_rows=None):
    """Run the full DN / crossings / shear pipeline for a given LMN set.

    Parameters
    ----------
    evt : dict
        Event data as returned by :func:`mms_mp.load_event` or ``_minimal_event``.
    vt : dict
        Cold-ion windows per probe.
    label : str
        Short label identifying the LMN source (e.g. ``"all_1243"``,
        ``"mixed_1230_1243"``, or ``"algorithmic"``).
    lmn_map : dict
        Mapping ``probe -> {"L", "M", "N"}`` with unit vectors in GSM.
    sav : dict, optional
        Optional IDL ``.sav`` payload providing VI_LMN fallback when DIS is
        missing or degenerate.
    summary_rows : list, optional
        Existing list of summary metric rows to extend.
    """
    if summary_rows is None:
        summary_rows = []

    BN, VN, DN = build_timeseries(evt, lmn_map, vt, sav=sav)
    plot_dn(DN, vt, label)

    # Export DN series per probe for this LMN set
    for p, series in DN.items():
        if series is not None and len(series) > 0:
            (series.to_frame(name='DN_km')).to_csv(EVENT_DIR / f'dn_mms{p}_{label}.csv')

    plot_bn_stack(BN, vt, label)
    cross_df, pred_df = crossings_and_predictions(BN, VN, evt, lmn_map, label)
    plot_pred_vs_actual(pred_df, label)
    shear_df = shear_and_xline(evt, lmn_map, BN, label)

    # Collect key metrics
    # DN stats per probe
    for p, series in DN.items():
        if series is None:
            continue
        vals = pd.Series(series.values)
        if vals.dropna().empty:
            continue
        summary_rows.append({
            'set': label,
            'probe': p,
            'dn_median_km': float(np.nanmedian(vals.values)),
            'dn_maxabs_km': float(np.nanmax(np.abs(vals.values))),
        })

    if not cross_df.empty:
        first_cross = cross_df.sort_values('time_utc').groupby('probe').head(1)
        for _, r in first_cross.iterrows():
            summary_rows.append({
                'set': label,
                'probe': r['probe'],
                'cross_time': r['time_utc'],
                'rN_km': r['rN_km'],
            })

    for _, r in shear_df.iterrows():
        summary_rows.append({
            'set': label,
            'probe': r['probe'],
            'time': r['time_utc'],
            'shear_deg': r['shear_deg'],
            'sigma_deg': r['sigma_deg'],
        })

    if not pred_df.empty:
        g = pred_df.groupby(['set', 'ref', 'tgt'])['err_s'].agg(['median', 'mean', 'std']).reset_index()
        for _, r in g.iterrows():
            summary_rows.append({
                'set': label,
                'ref': r['ref'],
                'tgt': r['tgt'],
                'pred_err_median_s': r['median'],
                'pred_err_mean_s': r['mean'],
                'pred_err_std_s': r['std'],
            })

    # Save per-set CSVs
    cross_df.to_csv(EVENT_DIR / f'crossings_{label}.csv', index=False)
    pred_df.to_csv(EVENT_DIR / f'predictions_{label}.csv', index=False)
    shear_df.to_csv(EVENT_DIR / f'shear_{label}.csv', index=False)

    return summary_rows


def _build_algorithmic_lmn_map(evt, window_half_width_s: float = 30.0):
    """Construct an LMN map using the physics-driven algorithmic LMN builder.

    This helper wraps :func:`mms_mp.coords.algorithmic_lmn` for the
    2019-01-27 event.  It uses **CDF-only inputs** (FGM B_gsm and MEC
    POS_gsm) plus manually curated boundary crossing times near 12:43 UT.

    The crossing times originate from earlier analysis using the IDL ``.sav``
    products, but are encoded here as simple UTC timestamps so that the *runtime*
    LMN construction depends only on CDF data.
    """
    from mms_mp.coords import algorithmic_lmn

    b_times = {}
    b_vals = {}
    pos_times = {}
    pos_vals = {}

    for p in PROBES:
        data = evt.get(p, {})
        if 'B_gsm' not in data or 'POS_gsm' not in data:
            continue
        tB, B = data['B_gsm']
        tP, P = data['POS_gsm']
        if tB is None or B is None or tP is None or P is None:
            continue
        b_times[p] = np.asarray(tB, dtype=float)
        b_vals[p] = np.asarray(B, dtype=float)
        pos_times[p] = np.asarray(tP, dtype=float)
        pos_vals[p] = np.asarray(P, dtype=float)

    # Crossing times near the main magnetopause crossing (~12:43 UT).
    # These are specified in UTC; pandas converts to epoch seconds.
    t_cross_utc = {
        '1': '2019-01-27T12:43:25',
        '2': '2019-01-27T12:43:26',
        '3': '2019-01-27T12:43:18',
        '4': '2019-01-27T12:43:26',
    }
    t_cross = {}
    for p, ts in t_cross_utc.items():
        if p in b_times:
            t_cross[p] = pd.Timestamp(ts, tz='UTC').timestamp()

    if len(t_cross) < 2:
        raise RuntimeError("algorithmic LMN requires at least two probes with crossing times.")

    lmn_per_probe = algorithmic_lmn(
        b_times=b_times,
        b_gsm=b_vals,
        pos_times=pos_times,
        pos_gsm_km=pos_vals,
        t_cross=t_cross,
        # The default normal_weights (0.8, 0.15, 0.05) were optimised for this
        # event via examples/algorithmic_lmn_param_sweep_20190127.py and give
        # mean N-angle differences < 7° and BN correlations > 0.9997 vs the
        # expert .sav LMN while remaining physically well-motivated.
        window_half_width_s=window_half_width_s,
        tangential_strategy="Bmean",
        normal_weights=(0.8, 0.15, 0.05),
    )

    # Convert LMN objects to the mapping expected by build_timeseries.
    lmn_map = {
        p: {'L': lm.L, 'M': lm.M, 'N': lm.N}
        for p, lm in lmn_per_probe.items()
    }
    return lmn_map


def main():
    try:
        evt = mp.load_event(
            list(TRANGE),
            probes=list(PROBES),
            include_ephem=True,
            data_rate_fgm='srvy',
            data_rate_fpi='fast',
            include_hpca=False,
        )
    except Exception:
        evt = _minimal_event(list(TRANGE), list(PROBES))

    # Option 2: infer cold-ion windows from local DIS moments (strict local cache)
    vt = infer_cold_ion_windows(evt)

    # Fallback: if DIS-based vt inference produced no windows, derive from .sav VI_LMN
    if sum(len(v) for v in vt.values()) == 0:
        try:
            first_label = next(iter(SAVS.keys()))
            sav_fb = load_idl_sav(SAVS[first_label])
            vt = infer_cold_ion_windows_from_sav(sav_fb)
            print('Note: DIS-based vt inference empty; using .sav-derived cold-ion windows fallback.')
        except Exception as e:
            print('Warning: failed to derive vt from .sav fallback:', e)

    summary_rows = []

    # 1) Original .sav-based LMN sets (authoritative reference frames)
    for label, savpath in SAVS.items():
        sav = load_idl_sav(savpath)
        lmn_map = sav.get('lmn', {})
        summary_rows = _run_analysis_for_lmn_set(evt, vt, label, lmn_map, sav=sav, summary_rows=summary_rows)

    # 2) Physics-driven algorithmic LMN (CDF-only at runtime)
    try:
        alg_label = 'algorithmic'
        lmn_alg_map = _build_algorithmic_lmn_map(evt, window_half_width_s=30.0)
        summary_rows = _run_analysis_for_lmn_set(evt, vt, alg_label, lmn_alg_map, sav=None, summary_rows=summary_rows)
    except Exception as e:
        print('Warning: algorithmic LMN analysis failed:', e)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(EVENT_DIR / 'summary_metrics.csv', index=False)
    print('Analysis outputs saved to', EVENT_DIR)


if __name__ == '__main__':
    main()

