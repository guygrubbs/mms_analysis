"""
MMS Event Visualization: 2019-01-27 12:00 UT

This example loads a 1-hour window centered at 2019-01-27 12:00:00 UT,
computes a local LMN frame per spacecraft, rotates magnetic field to LMN,
and generates quick-look summary plots. It is designed to compare with
published analyses for the Jan 27, 2019 magnetopause event.

Now supports:
- CLI options for center time, window length, probes, and output directory
- Saving figures with standardized filenames
- Optional overlay of published normal and phase speed from a JSON file

Requirements: pySPEDAS with MMS support, matplotlib, numpy, pandas.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Dict, Any

# Ensure repository root is on sys.path when running as a script
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive by default for batch runs
import matplotlib.pyplot as plt

from pyspedas.projects import mms
from pyspedas import get_data
from pytplot import data_quants

from mms_mp import data_loader, coords, visualize, spectra
from mms_mp import resample as mp_resample


def _window(center: datetime, minutes: int = 60):
    half = timedelta(minutes=minutes // 2)
    return [
        (center - half).strftime('%Y-%m-%d/%H:%M:%S'),
        (center + half).strftime('%Y-%m-%d/%H:%M:%S'),
    ]


def _to_datetime64_ns(tt2000_like: np.ndarray) -> np.ndarray:
    # data_loader.to_dataframe handles this, but we want raw time arrays for plotting
    # Use its helper indirectly by building a tiny DataFrame and extracting index
    df = data_loader.to_dataframe(tt2000_like, np.zeros((len(tt2000_like), 1)), ['stub'])
    return df.index.values.astype('datetime64[ns]')


def _load_overlay(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    # Normalize into per-probe mapping if needed
    if 'per_probe' in raw:
        return raw['per_probe']
    return {k: v for k, v in raw.items() if k in {'1','2','3','4'}} or {'all': raw}


def load_event(center: datetime, minutes: int, probes: list[str]) -> Dict[str, Dict[str, np.ndarray]]:
    trange = _window(center, minutes=minutes)
    return data_loader.load_event(
        trange=trange,
        probes=probes,
        data_rate_fgm='fast',
        data_rate_fpi='fast',
        data_rate_hpca='fast',
        include_brst=True,
        include_ephem=True,
        include_edp=False,
    )



def _ensure_spectrogram(probe: str, species: str, rates: list[str], trange: list[str]):
    """Ensure omni spectrogram vars are loaded via pyspedas if missing.
    species: 'des' or 'dis'
    """
    import fnmatch as _fn
    from pytplot import data_quants as _dq
    for rate in rates:
        keypat = f"mms{probe}_{species}_energyspectr_omni_{rate}*"
        if any(_fn.filter(_dq.keys(), keypat)):
            continue
        try:
            # Be explicit about varformat to encourage loading spectrogram products
            # First attempt the spectrogram product directly
            mms.fpi(
                trange=trange,
                probe=probe,
                data_rate=rate,
                datatype=f"{species}-spectr",
                time_clip=True,
                varformat=[
                    f"mms{probe}_{species}_energyspectr_*_{rate}*",
                    f"mms{probe}_{species}_energy_{rate}*",
                    f"mms{probe}_{species}_energyspectr_*",
                    f"mms{probe}_{species}_energy_*",
                ],
            )
        except Exception:
            continue
        # If omni still not present, try also loading distributions (may enable derived spectr)
        if not any(_fn.filter(_dq.keys(), keypat)):
            try:
                mms.fpi(
                    trange=trange,
                    probe=probe,
                    data_rate=rate,
                    datatype=f"{species}-dist",
                    time_clip=True,
                )
            except Exception:
                pass



def _log_spectrogram_diagnostics(probe: str, center: datetime, minutes: int):
    import fnmatch
    tr = _window(center, minutes)
    print(f"\n🔎 Spectrogram diagnostics for MMS{probe} in {tr}:")
    names = list(data_quants.keys())
    for species in ('des','dis'):
        print(f"  Species: {species}")
        for rate in ('brst','fast','srvy'):
            pats = [
                f"mms{probe}_{species}_energyspectr_omni_{rate}*",
                f"mms{probe}_{species}_energyspectr_*_{rate}*",
                f"mms{probe}_{species}_energy_{rate}*",
            ]
            matches = []
            for p in pats:
                matches.extend(fnmatch.filter(names, p))
            matches = sorted(set(matches))
            print(f"    {rate}: {len(matches)} matches")
            for v in matches[:6]:
                try:
                    g = get_data(v)
                    tt = g[0] if isinstance(g, (list, tuple)) and len(g) > 0 else None
                    dd = g[1] if isinstance(g, (list, tuple)) and len(g) > 1 else None
                    if tt is not None and dd is not None:
                        if hasattr(dd, 'ndim') and dd.ndim >= 2:
                            # compute finite min/max of slice to check it isn't empty/fill
                            dmin = np.nanmin(dd[np.isfinite(dd)]) if np.any(np.isfinite(dd)) else np.nan
                            dmax = np.nanmax(dd[np.isfinite(dd)]) if np.any(np.isfinite(dd)) else np.nan
                            print(f"      - {v}: shape={getattr(dd,'shape',None)}, min={dmin:.3g}, max={dmax:.3g}")
                        else:
                            print(f"      - {v}: shape={getattr(dd,'shape',None)}")
                except Exception as e:
                    print(f"      - {v}: error {e}")

# Build omni spectrogram by summing directional components if omni is missing
def _build_omni_from_directionals(probe: str, species: str, rates=('brst','fast','srvy')):
    import fnmatch
    comps = ['px','mx','py','my','pz','mz']
    for rate in rates:
        series = []
        t_ref = None
        for c in comps:
            pat = f"mms{probe}_{species}_energyspectr_{c}_{rate}*"
            matches = fnmatch.filter(data_quants.keys(), pat)
            if not matches:
                series = []
                break
            # Use the first match per component
            v = matches[0]
            g = get_data(v)
            # get_data can return (t, data) or (t, data, meta); normalize
            tt, dd = g[0], g[1] if isinstance(g, (list, tuple)) and len(g) > 1 else (None, None)
            if tt is None or dd is None:
                series = []
                break
            if t_ref is None:
                t_ref = tt
            series.append(dd)
        if series and t_ref is not None:
            try:
                omni = np.sum(np.stack(series, axis=0), axis=0)
                return t_ref, omni, rate
            except Exception:
                pass
    return None, None, None



def make_summary_plots(evt: Dict[str, Dict[str, np.ndarray]], *,
                       center: datetime,
                       minutes: int,
                       output_dir: Path,
                       overlay: Dict[str, Any],
                       show: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = center.strftime('%Y%m%d_%H%M%S')
    created = []

    for probe in ['1', '2', '3', '4']:
        if probe not in evt or 'B_gsm' not in evt[probe]:
            continue

        tB_raw, B_gsm = evt[probe]['B_gsm']
        if tB_raw is None or B_gsm is None or len(tB_raw) == 0:
            continue

        # Convert time to datetime64 for plotting
        t = _to_datetime64_ns(tB_raw)

        # Build LMN using hybrid method (MVA → PySPEDAS → Shue)
        lmn = coords.hybrid_lmn(B_gsm)
        B_lmn = lmn.to_lmn(B_gsm)

        # Densities and velocities (placeholders may be NaN if missing)
        Ni = evt[probe].get('N_tot', (tB_raw, np.full(len(t), np.nan)))[1]
        Ne = evt[probe].get('N_e',   (tB_raw, np.full(len(t), np.nan)))[1]
        Nhe = evt[probe].get('N_he', (tB_raw, np.full(len(t), np.nan)))[1]

        Vi = evt[probe].get('V_i_gse', (tB_raw, np.full((len(t), 3), np.nan)))[1]
        Ve = evt[probe].get('V_e_gse', (tB_raw, np.full((len(t), 3), np.nan)))[1]
        Vhe = evt[probe].get('V_he_gsm', (tB_raw, np.full((len(t), 3), np.nan)))[1]

        # Project bulk velocities to N in LMN
        R = lmn.R
        vN_i  = (Vi @ R.T)[:, 2] if np.isfinite(Vi).any() else np.full(len(t), np.nan)
        vN_e  = (Ve @ R.T)[:, 2] if np.isfinite(Ve).any() else np.full(len(t), np.nan)
        vN_he = (Vhe @ R.T)[:, 2] if np.isfinite(Vhe).any() else np.full(len(t), np.nan)

        # Ensure aligned lengths for plotting
        min_len = min(len(t), len(B_lmn), len(Ni), len(Ne), len(Nhe), len(vN_i), len(vN_e), len(vN_he))
        if min_len <= 1:
            continue
        t = t[:min_len]
        B_lmn = B_lmn[:min_len]
        Ni = Ni[:min_len]
        Ne = Ne[:min_len]
        Nhe = Nhe[:min_len]
        vN_i = vN_i[:min_len]
        vN_e = vN_e[:min_len]
        vN_he = vN_he[:min_len]

        title = (
            f"MMS{probe} — {center:%Y-%m-%d %H:%M} UT ±{minutes//2} min  "
            f"(LMN: {getattr(lmn, 'method', 'hybrid')})"
        )
        axes = visualize.summary_single(
            t=t,
            B_lmn=B_lmn,
            N_i=Ni,
            N_e=Ne,
            N_he=Nhe,
            vN_i=vN_i,
            vN_e=vN_e,
            vN_he=vN_he,
            title=title,
            show=False if not show else True,
        )

        # Add FPI spectrograms (electrons and ions) if available; prefer brst→fast→srvy
        # FPI burst 4-D flux variables (t,e,phi,theta): mms{p}_des_energyspectr_*_brst, mms{p}_dis_energyspectr_*_brst
        import fnmatch
        # Helper to find an energy axis variable for a base name (e.g., mms1_des)
        def _energy_var_for(base: str):
            for rate in ('brst','fast','srvy'):
                cand = f"{base}_energy_{rate}*"
                for v in fnmatch.filter(data_quants.keys(), cand):
                    g = get_data(v)
                    ee = g[1] if isinstance(g, (list, tuple)) and len(g) > 1 else None
                    if ee is not None and ee.ndim == 1 and len(ee) > 1:
                        return v
            return None

        import fnmatch
        # Prefer omni 2D spectrograms (much lighter), fallback to 4D burst
        def _first_omni_by_rate(base: str):
            for rate in ('brst','fast','srvy'):
                for v in fnmatch.filter(data_quants.keys(), f"{base}_energyspectr_omni_{rate}*"):
                    try:
                        tt, dat = get_data(v)
                        if tt is not None and dat is not None and dat.ndim == 2:
                            return v
                    except Exception:
                        continue
            return None

            # no omni 2D found for this rate
        return None

        # Fallback finder for 4D flux vars (t,e,phi,theta) by rate
        def _first_var_by_rate(base: str):
            for rate in ('brst','fast','srvy'):
                for v in fnmatch.filter(data_quants.keys(), f"{base}_*_{rate}*"):
                    try:
                        g = get_data(v)
                        tt = g[0] if isinstance(g, (list, tuple)) and len(g) > 0 else None
                        dd = g[1] if isinstance(g, (list, tuple)) and len(g) > 1 else None
                        if tt is not None and dd is not None and hasattr(dd, 'ndim') and dd.ndim == 4:
                            return v
                    except Exception:
                        continue
            return None

        # Ensure omni spectrogram variables are loaded (prefer brst→fast→srvy)
        tr = _window(center, minutes)
        _ensure_spectrogram(probe, 'des', ['brst','fast','srvy'], tr)
        _ensure_spectrogram(probe, 'dis', ['brst','fast','srvy'], tr)
        des_omni = _first_omni_by_rate(f"mms{probe}_des")
        dis_omni = _first_omni_by_rate(f"mms{probe}_dis")
        used_rate_e = None; used_rate_i = None
        # If omni is still missing, synthesize from directional components
        if not des_omni:
            t_tmp, omni_tmp, r_tmp = _build_omni_from_directionals(probe, 'des')
            if t_tmp is not None:
                used_rate_e = f'omni(sum dir {r_tmp})'
                t_des = data_loader._tt2000_to_datetime64_ns(t_tmp)
                evar = _energy_var_for(f"mms{probe}_des")
                _, e_des = get_data(evar) if evar else (None, None)
                if e_des is not None:
                    spectra.generic_spectrogram(t_des, e_des, omni_tmp, log10=True, show=False,
                                                ylabel='E$_e$ (eV)', title='Electron energy flux', ax=axes[3])
        if not dis_omni:
            t_tmp, omni_tmp, r_tmp = _build_omni_from_directionals(probe, 'dis')
            if t_tmp is not None:
                used_rate_i = f'omni(sum dir {r_tmp})'
                t_dis = data_loader._tt2000_to_datetime64_ns(t_tmp)
                evar = _energy_var_for(f"mms{probe}_dis")
                _, e_dis = get_data(evar) if evar else (None, None)
                if e_dis is not None:
                    spectra.generic_spectrogram(t_dis, e_dis, omni_tmp, log10=True, show=False,
                                                ylabel='E$_i$ (eV)', title='Ion energy flux', ax=axes[4])



        # Ensure spectrogram panels are visible
        for a in (axes[3], axes[4]):
            a.set_visible(True)
            a.set_frame_on(True)
            a.axis('on')

        if des_omni:
            t_des, omni_e = get_data(des_omni)
            evar = _energy_var_for(f"mms{probe}_des")
            _, e_des = get_data(evar) if evar else (None, None)
            if t_des is not None and e_des is not None:
                used_rate_e = 'omni'
                t_des = data_loader._tt2000_to_datetime64_ns(t_des)
                # Decimate if too many points
                if len(t_des) > 20000:
                    step = int(np.ceil(len(t_des)/20000))
                    t_des = t_des[::step]; omni_e = omni_e[::step]
                spectra.generic_spectrogram(t_des, e_des, omni_e, log10=True, show=False,
                                            ylabel='E$_e$ (eV)', title='Electron energy flux', ax=axes[3])
        else:
            des_var = _first_var_by_rate(f"mms{probe}_des_energyspectr")
            if des_var:
                t_des, flux4d_e = get_data(des_var)
                evar = _energy_var_for(f"mms{probe}_des")
                _, e_des = get_data(evar) if evar else (None, None)
                if t_des is not None and e_des is not None:
                    used_rate_e = '4D'
                    t_des = data_loader._tt2000_to_datetime64_ns(t_des)
                    spectra.fpi_electron_spectrogram(t_des, e_des, flux4d_e, log10=True, show=False, ax=axes[3])

        if dis_omni:
            t_dis, omni_i = get_data(dis_omni)
            evar = _energy_var_for(f"mms{probe}_dis")
            _, e_dis = get_data(evar) if evar else (None, None)

            # Log spectrogram diagnostics for this probe after plotting attempts
            _log_spectrogram_diagnostics(probe, center, minutes)

            if t_dis is not None and e_dis is not None:
                used_rate_i = 'omni'
                t_dis = data_loader._tt2000_to_datetime64_ns(t_dis)
                if len(t_dis) > 20000:
                    step = int(np.ceil(len(t_dis)/20000))
                    t_dis = t_dis[::step]; omni_i = omni_i[::step]
                spectra.generic_spectrogram(t_dis, e_dis, omni_i, log10=True, show=False,
                                            ylabel='E$_i$ (eV)', title='Ion energy flux', ax=axes[4])
        else:
            dis_var = _first_var_by_rate(f"mms{probe}_dis_energyspectr")
            if dis_var:
                t_dis, flux4d_i = get_data(dis_var)
                evar = _energy_var_for(f"mms{probe}_dis")
                _, e_dis = get_data(evar) if evar else (None, None)
                if t_dis is not None and e_dis is not None:
                    used_rate_i = '4D'
                    t_dis = data_loader._tt2000_to_datetime64_ns(t_dis)
                    spectra.fpi_ion_spectrogram(t_dis, e_dis, flux4d_i, log10=True, show=False, ax=axes[4])

        # Optional overlay annotations
        # Label used cadence on y-labels
        if used_rate_e:
            axes[3].set_ylabel(axes[3].get_ylabel() + f" [{used_rate_e}]")
        if used_rate_i:
            axes[4].set_ylabel(axes[4].get_ylabel() + f" [{used_rate_i}]")

        # Interpolate main time series to match spectrogram time grid if available
        # Priority: electron spectrogram time, else ion spectrogram time, else original t
        t_spec = None
        if des_omni:
            t_spec, _ = get_data(des_omni)
        elif dis_omni:
            t_spec, _ = get_data(dis_omni)
        if t_spec is not None:
            t_spec = data_loader._tt2000_to_datetime64_ns(t_spec)
            # Interpolate B_lmn components and scalars to t_spec for better alignment
            try:
                t_float = t.astype('datetime64[ns]').astype('int64').astype(float)
                t_spec_float = t_spec.astype('datetime64[ns]').astype('int64').astype(float)
                def _interp_series(y):
                    y = np.asarray(y)
                    if y.ndim == 1:
                        return np.interp(t_spec_float, t_float, y)
                    else:
                        return np.vstack([np.interp(t_spec_float, t_float, y[:, i]) for i in range(y.shape[1])]).T
                B_lmn_i = _interp_series(B_lmn)
                Ni_i = _interp_series(Ni); Ne_i = _interp_series(Ne); Nhe_i = _interp_series(Nhe)
                vN_i_i = _interp_series(vN_i); vN_e_i = _interp_series(vN_e); vN_he_i = _interp_series(vN_he)
                # Redraw top 3 panels on the spec grid
                axes[0].cla(); axes[1].cla(); axes[2].cla()
                visualize._safe_plot(axes[0], t_spec, B_lmn_i[:,0], label='B$_L$')
                visualize._safe_plot(axes[0], t_spec, B_lmn_i[:,1], label='B$_M$')
                visualize._safe_plot(axes[0], t_spec, B_lmn_i[:,2], label='B$_N$')
                axes[0].set_ylabel('B (nT)'); axes[0].legend(loc='upper right')
                visualize._safe_plot(axes[1], t_spec, Ni_i, label='N$_i$')
                visualize._safe_plot(axes[1], t_spec, Ne_i, label='N$_e$', lw=1.2)
                visualize._safe_plot(axes[1], t_spec, Nhe_i, label='N(He$^+$)', lw=1.2)
                axes[1].set_ylabel('Density (cm$^{-3}$)'); axes[1].legend(loc='upper right')
                visualize._safe_plot(axes[2], t_spec, vN_i_i, label='V$_N$ ions')
                visualize._safe_plot(axes[2], t_spec, vN_e_i, label='V$_N$ e$^-$', lw=1.2)
                visualize._safe_plot(axes[2], t_spec, vN_he_i, label='V$_N$ He$^+$', lw=1.2)
                axes[2].set_ylabel('V$_N$ (km/s)'); axes[2].legend(loc='upper right')
            except Exception:
                pass

        ov = overlay.get(probe) or overlay.get('all')
        if ov and isinstance(ov, dict):
            try:
                n_pub = np.asarray(ov.get('n_hat', []), dtype=float)
                V_pub = ov.get('V_ph_km_s', None)
                txts = []
                if n_pub.size == 3:
                    n_pub = n_pub / (np.linalg.norm(n_pub) + 1e-12)
                    cosang = float(np.dot(n_pub, lmn.N))
                    ang_deg = float(np.degrees(np.arccos(np.clip(abs(cosang), 0, 1))))
                    txts.append(f"n̂·N = {cosang:+.3f} (|Δ|≈{ang_deg:.1f}°)")
                if V_pub is not None:
                    txts.append(f"V_ph (pub) = {float(V_pub):.1f} km/s")
                if txts:
                    axes[0].text(0.01, 0.95, "\n".join(txts), transform=axes[0].transAxes,
                                 va='top', ha='left', fontsize=9,
                                 bbox=dict(boxstyle='round', facecolor='w', alpha=0.6))
            except Exception:
                pass

        # Save figure
        # Save figure PNG and an accessible HTML summary alongside
        fig = axes[0].figure
        fname = output_dir / f"mms{probe}_summary_{timestamp}.png"
        fig.savefig(fname, dpi=300, bbox_inches='tight')

        # Minimal HTML with image and LMN/N metadata for quick inspection
        html_path = output_dir / f"mms{probe}_summary_{timestamp}.html"
        try:
            with open(html_path, 'w', encoding='utf-8') as hf:
                hf.write("<html><head><meta charset='utf-8'><title>MMS {} Summary</title></head><body>".format(probe))
                hf.write(f"<h2>MMS{probe} — {center:%Y-%m-%d %H:%M} UT ±{minutes//2} min</h2>")
                hf.write(f"<p>LMN method: {getattr(lmn, 'method', 'hybrid')}</p>")
                hf.write(f"<p>L: {lmn.L.tolist()}<br>M: {lmn.M.tolist()}<br>N: {lmn.N.tolist()}</p>")
                if ov and isinstance(ov, dict) and ov.get('n_hat') is not None:
                    hf.write(f"<p>Published n̂: {np.asarray(ov['n_hat'], float).tolist()}</p>")
                if ov and isinstance(ov, dict) and ov.get('V_ph_km_s') is not None:
                    hf.write(f"<p>Published V_ph: {float(ov['V_ph_km_s']):.2f} km/s</p>")
                hf.write(f"<img src='{fname.name}' style='max-width:100%;height:auto;' />")
                hf.write("</body></html>")
        except Exception:
            pass

        # Optional CSV export for B_lmn and V_N time series (small subset)
        try:
            import pandas as pd
            df = pd.DataFrame({
                't': t.astype('datetime64[ns]').astype('datetime64[ms]'),
                'B_L': B_lmn[:,0], 'B_M': B_lmn[:,1], 'B_N': B_lmn[:,2],
                'Vn_i': vN_i, 'Vn_e': vN_e, 'Vn_he': vN_he,
            })
            csv_path = output_dir / f"mms{probe}_summary_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
        except Exception:
            pass

        plt.close(fig)
        created.append({
            'probe': probe,
            'png': str(fname),
            'html': str(html_path),
            'csv': str(csv_path) if 'csv_path' in locals() else None,
            'method': getattr(lmn, 'method', 'hybrid'),
        })


def main():
    ap = argparse.ArgumentParser(description="MMS 2019-01-27 12:00 UT visualization")
    ap.add_argument('--center', type=str, default='2019-01-27T12:00:00',
                    help='Center time in ISO format (UTC), e.g., 2019-01-27T12:00:00')
    ap.add_argument('--minutes', type=int, default=60, help='Total window length in minutes')
    ap.add_argument('--probes', nargs='+', default=['1','2','3','4'], help='Probe IDs to plot')
    ap.add_argument('--output-dir', type=str, default='results/visualizations', help='Directory to save figures')
    ap.add_argument('--overlay-json', type=str, default=None, help='Path to JSON with published normal/speed')
    ap.add_argument('--show', action='store_true', help='Also display figures interactively')
    args = ap.parse_args()

    try:
        center = datetime.fromisoformat(args.center.replace('Z',''))
    except Exception as e:
        raise SystemExit(f"Invalid --center: {e}")

    overlay = _load_overlay(args.overlay_json)
    evt = load_event(center, args.minutes, args.probes)
    make_summary_plots(evt,
                       center=center,
                       minutes=args.minutes,
                       output_dir=Path(args.output_dir),
                       overlay=overlay,
                       show=args.show)

    # Write an index.html to navigate outputs
    try:
        idx = Path(args.output_dir) / f"index_{center.strftime('%Y%m%d_%H%M%S')}.html"
        with open(idx, 'w', encoding='utf-8') as f:
            f.write("<html><head><meta charset='utf-8'><title>MMS 2019-01-27 Visualizations</title></head><body>")
            f.write(f"<h2>Visualizations — {center:%Y-%m-%d %H:%M} UT ±{args.minutes//2} min</h2>")
            f.write("<ul>")
            # Re-list files from output_dir
            for probe in ['1','2','3','4']:
                ts = center.strftime('%Y%m%d_%H%M%S')
                html_name = f"mms{probe}_summary_{ts}.html"
                png_name  = f"mms{probe}_summary_{ts}.png"
                csv_name  = f"mms{probe}_summary_{ts}.csv"
                f.write("<li>Probe {}: <a href='{}'>HTML</a> | <a href='{}'>PNG</a> | <a href='{}'>CSV</a></li>".format(
                    probe, html_name, png_name, csv_name
                ))
            f.write("</ul></body></html>")
    except Exception:
        pass


if __name__ == "__main__":
    main()
