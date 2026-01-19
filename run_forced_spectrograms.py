#!/usr/bin/env python3
"""
Run forced spectrogram loads across instruments and export plots.

Included:
- FPI DES/DIS (electrons/ions) with omni/4D fallback (force-loaded)
- HPCA species omni spectra (best-effort forced search)
- FEEPS (energetic electrons/ions) and EIS (energetic particles) best-effort
- Annotated colorbar units, and cadence/source tags on titles

Outputs are saved under results/visualizations/.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
# NumPy 2.x compatibility for third-party libraries (e.g., pytplot/bokeh)
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import matplotlib.pyplot as plt

from pyspedas.projects import mms
from pyspedas import get_data
from pytplot import data_quants

from mms_mp import spectra
from mms_mp.data_loader import (
    force_load_all_plasma_spectrometers,
    _tt2000_to_datetime64_ns,
)


# Debug helpers
import numpy as _np

def _dbg_summarize(name, arr):
    try:
        if arr is None:
            return f"{name}=None"
        shp = getattr(arr, 'shape', None); nd = getattr(arr, 'ndim', None)
        finite = _np.isfinite(arr) if hasattr(arr, '__array__') else None
        vmin = float(_np.nanmin(arr)) if hasattr(arr, '__array__') else None
        vmax = float(_np.nanmax(arr)) if hasattr(arr, '__array__') else None
        nnan = int(_np.size(arr) - _np.sum(finite)) if finite is not None else None
        return f"{name}: shape={shp} ndim={nd} min={vmin} max={vmax} nnan={nnan}"
    except Exception:
        return f"{name}: <err summarizing>"

def _to_1d_energy(e):
    if hasattr(e, 'ndim') and e.ndim == 2:
        return e[0] if e.shape[0] >= e.shape[1] else e[:, 0]
    return e

def _align_energy_to_z(e, z):
    e1 = _to_1d_energy(e)
    if e1 is None or not hasattr(z, 'ndim') or z.ndim != 2:
        return e1, z
    if z.shape[1] == e1.size:
        return e1, z
    if z.shape[0] == e1.size:
        return e1, z.T
    return e1, z

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# Synthesize omni from directional components when omni is absent
from typing import Optional, Tuple

def _build_omni_from_directionals(probe: str, species: str, rates=('brst','fast','srvy')) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    import fnmatch
    comps = ['px','mx','py','my','pz','mz']
    base = f"mms{probe}_{species}_energyspectr"
    for rate in rates:
        series = []
        t_ref = None
        for c in comps:
            for v in fnmatch.filter(data_quants.keys(), f"{base}_{c}_{rate}*"):
                try:
                    tt, dd = get_data(v)
                    if tt is None or dd is None:
                        continue
                    if t_ref is None:
                        t_ref = tt
                    if hasattr(dd, 'ndim') and dd.ndim == 2:
                        series.append(dd)
                except Exception:
                    continue
        if series and t_ref is not None:
            try:
                omni = np.sum(np.stack(series, axis=0), axis=0)
                return t_ref, omni, rate
            except Exception:
                pass
    return None, None, None


# ────────────────────────────────────────────────────────────────────────────
# Best-effort helpers to find 2D (time×energy) spectrogram-like variables
# ────────────────────────────────────────────────────────────────────────────

def _find_energy_var(prefix: str) -> Optional[str]:
    import fnmatch as _fn
    for pat in [f"{prefix}*energy*", f"{prefix}*_eenergy*", f"{prefix}*_logenergy*", f"{prefix}*energy"]:
        for v in _fn.filter(data_quants.keys(), pat):
            try:
                _, arr = get_data(v)
                if arr is not None and np.ndim(arr) == 1 and arr.size > 4:
                    return v
            except Exception:
                continue
    return None


def _find_2d_spectrograms(prefix: str, must_contain: Tuple[str, ...] = ("flux", "omni")) -> list[Tuple[str, Optional[str]]]:
    """Return list of (var, energy_var) pairs under the prefix that look like (t×E) spectrograms."""
    hits: list[Tuple[str, Optional[str]]] = []
    for name in list(data_quants.keys()):
        if not name.startswith(prefix):
            continue
        lname = name.lower()
        if not all(tok in lname for tok in must_contain):
            continue
        try:
            t, d = get_data(name)
            if t is None or d is None:
                continue
            if hasattr(d, 'ndim') and d.ndim == 2 and min(d.shape) >= 8:
                evar = _find_energy_var(prefix.split('*')[0]) or _find_energy_var(prefix)
                hits.append((name, evar))
        except Exception:
            continue
    return hits


def _try_load_hpca(trange: List[str], probe: str) -> dict:
    """Try to load HPCA and find species omni spectrograms (H+, He+, O+, O++)."""
    try:
        # HPCA commonly published in srvy; try burst too in case
        for rate in ['brst', 'srvy']:
            try:
                mms.mms_load_hpca(trange=trange, probe=probe, level='l2', data_rate=rate, time_clip=True)
            except Exception:
                continue
    except Exception:
        pass

    species_tokens = {
        'hplus': 'H+',
        'heplus': 'He+',
        'oplus': 'O+',
        'o2plus': 'O++',
    }
    found: dict = {}
    for tok, label in species_tokens.items():
        prefix = f"mms{probe}_hpca_{tok}"
        candidates = _find_2d_spectrograms(prefix, must_contain=("flux",))
        # Fallback: accept any 2D spectra even without explicit 'flux'
        if not candidates:
            candidates = _find_2d_spectrograms(prefix, must_contain=(tok,))
        if candidates:
            # Choose the first; later we could prefer ones including 'omni'
            found[label] = {
                'var': candidates[0][0],
                'energy_var': candidates[0][1]
            }
    return found


def _try_load_feeps(trange: List[str], probe: str) -> dict:
    try:
        for rate in ['brst', 'srvy']:
            for dt in ['electron', 'ion']:
                try:
                    mms.mms_load_feeps(trange=trange, probe=probe, data_rate=rate, level='l2', datatype=dt, time_clip=True)
                except Exception:
                    continue
    except Exception:
        pass
    out = {}
    # electrons
    e_hits = _find_2d_spectrograms(f"mms{probe}_fe", must_contain=("elec",)) or _find_2d_spectrograms(f"mms{probe}_feeps", must_contain=("elec",))
    if e_hits:
        out['e-'] = {'var': e_hits[0][0], 'energy_var': e_hits[0][1]}
    # ions
    i_hits = _find_2d_spectrograms(f"mms{probe}_fe", must_contain=("ion",)) or _find_2d_spectrograms(f"mms{probe}_feeps", must_contain=("ion",))
    if i_hits:
        out['i+'] = {'var': i_hits[0][0], 'energy_var': i_hits[0][1]}
    return out


def _try_load_eis(trange: List[str], probe: str) -> dict:
    try:
        for rate in ['brst', 'srvy']:
            try:
                mms.mms_load_eis(trange=trange, probe=probe, data_rate=rate, level='l2', time_clip=True)
            except Exception:
                continue
    except Exception:
        pass
    out = {}
    # Try to detect omni flux products
    for target, label in [("elec", "e-"), ("ion", "i+")]:
        hits = _find_2d_spectrograms(f"mms{probe}_eis", must_contain=(target,))
        if hits:
            out[label] = {'var': hits[0][0], 'energy_var': hits[0][1]}
    return out


# ────────────────────────────────────────────────────────────────────────────
# Main runner
# ────────────────────────────────────────────────────────────────────────────

def run(trange: List[str], probes: List[str], *, include_hpca: bool = True, include_eis: bool = True, include_feeps: bool = True) -> dict:
    out_dir = Path('results/visualizations')
    _ensure_dir(out_dir)

    print(f"🔄 Forcing FPI spectrogram loads for {trange} probes={probes}")
    info = force_load_all_plasma_spectrometers(trange, probes=probes, rates=['brst','fast','srvy'], verbose=True)

    for p in probes:
        # Optionally augment with other instruments
        hpca = _try_load_hpca(trange, p) if include_hpca else {}
        eis = _try_load_eis(trange, p) if include_eis else {}
        feps = _try_load_feeps(trange, p) if include_feeps else {}

        stamp = trange[0].replace(':','').replace('/','_') + '_' + trange[1].split('/')[-1].replace(':','')

        extra_rows = len(hpca) + len(eis) + len(feps)
        nrows = 2 + max(0, extra_rows)
        if nrows < 2:
            nrows = 2
        fig, axes = plt.subplots(nrows, 1, figsize=(10, 3 + 2.2*nrows), sharex=True)
        if nrows == 1:
            axes = [axes]
        fig.suptitle(f"MMS{p} spectrograms {trange[0]} to {trange[1]}")
        clabel = 'log10 dJ (keV/(cm^2 s sr keV))'
        # Log available keys and patterns to aid troubleshooting
        print(f"[DEBUG] Available tplot vars: {len(list(data_quants.keys()))}")
        for k in list(data_quants.keys())[:10]:
            print(f"   [DEBUG] var: {k}")


        # DES (electrons)
        des = info[p]['des']
        used_e = des.get('used_rate') or '?'
        src_e = des.get('source') or 'unknown'
        ax0 = axes[0]
        plotted_des = False
        try:
            if des.get('omni_var'):
                print(f"[DEBUG] DES omni_var: {des['omni_var']} energy_var: {des.get('energy_var')}")
                t, z = get_data(des['omni_var']); _, e = get_data(des.get('energy_var')) if des.get('energy_var') else (None, None)
                if t is not None and e is not None and z is not None:
                    print("[DEBUG] ", _dbg_summarize('DES t', t))
                    print("[DEBUG] ", _dbg_summarize('DES e', e))
                    print("[DEBUG] ", _dbg_summarize('DES z', z))
                    t = _tt2000_to_datetime64_ns(t)
                    e1, z1 = _align_energy_to_z(e, z)
                    print("[DEBUG] ", _dbg_summarize('DES e1(aligned)', e1))
                    print("[DEBUG] ", _dbg_summarize('DES z1(aligned)', z1))
                    spectra.generic_spectrogram(t, e1, z1, log10=True, ax=ax0, show=False,
                                                ylabel='E$_e$ (eV)', title=f'Electron energy flux [{src_e}/{used_e}]', clabel=clabel)
                    plotted_des = True
            elif des.get('flux4d_var'):
                print(f"[DEBUG] DES flux4d_var: {des['flux4d_var']} energy_var: {des.get('energy_var')}")
                t, f4 = get_data(des['flux4d_var']); _, e = get_data(des.get('energy_var')) if des.get('energy_var') else (None, None)
                if t is not None and e is not None and f4 is not None:
                    print("[DEBUG] ", _dbg_summarize('DES t', t))
                    print("[DEBUG] ", _dbg_summarize('DES e', e))
                    print("[DEBUG] ", _dbg_summarize('DES f4', f4))
                    t = _tt2000_to_datetime64_ns(t)
                    e = _to_1d_energy(e)
                    spectra.fpi_electron_spectrogram(t, e, f4, log10=True, ax=ax0, show=False,
                                                     title=f'Electron energy flux [4D/{used_e}]', clabel=clabel)
                    plotted_des = True
            elif des.get('flux3d_var'):
                print(f"[DEBUG] DES flux3d_var: {des['flux3d_var']} energy_var: {des.get('energy_var')}")
                t, f3 = get_data(des['flux3d_var']); _, e = get_data(des.get('energy_var')) if des.get('energy_var') else (None, None)
                if t is not None and e is not None and f3 is not None:
                    print("[DEBUG] ", _dbg_summarize('DES t', t))
                    print("[DEBUG] ", _dbg_summarize('DES e', e))
                    print("[DEBUG] ", _dbg_summarize('DES f3', f3))
                    # collapse pitch dimension by sum/mean depending on axis order
                    if hasattr(f3, 'ndim') and f3.ndim == 3:
                        if f3.shape[1] == e.size:  # (t, E, angle)
                            dat2d = np.nansum(f3, axis=2)
                        elif f3.shape[2] == e.size:  # (t, angle, E)
                            dat2d = np.nansum(f3, axis=1)
                        else:
                            dat2d = np.nansum(f3, axis=2)
                        t = _tt2000_to_datetime64_ns(t)
                        e = _to_1d_energy(e)
                        # Align energy index with dat2d axis
                        e1, z1 = _align_energy_to_z(e, dat2d)
                        print("[DEBUG] ", _dbg_summarize('DES e1(aligned)', e1))
                        print("[DEBUG] ", _dbg_summarize('DES z1(aligned)', z1))
                        spectra.generic_spectrogram(t, e1, z1, log10=True, ax=ax0, show=False,
                                                    ylabel='E$_e$ (eV)', title=f'Electron energy flux [3D pitch integ/{used_e}]', clabel=clabel)
                        plotted_des = True
            if not plotted_des:
                t_raw, omni, r = _build_omni_from_directionals(p, 'des')
                if t_raw is not None and omni is not None:
                    evar = _find_energy_var(f"mms{p}_des")
                    _, e = get_data(evar) if evar else (None, None)
                    if e is not None:
                        t = _tt2000_to_datetime64_ns(t_raw)
                        e1, z1 = _align_energy_to_z(e, omni)
                        print("[DEBUG] DES omni-from-dir:", _dbg_summarize('t', t), _dbg_summarize('e1', e1), _dbg_summarize('z1', z1))
                        spectra.generic_spectrogram(t, e1, z1, log10=True, ax=ax0, show=False,
                                                    ylabel='E$_e$ (eV)', title=f'Electron energy flux [omni(sum dir {r})]', clabel=clabel)
                        plotted_des = True
            # If plotted, overlay an inset text with min/max to ensure not flat
            if plotted_des:
                try:
                    _, testZ = get_data(des.get('omni_var') or des.get('flux4d_var') or des.get('flux3d_var'))
                    if testZ is not None:
                        zmin, zmax = np.nanmin(testZ), np.nanmax(testZ)
                        ax0.text(0.02, 0.9, f"z:[{zmin:.2g},{zmax:.2g}]", transform=ax0.transAxes, fontsize=8)
                except Exception:
                    pass
            # If plotted, overlay an inset text with min/max to ensure not flat
            if plotted_des:
                try:
                    _, testZ = get_data(des.get('omni_var') or des.get('flux4d_var') or des.get('flux3d_var'))
                    if testZ is not None:
                        zmin, zmax = np.nanmin(testZ), np.nanmax(testZ)
                        ax0.text(0.02, 0.9, f"z:[{zmin:.2g},{zmax:.2g}]", transform=ax0.transAxes, fontsize=8)
                except Exception:
                    pass

            if not plotted_des:
                ax0.text(0.5, 0.5, 'No DES spectrogram available', ha='center', va='center', transform=ax0.transAxes)
        except Exception as exc:
            ax0.text(0.5, 0.5, f'DES error: {exc}', ha='center', va='center', transform=ax0.transAxes)

        # DIS (ions)
        dis = info[p]['dis']
        # Dump DES/DIS variable choices
        print(f"[DEBUG] DES selection: {des}")
        print(f"[DEBUG] DIS selection: {dis}")
        used_i = dis.get('used_rate') or '?'
        src_i = dis.get('source') or 'unknown'
        ax1 = axes[1]
        plotted_dis = False
        try:
            if dis.get('omni_var'):
                print(f"[DEBUG] DIS omni_var: {dis['omni_var']} energy_var: {dis.get('energy_var')}")
                t, z = get_data(dis['omni_var']); _, e = get_data(dis.get('energy_var')) if dis.get('energy_var') else (None, None)
                if t is not None and e is not None and z is not None:
                    print("[DEBUG] ", _dbg_summarize('DIS t', t))
                    print("[DEBUG] ", _dbg_summarize('DIS e', e))
                    print("[DEBUG] ", _dbg_summarize('DIS z', z))
                    t = _tt2000_to_datetime64_ns(t)
                    e1, z1 = _align_energy_to_z(e, z)
                    print("[DEBUG] ", _dbg_summarize('DIS e1(aligned)', e1))
                    print("[DEBUG] ", _dbg_summarize('DIS z1(aligned)', z1))
                    spectra.generic_spectrogram(t, e1, z1, log10=True, ax=ax1, show=False,
                                                ylabel='E$_i$ (eV)', title=f'Ion energy flux [{src_i}/{used_i}]', clabel=clabel)
                    plotted_dis = True
            elif dis.get('flux4d_var'):
                print(f"[DEBUG] DIS flux4d_var: {dis['flux4d_var']} energy_var: {dis.get('energy_var')}")
                t, f4 = get_data(dis['flux4d_var']); _, e = get_data(dis.get('energy_var')) if dis.get('energy_var') else (None, None)
                if t is not None and e is not None and f4 is not None:
                    print("[DEBUG] ", _dbg_summarize('DIS t', t))
                    print("[DEBUG] ", _dbg_summarize('DIS e', e))
                    print("[DEBUG] ", _dbg_summarize('DIS f4', f4))
                    t = _tt2000_to_datetime64_ns(t)
                    e = _to_1d_energy(e)
                    spectra.fpi_ion_spectrogram(t, e, f4, log10=True, ax=ax1, show=False,
                                                title=f'Ion energy flux [4D/{used_i}]', clabel=clabel)
                    plotted_dis = True
            elif dis.get('flux3d_var'):
                print(f"[DEBUG] DIS flux3d_var: {dis['flux3d_var']} energy_var: {dis.get('energy_var')}")
                t, f3 = get_data(dis['flux3d_var']); _, e = get_data(dis.get('energy_var')) if dis.get('energy_var') else (None, None)
                if t is not None and e is not None and f3 is not None:
                    print("[DEBUG] ", _dbg_summarize('DIS t', t))
                    print("[DEBUG] ", _dbg_summarize('DIS e', e))
                    print("[DEBUG] ", _dbg_summarize('DIS f3', f3))
                    if hasattr(f3, 'ndim') and f3.ndim == 3:
                        if f3.shape[1] == e.size:
                            dat2d = np.nansum(f3, axis=2)
                        elif f3.shape[2] == e.size:
                            dat2d = np.nansum(f3, axis=1)
                        else:
                            dat2d = np.nansum(f3, axis=2)
                        t = _tt2000_to_datetime64_ns(t)
                        e = _to_1d_energy(e)
                        e1, z1 = _align_energy_to_z(e, dat2d)
                        print("[DEBUG] ", _dbg_summarize('DIS e1(aligned)', e1))
                        print("[DEBUG] ", _dbg_summarize('DIS z1(aligned)', z1))
                        spectra.generic_spectrogram(t, e1, z1, log10=True, ax=ax1, show=False,
                                                    ylabel='E$_i$ (eV)', title=f'Ion energy flux [3D pitch integ/{used_i}]', clabel=clabel)
                        plotted_dis = True
            # If plotted, annotate
            if plotted_dis:
                try:
                    _, testZ = get_data(dis.get('omni_var') or dis.get('flux4d_var') or dis.get('flux3d_var'))
                    if testZ is not None:
                        zmin, zmax = np.nanmin(testZ), np.nanmax(testZ)
                        ax1.text(0.02, 0.9, f"z:[{zmin:.2g},{zmax:.2g}]", transform=ax1.transAxes, fontsize=8)
                except Exception:
                    pass

            if not plotted_dis:
                t_raw, omni, r = _build_omni_from_directionals(p, 'dis')
                if t_raw is not None and omni is not None:
                    evar = _find_energy_var(f"mms{p}_dis")
                    _, e = get_data(evar) if evar else (None, None)
                    if e is not None:
                        t = _tt2000_to_datetime64_ns(t_raw)
                        spectra.generic_spectrogram(t, e, omni, log10=True, ax=ax1, show=False,
                                                    ylabel='E$_i$ (eV)', title=f'Ion energy flux [omni(sum dir {r})]', clabel=clabel)
                        plotted_dis = True
            if not plotted_dis:
                ax1.text(0.5, 0.5, 'No DIS spectrogram available', ha='center', va='center', transform=ax1.transAxes)
        except Exception as exc:
            ax1.text(0.5, 0.5, f'DIS error: {exc}', ha='center', va='center', transform=ax1.transAxes)

        # Additional rows index start
        row = 2
        # HPCA species
        for label, spec in hpca.items():
            try:
                t, z = get_data(spec['var']); evar = spec.get('energy_var'); _, e = get_data(evar) if evar else (None, None)
                if t is not None and z is not None and e is not None and row < len(axes):
                    t = _tt2000_to_datetime64_ns(t)
                    spectra.generic_spectrogram(t, e, z, log10=True, ax=axes[row], show=False,
                                                ylabel=f'E({label}) (eV)', title=f'HPCA {label} energy flux', clabel=clabel)
                    row += 1
            except Exception:
                continue
        # EIS
        for label, spec in eis.items():
            try:
                t, z = get_data(spec['var']); evar = spec.get('energy_var'); _, e = get_data(evar) if evar else (None, None)
                if t is not None and z is not None and e is not None and row < len(axes):
                    t = _tt2000_to_datetime64_ns(t)
                    spectra.generic_spectrogram(t, e, z, log10=True, ax=axes[row], show=False,
                                                ylabel=f'E({label}) (keV)', title=f'EIS {label} flux', clabel=clabel)
                    row += 1
            except Exception:
                continue
        # FEEPS
        for label, spec in feps.items():
            try:
                t, z = get_data(spec['var']); evar = spec.get('energy_var'); _, e = get_data(evar) if evar else (None, None)
                if t is not None and z is not None and e is not None and row < len(axes):
                    t = _tt2000_to_datetime64_ns(t)
                    spectra.generic_spectrogram(t, e, z, log10=True, ax=axes[row], show=False,
                                                ylabel=f'E({label}) (keV)', title=f'FEEPS {label} flux', clabel=clabel)
                    row += 1
            except Exception:
                continue

        fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        outpng = out_dir / f"forced_spectrograms_MMS{p}_{stamp}.png"
        fig.savefig(outpng, dpi=150)
        plt.close(fig)
        print(f"✅ Saved {outpng}")

    return info


if __name__ == '__main__':
    # Default event window used in the repository examples
    trange = ['2019-01-27/12:25:00', '2019-01-27/12:35:00']
    probes = ['1', '2', '3', '4']
    run(trange, probes, include_hpca=True, include_eis=True, include_feeps=True)

