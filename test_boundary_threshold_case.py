#!/usr/bin/env python3
"""
Boundary Threshold Test Case (≥0.4)
===================================

This test loads MMS data and performs the following, producing publication-ready plots:
- Compute a composite boundary score and flag all ≥0.4 crossings
- Estimate inter-spacecraft delays and boundary-normal speed (Vn)
- Determine boundary normal via MVA and project spacecraft distances (thickness)
- Compute MMS3 magnetic shear angles across specified sheath/sphere times

Event: 2019-01-27 (magnetopause)
Times of interest:
- Magnetosheath: 12:18, 12:30:50, 12:45
- Magnetosphere: 12:22, 12:33, 12:40

Outputs:
- boundary_threshold_overview.png
- boundary_timing_speed_thickness.png
- mms3_shear_angles.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import pandas as pd
from datetime import datetime, timedelta, timezone
import matplotlib.colors as mcolors

from pyspedas.projects import mms
from pytplot import get_data, data_quants
import os
from matplotlib.backends.backend_pdf import PdfPages

from publication_boundary_analysis import (
    load_mec_data_first,
    load_comprehensive_science_data,
    calculate_boundary_normal,
    transform_to_lmn,
    get_event_positions,
    safe_format_time_axis,
    ensure_datetime_format,
)

import warnings
warnings.filterwarnings('ignore')

# ---------- Config (energy caps for low-E focus, in eV) ----------
ION_LOW_E_MAX = 5000.0
ELECTRON_LOW_E_MAX = 3000.0

# ---------- Utility functions ----------

def composite_boundary_score(times_b, b_xyz, times_n, n_i, lmn_matrix=None):
    """Compute composite boundary score in [0,1] from |B| and Ni changes.
    - Smooth, gradient, robust normalization using percentile scaling
    - Optionally include Bn (LMN) if lmn_matrix provided (not required)
    Returns (times, score)
    """
    # Base time is B-field time
    if hasattr(times_b[0], 'timestamp'):
        t_b = np.array([t.timestamp() for t in times_b])
    else:
        t_b = np.asarray(times_b)

    # Interpolate density to B time base
    if hasattr(times_n[0], 'timestamp'):
        t_n = np.array([t.timestamp() for t in times_n])
    else:
        t_n = np.asarray(times_n)

    n_interp = np.interp(t_b, t_n, n_i)

    # |B|
    b_mag = np.sqrt(np.sum(b_xyz[:, :3]**2, axis=1))

    # Smooth and gradient
    def moving_average(x, window=51):
        w = max(3, min(window, len(x)))
        if w % 2 == 0:
            w -= 1
        if w < 3:
            return x
        kernel = np.ones(w) / w
        return np.convolve(x, kernel, mode='same')

    b_s = moving_average(b_mag, 51)
    n_s = moving_average(n_interp, 51)

    grad_b = np.abs(np.gradient(b_s))
    grad_n = np.abs(np.gradient(n_s))

    # Optional Bn contribution
    if lmn_matrix is not None:
        b_lmn = np.dot(b_xyz[:, :3], lmn_matrix)  # (N,3)
        bn = b_lmn[:, 2]
        bn_s = moving_average(bn, 51)
        grad_bn = np.abs(np.gradient(bn_s))
    else:
        grad_bn = np.zeros_like(grad_b)

    # Robust percentile scaling to [0,1] per channel, then average
    def scale01(x):
        p95 = np.percentile(x, 95) if np.any(np.isfinite(x)) else 1.0
        p05 = np.percentile(x, 5) if np.any(np.isfinite(x)) else 0.0
        denom = max(p95 - p05, 1e-6)
        y = (x - p05) / denom
        return np.clip(y, 0.0, 1.0)

    s_b = scale01(grad_b)
    s_n = scale01(grad_n)
    s_bn = scale01(grad_bn)

    score = np.clip(0.5 * s_b + 0.4 * s_n + 0.1 * s_bn, 0.0, 1.0)
    return times_b, score

def detect_crossings(times, score, threshold=0.4, min_separation_s=10):
    """Return list of crossing times where score >= threshold with min separation."""
    if hasattr(times[0], 'timestamp'):
        t = np.array([ti.timestamp() for ti in times])
    else:
        t = np.asarray(times)

    above = score >= threshold
    if not np.any(above):
        return []

    idx = np.where(above)[0]
    crossings = []
    last_t = -np.inf
    for i in idx:
        if t[i] - last_t >= min_separation_s:
            crossings.append(datetime.fromtimestamp(t[i], tz=timezone.utc))
            last_t = t[i]
    return crossings

def best_delay_by_xcorr(t_ref, s_ref, t_cmp, s_cmp, max_lag_s=120):
    """Estimate delay (cmp vs ref) via cross-correlation (s in [0,1]) within max_lag_s."""
    # Resample both onto a common uniform grid for robust xcorr
    t0 = max(t_ref[0], t_cmp[0])
    t1 = min(t_ref[-1], t_cmp[-1])
    if t1 - t0 < 10:
        return 0.0  # insufficient overlap

    fs = 5.0  # 5 Hz resample for xcorr
    grid = np.arange(t0, t1, 1.0 / fs)

    s_ref_i = np.interp(grid, t_ref, s_ref)
    s_cmp_i = np.interp(grid, t_cmp, s_cmp)

    # Detrend and normalize
    def z(x):
        x = x - np.nanmean(x)
        sd = np.nanstd(x) or 1.0
        return x / sd
    a = z(s_ref_i)
    b = z(s_cmp_i)

    # Limit search window to +/- max_lag_s
    max_lag = int(max_lag_s * fs)

    # Use NumPy correlation (SciPy-free)
    corr = np.correlate(b, a, mode='full')
    # For 1D correlation, lags range from -(N-1) to +(N-1) when len(a)==len(b)==N
    N = max(len(a), len(b))
    lags = np.arange(-len(a)+1, len(b))

    center = len(corr) // 2
    window = slice(max(center - max_lag, 0), min(center + max_lag + 1, len(corr)))
    corr_w = corr[window]
    lags_w = lags[window]

    best = int(lags_w[np.argmax(corr_w)])
    delay_s = best / fs  # cmp must be shifted by delay_s to align to ref
    return delay_s

def estimate_normal_speed(delays_s, positions_km, normal_vec, ref='1'):
    """Estimate Vn (km/s) by least squares fit: n·(ri - r_ref) = Vn * Δti."""
    n = normal_vec / (np.linalg.norm(normal_vec) or 1.0)
    r_ref = positions_km[ref]

    di = []
    dti = []
    for p, dt in delays_s.items():
        if p == ref:
            continue
        di.append(np.dot(n, positions_km[p] - r_ref))
        dti.append(dt)
    if len(dti) == 0 or np.sum(np.square(dti)) == 0:
        return np.nan
    di = np.array(di)
    dti = np.array(dti)
    vn = float(np.sum(di * dti) / np.sum(dti * dti))  # km/s
    return vn

def mean_vector_in_window(times, vec, center_dt, half_width_s=30):
    """Mean vector around a center time ± half_width_s.
    center_dt should be timezone-aware UTC; if naïve, assume UTC.
    """
    if hasattr(center_dt, 'tzinfo') and center_dt.tzinfo is None:
        center_dt = center_dt.replace(tzinfo=timezone.utc)
    if hasattr(times[0], 'timestamp'):
        # Ensure times are UTC-aware
        tt = [ti if getattr(ti, 'tzinfo', None) is not None else ti.replace(tzinfo=timezone.utc) for ti in times]
        t = np.array([ti.timestamp() for ti in tt])
    else:
        t = np.asarray(times)
    c = center_dt.timestamp()
    m = (t >= c - half_width_s) & (t <= c + half_width_s)
    if not np.any(m):
        return None
    return np.nanmean(vec[m, :3], axis=0)

# ---------- Main test ----------

def main():
    print("🧪 Boundary Threshold Test (≥0.4)")
    print("=" * 40)

    # CLI/env overrides for energy caps and time range
    import argparse, os
    parser = argparse.ArgumentParser(description='MMS boundary and spectrogram analysis')
    parser.add_argument('--start', default='2019-01-27/12:15:00')
    parser.add_argument('--end', default='2019-01-27/12:50:00')
    parser.add_argument('--ion-emax', type=float, default=float(os.getenv('ION_LOW_E_MAX', ION_LOW_E_MAX)))
    parser.add_argument('--electron-emax', type=float, default=float(os.getenv('ELECTRON_LOW_E_MAX', ELECTRON_LOW_E_MAX)))
    # PDF inclusion controls
    parser.add_argument('--include-grids', dest='include_grids', action='store_true')
    parser.add_argument('--no-grids', dest='include_grids', action='store_false')
    parser.add_argument('--include-combined', dest='include_combined', action='store_true')
    parser.add_argument('--no-combined', dest='include_combined', action='store_false')
    parser.set_defaults(include_grids=True, include_combined=True)
    # Diagnostics controls
    parser.add_argument('--diagnostics-only-pdf', dest='diagnostics_only_pdf', action='store_true', help='Export a standalone diagnostics PDF')
    parser.add_argument('--diag-filter-species', default='', help='Filter diagnostics by species (e.g., dis or des)')
    parser.add_argument('--diag-filter-probe', default='', help='Filter diagnostics by probe number (1..4)')
    parser.add_argument('--diag-filter-status', default='', help='Filter diagnostics by status (e.g., COVERAGE)')
    parser.add_argument('--diag-summarize', dest='diag_summarize', action='store_true', help='Print diagnostics summary to stdout after generation')
    args = parser.parse_args()

    # Analysis window
    trange = [args.start, args.end]
    event_dt = datetime(2019, 1, 27, 12, 30, 50, tzinfo=timezone.utc)

    # Override module-level caps from CLI if provided
    globals()['ION_LOW_E_MAX'] = args.ion_emax
    globals()['ELECTRON_LOW_E_MAX'] = args.electron_emax

    # Load data
    positions, _ = load_mec_data_first(trange, ['1', '2', '3', '4'])
    data = load_comprehensive_science_data(trange, ['1', '2', '3', '4'])

    # Boundary normal via MVA (MMS1)
    lmn_matrix, mva_results = calculate_boundary_normal(data, event_dt)

    # Transform to LMN for diagnostics (optional)
    if lmn_matrix is not None:
        data = transform_to_lmn(data, lmn_matrix)

    # Build composite boundary scores for each probe
    scores = {}
    btimes = {}
    for probe in ['1', '2', '3', '4']:
        if probe in data and 'B_gsm' in data[probe] and 'N_i' in data[probe]:
            tb, b_xyz = data[probe]['B_gsm']
            tn, n_i = data[probe]['N_i']
            t_score, score = composite_boundary_score(tb, b_xyz, tn, n_i, lmn_matrix)
            scores[probe] = score
            if hasattr(tb[0], 'timestamp'):
                btimes[probe] = np.array([t.timestamp() for t in t_score])
            else:
                btimes[probe] = np.asarray(t_score)

    # Detect threshold crossings
    threshold = 0.4
    crossings = {}
    for p in scores:
        ct = detect_crossings(btimes[p], scores[p], threshold=threshold, min_separation_s=10)
        crossings[p] = ct

    # Inter-spacecraft delays via cross-correlation around the event
    delays = {'1': 0.0}
    if '1' in scores:
        t_ref = btimes['1']
        s_ref = scores['1']
        for p in ['2', '3', '4']:
            if p in scores:
                delay = best_delay_by_xcorr(t_ref, s_ref, btimes[p], scores[p], max_lag_s=180)
                delays[p] = delay

    # Positions and normal speed estimate
    event_positions = get_event_positions(positions, event_dt)
    vn = np.nan
    if lmn_matrix is not None and len(event_positions) == 4:
        vn = estimate_normal_speed(delays, event_positions, mva_results['normal'] if mva_results else lmn_matrix[:, 2], ref='1')

    # Estimate crossing duration at MMS1 and thickness
    thickness_km = np.nan
    duration_s = np.nan
    if '1' in scores:
        s1 = scores['1']
        t1 = btimes['1']
        # find the main crossing region around event time where score >= threshold
        mask = (t1 >= event_dt.timestamp() - 180) & (t1 <= event_dt.timestamp() + 180)
        idx = np.where(mask & (s1 >= threshold))[0]
        if len(idx) > 0:
            duration_s = t1[idx[-1]] - t1[idx[0]]
            if np.isfinite(vn):
                thickness_km = abs(vn) * duration_s

    # ---------- Plot 1: Boundary score and thresholds ----------
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle('Boundary Score (≥0.4) and Crossings\n2019-01-27', fontsize=16, fontweight='bold')

    colors = {'1': '#1f77b4', '2': '#ff7f0e', '3': '#2ca02c', '4': '#d62728'}

    # Scores over time for all spacecraft
    for p in ['1', '2', '3', '4']:
        if p in scores:
            t_plot = ensure_datetime_format([datetime.fromtimestamp(tt, tz=timezone.utc) for tt in btimes[p]])
            axes[0].plot(t_plot, scores[p], color=colors[p], label=f'MMS{p}', linewidth=1.2)
    axes[0].axhline(threshold, color='red', linestyle='--', linewidth=1.5, label='Threshold 0.4')
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Boundary Score', fontweight='bold')
    axes[0].legend(ncol=5)

    # Mark detected crossings per spacecraft
    for p in ['1', '2', '3', '4']:
        if p in crossings:
            for ct in crossings[p]:
                axes[1].axvline(ct, color=colors[p], linestyle='--', alpha=0.7)
    axes[1].set_ylabel('Crossing markers', fontweight='bold')

    # Highlight event time
    for ax in axes:
        ax.axvline(event_dt, color='black', linestyle='-', linewidth=2, alpha=0.6)
    axes[2].set_ylabel(' ')
    axes[2].set_xlabel('Time (UT)', fontweight='bold')

    for ax in axes:
        safe_format_time_axis(ax, interval_minutes=2)

    plt.tight_layout()
    plt.savefig('boundary_threshold_overview.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # ---------- Plot 2: Timing, Vn, and thickness ----------
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('Timing and Boundary-normal Speed', fontsize=16, fontweight='bold')

    delay_text = 'Inter-spacecraft delays (s) relative to MMS1:\n'
    for p in ['1', '2', '3', '4']:
        if p in delays:
            delay_text += f"  MMS{p}: {delays[p]:+.2f} s\n"
    delay_text += f"\nEstimated Vn: {vn:.2f} km/s\n"
    if np.isfinite(duration_s):
        delay_text += f"Crossing duration (MMS1): {duration_s:.1f} s\n"
    if np.isfinite(thickness_km):
        delay_text += f"Estimated thickness: {thickness_km:.0f} km"

    ax.axis('off')
    ax.text(0.05, 0.95, delay_text, transform=ax.transAxes, fontsize=12,
            va='top', fontfamily='monospace', bbox=dict(boxstyle='round', fc='lavender', ec='gray'))

    # Plot formation with normal arrow
    if len(event_positions) == 4 and mva_results is not None:
        inset = fig.add_axes([0.58, 0.15, 0.35, 0.35])
        for p in ['1', '2', '3', '4']:
            r = event_positions[p]
            inset.scatter(r[0], r[1], s=80, color=colors[p], edgecolors='k', zorder=3)
            inset.text(r[0], r[1], f'  {p}', fontsize=10, weight='bold')
        # Normal vector projected into XY-GSM plane
        n = mva_results['normal']
        center = np.mean(np.vstack([event_positions[k] for k in ['1','2','3','4']]), axis=0)
        scale = 200.0
        inset.arrow(center[0], center[1], scale*n[0], scale*n[1], width=10.0, color='red', zorder=2)
        inset.set_title('Formation (XY-GSM) with Normal')
        inset.set_aspect('equal', adjustable='box')
        inset.grid(True, alpha=0.3)

    plt.savefig('boundary_timing_speed_thickness.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # ---------- Plot 3: FPI spectrograms (ion & electron if available) ----------
    def find_var(keys, probe, species, must_include=None, also_include=None):
        """Find a tplot variable name in keys for a given probe/species with substrings.
        Returns the first best match or None.
        """
        must_include = must_include or []
        also_include = also_include or []
        candidates = []
        prefix = f'mms{probe}_{species}_'
        for k in keys:
            if not k.startswith(prefix):
                continue
            ok = all(s in k for s in must_include)
            if ok and all(s in k for s in also_include):
                candidates.append(k)
        # fallback: relax also_include
        if not candidates and also_include:
            for k in keys:
                if not k.startswith(prefix):
                    continue
                if all(s in k for s in must_include):
                    candidates.append(k)
        return candidates[0] if candidates else None

    # For diagnostics
    spectro_diag_rows = []

    def record_var_coverage(varname, species, probe, status_tag):
        try:
            res = get_data(varname)
            if isinstance(res, tuple):
                t = res[0]
            elif isinstance(res, dict) and 'x' in res:
                t = res['x']
            else:
                t = None
            if t is not None and len(t) > 0:
                if hasattr(t[0], 'timestamp'):
                    times = [ti if getattr(ti, 'tzinfo', None) is not None else ti.replace(tzinfo=timezone.utc) for ti in t]
                else:
                    times = [datetime.fromtimestamp(tt, tz=timezone.utc) for tt in t]
                t_start = times[0].strftime('%Y-%m-%d %H:%M:%S')
                t_end = times[-1].strftime('%Y-%m-%d %H:%M:%S')
                spectro_diag_rows.append([species, probe, varname, '', '', status_tag, t_start, t_end, len(times)])
            else:
                spectro_diag_rows.append([species, probe, varname, '', '', status_tag, '', '', 0])
        except Exception:
            spectro_diag_rows.append([species, probe, varname, '', '', status_tag, '', '', ''])

    def _match_confidence(name: str, status: str) -> str:
        if not name:
            return ''
        n = name.lower()
        if 'energyspectr' in n and 'omni' in n and 'brst' in n:
            return 'high'
        if 'energyspectr' in n and ('omni' in n or 'brst' in n):
            return 'medium'
        if any(k in n for k in ['dist', 'distribution', 'energy']):
            return 'medium'
        # Otherwise low
        return 'low'

    def extract_spectrogram(varname, probe=None, species=None):
        # Try exact var, else search alternates if probe/species provided
        name = varname
        tried = [varname]
        if name not in data_quants and probe and species:
            # Try searching for other omni spectrogram names
            keys = list(data_quants.keys())
            # energyspectr_omni can vary: 'energyspectr_omni_brst', 'energyspectr_brst_omni', etc.
            cand = find_var(keys, probe, species,
                            must_include=['energyspectr'], also_include=['omni','brst'])
            if cand is None:
                cand = find_var(keys, probe, species, must_include=['energyspectr'])
            if cand is not None:
                name = cand
                tried.append(cand)
                record_var_coverage(name, species, probe, 'CANDIDATE_COVERAGE')
        if name not in data_quants:
            spectro_diag_rows.append([species, probe, ';'.join(tried), '', '', 'NOT_FOUND', '', '', ''])
            return None
        # Record coverage for chosen omni spectrogram
        record_var_coverage(name, species, probe, 'OMNI_COVERAGE')
        res = get_data(name)
        # Handle multiple return formats from pytplot.get_data
        if isinstance(res, tuple):
            if len(res) == 2:
                t, y = res
                v = None
            elif len(res) == 3:
                t, y, v = res
            else:
                t, y = res[0], res[1]
                v = res[2] if len(res) > 2 else None
        elif isinstance(res, dict):
            t = res.get('x') or res.get('times')
            y = res.get('y') or res.get('data')
            v = res.get('v') or res.get('energy')
        else:
            return None
        if t is None or y is None:
            return None
        times = ensure_datetime_format(t)
        data2d = np.array(y)
        # Ensure shape (N_time, N_energy)
        if data2d.ndim == 1:
            data2d = data2d[:, None]
        if data2d.shape[0] != len(times):
            # Try transpose
            if data2d.T.shape[0] == len(times):
                data2d = data2d.T
            else:
                return None
        # Determine energy bins
        if v is None:
            energy = np.arange(data2d.shape[1])
            spectro_diag_rows.append([species, probe, name, '', f'imputed:{len(energy)}bins', 'OMNI_NO_ENERGY_VECTOR'])
        else:
            energy = np.array(v)
            # If v is 2D (Nt, Ne), take first row or median across time
            if energy.ndim == 2:
                if energy.shape[0] == len(times) and energy.shape[1] == data2d.shape[1]:
                    # Use first time's energy bins (assumed constant)
                    energy = energy[0]
                elif energy.shape[0] == data2d.shape[1] and energy.shape[1] == len(times):
                    # Possibly transposed
                    energy = energy[:, 0]
                else:
                    # Fallback to median across axis 0 if possible
                    try:
                        energy = np.nanmedian(energy, axis=0)
                    except Exception:
                        energy = np.arange(data2d.shape[1])
            elif energy.ndim != 1 or energy.size != data2d.shape[1]:
                energy = np.arange(data2d.shape[1])
        # If energy is descending, reverse both axes
        if energy.ndim == 1 and energy.size == data2d.shape[1]:
            if energy[0] > energy[-1]:
                energy = energy[::-1]
                data2d = data2d[:, ::-1]
        return times, energy, data2d

    def build_spectrogram_from_dist(species: str, probe: str = '1'):
        """Fallback: build time×energy spectrogram by averaging the distribution over angles.
        Accepts multiple variable name variants.
        Returns (times_datetime, energy_1d, spec2d) or None.
        """
        # Try known dist var names
        dist_names = [
            f'mms{probe}_{species}_dist_brst',
            f'mms{probe}_{species}_dist',
            f'mms{probe}_{species}_distribution_brst',
            f'mms{probe}_{species}_dist_omni_brst',
        ]
        en_names = [
            f'mms{probe}_{species}_energy_brst',
            f'mms{probe}_{species}_energy',
            f'mms{probe}_{species}_en_brst',
        ]
        dist_var = next((n for n in dist_names if n in data_quants), None)
        en_var = next((n for n in en_names if n in data_quants), None)
        # Try search if not found
        if dist_var is None:
            dist_var = find_var(list(data_quants.keys()), probe, species, must_include=['dist'])
        if en_var is None:
            en_var = find_var(list(data_quants.keys()), probe, species, must_include=['energy'])
        if dist_var is None or en_var is None:
            spectro_diag_rows.append([species, probe, dist_var or '', en_var or '', '', 'MISSING_DIST_OR_ENERGY'])
            return None
        # Record coverage of dist and energy vars
        record_var_coverage(dist_var, species, probe, 'DIST_COVERAGE')
        record_var_coverage(en_var, species, probe, 'ENERGY_COVERAGE')
        # Get distribution
        dist_res = get_data(dist_var)
        if isinstance(dist_res, tuple):
            t_dist, dist = dist_res[0], dist_res[1]
        elif isinstance(dist_res, dict):
            t_dist = dist_res.get('x') or dist_res.get('times')
            dist = dist_res.get('y') if dist_res.get('y') is not None else dist_res.get('data')
        else:
            return None
        if t_dist is None or dist is None:
            return None
        times = ensure_datetime_format(t_dist)
        dist = np.asarray(dist)
        # Get energy bins (may be (Nt, Ne) or (Ne,))
        en_res = get_data(en_var)
        if isinstance(en_res, tuple):
            _, en = en_res[0], en_res[1]
        elif isinstance(en_res, dict):
            en = en_res.get('v') or en_res.get('y') or en_res.get('data')
        else:
            en = None
        if en is None:
            # Infer energy axis by dimension matching later or use metadata defaults
            energy = None
        else:
            en = np.asarray(en)
            if en.ndim == 2 and en.shape[0] == len(times):
                energy = en[0]
            elif en.ndim == 2 and en.shape[1] == len(times):
                energy = en[:, 0]
            elif en.ndim == 1:
                energy = en
            else:
                # Try median across time-like axis
                try:
                    energy = np.nanmedian(en, axis=0)
                except Exception:
                    energy = None
        # If still missing energy, construct a plausible energy vector
        if energy is None:
            # Construct a log-spaced vector representing typical FPI ranges
            if species == 'dis':
                energy = np.geomspace(10, 30000, 32)  # ions ~10 eV to 30 keV
            else:
                energy = np.geomspace(5, 30000, 32)   # electrons ~5 eV to 30 keV
            spectro_diag_rows.append([species, probe, dist_var, en_var, f'imputed:{len(energy)}bins', 'IMPUTED_ENERGY_VECTOR'])
        # Identify axes: assume time is axis 0 if matches
        time_axis = 0 if dist.shape[0] == len(times) else None
        if time_axis is None:
            # Try last axis as time
            if dist.shape[-1] == len(times):
                time_axis = dist.ndim - 1
                dist = np.moveaxis(dist, time_axis, 0)
            else:
                return None
        # Now dist.shape[0] == Nt
        Nt = dist.shape[0]
        # Determine energy axis by matching energy length or choosing max varying axis
        if energy is not None:
            ne = energy.size
            energy_axis = None
            for ax in range(1, dist.ndim):
                if dist.shape[ax] == ne:
                    energy_axis = ax
                    break
        else:
            # Pick the axis (excluding time) with largest size as energy
            energy_axis = 1 if dist.ndim > 1 else None
            ne = dist.shape[energy_axis] if energy_axis is not None else None
            energy = np.arange(ne) if ne is not None else None
        if energy_axis is None or ne is None:
            return None
        # Average over all other non-time, non-energy axes
        axes_to_avg = tuple(ax for ax in range(1, dist.ndim) if ax != energy_axis)
        if len(axes_to_avg) > 0:
            spec2d = np.nanmean(dist, axis=axes_to_avg)
        else:
            spec2d = dist
        # Ensure shape (Nt, Ne)
        if spec2d.shape[0] == Nt and spec2d.shape[1] == ne:
            pass
        elif spec2d.shape[0] == Nt and spec2d.shape[-1] == ne:
            spec2d = spec2d.reshape((Nt, ne))
        elif spec2d.shape[0] == ne and spec2d.shape[1] == Nt:
            spec2d = spec2d.T
        else:
            # Try to find last axis as energy
            if spec2d.ndim == 2 and spec2d.shape[-1] == ne:
                spec2d = spec2d
            else:
                return None
        return times, np.asarray(energy), spec2d

    def plot_spectrogram(times, energy, spec2d, outname, title, interval_minutes=2, emax=None):
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        # Optionally clip to lower-energy focus
        if emax is not None and np.isfinite(emax):
            m = (energy <= emax)
            if np.any(m):
                energy = energy[m]
                spec2d = spec2d[:, m]
        # Time edges
        Xc = mdates.date2num(times)
        Xe = np.empty(len(Xc) + 1)
        if len(Xc) > 1:
            Xe[1:-1] = 0.5 * (Xc[:-1] + Xc[1:])
            Xe[0] = Xc[0] - (Xc[1] - Xc[0]) / 2.0
            Xe[-1] = Xc[-1] + (Xc[-1] - Xc[-2]) / 2.0
        else:
            Xe[:] = [Xc[0] - 1.0/1440, Xc[0] + 1.0/1440]
        # Energy edges (geometric if strongly log-spaced)
        En = np.asarray(energy)
        Ee = np.empty(En.size + 1)
        if En.size > 1 and np.all(En > 0) and (np.max(En)/max(np.min(En),1e-9) > 50):
            Ee[1:-1] = np.sqrt(En[:-1] * En[1:])
            Ee[0] = En[0]**2 / Ee[1]
            Ee[-1] = En[-1]**2 / Ee[-2]
        else:
            Ee[1:-1] = 0.5 * (En[:-1] + En[1:]) if En.size > 1 else En[0]
            Ee[0] = En[0] - (En[1] - En[0]) / 2.0 if En.size > 1 else En[0] - 1.0
            Ee[-1] = En[-1] + (En[-1] - En[-2]) / 2.0 if En.size > 1 else En[-1] + 1.0
        # Color scaling
        vmin = np.nanpercentile(spec2d, 5)
        vmax = np.nanpercentile(spec2d, 99)
        # Handle non-positive values for log
        spec_plot = np.where(spec2d <= 0, np.nan, spec2d)
        pcm = ax.pcolormesh(Xe, Ee, spec_plot.T, shading='auto', cmap='viridis',
                            norm=mcolors.LogNorm(vmin=max(vmin, 1e-2), vmax=max(vmax, 1.0)))
        cbar = fig.colorbar(pcm, ax=ax, pad=0.01)
        cbar.set_label('Flux / Counts (arb.)')
        ax.set_ylabel('Energy (eV)')
        safe_format_time_axis(ax, interval_minutes=interval_minutes)
        ax.set_xlabel('Time (UT)')
        plt.tight_layout()
        plt.savefig(outname, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    try:
        # Explicitly load FPI distribution data for spectrograms; fallback to build if omni missing
        for species in ['dis', 'des']:
            sp_name = {'dis': 'Ion', 'des': 'Electron'}[species]
            # Per-species lower energy focus (eV)
            lowE_focus = {'dis': 5000.0, 'des': 3000.0}[species]
            for probe in ['1', '2', '3', '4']:
                try:
                    mms.mms_load_fpi(trange=trange, probe=probe, data_rate='brst', level='l2',
                                     datatype=f'{species}-dist', time_clip=True)
                except Exception as e:
                    print(f"   ⚠️ Could not load FPI {sp_name} distributions for MMS{probe}: {e}")
                    continue

                e_var = f'mms{probe}_{species}_energyspectr_omni_brst'
                spec = extract_spectrogram(e_var)
                if spec is None:
                    spec = build_spectrogram_from_dist(species, probe=probe)
                if spec is None:
                    print(f"   ⚠️ No spectrogram available for MMS{probe} {sp_name}")
                    continue
                times, energy, spec2d = spec
                # Add UT coverage to diagnostics
                try:
                    t_start = times[0].strftime('%Y-%m-%d %H:%M:%S') if len(times) else ''
                    t_end = times[-1].strftime('%Y-%m-%d %H:%M:%S') if len(times) else ''
                    spectro_diag_rows.append([species, probe, e_var, '', f'{len(energy)}bins', 'COVERAGE', t_start, t_end, len(times)])
                except Exception:
                    pass
                # Always save full-range and low-energy focused versions
                plot_spectrogram(times, energy, spec2d,
                                 outname=f'fpi_{species}_spectrogram_full_mms{probe}.png',
                                 title=f'FPI {sp_name} Spectrogram (Full) — MMS{probe}',
                                 interval_minutes=2, emax=None)
                plot_spectrogram(times, energy, spec2d,
                                 outname=f'fpi_{species}_spectrogram_lowE_mms{probe}.png',
                                 title=f'FPI {sp_name} Spectrogram (≤{int(lowE_focus)} eV) — MMS{probe}',
                                 interval_minutes=2, emax=lowE_focus)
    except Exception as e:
        print(f"   ⚠️ FPI spectrogram warning: {e}")

    # ---------- Plot 4: MMS3 shear angles ----------
    if '3' in data and 'B_gsm' in data['3']:
        times3, b3 = data['3']['B_gsm']
        sheath_times = [
            datetime(2019,1,27,12,18,0, tzinfo=timezone.utc),
            datetime(2019,1,27,12,30,50, tzinfo=timezone.utc),
            datetime(2019,1,27,12,45,0, tzinfo=timezone.utc),
        ]
        sphere_times = [
            datetime(2019,1,27,12,22,0, tzinfo=timezone.utc),
            datetime(2019,1,27,12,33,0, tzinfo=timezone.utc),
            datetime(2019,1,27,12,40,0, tzinfo=timezone.utc),
        ]

        angles = []
        for sh, sp in zip(sheath_times, sphere_times):
            v_sh = mean_vector_in_window(times3, b3, sh, 30)
            v_sp = mean_vector_in_window(times3, b3, sp, 30)
            if v_sh is None or v_sp is None:
                angles.append(np.nan)
            else:
                cosang = np.dot(v_sh, v_sp) / (np.linalg.norm(v_sh) * np.linalg.norm(v_sp) + 1e-12)
                cosang = np.clip(cosang, -1.0, 1.0)
                ang = np.degrees(np.arccos(cosang))
                angles.append(float(ang))

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        fig.suptitle('MMS3 Magnetic Shear Angles (GSM)', fontsize=14, fontweight='bold')
        labels = ['12:18 vs 12:22', '12:30:50 vs 12:33', '12:45 vs 12:40']
        ax.bar(labels, angles, color='#1f77b4')
        for i, ang in enumerate(angles):
            if np.isfinite(ang):
                ax.text(i, ang + 1, f"{ang:.1f}°", ha='center')
        ax.set_ylabel('Shear Angle (deg)', fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('mms3_shear_angles.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    # ---------- Export CSV/JSON summary ----------
    import json, csv

    # Spectrogram diagnostics CSV (what variables found/used; UT coverage)
    try:
        # extend rows with UT coverage; also include analysis window overlap
        from datetime import datetime
        tr_start = datetime.strptime(args.start.replace('/', ' '), '%Y-%m-%d %H:%M:%S')
        tr_end = datetime.strptime(args.end.replace('/', ' '), '%Y-%m-%d %H:%M:%S')
        with open('spectrogram_diagnostics.csv', 'w', newline='') as df:
            dw = csv.writer(df)
            dw.writerow(['species', 'probe', 'candidate_or_dist_var', 'energy_var', 'energy_info', 'status', 't_start_UT', 't_end_UT', 'n_times', 'overlaps_window', 'match_confidence'])
            for row in spectro_diag_rows:
                # pad with blanks for coverage fields
                if len(row) < 9:
                    row = row + ['', '', '']
                # compute overlap and match confidence if possible
                t0, t1 = row[6], row[7]
                overlaps = ''
                try:
                    if t0 and t1:
                        dt0 = datetime.strptime(t0, '%Y-%m-%d %H:%M:%S')
                        dt1 = datetime.strptime(t1, '%Y-%m-%d %H:%M:%S')
                        overlaps = 'Y' if (dt1 >= tr_start and dt0 <= tr_end) else 'N'
                except Exception:
                    overlaps = ''
                # append overlap and confidence
                cand = row[2] if len(row) > 2 else ''
                conf = _match_confidence(cand, row[5] if len(row) > 5 else '')
                row_out = row[:9] + [overlaps, conf]
                dw.writerow(row_out)
        print(f"   ⚠️ Could not write spectrogram_diagnostics.csv: {e}")

    except Exception as e:
    # Optional: print filtered diagnostics summary and write filtered CSV
    if args.diag_summarize or args.diag_filter_species or args.diag_filter_probe or args.diag_filter_status:
        try:
            import pandas as pd
            if os.path.exists('spectrogram_diagnostics.csv'):
                diag = pd.read_csv('spectrogram_diagnostics.csv')
                filt = diag.copy()
                if args.diag_filter_species:
                    filt = filt[filt['species'].astype(str).str.lower() == args.diag_filter_species.lower()]
                if args.diag_filter_probe:
                    filt = filt[filt['probe'].astype(str) == str(args.diag_filter_probe)]
                if args.diag_filter_status:
                    filt = filt[filt['status'].astype(str).str.upper() == args.diag_filter_status.upper()]
                # Console summary
                if args.diag_summarize:
                    print('\nDiagnostics summary (filtered):')
                    if len(filt) == 0:
                        print('  No rows after filters')
                    else:
                        by_status = filt.groupby('status').size().reset_index(name='count')
                        for _, r in by_status.iterrows():
                            print(f"  {r['status']}: {r['count']}")
                        if 'match_confidence' in filt.columns:
                            by_conf = filt.groupby('match_confidence').size().reset_index(name='count')
                            print('  Confidence tiers:')
                            for _, r in by_conf.iterrows():
                                print(f"    {r['match_confidence']}: {r['count']}")
                # Write filtered CSV for convenience
                out_cols = [c for c in ['species','probe','candidate_or_dist_var','energy_var','energy_info','status','t_start_UT','t_end_UT','n_times','overlaps_window','match_confidence'] if c in filt.columns]
                filt[out_cols].to_csv('spectrogram_diagnostics_filtered.csv', index=False)
        except Exception as e:
            print(f"   ⚠️ Could not produce filtered diagnostics: {e}")

        print(f"   ⚠️ Could not write spectrogram_diagnostics.csv: {e}")
    # Build per-pair delays and Vn projections for CSV
    pairs = []
    for p in ['2', '3', '4']:
        if p in delays:
            pairs.append((f'MMS1-MMS{p}', delays[p]))

    summary = {
        'delays_s': delays,
        'pairs': {name: dt for name, dt in pairs},
        'vn_km_per_s': None if not np.isfinite(vn) else float(vn),
        'crossing_duration_s': None if not np.isfinite(duration_s) else float(duration_s),
        'thickness_km': None if not np.isfinite(thickness_km) else float(thickness_km),
        'crossings': {p: [dt.strftime('%Y-%m-%dT%H:%M:%S') for dt in cts] for p, cts in crossings.items()}
    }
    with open('boundary_summary.json', 'w') as jf:
        json.dump(summary, jf, indent=2)

    # CSV with crossings per spacecraft
    with open('boundary_crossings.csv', 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['spacecraft', 'crossing_time_UT'])
        for p, cts in crossings.items():
            for dt in cts:
                w.writerow([f'MMS{p}', dt.strftime('%Y-%m-%d %H:%M:%S')])

    # CSV with per-pair delays relative to MMS1 and Vn (and MVA metrics)
    with open('boundary_delays_vn.csv', 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['pair', 'delay_s'])
        for name, dt in pairs:
            w.writerow([name, f'{dt:.3f}'])
        w.writerow([])
        w.writerow(['Vn (km/s)', f'{vn:.3f}' if np.isfinite(vn) else 'NaN'])
        if mva_results is not None:
            nx, ny, nz = mva_results['normal']
            l21, l32 = mva_results['lambda_ratios']
            qual = mva_results.get('quality', '')
            w.writerow(['MVA_normal_x', f'{nx:.4f}'])
            w.writerow(['MVA_normal_y', f'{ny:.4f}'])
            w.writerow(['MVA_normal_z', f'{nz:.4f}'])
            w.writerow(['lambda2_over_lambda1', f'{l21:.3f}'])
            w.writerow(['lambda3_over_lambda2', f'{l32:.3f}'])
            w.writerow(['MVA_quality', qual])

    # Consolidated CSV: delays, Vn, duration, thickness, first crossings, and MVA metrics
    with open('boundary_consolidated.csv', 'w', newline='') as cf:
        w = csv.writer(cf)
        headers = [
            'pair', 'delay_s', 'Vn_km_per_s', 'duration_s', 'thickness_km',
            'MMS1_crossing', 'MMS2_crossing', 'MMS3_crossing', 'MMS4_crossing',
            'MVA_normal_x', 'MVA_normal_y', 'MVA_normal_z', 'lambda2_over_lambda1', 'lambda3_over_lambda2', 'MVA_quality'
        ]
        w.writerow(headers)
        # delays in MMS1-MMS2, MMS1-MMS3, MMS1-MMS4 order
        delays_ordered = [dict(pairs).get('MMS1-MMS2', np.nan), dict(pairs).get('MMS1-MMS3', np.nan), dict(pairs).get('MMS1-MMS4', np.nan)]
        c1 = crossings.get('1', [])
        c2 = crossings.get('2', [])
        c3 = crossings.get('3', [])
        c4 = crossings.get('4', [])
        first = lambda L: L[0].strftime('%Y-%m-%d %H:%M:%S') if L else ''
        nx = ny = nz = l21 = l32 = np.nan
        qual = ''
        if mva_results is not None:
            nx, ny, nz = mva_results['normal']
            l21, l32 = mva_results['lambda_ratios']
            qual = mva_results.get('quality', '')
        w.writerow([
            'ALL',
            ';'.join([f'{d:.3f}' if np.isfinite(d) else 'NaN' for d in delays_ordered]),
            f'{vn:.3f}' if np.isfinite(vn) else 'NaN',
            f'{duration_s:.1f}' if np.isfinite(duration_s) else 'NaN',
            f'{thickness_km:.0f}' if np.isfinite(thickness_km) else 'NaN',
            first(c1), first(c2), first(c3), first(c4),
            f'{nx:.4f}' if np.isfinite(nx) else 'NaN',
            f'{ny:.4f}' if np.isfinite(ny) else 'NaN',
            f'{nz:.4f}' if np.isfinite(nz) else 'NaN',
            f'{l21:.3f}' if np.isfinite(l21) else 'NaN',
            f'{l32:.3f}' if np.isfinite(l32) else 'NaN',
            qual
        ])

    # ---------- Assemble a single PDF with key figures and per-MMS grids ----------
    try:
        with PdfPages('mms_2019_01_27_publication_bundle.pdf') as pdf:
            # Diagnostics page: quick summary
            try:
                import pandas as pd
                if os.path.exists('spectrogram_diagnostics.csv'):
                    diag = pd.read_csv('spectrogram_diagnostics.csv')
                    # Summary counts page
                    fig, ax = plt.subplots(figsize=(11, 8.5))
                    ax.axis('off')
                    ax.set_title('Spectrogram Diagnostics (Summary)', fontsize=14, fontweight='bold')
                    summary = diag.groupby(['species','probe','status']).size().reset_index(name='count')
                    text = 'Status counts by species/probe:\n\n' + '\n'.join(
                        f"{r.species}/MMS{r.probe}: {r.status} = {r['count']}" for _, r in summary.iterrows()
                    )
                    # Confidence breakdown
                    if 'match_confidence' in diag.columns:
                        by_conf = diag.groupby('match_confidence').size().reset_index(name='count')
                        text += '\n\nConfidence tiers:\n' + '\n'.join(
                            f"{r.match_confidence}: {r['count']}" for _, r in by_conf.iterrows()
                        )
                    ax.text(0.01, 0.98, text, va='top', ha='left', family='monospace')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    # Detailed table page only if we have rows
                    if len(diag) > 0:
                        cols = ['species','probe','candidate_or_dist_var','status','t_start_UT','t_end_UT','n_times','overlaps_window']
                        # Gracefully limit columns if missing
                        cols = [c for c in cols if c in diag.columns]
                        sub = diag[cols].copy()
                        sub['probe'] = sub['probe'].astype(str)
                        # Sort rows for easy scanning
                        sort_cols = [c for c in ['species','probe','status','t_start_UT'] if c in sub.columns]
                        if sort_cols:
                            sub = sub.sort_values(sort_cols)
                        max_rows = 30
                        shown = sub.head(max_rows)
                        fig, ax = plt.subplots(figsize=(11, 8.5))
    # ---------- Optionally assemble a standalone diagnostics PDF ----------
    if args.diagnostics_only_pdf:
        try:
            if os.path.exists('spectrogram_diagnostics.csv'):
                import pandas as pd
                diag = pd.read_csv('spectrogram_diagnostics.csv')
                with PdfPages('spectrogram_diagnostics_report.pdf') as dpdf:
                    # Summary
                    fig, ax = plt.subplots(figsize=(11, 8.5))
                    ax.axis('off')
                    ax.set_title('Spectrogram Diagnostics (Summary)', fontsize=14, fontweight='bold')
                    if len(diag) > 0:
                        summary = diag.groupby(['species','probe','status']).size().reset_index(name='count')
                        text = 'Status counts by species/probe:\n\n' + '\n'.join(
                            f"{r.species}/MMS{r.probe}: {r.status} = {r['count']}" for _, r in summary.iterrows()
                        )
                        if 'match_confidence' in diag.columns:
                            by_conf = diag.groupby('match_confidence').size().reset_index(name='count')
                            text += '\n\nConfidence tiers:\n' + '\n'.join(
                                f"{r.match_confidence}: {r['count']}" for _, r in by_conf.iterrows()
                            )
                    else:
                        text = 'No diagnostics rows.'
                    ax.text(0.01, 0.98, text, va='top', ha='left', family='monospace')
                    dpdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    # Detailed (with filters applied if provided)
                    if len(diag) > 0:
                        filt = diag.copy()
                        if args.diag_filter_species:
                            filt = filt[filt['species'].astype(str).str.lower() == args.diag_filter_species.lower()]
                        if args.diag_filter_probe:
                            filt = filt[filt['probe'].astype(str) == str(args.diag_filter_probe)]
                        if args.diag_filter_status:
                            filt = filt[filt['status'].astype(str).str.upper() == args.diag_filter_status.upper()]
                        cols = ['species','probe','candidate_or_dist_var','status','t_start_UT','t_end_UT','n_times','overlaps_window','match_confidence']
                        cols = [c for c in cols if c in filt.columns]
                        if len(filt) > 0 and cols:
                            sort_cols = [c for c in ['species','probe','status','t_start_UT'] if c in filt.columns]
                            if sort_cols:
                                filt = filt.sort_values(sort_cols)
                            shown = filt[cols].head(30)
                            fig, ax = plt.subplots(figsize=(11, 8.5))
                            ax.axis('off')
                            ax.set_title('Spectrogram Diagnostics (Detailed)', fontsize=14, fontweight='bold')
                            table = ax.table(cellText=shown.values,
                                             colLabels=[c.replace('_',' ') for c in cols],
                                             loc='upper left', cellLoc='left')
                            table.auto_set_font_size(False)
                            table.set_fontsize(8)
                            table.scale(1, 1.2)
                            if len(filt) > len(shown):
                                ax.text(0.01, 0.02, f"… {len(filt)-len(shown)} additional rows omitted", fontsize=9)
                            dpdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)
        except Exception as e:
            print(f"   ⚠️ Could not build diagnostics-only PDF: {e}")

                        ax.axis('off')
                        ax.set_title('Spectrogram Diagnostics (Detailed)', fontsize=14, fontweight='bold')
                        table = ax.table(cellText=shown.values,
                                         colLabels=[c.replace('_',' ') for c in cols],
                                         loc='upper left', cellLoc='left')
                        table.auto_set_font_size(False)
                        table.set_fontsize(8)
                        table.scale(1, 1.2)
                        if len(sub) > max_rows:
                            ax.text(0.01, 0.02, f"… {len(sub)-max_rows} additional rows omitted", fontsize=9)
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
            except Exception as e:
                print(f"   ⚠️ Could not add diagnostics summary page: {e}")
            # Key figures
            fig_list = [
                'boundary_threshold_overview.png',
                'boundary_timing_speed_thickness.png',
                'mms3_shear_angles.png'
            ]
            if args.include_combined:
                fig_list.append('mms_combined_publication_2019_01_27.png')
            for f in fig_list:
                if os.path.exists(f):
                    img = plt.imread(f)
                    fig = plt.figure(figsize=(11, 8.5))
                    plt.imshow(img)
                    plt.axis('off')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            # Per-MMS grids: Ion full/lowE and Electron full/lowE
            if args.include_grids:
                for species, label in [('dis','Ion'), ('des','Electron')]:
                    for variant, vlabel in [('full','Full'), ('lowE','Low-E')]:
                        # 2x2 grid for MMS1..4
                        fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
                        fig.suptitle(f'FPI {label} Spectrograms ({vlabel}) — MMS1–4', fontsize=14, fontweight='bold')
                        idx = 0
                        for r in range(2):
                            for c in range(2):
                                probe = str(idx+1)
                                ax = axs[r, c]
                                f = f'fpi_{species}_spectrogram_{variant}_mms{probe}.png'
                                if os.path.exists(f):
                                    img = plt.imread(f)
                                    ax.imshow(img)
                                    ax.axis('off')
                                    ax.set_title(f'MMS{probe}', loc='left', fontsize=10)
                                else:
                                    ax.text(0.5, 0.5, f'MMS{probe}: not available', transform=ax.transAxes,
                                            ha='center', va='center')
                                    ax.axis('off')
                                idx += 1
                        plt.tight_layout()
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
    except Exception as e:
        print(f"   ⚠️ PDF assembly warning: {e}")

    print("\n✅ Test complete. Generated files:")
    print("  - boundary_threshold_overview.png")
    print("  - boundary_timing_speed_thickness.png")
    print("  - mms3_shear_angles.png")
    print("  - boundary_summary.json, boundary_crossings.csv, boundary_delays_vn.csv")
    print("  - Spectrograms per MMS (full & lowE) when available")
    print("  - mms_2019_01_27_publication_bundle.pdf")

if __name__ == '__main__':
    main()

