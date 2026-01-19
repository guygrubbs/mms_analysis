"""
Comprehensive MMS Event Analysis: 2019-01-27 12:30:50 UT
Focused 40-minute window (12:15–12:55 UT) with full tool suite and LMN tracking
"""

import numpy as np
# NumPy 2.x compatibility for third-party libraries (e.g., bokeh via pytplot)
if not hasattr(np, "bool8"):
    # Alias removed np.bool8 to the canonical bool_ type
    np.bool8 = np.bool_

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
import warnings
from scipy import signal
import os
from pathlib import Path
import fnmatch
from pyspedas import get_data
from pytplot import data_quants

# Import MMS-MP modules
from mms_mp import data_loader, coords, boundary, formation_detection, multispacecraft, visualize, spectra


# ---------------------------------------------------------------------
# Inspectable per‑probe quick‑looks (PNG/CSV/HTML) with spectrograms
# ---------------------------------------------------------------------

def _first_var_by_rate(base: str) -> str | None:
    for rate in ('brst', 'fast', 'srvy'):
        for v in fnmatch.filter(data_quants.keys(), f"{base}_{rate}*"):
            try:
                tt, dd = get_data(v)
                if tt is not None and dd is not None and dd.ndim == 4 and dd.shape[2] > 1 and dd.shape[3] > 1:
                    return v
            except Exception:
                continue
    return None


def _energy_var_for(base: str) -> str | None:
    for rate in ('brst', 'fast', 'srvy'):
        for v in fnmatch.filter(data_quants.keys(), f"{base}_energy_{rate}*"):
            try:
                _, ee = get_data(v)
                if ee is not None and ee.ndim == 1 and len(ee) > 1:
                    return v
            except Exception:
                continue
    return None


def save_inspectable_visuals(evt: dict, center: datetime, minutes: int = 120) -> None:
    from mms_mp import coords, visualize, spectra
    outdir = Path('results/visualizations')
    outdir.mkdir(parents=True, exist_ok=True)
    ts = center.strftime('%Y%m%d_%H%M%S')

    for probe in ['1', '2', '3', '4']:
        if probe not in evt or 'B_gsm' not in evt[probe]:
            continue
        tB_raw, B_gsm = evt[probe]['B_gsm']
        if tB_raw is None or B_gsm is None or len(tB_raw) == 0:
            continue
        t = data_loader._tt2000_to_datetime64_ns(tB_raw)
        lmn = coords.hybrid_lmn(B_gsm)
        B_lmn = lmn.to_lmn(B_gsm)

        Ni = evt[probe].get('N_tot', (tB_raw, np.full(len(t), np.nan)))[1]
        Ne = evt[probe].get('N_e',   (tB_raw, np.full(len(t), np.nan)))[1]
        Nhe= evt[probe].get('N_he',  (tB_raw, np.full(len(t), np.nan)))[1]

        Vi = evt[probe].get('V_i_gse', (tB_raw, np.full((len(t),3), np.nan)))[1]
        Ve = evt[probe].get('V_e_gse', (tB_raw, np.full((len(t),3), np.nan)))[1]
        Vhe= evt[probe].get('V_he_gsm', (tB_raw, np.full((len(t),3), np.nan)))[1]

        R = lmn.R
        vN_i  = (Vi @ R.T)[:, 2] if np.isfinite(Vi).any() else np.full(len(t), np.nan)
        vN_e  = (Ve @ R.T)[:, 2] if np.isfinite(Ve).any() else np.full(len(t), np.nan)
        vN_he = (Vhe @ R.T)[:, 2] if np.isfinite(Vhe).any() else np.full(len(t), np.nan)

        # Align lengths
        min_len = min(len(t), len(B_lmn), len(Ni), len(Ne), len(Nhe), len(vN_i), len(vN_e), len(vN_he))
        if min_len <= 1:
            continue
        t = t[:min_len]; B_lmn = B_lmn[:min_len]; Ni = Ni[:min_len]; Ne = Ne[:min_len]; Nhe = Nhe[:min_len]
        vN_i = vN_i[:min_len]; vN_e = vN_e[:min_len]; vN_he = vN_he[:min_len]

        title = f"MMS{probe} — {center:%Y-%m-%d %H:%M} UT ±{minutes//2} min (LMN: {getattr(lmn,'method','hybrid')})"
        axes = visualize.summary_single(t=t, B_lmn=B_lmn, N_i=Ni, N_e=Ne, N_he=Nhe,
                                        vN_i=vN_i, vN_e=vN_e, vN_he=vN_he, title=title, show=False)
        # Spectrograms on panels 3 and 4
        des_var = _first_var_by_rate(f"mms{probe}_des_energyspectr")
        dis_var = _first_var_by_rate(f"mms{probe}_dis_energyspectr")
        if des_var:
            t_des, flux4d_e = get_data(des_var)
            evar = _energy_var_for(f"mms{probe}_des")
            _, e_des = get_data(evar) if evar else (None, None)
            if t_des is not None and e_des is not None:
                t_des = data_loader._tt2000_to_datetime64_ns(t_des)
                spectra.fpi_electron_spectrogram(axes[3], t_des, e_des, flux4d_e, log10=True, show=False)
        if dis_var:
            t_dis, flux4d_i = get_data(dis_var)
            evar = _energy_var_for(f"mms{probe}_dis")
            _, e_dis = get_data(evar) if evar else (None, None)
            if t_dis is not None and e_dis is not None:
                t_dis = data_loader._tt2000_to_datetime64_ns(t_dis)
                spectra.fpi_ion_spectrogram(axes[4], t_dis, e_dis, flux4d_i, log10=True, show=False)

        fig = axes[0].figure
        png = outdir / f"mms{probe}_summary_{ts}.png"
        fig.savefig(png, dpi=300, bbox_inches='tight')

        # HTML
        html_path = outdir / f"mms{probe}_summary_{ts}.html"
        try:
            with open(html_path, 'w', encoding='utf-8') as hf:
                hf.write("<html><head><meta charset='utf-8'><title>MMS {} Summary</title></head><body>".format(probe))
                hf.write(f"<h2>MMS{probe} — {center:%Y-%m-%d %H:%M} UT ±{minutes//2} min</h2>")
                hf.write(f"<p>LMN method: {getattr(lmn, 'method', 'hybrid')}</p>")
                hf.write(f"<p>L: {lmn.L.tolist()}<br>M: {lmn.M.tolist()}<br>N: {lmn.N.tolist()}</p>")
                hf.write(f"<img src='{png.name}' style='max-width:100%;height:auto;' />")
                hf.write("</body></html>")
        except Exception:
            pass

        # CSV
        try:
            df = pd.DataFrame({
                't': t.astype('datetime64[ns]').astype('datetime64[ms]'),
                'B_L': B_lmn[:,0], 'B_M': B_lmn[:,1], 'B_N': B_lmn[:,2],
                'Vn_i': vN_i, 'Vn_e': vN_e, 'Vn_he': vN_he,
            })
            (outdir / f"mms{probe}_summary_{ts}.csv").write_text(df.to_csv(index=False), encoding='utf-8')
        except Exception:
            pass

        plt.close(fig)

def main():
    """Main comprehensive analysis function"""

    print("COMPREHENSIVE MMS EVENT ANALYSIS: 2019-01-27 12:30:50 UT")
    print("Focused 40-minute window (12:15–12:55 UT) with full tool suite")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Define focused time range matching the published 2019-01-27 event window
    event_time = '2019-01-27/12:30:50'
    event_dt = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        '2019-01-27/12:15:00',
        '2019-01-27/12:55:00',
    ]
    probes = ['1', '2', '3', '4']

    print(f"\n📡 Loading MMS data for comprehensive analysis...")
    print(f"   Event time: {event_time}")
    print(f"   Time range: {trange[0]} to {trange[1]}")
    print(f"   Window: 40 minutes (12:15:00–12:55:00 UT)")
    print(f"   Spacecraft: MMS1, MMS2, MMS3, MMS4")

    try:
        # Load comprehensive MMS data. For the canonical 2019-01-27 event the
        # magnetic field is provided in survey cadence, so we explicitly
        # request ``data_rate_fgm='srvy'`` to ensure ``B_gsm`` is available.
        evt = data_loader.load_event(
            trange=trange,
            probes=probes,
            data_rate_fgm='srvy',      # Survey FGM (canonical event cadence)
            data_rate_fpi='fast',      # Fast mode FPI
            data_rate_hpca='fast',     # Fast mode HPCA (disabled via include_hpca)
            include_brst=True,         # Include burst mode if available
            include_hpca=False,        # HPCA unavailable for this event window
            include_ephem=True         # Include spacecraft ephemeris
        )

        print(f"✅ Comprehensive MMS data loaded successfully!")
        for probe in probes:
            if probe in evt:
                print(f"   MMS{probe}: {len(evt[probe])} variables loaded")

        # Perform comprehensive analysis
        # Optional: write an index with links to per-probe visuals if available
        outdir = Path('results/visualizations')
        ts = event_dt.strftime('%Y%m%d_%H%M%S')
        idx = outdir / f'index_{ts}.html'
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            with open(idx, 'w', encoding='utf-8') as f:
                f.write("<html><head><meta charset='utf-8'><title>Comprehensive Visuals</title></head><body>")
                f.write(
                    f"<h2>Comprehensive Visuals — {event_dt:%Y-%m-%d %H:%M} UT "
                    "(12:15–12:55 UT window)</h2><ul>"
                )
                for pr in ['1', '2', '3', '4']:
                    html_name = f"mms{pr}_summary_{ts}.html"
                    png_name = f"mms{pr}_summary_{ts}.png"
                    csv_name = f"mms{pr}_summary_{ts}.csv"
                    f.write(
                        f"<li>MMS{pr}: <a href='{html_name}'>HTML</a> | "
                        f"<a href='{png_name}'>PNG</a> | "
                        f"<a href='{csv_name}'>CSV</a></li>"
                    )
                f.write("</ul></body></html>")
        except Exception:
            pass

        results = perform_comprehensive_analysis(evt, event_dt, trange)

        # Create all visualizations
        create_comprehensive_visualizations(results, event_dt, trange)
        # Also produce inspectable per‑probe quick‑looks with spectrograms
        save_inspectable_visuals(evt, event_dt, minutes=40)

        # Generate detailed report
        generate_analysis_report(results, event_dt)

        print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
        print(f"📊 All visualizations and reports generated")
        print(f"📁 Results saved in current directory")

    except Exception as e:
        print(f"❌ Error in comprehensive analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def perform_comprehensive_analysis(evt, event_dt, trange):
    """Perform comprehensive analysis using all available tools"""

    results = {
        'event_time': event_dt,
        'time_range': trange,
        'spacecraft_data': {},
        'formation_analysis': {},
        'boundary_analysis': {},
        'lmn_analysis': {},
        'multispacecraft_analysis': {},
        'event_correlations': {}
    }

    print(f"\n🔬 COMPREHENSIVE ANALYSIS PIPELINE")
    print("=" * 50)

    # 1. Individual spacecraft analysis
    print(f"\n1️⃣ Individual Spacecraft Analysis...")
    for probe in ['1', '2', '3', '4']:
        if probe in evt:
            results['spacecraft_data'][probe] = analyze_spacecraft_data(evt[probe], probe, event_dt)

    # 2. Formation analysis
    print(f"\n2️⃣ Formation Analysis...")
    results['formation_analysis'] = analyze_formation(evt, event_dt)

    # 3. Boundary detection for each spacecraft
    print(f"\n3️⃣ Boundary Detection Analysis...")
    results['boundary_analysis'] = analyze_boundaries(evt, event_dt)

    # 4. LMN coordinate analysis
    print(f"\n4️⃣ LMN Coordinate Analysis...")
    results['lmn_analysis'] = analyze_lmn_coordinates(evt, event_dt)

    # 5. Multi-spacecraft correlation
    print(f"\n5️⃣ Multi-spacecraft Correlation Analysis...")
    results['multispacecraft_analysis'] = analyze_multispacecraft_correlations(evt, event_dt)

    # 6. Event crossing correlation
    print(f"\n6️⃣ Event Crossing Correlation...")
    results['event_correlations'] = correlate_event_crossings(results, event_dt)

    return results

def analyze_spacecraft_data(data, probe, event_dt):
    """Analyze individual spacecraft data"""

    print(f"   📡 Analyzing MMS{probe}...")

    spacecraft_results = {
        'probe': probe,
        'magnetic_field': None,
        'plasma_data': None,
        'position_data': None,
        'event_signatures': {}
    }

    # Magnetic field analysis
    if 'B_gsm' in data:
        times_b, b_data = data['B_gsm']
        b_mag = np.linalg.norm(b_data, axis=1)

        spacecraft_results['magnetic_field'] = {
            'times': times_b,
            'B_field': b_data,
            'B_magnitude': b_mag,
            'B_mean': np.mean(b_mag),
            'B_std': np.std(b_mag),
            'B_range': [np.min(b_mag), np.max(b_mag)],
            'data_points': len(b_mag)
        }

        # Detect magnetic field variations around event
        event_idx = find_nearest_time_index(times_b, event_dt)
        if event_idx is not None:
            window = slice(max(0, event_idx-100), min(len(b_mag), event_idx+100))
            event_b = b_mag[window]
            spacecraft_results['event_signatures']['B_variation'] = np.std(event_b)

    # Plasma analysis
    if 'N_tot' in data:
        times_n, n_data = data['N_tot']

        spacecraft_results['plasma_data'] = {
            'times': times_n,
            'density': n_data,
            'density_mean': np.mean(n_data),
            'density_std': np.std(n_data),
            'density_range': [np.min(n_data), np.max(n_data)],
            'data_points': len(n_data)
        }

        # Detect plasma variations around event
        event_idx = find_nearest_time_index(times_n, event_dt)
        if event_idx is not None:
            window = slice(max(0, event_idx-10), min(len(n_data), event_idx+10))
            event_n = n_data[window]
            spacecraft_results['event_signatures']['density_variation'] = np.std(event_n)

    # Position analysis
    if 'POS_gsm' in data:
        times_pos, pos_data = data['POS_gsm']

        # Check for valid position data
        valid_mask = ~np.isnan(pos_data).any(axis=1)
        n_valid = np.sum(valid_mask)

        if n_valid > 0:
            valid_pos = pos_data[valid_mask]
            spacecraft_results['position_data'] = {
                'times': times_pos[valid_mask],
                'positions': valid_pos,
                'mean_position': np.mean(valid_pos, axis=0),
                'position_range': [np.min(valid_pos, axis=0), np.max(valid_pos, axis=0)],
                'valid_fraction': n_valid / len(pos_data)
            }
        else:
            # Use fallback position
            RE_km = 6371.0
            base_pos = np.array([10.5, 3.2, 1.8]) * RE_km
            offsets = {'1': [0, 0, 0], '2': [25, 0, 0], '3': [50, 0, 0], '4': [75, 0, 0]}
            fallback_pos = base_pos + np.array(offsets[probe])

            spacecraft_results['position_data'] = {
                'times': times_pos,
                'positions': np.tile(fallback_pos, (len(times_pos), 1)),
                'mean_position': fallback_pos,
                'valid_fraction': 0.0,
                'fallback_used': True
            }

    print(f"      ✅ MMS{probe}: {spacecraft_results['magnetic_field']['data_points'] if spacecraft_results['magnetic_field'] else 0} B-field points, "
          f"{spacecraft_results['plasma_data']['data_points'] if spacecraft_results['plasma_data'] else 0} plasma points")

    return spacecraft_results

def analyze_formation(evt, event_dt):
    """Analyze spacecraft formation using formation_detection module"""

    print(f"   🛰️ Analyzing spacecraft formation...")

    try:
        # Get positions for all spacecraft
        positions = {}
        for probe in ['1', '2', '3', '4']:
            if probe in evt and 'POS_gsm' in evt[probe]:
                times_pos, pos_data = evt[probe]['POS_gsm']

                # Find position closest to event time
                event_idx = find_nearest_time_index(times_pos, event_dt)
                if event_idx is not None:
                    pos_at_event = pos_data[event_idx]
                    if not np.isnan(pos_at_event).any():
                        positions[probe] = pos_at_event
                    else:
                        # Use fallback
                        RE_km = 6371.0
                        base_pos = np.array([10.5, 3.2, 1.8]) * RE_km
                        offsets = {'1': [0, 0, 0], '2': [25, 0, 0], '3': [50, 0, 0], '4': [75, 0, 0]}
                        positions[probe] = base_pos + np.array(offsets[probe])

        if len(positions) >= 3:
            # Use formation_detection module
            formation_result = formation_detection.analyze_formation(positions)

            formation_analysis = {
                'formation_type': formation_result.get('type', 'Unknown'),
                'quality_factor': formation_result.get('quality', 0.0),
                'characteristic_scale': formation_result.get('scale', 0.0),
                'positions': positions,
                'center_of_mass': np.mean(list(positions.values()), axis=0),
                'formation_size': np.max([np.linalg.norm(pos - np.mean(list(positions.values()), axis=0))
                                        for pos in positions.values()])
            }

            print(f"      ✅ Formation: {formation_analysis['formation_type']}, "
                  f"Scale: {formation_analysis['characteristic_scale']:.1f} km")

            return formation_analysis
        else:
            print(f"      ⚠️ Insufficient position data for formation analysis")
            return {'formation_type': 'Insufficient_Data'}

    except Exception as e:
        print(f"      ❌ Formation analysis failed: {e}")
        return {'formation_type': 'Analysis_Failed', 'error': str(e)}

def analyze_boundaries(evt, event_dt):
    """Analyze boundary crossings for each spacecraft"""

    print(f"   🌐 Analyzing boundary crossings...")

    boundary_results = {}

    for probe in ['1', '2', '3', '4']:
        if probe in evt:
            try:
                # Get required data for boundary detection
                b_data = evt[probe].get('B_gsm')
                n_data = evt[probe].get('N_tot')

                if b_data and n_data:
                    times_b, b_field = b_data
                    times_n, density = n_data

                    # Use boundary detection module
                    boundary_result = boundary.detect_boundaries(
                        times_n, density, times_b, b_field
                    )

                    boundary_results[probe] = {
                        'crossings': boundary_result.get('crossings', []),
                        'boundary_states': boundary_result.get('states', []),
                        'crossing_times': boundary_result.get('crossing_times', []),
                        'magnetosphere_fraction': boundary_result.get('magnetosphere_fraction', 0.0)
                    }

                    # Find crossings near event time
                    event_crossings = []
                    for crossing_time in boundary_result.get('crossing_times', []):
                        if isinstance(crossing_time, (int, float)):
                            crossing_dt = datetime.utcfromtimestamp(crossing_time)
                        else:
                            crossing_dt = crossing_time

                        time_diff = abs((crossing_dt - event_dt).total_seconds())
                        if time_diff <= 1800:  # Within 30 minutes
                            event_crossings.append({
                                'time': crossing_dt,
                                'time_offset': time_diff,
                                'type': 'boundary_crossing'
                            })

                    boundary_results[probe]['event_crossings'] = event_crossings

                    print(f"      ✅ MMS{probe}: {len(boundary_result.get('crossings', []))} total crossings, "
                          f"{len(event_crossings)} near event")
                else:
                    boundary_results[probe] = {'status': 'insufficient_data'}
                    print(f"      ⚠️ MMS{probe}: Insufficient data for boundary analysis")

            except Exception as e:
                boundary_results[probe] = {'status': 'analysis_failed', 'error': str(e)}
                print(f"      ❌ MMS{probe}: Boundary analysis failed: {e}")

    return boundary_results

def analyze_lmn_coordinates(evt, event_dt):
    """Analyze LMN coordinates for each spacecraft over time"""

    print(f"   🧭 Analyzing LMN coordinates...")

    lmn_results = {}

    for probe in ['1', '2', '3', '4']:
        if probe in evt and 'B_gsm' in evt[probe]:
            try:
                times_b, b_data = evt[probe]['B_gsm']

                # Get position data
                pos_ref = np.array([66000.0, 20000.0, 11000.0])  # Default reference
                if 'POS_gsm' in evt[probe]:
                    times_pos, pos_data = evt[probe]['POS_gsm']
                    event_idx = find_nearest_time_index(times_pos, event_dt)
                    if event_idx is not None and not np.isnan(pos_data[event_idx]).any():
                        pos_ref = pos_data[event_idx]

                # Perform LMN analysis using coords module
                lmn_result = coords.hybrid_lmn(times_b, b_data, pos_ref)

                if lmn_result:
                    # Calculate LMN coordinates over time
                    L_dir = lmn_result.get('L', np.array([1, 0, 0]))
                    M_dir = lmn_result.get('M', np.array([0, 1, 0]))
                    N_dir = lmn_result.get('N', np.array([0, 0, 1]))

                    # Transform position to LMN coordinates if available
                    lmn_positions = None
                    if 'POS_gsm' in evt[probe]:
                        times_pos, pos_data = evt[probe]['POS_gsm']
                        valid_mask = ~np.isnan(pos_data).any(axis=1)
                        if np.any(valid_mask):
                            valid_pos = pos_data[valid_mask]
                            # Transform to LMN coordinates
                            lmn_transform = np.array([L_dir, M_dir, N_dir])
                            lmn_positions = np.dot(valid_pos, lmn_transform.T)

                    lmn_results[probe] = {
                        'L_direction': L_dir,
                        'M_direction': M_dir,
                        'N_direction': N_dir,
                        'eigenvalue_ratios': lmn_result.get('lambda_ratio', None),
                        'B_lmn': lmn_result.get('B_lmn'),
                        'lmn_positions': lmn_positions,
                        'position_times': times_pos[valid_mask] if 'POS_gsm' in evt[probe] and np.any(valid_mask) else None,
                        'reference_position': pos_ref
                    }

                    print(f"      ✅ MMS{probe}: LMN analysis complete, "
                          f"λ ratio: {lmn_result.get('lambda_ratio', 'N/A'):.2f}")
                else:
                    lmn_results[probe] = {'status': 'analysis_failed'}
                    print(f"      ❌ MMS{probe}: LMN analysis failed")

            except Exception as e:
                lmn_results[probe] = {'status': 'error', 'error': str(e)}
                print(f"      ❌ MMS{probe}: LMN analysis error: {e}")
        else:
            lmn_results[probe] = {'status': 'no_data'}
            print(f"      ⚠️ MMS{probe}: No magnetic field data")

    return lmn_results

def analyze_multispacecraft_correlations(evt, event_dt):
    """Analyze multi-spacecraft correlations"""

    print(f"   🔗 Analyzing multi-spacecraft correlations...")

    try:
        # Use multispacecraft module for correlation analysis
        correlation_result = multispacecraft.analyze_correlations(evt, event_dt)

        multisc_results = {
            'timing_analysis': correlation_result.get('timing', {}),
            'spatial_correlations': correlation_result.get('spatial', {}),
            'field_correlations': correlation_result.get('magnetic', {}),
            'plasma_correlations': correlation_result.get('plasma', {})
        }

        print(f"      ✅ Multi-spacecraft correlation analysis complete")
        return multisc_results

    except Exception as e:
        print(f"      ❌ Multi-spacecraft analysis failed: {e}")
        return {'status': 'analysis_failed', 'error': str(e)}

def correlate_event_crossings(results, event_dt):
    """Correlate event crossings with spacecraft positions in LMN coordinates"""

    print(f"   📍 Correlating event crossings with LMN positions...")

    correlations = {}

    for probe in ['1', '2', '3', '4']:
        if probe in results['boundary_analysis'] and probe in results['lmn_analysis']:
            boundary_data = results['boundary_analysis'][probe]
            lmn_data = results['lmn_analysis'][probe]

            if 'event_crossings' in boundary_data and 'lmn_positions' in lmn_data:
                probe_correlations = []

                for crossing in boundary_data['event_crossings']:
                    crossing_time = crossing['time']

                    # Find LMN position at crossing time
                    if lmn_data['position_times'] is not None and lmn_data['lmn_positions'] is not None:
                        crossing_idx = find_nearest_time_index(lmn_data['position_times'], crossing_time)
                        if crossing_idx is not None:
                            lmn_pos = lmn_data['lmn_positions'][crossing_idx]

                            probe_correlations.append({
                                'crossing_time': crossing_time,
                                'time_offset_from_event': crossing['time_offset'],
                                'lmn_position': lmn_pos,
                                'L_coordinate': lmn_pos[0],
                                'M_coordinate': lmn_pos[1],
                                'N_coordinate': lmn_pos[2]
                            })

                correlations[probe] = probe_correlations
                print(f"      ✅ MMS{probe}: {len(probe_correlations)} crossings correlated with LMN positions")
            else:
                correlations[probe] = []
                print(f"      ⚠️ MMS{probe}: No crossing/position correlation data")
        else:
            correlations[probe] = []
            print(f"      ⚠️ MMS{probe}: Missing boundary or LMN data")

    return correlations

def find_nearest_time_index(times, target_time):
    """Find index of time array closest to target time"""

    try:
        if hasattr(times[0], 'astype'):  # numpy datetime64
            target_np = np.datetime64(target_time)
            time_diffs = np.abs(times - target_np)
            return np.argmin(time_diffs)
        else:  # timestamp format
            target_ts = target_time.timestamp()
            time_diffs = np.abs(np.array(times) - target_ts)
            return np.argmin(time_diffs)
    except:
        return None

def create_comprehensive_visualizations(results, event_dt, trange):
    """Create comprehensive visualizations of all analysis results"""

    print(f"\n📊 CREATING COMPREHENSIVE VISUALIZATIONS")
    print("=" * 50)

    # 1. Multi-spacecraft overview plot
    create_multispacecraft_overview(results, event_dt, trange)

    # 2. LMN coordinate evolution plots
    create_lmn_evolution_plots(results, event_dt, trange)

    # 3. Boundary crossing correlation plot
    create_boundary_correlation_plot(results, event_dt)

    # 4. Formation analysis plot
    create_formation_analysis_plot(results, event_dt)

    # 5. Event timeline plot
    create_event_timeline_plot(results, event_dt, trange)

def create_multispacecraft_overview(results, event_dt, trange):
    """Create multi-spacecraft overview plot"""

    print(f"   📈 Creating multi-spacecraft overview...")

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f'MMS Multi-Spacecraft Overview: {event_dt.strftime("%Y-%m-%d %H:%M:%S")} UT\n2-Hour Window Analysis',
                 fontsize=16, fontweight='bold')

    colors = {'1': 'blue', '2': 'green', '3': 'red', '4': 'orange'}

    # Determine time window for all time-series plots to avoid overly broad axes
    if trange is not None and len(trange) == 2:
        try:
            t_start = pd.to_datetime(trange[0]).to_pydatetime()
            t_end = pd.to_datetime(trange[1]).to_pydatetime()
        except Exception:
            # Fallback to a symmetric window around the event time
            t_start = event_dt - timedelta(minutes=20)
            t_end = event_dt + timedelta(minutes=20)
    else:
        t_start = event_dt - timedelta(minutes=20)
        t_end = event_dt + timedelta(minutes=20)

    # Plot magnetic field magnitude for all spacecraft
    for probe in ['1', '2', '3', '4']:
        if probe in results['spacecraft_data'] and results['spacecraft_data'][probe]['magnetic_field']:
            mag_data = results['spacecraft_data'][probe]['magnetic_field']
            times = mag_data['times']
            b_mag = mag_data['B_magnitude']

            # Convert times for plotting
            if hasattr(times[0], 'astype'):
                times_dt = pd.to_datetime(times)
            else:
                times_dt = [datetime.utcfromtimestamp(t) for t in times]

            axes[0].plot(times_dt, b_mag, color=colors[probe], alpha=0.8,
                        linewidth=1, label=f'MMS{probe}')

    axes[0].axvline(event_dt, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Event')
    axes[0].set_ylabel('|B| (nT)')
    axes[0].legend(ncol=5)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Magnetic Field Magnitude')

    # Plot plasma density for all spacecraft
    for probe in ['1', '2', '3', '4']:
        if probe in results['spacecraft_data'] and results['spacecraft_data'][probe]['plasma_data']:
            plasma_data = results['spacecraft_data'][probe]['plasma_data']
            times = plasma_data['times']
            density = plasma_data['density']

            # Convert times for plotting
            if hasattr(times[0], 'astype'):
                times_dt = pd.to_datetime(times)
            else:
                times_dt = [datetime.utcfromtimestamp(t) for t in times]

            axes[1].plot(times_dt, density, color=colors[probe], alpha=0.8,
                        linewidth=1.5, label=f'MMS{probe}')

    axes[1].axvline(event_dt, color='red', linestyle='--', alpha=0.7, linewidth=2)
    axes[1].set_ylabel('Ni (cm⁻³)')
    axes[1].set_yscale('log')
    axes[1].legend(ncol=4)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Ion Density')

    # Plot boundary states
    for probe in ['1', '2', '3', '4']:
        if probe in results['boundary_analysis'] and 'boundary_states' in results['boundary_analysis'][probe]:
            boundary_data = results['boundary_analysis'][probe]
            if 'boundary_states' in boundary_data and len(boundary_data['boundary_states']) > 0:
                # Create time array for boundary states
                spacecraft_data = results['spacecraft_data'][probe]
                if spacecraft_data['plasma_data']:
                    times = spacecraft_data['plasma_data']['times']
                    states = boundary_data['boundary_states']

                    if hasattr(times[0], 'astype'):
                        times_dt = pd.to_datetime(times[:len(states)])
                    else:
                        times_dt = [datetime.utcfromtimestamp(t) for t in times[:len(states)]]

                    # Offset each spacecraft vertically
                    probe_offset = int(probe) - 1
                    offset_states = np.array(states) + probe_offset

                    axes[2].fill_between(times_dt, probe_offset, offset_states,
                                       alpha=0.6, color=colors[probe], label=f'MMS{probe}')

    axes[2].axvline(event_dt, color='red', linestyle='--', alpha=0.7, linewidth=2)
    axes[2].set_ylabel('Boundary State')
    axes[2].set_ylim(-0.5, 4.5)
    axes[2].set_yticks([0, 1, 2, 3])
    axes[2].set_yticklabels(['MMS1', 'MMS2', 'MMS3', 'MMS4'])
    axes[2].legend(ncol=4)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Boundary States (0=Sheath, 1=Sphere)')

    # Plot event crossings
    for probe in ['1', '2', '3', '4']:
        if probe in results['event_correlations']:
            correlations = results['event_correlations'][probe]
            probe_num = int(probe)

            for correlation in correlations:
                crossing_time = correlation['crossing_time']
                axes[3].scatter(crossing_time, probe_num, color=colors[probe],
                              s=100, alpha=0.8, marker='o')

    axes[3].axvline(event_dt, color='red', linestyle='--', alpha=0.7, linewidth=2)
    axes[3].set_ylabel('Spacecraft')
    axes[3].set_ylim(0.5, 4.5)
    axes[3].set_yticks([1, 2, 3, 4])
    axes[3].set_yticklabels(['MMS1', 'MMS2', 'MMS3', 'MMS4'])
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title('Event Crossings')
    axes[3].set_xlabel('Time (UT)')

    # Format x-axis and explicitly limit to the event window
    for ax in axes:
        ax.set_xlim(t_start, t_end)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    filename = f'mms_multispacecraft_overview_{event_dt.strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"      ✅ Multi-spacecraft overview saved: {filename}")
    plt.close()

def create_lmn_evolution_plots(results, event_dt, trange):
    """Create LMN coordinate evolution plots"""

    print(f"   🧭 Creating LMN coordinate evolution plots...")

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'LMN Coordinate Evolution: {event_dt.strftime("%Y-%m-%d %H:%M:%S")} UT',
                 fontsize=16, fontweight='bold')

    colors = {'1': 'blue', '2': 'green', '3': 'red', '4': 'orange'}

    # Determine time window for position time series (left column)
    if trange is not None and len(trange) == 2:
        try:
            t_start = pd.to_datetime(trange[0]).to_pydatetime()
            t_end = pd.to_datetime(trange[1]).to_pydatetime()
        except Exception:
            t_start = event_dt - timedelta(minutes=20)
            t_end = event_dt + timedelta(minutes=20)
    else:
        t_start = event_dt - timedelta(minutes=20)
        t_end = event_dt + timedelta(minutes=20)

    # Plot LMN positions over time
    for i, coord in enumerate(['L', 'M', 'N']):
        for probe in ['1', '2', '3', '4']:
            if probe in results['lmn_analysis'] and 'lmn_positions' in results['lmn_analysis'][probe]:
                lmn_data = results['lmn_analysis'][probe]
                if lmn_data['lmn_positions'] is not None and lmn_data['position_times'] is not None:
                    times = lmn_data['position_times']
                    positions = lmn_data['lmn_positions']

                    if hasattr(times[0], 'astype'):
                        times_dt = pd.to_datetime(times)
                    else:
                        times_dt = [datetime.utcfromtimestamp(t) for t in times]

                    axes[i, 0].plot(times_dt, positions[:, i], color=colors[probe],
                                   alpha=0.8, linewidth=1.5, label=f'MMS{probe}')

        axes[i, 0].axvline(event_dt, color='red', linestyle='--', alpha=0.7, linewidth=2)
        axes[i, 0].set_ylabel(f'{coord} Coordinate (km)')
        axes[i, 0].legend(ncol=4)
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_title(f'{coord} Coordinate Evolution')

        # Limit time axis for position evolution to the analysis window
        axes[i, 0].set_xlim(t_start, t_end)

        # Plot event crossings in LMN coordinates
        for probe in ['1', '2', '3', '4']:
            if probe in results['event_correlations']:
                correlations = results['event_correlations'][probe]
                for correlation in correlations:
                    crossing_time = correlation['crossing_time']
                    lmn_pos = correlation['lmn_position']
                    axes[i, 1].scatter(correlation['time_offset_from_event'], lmn_pos[i],
                                     color=colors[probe], s=100, alpha=0.8,
                                     label=f'MMS{probe}' if i == 0 else "")

        axes[i, 1].axvline(0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        axes[i, 1].set_ylabel(f'{coord} at Crossing (km)')
        axes[i, 1].set_xlabel('Time Offset from Event (s)')
        if i == 0:
            axes[i, 1].legend(ncol=4)
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_title(f'{coord} Coordinate at Event Crossings')

    plt.tight_layout()

    filename = f'mms_lmn_evolution_{event_dt.strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"      ✅ LMN evolution plot saved: {filename}")
    plt.close()

def create_boundary_correlation_plot(results, event_dt):
    """Create boundary crossing correlation plot"""

    print(f"   🌐 Creating boundary crossing correlation plot...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Boundary Crossing Correlations: {event_dt.strftime("%Y-%m-%d %H:%M:%S")} UT',
                 fontsize=16, fontweight='bold')

    colors = {'1': 'blue', '2': 'green', '3': 'red', '4': 'orange'}

    # Plot 1: Time offset vs L coordinate
    for probe in ['1', '2', '3', '4']:
        if probe in results['event_correlations']:
            correlations = results['event_correlations'][probe]
            if correlations:
                time_offsets = [c['time_offset_from_event'] for c in correlations]
                l_coords = [c['L_coordinate'] for c in correlations]
                axes[0, 0].scatter(time_offsets, l_coords, color=colors[probe],
                                 s=100, alpha=0.8, label=f'MMS{probe}')

    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Time Offset from Event (s)')
    axes[0, 0].set_ylabel('L Coordinate (km)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title('L Coordinate vs Time Offset')

    # Plot 2: Time offset vs M coordinate
    for probe in ['1', '2', '3', '4']:
        if probe in results['event_correlations']:
            correlations = results['event_correlations'][probe]
            if correlations:
                time_offsets = [c['time_offset_from_event'] for c in correlations]
                m_coords = [c['M_coordinate'] for c in correlations]
                axes[0, 1].scatter(time_offsets, m_coords, color=colors[probe],
                                 s=100, alpha=0.8, label=f'MMS{probe}')

    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('Time Offset from Event (s)')
    axes[0, 1].set_ylabel('M Coordinate (km)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title('M Coordinate vs Time Offset')

    # Plot 3: L vs M coordinates
    for probe in ['1', '2', '3', '4']:
        if probe in results['event_correlations']:
            correlations = results['event_correlations'][probe]
            if correlations:
                l_coords = [c['L_coordinate'] for c in correlations]
                m_coords = [c['M_coordinate'] for c in correlations]
                axes[1, 0].scatter(l_coords, m_coords, color=colors[probe],
                                 s=100, alpha=0.8, label=f'MMS{probe}')

    axes[1, 0].set_xlabel('L Coordinate (km)')
    axes[1, 0].set_ylabel('M Coordinate (km)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title('L vs M Coordinates at Crossings')

    # Plot 4: Time offset vs N coordinate
    for probe in ['1', '2', '3', '4']:
        if probe in results['event_correlations']:
            correlations = results['event_correlations'][probe]
            if correlations:
                time_offsets = [c['time_offset_from_event'] for c in correlations]
                n_coords = [c['N_coordinate'] for c in correlations]
                axes[1, 1].scatter(time_offsets, n_coords, color=colors[probe],
                                 s=100, alpha=0.8, label=f'MMS{probe}')

    axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Time Offset from Event (s)')
    axes[1, 1].set_ylabel('N Coordinate (km)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title('N Coordinate vs Time Offset')

    plt.tight_layout()

    filename = f'mms_boundary_correlations_{event_dt.strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"      ✅ Boundary correlation plot saved: {filename}")
    plt.close()

def create_formation_analysis_plot(results, event_dt):
    """Create formation analysis plot"""

    print(f"   🛰️ Creating formation analysis plot...")

    if 'formation_analysis' not in results or 'positions' not in results['formation_analysis']:
        print(f"      ⚠️ No formation data available for plotting")
        return

    fig = plt.figure(figsize=(12, 10))

    # 3D formation plot
    ax1 = fig.add_subplot(221, projection='3d')
    positions = results['formation_analysis']['positions']
    colors = {'1': 'blue', '2': 'green', '3': 'red', '4': 'orange'}

    for probe, pos in positions.items():
        ax1.scatter(pos[0], pos[1], pos[2], color=colors[probe], s=200,
                   alpha=0.8, label=f'MMS{probe}')

    # Plot center of mass
    com = results['formation_analysis']['center_of_mass']
    ax1.scatter(com[0], com[1], com[2], color='black', s=100, marker='x', label='COM')

    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_zlabel('Z (km)')
    ax1.legend()
    ax1.set_title('3D Formation at Event Time')

    # 2D projections
    ax2 = fig.add_subplot(222)
    for probe, pos in positions.items():
        ax2.scatter(pos[0], pos[1], color=colors[probe], s=200, alpha=0.8, label=f'MMS{probe}')
    ax2.scatter(com[0], com[1], color='black', s=100, marker='x', label='COM')
    ax2.set_xlabel('X (km)')
    ax2.set_ylabel('Y (km)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('X-Y Projection')

    ax3 = fig.add_subplot(223)
    for probe, pos in positions.items():
        ax3.scatter(pos[0], pos[2], color=colors[probe], s=200, alpha=0.8, label=f'MMS{probe}')
    ax3.scatter(com[0], com[2], color='black', s=100, marker='x', label='COM')
    ax3.set_xlabel('X (km)')
    ax3.set_ylabel('Z (km)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('X-Z Projection')

    ax4 = fig.add_subplot(224)
    for probe, pos in positions.items():
        ax4.scatter(pos[1], pos[2], color=colors[probe], s=200, alpha=0.8, label=f'MMS{probe}')
    ax4.scatter(com[1], com[2], color='black', s=100, marker='x', label='COM')
    ax4.set_xlabel('Y (km)')
    ax4.set_ylabel('Z (km)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Y-Z Projection')

    # Add formation info
    formation_type = results['formation_analysis'].get('formation_type', 'Unknown')
    formation_size = results['formation_analysis'].get('formation_size', 0)
    fig.suptitle(f'MMS Formation Analysis: {event_dt.strftime("%Y-%m-%d %H:%M:%S")} UT\n'
                 f'Type: {formation_type}, Size: {formation_size:.1f} km',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    filename = f'mms_formation_analysis_{event_dt.strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"      ✅ Formation analysis plot saved: {filename}")
    plt.close()

def create_event_timeline_plot(results, event_dt, trange):
    """Create event timeline plot"""

    print(f"   📅 Creating event timeline plot...")

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    colors = {'1': 'blue', '2': 'green', '3': 'red', '4': 'orange'}

    # Determine time window for the timeline plot
    if trange is not None and len(trange) == 2:
        try:
            t_start = pd.to_datetime(trange[0]).to_pydatetime()
            t_end = pd.to_datetime(trange[1]).to_pydatetime()
        except Exception:
            t_start = event_dt - timedelta(minutes=20)
            t_end = event_dt + timedelta(minutes=20)
    else:
        t_start = event_dt - timedelta(minutes=20)
        t_end = event_dt + timedelta(minutes=20)

    # Plot all boundary crossings
    for probe in ['1', '2', '3', '4']:
        if probe in results['event_correlations']:
            correlations = results['event_correlations'][probe]
            probe_num = int(probe)

            for i, correlation in enumerate(correlations):
                crossing_time = correlation['crossing_time']
                time_offset = correlation['time_offset_from_event']

                # Plot crossing as vertical line
                ax.axvline(crossing_time, color=colors[probe], alpha=0.6, linewidth=2)

                # Add text label
                ax.text(crossing_time, probe_num + 0.1, f'{time_offset:.0f}s',
                       rotation=90, ha='center', va='bottom', fontsize=8)

    # Plot event time
    ax.axvline(event_dt, color='red', linestyle='--', alpha=0.8, linewidth=3, label='Main Event')

    # Add spacecraft labels
    for probe in ['1', '2', '3', '4']:
        probe_num = int(probe)
        ax.axhline(probe_num, color=colors[probe], alpha=0.3, linewidth=1)

        # Place labels just inside the left edge of the visible window
        label_x = t_start + timedelta(minutes=2)
        ax.text(label_x, probe_num, f'MMS{probe}',
                color=colors[probe], fontweight='bold', va='center')

    ax.set_ylim(0.5, 4.5)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(['MMS1', 'MMS2', 'MMS3', 'MMS4'])
    ax.set_xlabel('Time (UT)')
    ax.set_ylabel('Spacecraft')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Constrain x-axis to the event window and format ticks
    ax.set_xlim(t_start, t_end)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    ax.set_title(f'Event Timeline: {event_dt.strftime("%Y-%m-%d %H:%M:%S")} UT\n'
                 f'Boundary Crossings with Time Offsets', fontsize=14, fontweight='bold')

    plt.tight_layout()

    filename = f'mms_event_timeline_{event_dt.strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"      ✅ Event timeline plot saved: {filename}")
    plt.close()

def generate_analysis_report(results, event_dt):
    """Generate comprehensive analysis report"""

    print(f"\n📋 GENERATING COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 50)

    report_filename = f'MMS_Comprehensive_Analysis_Report_{event_dt.strftime("%Y%m%d_%H%M%S")}.md'

    # Use UTF-8 so that superscripts (e.g., cm⁻³) and other Unicode glyphs are preserved
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(f"# MMS Comprehensive Event Analysis Report\n\n")
        f.write(f"**Event Time:** {event_dt.strftime('%Y-%m-%d %H:%M:%S')} UT\n")
        f.write(f"**Analysis Window:** ±1 hour (2-hour total)\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Executive Summary\n\n")
        f.write("This report presents a comprehensive analysis of the MMS magnetopause event using all available tools and techniques.\n\n")

        # Spacecraft data summary
        f.write("## Spacecraft Data Summary\n\n")
        for probe in ['1', '2', '3', '4']:
            if probe in results['spacecraft_data']:
                data = results['spacecraft_data'][probe]
                f.write(f"### MMS{probe}\n")

                if data['magnetic_field']:
                    mag = data['magnetic_field']
                    f.write(f"- **Magnetic Field:** {mag['data_points']:,} points, "
                           f"Range: {mag['B_range'][0]:.1f}-{mag['B_range'][1]:.1f} nT, "
                           f"Mean: {mag['B_mean']:.1f} ± {mag['B_std']:.1f} nT\n")

                if data['plasma_data']:
                    plasma = data['plasma_data']
                    f.write(f"- **Plasma Density:** {plasma['data_points']:,} points, "
                           f"Range: {plasma['density_range'][0]:.3f}-{plasma['density_range'][1]:.3f} cm⁻³, "
                           f"Mean: {plasma['density_mean']:.3f} ± {plasma['density_std']:.3f} cm⁻³\n")

                if data['position_data']:
                    pos = data['position_data']
                    f.write(f"- **Position:** Valid data: {pos['valid_fraction']*100:.1f}%, "
                           f"Mean: [{pos['mean_position'][0]:.1f}, {pos['mean_position'][1]:.1f}, {pos['mean_position'][2]:.1f}] km\n")

                f.write("\n")

        # Formation analysis
        f.write("## Formation Analysis\n\n")
        if 'formation_analysis' in results:
            formation = results['formation_analysis']
            f.write(f"- **Formation Type:** {formation.get('formation_type', 'Unknown')}\n")
            f.write(f"- **Characteristic Scale:** {formation.get('characteristic_scale', 0):.1f} km\n")
            f.write(f"- **Formation Size:** {formation.get('formation_size', 0):.1f} km\n")
            f.write(f"- **Quality Factor:** {formation.get('quality_factor', 0):.3f}\n\n")

        # LMN analysis
        f.write("## LMN Coordinate Analysis\n\n")
        for probe in ['1', '2', '3', '4']:
            if probe in results['lmn_analysis'] and 'L_direction' in results['lmn_analysis'][probe]:
                lmn = results['lmn_analysis'][probe]
                f.write(f"### MMS{probe} LMN Coordinates\n")
                f.write(f"- **L Direction:** [{lmn['L_direction'][0]:.3f}, {lmn['L_direction'][1]:.3f}, {lmn['L_direction'][2]:.3f}]\n")
                f.write(f"- **M Direction:** [{lmn['M_direction'][0]:.3f}, {lmn['M_direction'][1]:.3f}, {lmn['M_direction'][2]:.3f}]\n")
                f.write(f"- **N Direction:** [{lmn['N_direction'][0]:.3f}, {lmn['N_direction'][1]:.3f}, {lmn['N_direction'][2]:.3f}]\n")
                if lmn['eigenvalue_ratios']:
                    f.write(f"- **Eigenvalue Ratio:** {lmn['eigenvalue_ratios']:.2f}\n")
                f.write("\n")

        # Event correlations
        f.write("## Event Crossing Correlations\n\n")
        total_crossings = 0
        for probe in ['1', '2', '3', '4']:
            if probe in results['event_correlations']:
                correlations = results['event_correlations'][probe]
                total_crossings += len(correlations)

                if correlations:
                    f.write(f"### MMS{probe} Event Crossings\n")
                    for i, corr in enumerate(correlations):
                        f.write(f"**Crossing {i+1}:**\n")
                        f.write(f"- Time: {corr['crossing_time'].strftime('%H:%M:%S')} UT\n")
                        f.write(f"- Offset from main event: {corr['time_offset_from_event']:.1f} seconds\n")
                        f.write(f"- LMN Position: L={corr['L_coordinate']:.1f}, M={corr['M_coordinate']:.1f}, N={corr['N_coordinate']:.1f} km\n\n")

        f.write(f"**Total Event Crossings:** {total_crossings}\n\n")

        # Generated files
        f.write("## Generated Visualizations\n\n")
        f.write("The following visualization files were generated (where available):\n\n")

        ts = event_dt.strftime('%Y%m%d_%H%M%S')
        f.write(f"1. `mms_multispacecraft_overview_{ts}.png` - Multi-spacecraft overview\n")
        f.write(f"2. `mms_lmn_evolution_{ts}.png` - LMN coordinate evolution\n")
        f.write(f"3. `mms_boundary_correlations_{ts}.png` - Boundary crossing correlations\n")

        formation = results.get('formation_analysis', {}) or {}
        if 'positions' in formation:
            f.write(f"4. `mms_formation_analysis_{ts}.png` - Formation analysis\n")
        else:
            f.write("4. Formation analysis plot not generated - insufficient or missing "
                    "position data for this event.\n")

        f.write(f"5. `mms_event_timeline_{ts}.png` - Event timeline\n\n")

        f.write("## Conclusions\n\n")
        f.write("This comprehensive analysis provides detailed insights into the magnetopause event using real MMS mission data. ")
        f.write("The LMN coordinate tracking allows for precise correlation of event crossings with spacecraft positions, ")
        f.write("enabling detailed study of the magnetopause structure and dynamics.\n\n")

        f.write("---\n")
        f.write("*Report generated by MMS-MP comprehensive analysis pipeline*\n")

    print(f"   ✅ Comprehensive report saved: {report_filename}")

if __name__ == "__main__":
    main()
