# mms_mp/cli.py
# ------------------------------------------------------------
# Command-line interface for the MMS-Magnetopause toolkit
#   $ python -m mms_mp.cli --start 2019-11-12T04:00 \
#         --end 2019-11-12T05:00 --probes 1 2 3 4 \
#         --cadence 150ms --plot --edp
# ------------------------------------------------------------
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import List
from scipy.interpolate import interp1d

# Local imports
from . import (data_loader, resample, coords, boundary,
               electric, motion, multispacecraft, visualize)

# ------------------------------------------------------------------
# CLI helpers
# ------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MMS magnetopause auto-analysis")
    p.add_argument('--start', required=True,
                   help='UTC start ISO8601 (e.g. 2019-11-12T04:00)')
    p.add_argument('--end', required=True,
                   help='UTC end   ISO8601 (e.g. 2019-11-12T05:00)')
    p.add_argument('--probes', nargs='+', default=['1', '2', '3', '4'],
                   choices=['1', '2', '3', '4'],
                   help='Spacecraft to include (default all)')
    p.add_argument('--cadence', default='150ms',
                   help='Uniform resample cadence (e.g. 100ms, 1s)')
    p.add_argument('--edp', action='store_true',
                   help='Load EDP and compute E×B drift (default off)')
    p.add_argument('--plot', action='store_true',
                   help='Quick-look PNGs')
    p.add_argument('--outdir', default='results',
                   help='Directory for JSON/CSV/figures')
    return p.parse_args()


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    trange = [args.start, args.end]
    out_dir = Path(args.outdir)
    (out_dir / 'figures').mkdir(parents=True, exist_ok=True)

    include_edp = args.edp

    # 1 ── Load MMS data
    print('[1] Downloading/loading MMS data …')
    evt = data_loader.load_event(
        trange, probes=args.probes,
        include_edp=include_edp, include_ephem=True)

    crossings, positions, layer_results = {}, {}, {}

    # 2 ── Loop spacecraft
    for p in args.probes:
        print(f'  → Processing MMS{p}')
        d = evt[p]

        # 2.1 ─ Resample to common cadence
        t_grid, vars_grid, good = resample.merge_vars({
            'Ni': (d['N_tot'][0],  d['N_tot'][1]),
            'Ne': (d['N_e'][0],    d['N_e'][1]),
            'He': (d['N_he'][0],   d['N_he'][1]),
            'B' : (d['B_gsm'][0],  d['B_gsm'][1]),
            'Vi': (d['V_i_gse'][0], d['V_i_gse'][1]),
            'Ve': (d['V_e_gse'][0], d['V_e_gse'][1]),
            'Vh': (d['V_he_gsm'][0], d['V_he_gsm'][1]),
            'E' : (d['E_gse'][0],  d['E_gse'][1]) if include_edp else
                   (d['B_gsm'][0], np.full_like(d['B_gsm'][1][:, 0], np.nan)),
        }, cadence=args.cadence, method='linear')

        # 2.2 ─ LMN transform
        # Build LMN (B burst vs. POS 1-min → interpolate)
        mid_B = len(d['B_gsm'][0]) // 2
        t_mid = d['B_gsm'][0][mid_B]
        B_seg = d['B_gsm'][1][mid_B-64:mid_B+64]
        interp_pos = interp1d(data['POS_gsm'][0], data['POS_gsm'][1],
                              axis=0, bounds_error=False,
                              fill_value='extrapolate')
        pos_mid = interp_pos(t_mid)
        lm = coords.hybrid_lmn(B_seg, pos_gsm=pos_mid)

        B_lmn = lm.to_lmn(d['B_gsm'][1])
        BN = B_lmn[:, 2]

        # 2.3 ─ Boundary detection
        layers = boundary.detect_crossings_multi(
            t_grid, vars_grid['He'], BN[np.searchsorted(d['B_gsm'][0], t_grid)],
            good_mask=good['He'])
        xings = boundary.extract_enter_exit(layers, t_grid)
        crossings[p] = xings[0][0] if xings else np.nan

        # 2.4 ─ Normal velocities
        v_i_lmn = lm.to_lmn(d['V_i_gse'][1]); vN_i = v_i_lmn[:, 2]
        v_e_lmn = lm.to_lmn(d['V_e_gse'][1]); vN_e = v_e_lmn[:, 2]
        v_h_lmn = lm.to_lmn(d['V_he_gsm'][1]); vN_he = v_h_lmn[:, 2]

        if include_edp:
            t_exb, v_exb = electric.exb_velocity_sync(
                d['E_gse'][0], d['E_gse'][1],
                d['B_gsm'][0], d['B_gsm'][1])
            vN_exb = motion.normal_velocity(v_exb, lm.R)
            vN_exb = np.interp(d['V_he_gsm'][0], t_exb, vN_exb)
            # blend ExB + He+
            vN = electric.normal_velocity(
                v_bulk_lmn=np.column_stack([np.zeros_like(vN_he),
                                            np.zeros_like(vN_he),
                                            vN_he]),
                v_exb_lmn=np.column_stack([np.zeros_like(vN_exb),
                                           np.zeros_like(vN_exb),
                                           vN_exb]),
                strategy='prefer_exb')
        else:
            vN = vN_he   # fall back to cold-ion bulk

        disp = motion.integrate_disp(d['V_he_gsm'][0], vN)

        # 2.5 ─ Layer thickness
        layer_results[p] = {typ: abs(disp.disp_km[e] - disp.disp_km[s])
                            for typ, s, e in layers}

        # 2.6 ─ Position at crossing
        idx_pos = np.searchsorted(d['POS_gsm'][0], crossings[p])
        positions[p] = d['POS_gsm'][1][idx_pos]

        # 2.7 ─ Quick-look plots
        if args.plot:
            visualize.summary_single(
                t_grid, B_lmn,
                vars_grid['Ni'], vars_grid['Ne'], vars_grid['He'],
                vN_i, vN_e, vN_he,
                layers=layers, title=f'MMS{p}')
            visualize.plot_displacement(
                d['V_he_gsm'][0], disp.disp_km,
                sigma=disp.sigma_km,
                title=f'MMS{p} displacement')

    # 3 ── Multi-SC timing
    print('[2] Multi-spacecraft boundary normal …')
    n_hat, V_phase, sigV = multispacecraft.timing_normal(positions, crossings)
    print(f'    n̂ = {n_hat}   V_ph = {V_phase:.1f} ± {sigV:.1f} km/s')

    # 4 ── Save JSON / CSV
    out = {
        'crossings_sec': crossings,
        'positions_km': {k: v.tolist() for k, v in positions.items()},
        'boundary_normal': n_hat.tolist(),
        'phase_speed_km_s': V_phase,
        'phase_speed_sigma': sigV,
        'layer_thickness_km': layer_results
    }
    json_path = out_dir / 'mp_analysis.json'
    json_path.write_text(json.dumps(out, indent=2))
    print(f'[✓] JSON  → {json_path}')

    rows = [{'probe': p, 'layer': typ, 'thickness_km': val}
            for p, lt in layer_results.items() for typ, val in lt.items()]
    pd.DataFrame(rows).to_csv(out_dir / 'layer_thickness.csv', index=False)
    print(f'[✓] CSV   → {out_dir / "layer_thickness.csv"}')


if __name__ == '__main__':
    main()
