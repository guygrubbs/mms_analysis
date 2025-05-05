# mms_mp/cli.py
# ------------------------------------------------------------
# Command-line interface for the MMS-Magnetopause toolkit
# ------------------------------------------------------------
# Run from shell:
#   python -m mms_mp.cli --start 2019-11-12T04:00 --end 2019-11-12T05:00 \
#       --probes 1 2 3 4 --plot
#
# Produces:
#   • STDOUT summary (crossing times, normal, phase speed, layer thickness)
#   • Optional PNGs in ./figures/
#   • Optional CSV results in ./results/
# ------------------------------------------------------------
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Local imports
from . import (data_loader, resample, coords, boundary,
               electric, motion, multispacecraft, visualize)

# ------------------------------------------------------------
# CLI helpers
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MMS Magnetopause Auto-Analysis")
    p.add_argument('--start', required=True,
                   help='UTC start ISO8601 (e.g. 2019-11-12T04:00)')
    p.add_argument('--end',   required=True,
                   help='UTC end   ISO8601 (e.g. 2019-11-12T05:00)')
    p.add_argument('--probes', nargs='+', default=['1', '2', '3', '4'],
                   choices=['1', '2', '3', '4'],
                   help='Spacecraft to include (default all)')
    p.add_argument('--cadence', default='150ms',
                   help='Uniform resample cadence (e.g. 100ms, 1s)')
    p.add_argument('--plot', action='store_true',
                   help='Generate quick-look PNGs')
    p.add_argument('--outdir', default='results',
                   help='Directory for figures & CSV/JSON output')
    return p.parse_args()


def main():
    args = parse_args()
    trange = [args.start, args.end]
    Path(args.outdir, 'figures').mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print('[1] Downloading / loading MMS data …')
    evt = data_loader.load_event(trange, probes=args.probes,
                                 include_edp=True, include_ephem=True)

    # Containers
    crossings = {}
    positions = {}
    layer_results = {}

    # ---- Per-probe pipeline ----
    for p in args.probes:
        print(f'  → Processing MMS{p}')
        data = evt[p]
        # 1) Resample B_N & He density for detector
        t_grid, vars_grid, good = resample.merge_vars({
            'He': (data['N_he'][0], data['N_he'][1]),
            'B':  (data['B_gsm'][0], data['B_gsm'][1])
        }, cadence=args.cadence, method='linear')

        # Build LMN using hybrid (need 1-min slice around mid-interval)
        mid = len(data['B_gsm'][0]) // 2
        lm   = coords.hybrid_lmn(data['B_gsm'][1][mid-64:mid+64],
                                 pos_gsm=data['POS_gsm'][1][mid])

        B_lmn = lm.to_lmn(data['B_gsm'][1])
        BN    = B_lmn[:, 2]

        # Boundary detection
        layers = boundary.detect_crossings_multi(
            t_grid, vars_grid['He'], BN[np.searchsorted(
                 data['B_gsm'][0], t_grid)], good_mask=good['He'])
        xings = boundary.extract_enter_exit(layers, t_grid)
        crossings[p] = xings[0][0] if xings else np.nan  # take first entry time

        # Magnetopause motion
        # Build velocity arrays (He+ bulk and ExB)
        vN_bulk = motion.normal_velocity(
            data['V_he_gsm'][1], lm.R)  # using He⁺ bulk
        t_v     = data['V_he_gsm'][0]
        t_exb, v_exb = electric.exb_velocity_sync(
            data['E_gse'][0], data['E_gse'][1],
            data['B_gsm'][0], data['B_gsm'][1])
        vN_exb = motion.normal_velocity(v_exb, lm.R)
        # Interpolate v_exb on v_bulk clock
        vN_exb = np.interp(t_v, t_exb, vN_exb)

        vN = electric.normal_velocity(v_bulk_lmn=np.column_stack([np.zeros_like(vN_bulk),
                                                                  np.zeros_like(vN_bulk),
                                                                  vN_bulk]),
                                      v_exb_lmn=np.column_stack([np.zeros_like(vN_exb),
                                                                 np.zeros_like(vN_exb),
                                                                 vN_exb]),
                                      strategy='prefer_exb')
        disp = motion.integrate_disp(t_v, vN)

        # Layer thickness
        layer_thick = {}
        for typ, i1, i2 in layers:
            layer_thick[typ] = abs(disp.disp_km[i2] - disp.disp_km[i1])
        layer_results[p] = layer_thick

        # Position at crossing (for timing analysis)
        idx_pos = np.searchsorted(data['POS_gsm'][0], crossings[p])
        positions[p] = data['POS_gsm'][1][idx_pos]  # km GSM

        # ---- Plots ----
        if args.plot:
            visualize.summary_single(t_grid, lm.to_lmn(data['B_gsm'][1]),
                                     vars_grid['He']*0+np.nan,  # skip N_tot quick-look
                                     vars_grid['He'], vN_bulk, vN_exb,
                                     layers=layers,
                                     title=f'MMS{p}')
            visualize.plot_displacement(t_v, disp.disp_km,
                                        sigma=disp.sigma_km,
                                        title=f'MMS{p} displacement')

    # ---- Multi-SC timing ----
    print('[2] Multi-spacecraft boundary normal …')
    n_hat, V_phase, sigV = multispacecraft.timing_normal(positions, crossings)
    print(f'   n̂ = {n_hat} ,  V_phase = {V_phase:.1f} ± {sigV:.1f} km/s')

    # ---- Save outputs ----
    out = {
        'crossings_sec': crossings,
        'positions_km':  {k: v.tolist() for k, v in positions.items()},
        'boundary_normal': n_hat.tolist(),
        'phase_speed_km_s': V_phase,
        'phase_speed_sigma': sigV,
        'layer_thickness_km': layer_results
    }
    json_path = Path(args.outdir, 'mp_analysis.json')
    json_path.write_text(json.dumps(out, indent=2))
    print(f'[✓] Saved JSON → {json_path}')

    # CSV table of layers
    rows = []
    for p, lt in layer_results.items():
        for typ, val in lt.items():
            rows.append({'probe': p, 'layer': typ, 'thickness_km': val})
    df = pd.DataFrame(rows)
    csv_path = Path(args.outdir, 'layer_thickness.csv')
    df.to_csv(csv_path, index=False)
    print(f'[✓] Saved CSV  → {csv_path}')


if __name__ == '__main__':
    main()
