# main_analysis.py  –  quick-look demo for the MMS magnetopause toolkit
# ============================================================

import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d

from mms_mp import (
    data_loader,    # CDF download / load
    resample,       # multi-var resample
    coords,         # LMN transform
    electric,       # E×B drift
    boundary,       # crossing detector
    motion,         # displacement integrator
    multispacecraft,
    visualize
)

# ──────────────────────────────────────────────────────────────
# 1. User settings
# ──────────────────────────────────────────────────────────────
TRANGE   = ['2019-01-27/12:20:00', '2019-01-27/12:40:00']
PROBES   = ['1', '2', '3', '4']
CADENCE  = '150ms'
THR_CFG  = boundary.DetectorCfg(he_in=0.25, he_out=0.15, BN_tol=1.0)
include_edp = True          # set False to skip E×B drift

# ──────────────────────────────────────────────────────────────
# 2. Load data
# ──────────────────────────────────────────────────────────────
evt = data_loader.load_event(
    TRANGE, probes=PROBES,
    include_edp=include_edp, include_ephem=True)

crossing_times, positions_gsm = {}, {}

for p in PROBES:
    print(f'\n=== MMS{p} ===')
    d = evt[p]

    # ── 2.1  resample to uniform clock ──────────────────────────
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
    }, cadence=CADENCE, method='linear')

    # ── 2.2  LMN triad (hybrid) ─────────────────────────────────
    mid_B_idx = len(d['B_gsm'][0]) // 2
    t_mid     = d['B_gsm'][0][mid_B_idx]
    B_slice   = d['B_gsm'][1][mid_B_idx-64:mid_B_idx+64, :3]

    interp_pos = interp1d(d['POS_gsm'][0], d['POS_gsm'][1],
                          axis=0, bounds_error=False,
                          fill_value='extrapolate')
    pos_mid = interp_pos(t_mid)
    lm      = coords.hybrid_lmn(B_slice, pos_gsm_km=pos_mid)

    B_lmn = lm.to_lmn(vars_grid['B'][:, :3])
    BN    = B_lmn[:, 2]

    # ── 2.3  boundary detection ─────────────────────────────────
    mask_all = good['He'] & good['B']
    layers   = boundary.detect_crossings_multi(
        t_grid, vars_grid['He'], BN,
        cfg=THR_CFG, good_mask=mask_all)
    xings    = boundary.extract_enter_exit(layers, t_grid)
    crossing_found = bool(xings)

    if crossing_found:
        t_entry = xings[0][0]
        crossing_times[p] = t_entry
        idx_pos = np.searchsorted(d['POS_gsm'][0], t_entry)
        positions_gsm[p] = d['POS_gsm'][1][idx_pos]

    # ── 2.4  velocities → V_N ───────────────────────────────────
    vN_i  = lm.to_lmn(vars_grid['Vi'])[:, 2]
    vN_e  = lm.to_lmn(vars_grid['Ve'])[:, 2]
    vN_he = lm.to_lmn(vars_grid['Vh'])[:, 2]

    if include_edp:
        t_exb, v_exb = electric.exb_velocity_sync(
            vars_grid['E'][:, 0], vars_grid['E'],
            vars_grid['B'][:, 0], vars_grid['B'],
            cadence=CADENCE)
        vN_exb = lm.to_lmn(v_exb)[:, 2]
        vN = electric.normal_velocity(
            v_bulk_lmn=np.column_stack([np.zeros_like(vN_he),
                                        np.zeros_like(vN_he),
                                        vN_he]),
            v_exb_lmn=np.column_stack([np.zeros_like(vN_exb),
                                       np.zeros_like(vN_exb),
                                       vN_exb]),
            strategy='prefer_exb')
    else:
        vN = vN_he

    if np.all(~np.isfinite(vN)):
        vN = vN_i         # use ion bulk V_N as a last resort

    # ── 2.5  integrate displacement ────────────────────────────
    disp = motion.integrate_disp(t_grid, vN, good_mask=mask_all)

    # ── 2.6  console summary ───────────────────────────────────
    if crossing_found:
        print(f'  Entry at {datetime.utcfromtimestamp(t_entry):%H:%M:%S}, '
              f'disp max {np.nanmax(disp.disp_km):.0f} km')
    else:
        print('  !!  No magnetopause entry detected in this interval')
        print(f'  disp max {np.nanmax(disp.disp_km):.0f} km')

    # ── 2.7  quick-look plot ───────────────────────────────────
    figure_title = (f'MMS{p} – crossing detected'
                    if crossing_found else
                    f'MMS{p} – no boundary crossing in window')
    visualize.summary_single(
        t_grid, B_lmn,
        vars_grid['Ni'], vars_grid['Ne'], vars_grid['He'],
        vN_i, vN_e, vN_he,
        layers=layers if crossing_found else [],
        title=figure_title)


# 3.  multi-spacecraft timing
if len(crossing_times) >= 2:
    n_hat, V_phase, sigV = multispacecraft.timing_normal(
        positions_gsm, crossing_times)
    print('\n>>> timing normal',
          '\n    n̂  =', np.round(n_hat, 3),
          '\n    Vph = %.1f ± %.1f km/s' % (V_phase, sigV))
else:
    print('not enough spacecraft with crossings for timing analysis')
