# main_analysis.py  –  high-level demo for the MMS magnetopause toolkit
# ============================================================
# Analyse one event, show quick-look plots, print a concise
# result summary.  NOT as full-featured as cli.py (which does
# CSV/JSON/PNG output), but great for notebooks & tinkering.
# ------------------------------------------------------------
import numpy as np
from datetime import datetime, timedelta

from mms_mp import (
    data_loader,        # download + load CDFs
    resample,           # align variables to uniform clock
    coords,             # LMN transformer
    electric,           # ExB drift
    quality,            # quality-flag handling
    boundary,           # multi-parameter crossing detector
    motion,             # integrate v_N -> displacement
    multispacecraft,    # normal + phase speed
    visualize           # publication plots
)

# ------------------------------------------------------------
# 1. Event time and spacecraft selection
# ------------------------------------------------------------
TRANGE   = ['2019-11-12/04:00:00', '2019-11-12/05:00:00']
PROBES   = ['1', '2', '3', '4']          # MMS1–4
CADENCE  = '150ms'                       # resample cadence
THR_CFG  = boundary.DetectorCfg(he_in=0.25, he_out=0.15, BN_tol=1.0)

# ------------------------------------------------------------
# 2. Load MMS data (FGM, FPI, HPCA, EDP, EPHEM)
# ------------------------------------------------------------
event = data_loader.load_event(TRANGE, probes=PROBES,
                               include_edp=True, include_ephem=True)

# Containers for multi-SC timing
crossing_times = {}
positions_gsm  = {}

for p in PROBES:
    print(f'\n=== MMS{p} ===')
    d = event[p]

    # --------------------------------------------------------
    # 3. Resample key variables onto common clock
    # --------------------------------------------------------
    t_grid, vars_grid, good = resample.merge_vars({
        'He' : (d['N_he'][0], d['N_he'][1]),           # He+ density
        'B'  : (d['B_gsm'][0], d['B_gsm'][1]),         # B-vector
        'E'  : (d['E_gse'][0], d['E_gse'][1]),         # electric field
        'Vh' : (d['V_he_gsm'][0], d['V_he_gsm'][1]),   # He+ bulk V
    }, cadence=CADENCE, method='linear')

    # --------------------------------------------------------
    # 4. Build LMN coordinate system (hybrid)
    # --------------------------------------------------------
    # Use a small B slice around mid-interval for MVA
    mid = len(d['B_gsm'][1]) // 2
    lm  = coords.hybrid_lmn(d['B_gsm'][1][mid-64:mid+64],
                            pos_gsm=d['POS_gsm'][1][mid])

    B_lmn = lm.to_lmn(vars_grid['B'])
    BN    = B_lmn[:, 2]

    # --------------------------------------------------------
    # 5. Boundary detection (He+ + B_N)
    # --------------------------------------------------------
    mask_all = good['He'] & good['B']
    layers = boundary.detect_crossings_multi(t_grid,
                                             vars_grid['He'],
                                             BN,
                                             cfg=THR_CFG,
                                             good_mask=mask_all)
    xings = boundary.extract_enter_exit(layers, t_grid)
    if not xings:
        print('  !!  No magnetosphere entry detected – skipping spacecraft')
        continue
    t_entry = xings[0][0]   # first entry epoch (sec)

    crossing_times[p] = t_entry
    idx_pos = np.searchsorted(d['POS_gsm'][0], t_entry)
    positions_gsm[p] = d['POS_gsm'][1][idx_pos]

    # --------------------------------------------------------
    # 6. Velocity (ExB vs He+ bulk)  → normal component
    # --------------------------------------------------------
    # ExB drift
    v_exb_time, v_exb_xyz = electric.exb_velocity_sync(
        vars_grid['E'][:,0], vars_grid['E'],
        vars_grid['B'][:,0], vars_grid['B'],
        cadence=CADENCE)
    v_exb_lmn = lm.to_lmn(v_exb_xyz)
    vN_exb = v_exb_lmn[:, 2]

    # He+ bulk in LMN
    v_he_lmn = lm.to_lmn(vars_grid['Vh'])
    vN_bulk  = v_he_lmn[:, 2]

    vN = electric.normal_velocity(
        v_bulk_lmn=np.column_stack([np.zeros_like(vN_bulk),
                                    np.zeros_like(vN_bulk),
                                    vN_bulk]),
        v_exb_lmn=np.column_stack([np.zeros_like(vN_exb),
                                   np.zeros_like(vN_exb),
                                   vN_exb]),
        strategy='prefer_exb')

    # --------------------------------------------------------
    # 7. Displacement integration
    # --------------------------------------------------------
    disp = motion.integrate_disp(t_grid, vN,
                                 good_mask=mask_all,
                                 scheme='trap')

    # Layer thickness
    thicks = {typ: abs(disp.disp_km[e] - disp.disp_km[s])
              for typ, s, e in layers}

    print(f'  Entry at {datetime.utcfromtimestamp(t_entry):%H:%M:%S}; '
          f'Layers: {thicks}')

    # --------------------------------------------------------
    # 8. Quick-look plots
    # --------------------------------------------------------
    visualize.summary_single(
        t_grid, B_lmn, vars_grid['He']*np.nan, vars_grid['He'],
        vN_bulk, vN_exb, layers=layers,
        title=f'MMS{p}  ({TRANGE[0][:10]})', show=True)

    visualize.plot_displacement(t_grid, disp.disp_km,
                                title=f'MMS{p} magnetopause displacement',
                                show=True)

# ------------------------------------------------------------
# 9. Multi-spacecraft timing   (if ≥2 SC had crossings)
# ------------------------------------------------------------
if len(crossing_times) >= 2:
    n_hat, V_phase, sigV = multispacecraft.timing_normal(
        positions_gsm, crossing_times)
    print('\n>>> Boundary normal & speed (timing method)')
    print('    n̂  =', np.round(n_hat, 3),
          '\n    V_ph= %.1f ± %.1f km/s' % (V_phase, sigV))
else:
    print('Not enough spacecraft with valid crossings for timing analysis.')
