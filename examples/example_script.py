#!/usr/bin/env python3
"""
Example script demonstrating MMS Magnetopause Analysis Toolkit

This script shows a complete workflow for analyzing magnetopause crossings
using MMS data. It can be run standalone without Jupyter.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import the MMS-MP toolkit
import mms_mp

def main():
    """Main analysis function"""
    print("🚀 MMS Magnetopause Analysis Example")
    print("=" * 50)
    
    # Configuration
    trange = ['2019-01-27/12:20:00', '2019-01-27/12:40:00']
    probes = ['1', '2', '3', '4']
    cadence = '150ms'
    
    print(f"Time range: {trange[0]} to {trange[1]}")
    print(f"Spacecraft: MMS{', MMS'.join(probes)}")
    print(f"Cadence: {cadence}")
    print()
    
    # Step 1: Load data
    print("📡 Loading MMS data...")
    try:
        evt = mms_mp.load_event(
            trange, 
            probes=probes,
            include_edp=True,  # Include electric field for E×B drift
            include_ephem=True
        )
        print("✓ Data loaded successfully")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return
    
    # Step 2: Process each spacecraft
    crossing_times = {}
    positions_gsm = {}
    
    for probe in probes:
        print(f"\n🛰️  Processing MMS{probe}...")
        
        try:
            # Get data for this probe
            d = evt[probe]
            
            # Resample to uniform grid
            t_grid, vars_grid, good = mms_mp.merge_vars({
                'Ni': (d['N_tot'][0], d['N_tot'][1]),
                'Ne': (d['N_e'][0], d['N_e'][1]),
                'He': (d['N_he'][0], d['N_he'][1]),
                'B': (d['B_gsm'][0], d['B_gsm'][1]),
                'Vi': (d['V_i_gse'][0], d['V_i_gse'][1]),
                'Ve': (d['V_e_gse'][0], d['V_e_gse'][1]),
                'Vh': (d['V_he_gsm'][0], d['V_he_gsm'][1]),
                'E': (d['E_gse'][0], d['E_gse'][1]),
            }, cadence=cadence, method='linear')
            
            # Coordinate transformation
            mid_idx = len(d['B_gsm'][0]) // 2
            B_slice = d['B_gsm'][1][mid_idx-64:mid_idx+64, :3]
            
            # Get position at middle time for LMN calculation
            from scipy.interpolate import interp1d
            t_mid = d['B_gsm'][0][mid_idx]
            interp_pos = interp1d(d['POS_gsm'][0], d['POS_gsm'][1],
                                axis=0, bounds_error=False,
                                fill_value='extrapolate')
            pos_mid = interp_pos(t_mid)
            
            # Hybrid LMN coordinate system
            lmn = mms_mp.hybrid_lmn(B_slice, pos_gsm_km=pos_mid)
            B_lmn = lmn.to_lmn(vars_grid['B'][:, :3])
            BN = B_lmn[:, 2]  # Normal component
            
            # Boundary detection
            cfg = mms_mp.DetectorCfg(he_in=0.25, he_out=0.15, BN_tol=1.0)
            mask_all = good['He'] & good['B']
            
            layers = mms_mp.detect_crossings_multi(
                t_grid, vars_grid['He'], BN,
                cfg=cfg, good_mask=mask_all
            )
            
            # Extract crossing times
            from mms_mp.boundary import extract_enter_exit
            xings = extract_enter_exit(layers, t_grid)
            
            if xings:
                t_entry = xings[0][0]
                crossing_times[probe] = t_entry
                
                # Get position at crossing
                idx_pos = np.searchsorted(d['POS_gsm'][0], t_entry)
                positions_gsm[probe] = d['POS_gsm'][1][idx_pos]
                
                print(f"   ✓ Boundary crossing detected at {datetime.utcfromtimestamp(t_entry):%H:%M:%S}")
            else:
                print(f"   ⚠️  No boundary crossing detected")
                
        except Exception as e:
            print(f"   ❌ Processing failed: {e}")
            continue
    
    # Step 3: Multi-spacecraft timing analysis
    if len(crossing_times) >= 2:
        print(f"\n🎯 Multi-spacecraft timing analysis...")
        print(f"   Using {len(crossing_times)} spacecraft")
        
        try:
            n_hat, V_phase, sigV = mms_mp.timing_normal(
                positions_gsm, crossing_times
            )
            
            print(f"   Boundary normal: [{n_hat[0]:.3f}, {n_hat[1]:.3f}, {n_hat[2]:.3f}]")
            print(f"   Phase velocity: {V_phase:.1f} ± {sigV:.1f} km/s")
            
        except Exception as e:
            print(f"   ❌ Timing analysis failed: {e}")
    else:
        print(f"\n⚠️  Not enough spacecraft for timing analysis ({len(crossing_times)} < 2)")
    
    # Step 4: Summary
    print(f"\n📊 Analysis Summary")
    print("=" * 30)
    print(f"Spacecraft processed: {len(probes)}")
    print(f"Crossings detected: {len(crossing_times)}")
    
    if crossing_times:
        times_str = [f"MMS{p}: {datetime.utcfromtimestamp(t):%H:%M:%S}" 
                    for p, t in crossing_times.items()]
        print("Crossing times:")
        for time_str in times_str:
            print(f"  {time_str}")
    
    print("\n🎉 Analysis complete!")
    print("\nFor interactive plots and more detailed analysis,")
    print("see the Jupyter notebooks in this directory.")


if __name__ == "__main__":
    main()
