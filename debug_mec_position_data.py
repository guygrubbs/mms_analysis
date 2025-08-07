#!/usr/bin/env python3
"""
Debug MEC Position Data Loading
===============================

This script investigates why the MEC position data is returning NaN values
instead of real spacecraft positions for the 2019-01-27 event.
"""

import numpy as np
from datetime import datetime
from mms_mp.data_loader import load_event
import warnings
warnings.filterwarnings('ignore')

def main():
    """Debug MEC position data loading"""
    
    print("🔍 DEBUGGING MEC POSITION DATA LOADING")
    print("=" * 50)
    
    # Event parameters
    event_time = '2019-01-27/12:30:50'
    event_dt = datetime(2019, 1, 27, 12, 30, 50)
    trange = ['2019-01-27/12:25:00', '2019-01-27/12:35:00']
    
    print(f"📡 Loading MMS data for: {event_time}")
    print(f"   Time range: {trange[0]} to {trange[1]}")
    
    try:
        # Load data with detailed inspection
        data = load_event(
            trange=trange,
            probes=['1', '2', '3', '4'],
            data_rate_fgm='brst',
            data_rate_fpi='brst'
        )
        
        print("✅ Data loaded successfully")
        print("\n🔍 DETAILED MEC DATA INSPECTION:")
        print("=" * 40)
        
        for probe in ['1', '2', '3', '4']:
            if probe in data:
                probe_data = data[probe]
                print(f"\nMMS{probe} available variables:")
                for key in sorted(probe_data.keys()):
                    print(f"   {key}")
                
                # Check each possible position variable
                position_vars = ['POS_gsm', 'pos_gsm', 'R_gsm', 'mms1_mec_r_gsm', 
                               f'mms{probe}_mec_r_gsm', 'r_gsm', 'position_gsm']
                
                print(f"\nMMS{probe} position variable inspection:")
                for pos_var in position_vars:
                    if pos_var in probe_data:
                        times, pos_data = probe_data[pos_var]
                        print(f"   ✅ Found {pos_var}:")
                        print(f"      Times shape: {np.array(times).shape}")
                        print(f"      Data shape: {np.array(pos_data).shape}")
                        print(f"      Data type: {type(pos_data)}")
                        
                        if len(pos_data) > 0:
                            # Check first few values
                            first_pos = pos_data[0] if len(pos_data) > 0 else None
                            mid_pos = pos_data[len(pos_data)//2] if len(pos_data) > 0 else None
                            last_pos = pos_data[-1] if len(pos_data) > 0 else None
                            
                            print(f"      First position: {first_pos}")
                            print(f"      Middle position: {mid_pos}")
                            print(f"      Last position: {last_pos}")
                            
                            # Check for NaN values
                            if isinstance(pos_data, np.ndarray):
                                nan_count = np.isnan(pos_data).sum()
                                total_elements = pos_data.size
                                print(f"      NaN count: {nan_count}/{total_elements}")
                                
                                if nan_count == 0:
                                    print(f"      ✅ No NaN values - data looks good!")
                                    # Calculate distance from Earth
                                    distances = np.linalg.norm(pos_data, axis=1)
                                    print(f"      Distance range: {distances.min():.1f} - {distances.max():.1f} km")
                                    print(f"      Distance in RE: {distances.min()/6371:.2f} - {distances.max()/6371:.2f} RE")
                                else:
                                    print(f"      ❌ Contains {nan_count} NaN values")
                        else:
                            print(f"      ❌ Empty data array")
                    else:
                        print(f"   ❌ {pos_var} not found")
                
                # Check velocity variables too
                velocity_vars = ['VEL_gsm', 'vel_gsm', 'V_gsm', f'mms{probe}_mec_v_gsm', 'v_gsm']
                
                print(f"\nMMS{probe} velocity variable inspection:")
                for vel_var in velocity_vars:
                    if vel_var in probe_data:
                        times, vel_data = probe_data[vel_var]
                        print(f"   ✅ Found {vel_var}:")
                        print(f"      Times shape: {np.array(times).shape}")
                        print(f"      Data shape: {np.array(vel_data).shape}")
                        
                        if len(vel_data) > 0:
                            mid_vel = vel_data[len(vel_data)//2] if len(vel_data) > 0 else None
                            print(f"      Middle velocity: {mid_vel}")
                            
                            if isinstance(vel_data, np.ndarray):
                                nan_count = np.isnan(vel_data).sum()
                                total_elements = vel_data.size
                                print(f"      NaN count: {nan_count}/{total_elements}")
                        else:
                            print(f"      ❌ Empty data array")
                    else:
                        print(f"   ❌ {vel_var} not found")
            else:
                print(f"\n❌ No data for MMS{probe}")
        
        # Try to find any working position data
        print(f"\n🎯 SEARCHING FOR WORKING POSITION DATA:")
        print("=" * 45)
        
        working_positions = {}
        for probe in ['1', '2', '3', '4']:
            if probe in data:
                probe_data = data[probe]
                
                # Try all possible position variables
                for key in probe_data.keys():
                    if any(pos_term in key.lower() for pos_term in ['pos', 'r_', 'position']):
                        times, pos_data = probe_data[key]
                        if len(pos_data) > 0 and isinstance(pos_data, np.ndarray):
                            if not np.isnan(pos_data).any():
                                mid_pos = pos_data[len(pos_data)//2]
                                working_positions[probe] = mid_pos
                                print(f"   ✅ MMS{probe} working position from {key}: [{mid_pos[0]:.1f}, {mid_pos[1]:.1f}, {mid_pos[2]:.1f}] km")
                                break
                
                if probe not in working_positions:
                    print(f"   ❌ No working position data found for MMS{probe}")
        
        if len(working_positions) == 4:
            print(f"\n🎉 SUCCESS! Found working positions for all spacecraft")
            
            # Calculate the correct ordering
            x_positions = {probe: working_positions[probe][0] for probe in ['1', '2', '3', '4']}
            x_ordered = sorted(['1', '2', '3', '4'], key=lambda p: x_positions[p])
            
            print(f"\nX-GSM ordering (most negative to most positive):")
            for probe in x_ordered:
                print(f"   MMS{probe}: X = {x_positions[probe]:8.1f} km")
            
            print(f"\nCalculated order: {'-'.join(x_ordered)}")
            print(f"Expected order:   2-1-4-3")
            print(f"Match: {'✅ YES' if x_ordered == ['2', '1', '4', '3'] else '❌ NO'}")
            
        else:
            print(f"\n❌ Could not find working position data for all spacecraft")
            print(f"   Found positions for: {list(working_positions.keys())}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
