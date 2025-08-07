#!/usr/bin/env python3
"""
Debug MEC Raw Data Access
=========================

This script directly accesses the raw MEC data from pytplot to see
what's actually being loaded before any processing.
"""

import numpy as np
from datetime import datetime
from mms_mp.data_loader import load_event
from pytplot import data_quants, get_data
import warnings
warnings.filterwarnings('ignore')

def main():
    """Debug raw MEC data access"""
    
    print("🔍 DEBUGGING RAW MEC DATA ACCESS")
    print("=" * 40)
    
    # Event parameters
    event_time = '2019-01-27/12:30:50'
    event_dt = datetime(2019, 1, 27, 12, 30, 50)
    trange = ['2019-01-27/12:25:00', '2019-01-27/12:35:00']
    
    print(f"📡 Loading MMS data for: {event_time}")
    print(f"   Time range: {trange[0]} to {trange[1]}")
    
    try:
        # Load data
        data = load_event(
            trange=trange,
            probes=['1', '2', '3', '4'],
            data_rate_fgm='brst',
            data_rate_fpi='brst'
        )
        
        print("✅ Data loaded successfully")
        
        # Check what's actually in pytplot
        print(f"\n🔍 RAW PYTPLOT VARIABLES:")
        print("=" * 30)
        
        all_vars = list(data_quants.keys())
        mec_vars = [var for var in all_vars if 'mec' in var.lower()]
        
        print(f"Total pytplot variables: {len(all_vars)}")
        print(f"MEC variables found: {len(mec_vars)}")
        
        for var in sorted(mec_vars):
            print(f"   {var}")
        
        # Check each MEC variable directly
        print(f"\n🔬 DIRECT MEC VARIABLE INSPECTION:")
        print("=" * 45)
        
        for probe in ['1', '2', '3', '4']:
            print(f"\nMMS{probe} MEC variables:")
            
            # Position variables
            pos_vars = [f'mms{probe}_mec_r_gsm', f'mms{probe}_mec_r_gse']
            vel_vars = [f'mms{probe}_mec_v_gsm', f'mms{probe}_mec_v_gse']
            
            for pos_var in pos_vars:
                if pos_var in data_quants:
                    try:
                        times, pos_data = get_data(pos_var)
                        print(f"   ✅ {pos_var}:")
                        print(f"      Times: {len(times)} points")
                        print(f"      Data shape: {pos_data.shape}")
                        print(f"      Time range: {times[0]} to {times[-1]}")
                        
                        # Check for NaN values
                        nan_count = np.isnan(pos_data).sum()
                        total_elements = pos_data.size
                        print(f"      NaN count: {nan_count}/{total_elements}")
                        
                        if nan_count == 0:
                            # Show some actual values
                            mid_idx = len(pos_data) // 2
                            mid_pos = pos_data[mid_idx]
                            print(f"      Middle position: [{mid_pos[0]:.1f}, {mid_pos[1]:.1f}, {mid_pos[2]:.1f}] km")
                            
                            # Check distance from Earth
                            distances = np.linalg.norm(pos_data, axis=1)
                            print(f"      Distance range: {distances.min():.1f} - {distances.max():.1f} km")
                            print(f"      Distance in RE: {distances.min()/6371:.2f} - {distances.max()/6371:.2f} RE")
                        else:
                            print(f"      ❌ Contains NaN values")
                            
                    except Exception as e:
                        print(f"   ❌ Error accessing {pos_var}: {e}")
                else:
                    print(f"   ❌ {pos_var} not found in pytplot")
            
            for vel_var in vel_vars:
                if vel_var in data_quants:
                    try:
                        times, vel_data = get_data(vel_var)
                        print(f"   ✅ {vel_var}:")
                        print(f"      Times: {len(times)} points")
                        print(f"      Data shape: {vel_data.shape}")
                        
                        # Check for NaN values
                        nan_count = np.isnan(vel_data).sum()
                        total_elements = vel_data.size
                        print(f"      NaN count: {nan_count}/{total_elements}")
                        
                        if nan_count == 0:
                            mid_idx = len(vel_data) // 2
                            mid_vel = vel_data[mid_idx]
                            print(f"      Middle velocity: [{mid_vel[0]:.3f}, {mid_vel[1]:.3f}, {mid_vel[2]:.3f}] km/s")
                        else:
                            print(f"      ❌ Contains NaN values")
                            
                    except Exception as e:
                        print(f"   ❌ Error accessing {vel_var}: {e}")
                else:
                    print(f"   ❌ {vel_var} not found in pytplot")
        
        # Now check what happens during the data processing
        print(f"\n🔧 DATA PROCESSING INVESTIGATION:")
        print("=" * 40)
        
        # Check the time ranges
        for probe in ['1', '2', '3', '4']:
            if probe in data:
                probe_data = data[probe]
                
                # Get B-field time range (burst mode)
                if 'B_gsm' in probe_data:
                    b_times, b_data = probe_data['B_gsm']
                    print(f"\nMMS{probe} B-field times:")
                    print(f"   Count: {len(b_times)}")
                    print(f"   Range: {b_times[0]} to {b_times[-1]}")
                    print(f"   Type: {type(b_times[0])}")
                
                # Get position time range (MEC)
                if 'POS_gsm' in probe_data:
                    pos_times, pos_data = probe_data['POS_gsm']
                    print(f"\nMMS{probe} Position times:")
                    print(f"   Count: {len(pos_times)}")
                    print(f"   Range: {pos_times[0]} to {pos_times[-1]}")
                    print(f"   Type: {type(pos_times[0])}")
                    print(f"   All NaN: {np.isnan(pos_data).all()}")
                    
                    # Check if times match
                    if len(b_times) == len(pos_times):
                        print(f"   ✅ Time arrays same length")
                        time_match = np.array_equal(b_times, pos_times)
                        print(f"   Times identical: {time_match}")
                    else:
                        print(f"   ⚠️ Time arrays different lengths: B={len(b_times)}, POS={len(pos_times)}")
        
        # Try to manually get the correct spacecraft positions
        print(f"\n🎯 MANUAL POSITION EXTRACTION:")
        print("=" * 35)
        
        working_positions = {}
        for probe in ['1', '2', '3', '4']:
            # Try to get raw MEC data directly
            mec_var = f'mms{probe}_mec_r_gsm'
            if mec_var in data_quants:
                try:
                    mec_times, mec_pos = get_data(mec_var)
                    
                    # Find the time closest to our event
                    event_timestamp = event_dt.timestamp()
                    
                    # Convert MEC times to timestamps for comparison
                    if hasattr(mec_times[0], 'timestamp'):
                        mec_timestamps = np.array([t.timestamp() for t in mec_times])
                    else:
                        # Handle different time formats
                        mec_timestamps = mec_times
                    
                    # Find closest time index
                    closest_idx = np.argmin(np.abs(mec_timestamps - event_timestamp))
                    event_pos = mec_pos[closest_idx]
                    
                    if not np.isnan(event_pos).any():
                        working_positions[probe] = event_pos
                        print(f"   ✅ MMS{probe}: [{event_pos[0]:.1f}, {event_pos[1]:.1f}, {event_pos[2]:.1f}] km")
                    else:
                        print(f"   ❌ MMS{probe}: Position is NaN at event time")
                        
                except Exception as e:
                    print(f"   ❌ MMS{probe}: Error extracting position: {e}")
            else:
                print(f"   ❌ MMS{probe}: No MEC data found")
        
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
            
            if x_ordered != ['2', '1', '4', '3']:
                print(f"\n🤔 ANALYSIS:")
                print("The raw MEC data gives a different order than expected.")
                print("This could mean:")
                print("1. The expected order 2-1-4-3 is for a different time")
                print("2. The expected order is in a different coordinate system")
                print("3. The expected order refers to a different type of ordering")
                
        else:
            print(f"\n❌ Could not extract working positions for all spacecraft")
            print(f"   Found positions for: {list(working_positions.keys())}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
