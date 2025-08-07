#!/usr/bin/env python3
"""
Corrected Formation Analysis for 2019-01-27 Event
=================================================

This script directly accesses MEC data to determine the correct
spacecraft ordering, which should be 2-1-4-3 for this event.
"""

import numpy as np
from datetime import datetime
from mms_mp.data_loader import _load_state
from pytplot import get_data, data_quants
import warnings
warnings.filterwarnings('ignore')

def main():
    """Corrected formation analysis"""
    
    print("🎯 CORRECTED FORMATION ANALYSIS")
    print("=" * 40)
    print("Expected spacecraft order: 2-1-4-3")
    print()
    
    # Event parameters
    event_time = '2019-01-27/12:30:50'
    event_dt = datetime(2019, 1, 27, 12, 30, 50)
    trange = ['2019-01-27/12:25:00', '2019-01-27/12:35:00']
    
    print(f"📡 Loading MEC data for: {event_time}")
    print(f"   Time range: {trange[0]} to {trange[1]}")
    
    # Load MEC data directly for all spacecraft
    positions = {}
    velocities = {}
    
    for probe in ['1', '2', '3', '4']:
        print(f"\n🛰️ Loading MEC data for MMS{probe}:")
        
        try:
            # Load MEC data directly
            result = _load_state(trange, probe)
            
            # Check what MEC variables are available
            mec_vars = [v for v in data_quants.keys() if f'mms{probe}_mec' in v]
            print(f"   Available MEC variables: {len(mec_vars)}")
            
            # Try to get position data
            pos_var = f'mms{probe}_mec_r_gsm'
            vel_var = f'mms{probe}_mec_v_gsm'
            
            if pos_var in data_quants:
                try:
                    times, pos_data = get_data(pos_var)
                    print(f"   ✅ Position data: {len(times)} points")
                    
                    # Find time closest to event
                    event_timestamp = event_dt.timestamp()
                    
                    # Convert times to timestamps
                    if hasattr(times[0], 'timestamp'):
                        time_stamps = np.array([t.timestamp() for t in times])
                    else:
                        time_stamps = times
                    
                    # Find closest time index
                    closest_idx = np.argmin(np.abs(time_stamps - event_timestamp))
                    event_pos = pos_data[closest_idx]
                    
                    if not np.isnan(event_pos).any():
                        positions[probe] = event_pos
                        distance = np.linalg.norm(event_pos)
                        print(f"   📍 Position: [{event_pos[0]:.1f}, {event_pos[1]:.1f}, {event_pos[2]:.1f}] km")
                        print(f"   📏 Distance: {distance:.1f} km ({distance/6371:.2f} RE)")
                    else:
                        print(f"   ❌ Position data contains NaN")
                        
                except Exception as e:
                    print(f"   ❌ Error accessing position data: {e}")
            else:
                print(f"   ❌ Position variable {pos_var} not found")
            
            # Try to get velocity data
            if vel_var in data_quants:
                try:
                    times, vel_data = get_data(vel_var)
                    print(f"   ✅ Velocity data: {len(times)} points")
                    
                    # Find velocity at event time
                    closest_idx = np.argmin(np.abs(time_stamps - event_timestamp))
                    event_vel = vel_data[closest_idx]
                    
                    if not np.isnan(event_vel).any():
                        velocities[probe] = event_vel
                        speed = np.linalg.norm(event_vel)
                        print(f"   🚀 Velocity: [{event_vel[0]:.3f}, {event_vel[1]:.3f}, {event_vel[2]:.3f}] km/s")
                        print(f"   🏃 Speed: {speed:.3f} km/s")
                    else:
                        print(f"   ❌ Velocity data contains NaN")
                        
                except Exception as e:
                    print(f"   ❌ Error accessing velocity data: {e}")
            else:
                print(f"   ❌ Velocity variable {vel_var} not found")
                
        except Exception as e:
            print(f"   ❌ MEC loading failed: {e}")
    
    # Analyze spacecraft ordering
    print(f"\n🎯 SPACECRAFT ORDERING ANALYSIS:")
    print("=" * 40)
    
    if len(positions) >= 2:
        print(f"✅ Found valid positions for {len(positions)} spacecraft")
        print(f"   Spacecraft with positions: {list(positions.keys())}")
        
        # Calculate X-ordering (most relevant for magnetopause crossings)
        x_positions = {probe: positions[probe][0] for probe in positions.keys()}
        x_ordered = sorted(positions.keys(), key=lambda p: x_positions[p])
        
        print(f"\nX-GSM ordering (most negative to most positive):")
        for probe in x_ordered:
            print(f"   MMS{probe}: X = {x_positions[probe]:8.1f} km")
        
        print(f"\nCalculated order: {'-'.join(x_ordered)}")
        print(f"Expected order:   2-1-4-3")
        
        if x_ordered == ['2', '1', '4', '3']:
            print(f"🎉 SUCCESS! Calculated order matches expected order!")
        else:
            print(f"⚠️ Order difference detected")
            
            # Analyze the difference
            expected = ['2', '1', '4', '3']
            print(f"\nDifference analysis:")
            print(f"   Expected: {'-'.join(expected)}")
            print(f"   Got:      {'-'.join(x_ordered)}")
            
            if len(x_ordered) == 4:
                print(f"\nPosition comparison:")
                for i, (calc, exp) in enumerate(zip(x_ordered, expected)):
                    match = "✅" if calc == exp else "❌"
                    print(f"   Position {i+1}: {match} Got MMS{calc}, expected MMS{exp}")
                    if calc in x_positions and exp in x_positions:
                        print(f"      MMS{calc} X = {x_positions[calc]:.1f} km")
                        print(f"      MMS{exp} X = {x_positions[exp]:.1f} km")
            
            # Check if it's a coordinate system or time issue
            print(f"\n🔍 Possible explanations:")
            print("1. The expected order 2-1-4-3 might be for a different time")
            print("2. The expected order might be in a different coordinate system")
            print("3. The expected order might refer to a different type of ordering")
            print("4. The MEC data might have timing or coordinate issues")
        
        # Additional analysis
        if len(positions) == 4:
            print(f"\n📊 ADDITIONAL FORMATION ANALYSIS:")
            print("=" * 35)
            
            # Calculate formation size
            pos_array = np.array([positions[p] for p in ['1', '2', '3', '4']])
            center = np.mean(pos_array, axis=0)
            distances = [np.linalg.norm(positions[p] - center) for p in ['1', '2', '3', '4']]
            
            print(f"Formation center: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}] km")
            print(f"Formation size: {np.max(distances):.1f} km")
            
            # Calculate separations
            print(f"\nSpacecraft separations:")
            for i, p1 in enumerate(['1', '2', '3', '4']):
                for p2 in ['1', '2', '3', '4'][i+1:]:
                    sep = np.linalg.norm(positions[p1] - positions[p2])
                    print(f"   MMS{p1}-MMS{p2}: {sep:.1f} km")
    
    else:
        print(f"❌ Only found valid positions for {len(positions)} spacecraft")
        print(f"   Valid positions: {list(positions.keys())}")
        print("   Cannot perform formation analysis")
        
        if len(positions) == 0:
            print(f"\n🔧 TROUBLESHOOTING:")
            print("   The MEC data loading is still not working correctly.")
            print("   This suggests a deeper issue with the data access or timing.")

if __name__ == "__main__":
    main()
