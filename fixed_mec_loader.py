#!/usr/bin/env python3
"""
Fixed MEC Data Loader
====================

This script fixes the MEC data persistence issue by capturing
the data immediately after loading each spacecraft.
"""

import numpy as np
from datetime import datetime
from pyspedas.projects import mms
from pytplot import data_quants, get_data
import warnings
warnings.filterwarnings('ignore')

def load_mec_data_fixed(trange, probes):
    """Load MEC data with immediate capture to avoid pytplot persistence issues"""
    
    print("🔧 FIXED MEC DATA LOADER")
    print("=" * 30)
    
    all_positions = {}
    all_velocities = {}
    
    for probe in probes:
        print(f"\n🛰️ Loading MEC data for MMS{probe}:")
        
        try:
            # Load MEC data for this spacecraft only
            result = mms.mms_load_mec(
                trange=trange,
                probe=probe,
                data_rate='srvy',
                level='l2',
                datatype='epht89q',
                time_clip=True
            )
            
            # IMMEDIATELY capture the data before it can be overwritten
            pos_var = f'mms{probe}_mec_r_gsm'
            vel_var = f'mms{probe}_mec_v_gsm'
            
            if pos_var in data_quants and vel_var in data_quants:
                # Get the data immediately
                times_pos, pos_data = get_data(pos_var)
                times_vel, vel_data = get_data(vel_var)
                
                print(f"   ✅ Captured position data: {len(times_pos)} points")
                print(f"   ✅ Captured velocity data: {len(times_vel)} points")
                
                # Store the data in our own dictionaries
                all_positions[probe] = {
                    'times': times_pos,
                    'data': pos_data
                }
                all_velocities[probe] = {
                    'times': times_vel,
                    'data': vel_data
                }
                
                # Verify data quality
                nan_count_pos = np.isnan(pos_data).sum()
                nan_count_vel = np.isnan(vel_data).sum()
                
                print(f"   📊 Position NaN count: {nan_count_pos}/{pos_data.size}")
                print(f"   📊 Velocity NaN count: {nan_count_vel}/{vel_data.size}")
                
                if nan_count_pos == 0 and nan_count_vel == 0:
                    mid_idx = len(pos_data) // 2
                    mid_pos = pos_data[mid_idx]
                    mid_vel = vel_data[mid_idx]
                    
                    print(f"   📍 Sample position: [{mid_pos[0]:.1f}, {mid_pos[1]:.1f}, {mid_pos[2]:.1f}] km")
                    print(f"   🚀 Sample velocity: [{mid_vel[0]:.3f}, {mid_vel[1]:.3f}, {mid_vel[2]:.3f}] km/s")
                
            else:
                print(f"   ❌ MEC variables not found in pytplot")
                print(f"      Looking for: {pos_var}, {vel_var}")
                available_vars = [v for v in data_quants.keys() if f'mms{probe}' in v]
                print(f"      Available: {available_vars[:5]}...")
                
        except Exception as e:
            print(f"   ❌ Error loading MEC data: {e}")
    
    return all_positions, all_velocities

def calculate_spacecraft_ordering(positions, event_dt):
    """Calculate spacecraft ordering at event time"""
    
    print(f"\n📊 SPACECRAFT ORDERING ANALYSIS")
    print("=" * 40)
    
    if len(positions) < 4:
        print(f"❌ Insufficient position data: {len(positions)} spacecraft")
        return None
    
    # Find positions at event time
    event_positions = {}
    
    for probe in ['1', '2', '3', '4']:
        if probe in positions:
            times = positions[probe]['times']
            pos_data = positions[probe]['data']
            
            # Convert event time to timestamp
            event_timestamp = event_dt.timestamp()
            
            # Convert times to timestamps
            if hasattr(times[0], 'timestamp'):
                time_stamps = np.array([t.timestamp() for t in times])
            else:
                time_stamps = times
            
            # Find closest time index
            closest_idx = np.argmin(np.abs(time_stamps - event_timestamp))
            event_pos = pos_data[closest_idx]
            
            event_positions[probe] = event_pos
            
            distance = np.linalg.norm(event_pos)
            print(f"   MMS{probe}: [{event_pos[0]:.1f}, {event_pos[1]:.1f}, {event_pos[2]:.1f}] km")
            print(f"           Distance: {distance:.1f} km ({distance/6371:.2f} RE)")
    
    # Calculate X-GSM ordering
    if len(event_positions) == 4:
        x_positions = {probe: event_positions[probe][0] for probe in event_positions.keys()}
        x_ordered = sorted(event_positions.keys(), key=lambda p: x_positions[p])
        
        print(f"\n🎯 SPACECRAFT ORDERING:")
        print(f"   X-GSM order: {'-'.join(x_ordered)}")
        print(f"   Expected:    2-1-4-3")
        
        if x_ordered == ['2', '1', '4', '3']:
            print(f"   ✅ CORRECT ORDERING CONFIRMED!")
        else:
            print(f"   ⚠️ Unexpected ordering")
        
        # Calculate formation parameters
        pos_array = np.array([event_positions[p] for p in ['1', '2', '3', '4']])
        center = np.mean(pos_array, axis=0)
        distances = [np.linalg.norm(event_positions[p] - center) for p in ['1', '2', '3', '4']]
        
        print(f"\n📐 FORMATION PARAMETERS:")
        print(f"   Center: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}] km")
        print(f"   Center distance: {np.linalg.norm(center):.1f} km ({np.linalg.norm(center)/6371:.2f} RE)")
        print(f"   Formation size: {np.max(distances):.1f} km")
        
        # Inter-spacecraft distances
        print(f"\n📏 INTER-SPACECRAFT DISTANCES:")
        for i, p1 in enumerate(['1', '2', '3', '4']):
            for p2 in ['1', '2', '3', '4'][i+1:]:
                dist = np.linalg.norm(event_positions[p1] - event_positions[p2])
                print(f"   MMS{p1}-MMS{p2}: {dist:.1f} km")
        
        return event_positions
    
    else:
        print(f"❌ Incomplete position data: {len(event_positions)} spacecraft")
        return None

def main():
    """Test the fixed MEC loader"""
    
    # Event parameters
    event_time = '2019-01-27/12:30:50'
    event_dt = datetime(2019, 1, 27, 12, 30, 50)
    trange = ['2019-01-27/12:25:00', '2019-01-27/12:35:00']
    
    print(f"📡 Testing fixed MEC loader for: {event_time}")
    print(f"   Time range: {trange[0]} to {trange[1]}")
    
    # Load MEC data with immediate capture
    positions, velocities = load_mec_data_fixed(trange, ['1', '2', '3', '4'])
    
    # Analyze spacecraft ordering
    event_positions = calculate_spacecraft_ordering(positions, event_dt)
    
    if event_positions:
        print(f"\n🎉 SUCCESS! MEC data captured and analyzed")
        print(f"   All spacecraft positions available at event time")
        print(f"   Ready for comprehensive visualizations")
    else:
        print(f"\n❌ FAILED to capture complete MEC data")

if __name__ == "__main__":
    main()
