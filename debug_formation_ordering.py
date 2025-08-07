#!/usr/bin/env python3
"""
Debug Formation Ordering for 2019-01-27 Event
==============================================

This script specifically investigates why the formation calculation 
is not showing the correct spacecraft order (2-1-4-3) for the 
2019-01-27 magnetopause crossing event.
"""

import numpy as np
from datetime import datetime
from mms_mp.data_loader import load_event
from mms_mp.formation_detection import detect_formation_type
from mms_mp.ephemeris import EphemerisManager
import warnings
warnings.filterwarnings('ignore')

def main():
    """Debug the formation ordering issue"""
    
    print("🔍 DEBUGGING FORMATION ORDERING FOR 2019-01-27 EVENT")
    print("=" * 60)
    print("Known correct order: 2-1-4-3")
    print("Investigating why our calculation differs...")
    print()
    
    # Event parameters
    event_time = '2019-01-27/12:30:50'
    event_dt = datetime(2019, 1, 27, 12, 30, 50)
    trange = ['2019-01-27/12:25:00', '2019-01-27/12:35:00']  # Short window around event
    
    print(f"📡 Loading MMS data for: {event_time}")
    print(f"   Time range: {trange[0]} to {trange[1]}")
    
    try:
        # Load real MMS data
        data = load_event(
            trange=trange,
            probes=['1', '2', '3', '4'],
            data_rate_fgm='brst',
            data_rate_fpi='brst'
        )
        
        print("✅ Data loaded successfully")
        
        # Extract real spacecraft positions from MEC data
        positions = {}
        velocities = {}
        
        for probe in ['1', '2', '3', '4']:
            if probe in data:
                probe_data = data[probe]
                
                # Try to get position data
                pos_data = None
                vel_data = None
                
                # Check for position data
                for pos_key in ['POS_gsm', 'pos_gsm', 'R_gsm']:
                    if pos_key in probe_data:
                        times, pos_array = probe_data[pos_key]
                        if len(pos_array) > 0:
                            # Use middle position
                            mid_idx = len(pos_array) // 2
                            pos_data = pos_array[mid_idx]
                            print(f"   MMS{probe} position from {pos_key}: [{pos_data[0]:.1f}, {pos_data[1]:.1f}, {pos_data[2]:.1f}] km")
                            break
                
                # Check for velocity data
                for vel_key in ['V_gsm', 'vel_gsm', 'VEL_gsm']:
                    if vel_key in probe_data:
                        times, vel_array = probe_data[vel_key]
                        if len(vel_array) > 0:
                            mid_idx = len(vel_array) // 2
                            vel_data = vel_array[mid_idx]
                            print(f"   MMS{probe} velocity from {vel_key}: [{vel_data[0]:.3f}, {vel_data[1]:.3f}, {vel_data[2]:.3f}] km/s")
                            break
                
                if pos_data is not None:
                    positions[probe] = pos_data
                else:
                    print(f"   ⚠️ No position data found for MMS{probe}")
                    print(f"      Available keys: {list(probe_data.keys())}")
                
                if vel_data is not None:
                    velocities[probe] = vel_data
                else:
                    print(f"   ⚠️ No velocity data found for MMS{probe}")
        
        print(f"\n📊 POSITION ANALYSIS:")
        print("=" * 30)
        
        if len(positions) == 4:
            # Analyze the actual positions
            print("Real spacecraft positions (GSM coordinates):")
            for probe in ['1', '2', '3', '4']:
                pos = positions[probe]
                print(f"   MMS{probe}: X={pos[0]:8.1f}, Y={pos[1]:8.1f}, Z={pos[2]:8.1f} km")
            
            # Calculate X-ordering (most relevant for magnetopause crossings)
            x_positions = {probe: positions[probe][0] for probe in ['1', '2', '3', '4']}
            x_ordered = sorted(['1', '2', '3', '4'], key=lambda p: x_positions[p])
            
            print(f"\nX-GSM ordering (most negative to most positive):")
            print(f"   Calculated: {'-'.join(x_ordered)}")
            print(f"   Known correct: 2-1-4-3")
            print(f"   Match: {'✅ YES' if x_ordered == ['2', '1', '4', '3'] else '❌ NO'}")
            
            # Try formation detection
            print(f"\n🔬 FORMATION DETECTION ANALYSIS:")
            print("=" * 40)
            
            if len(velocities) == 4:
                analysis = detect_formation_type(positions, velocities)
            else:
                analysis = detect_formation_type(positions)
            
            print(f"Formation type: {analysis.formation_type}")
            print(f"Confidence: {analysis.confidence:.3f}")
            print(f"Linearity: {analysis.linearity:.3f}")
            print(f"Planarity: {analysis.planarity:.3f}")
            
            print(f"\nCalculated orderings:")
            for ordering_type, order in analysis.spacecraft_ordering.items():
                order_str = '-'.join(order)
                is_correct = order == ['2', '1', '4', '3']
                print(f"   {ordering_type:20}: {order_str} {'✅' if is_correct else '❌'}")
            
            # Manual calculation of what the ordering should be
            print(f"\n🧮 MANUAL VERIFICATION:")
            print("=" * 25)
            
            # Sort by X position manually
            manual_x_sort = []
            x_values = []
            for probe in ['1', '2', '3', '4']:
                x_val = positions[probe][0]
                x_values.append((probe, x_val))
            
            x_values.sort(key=lambda x: x[1])  # Sort by X value
            manual_order = [item[0] for item in x_values]
            
            print("Manual X-position sorting:")
            for probe, x_val in x_values:
                print(f"   MMS{probe}: X = {x_val:8.1f} km")
            
            print(f"\nManual order: {'-'.join(manual_order)}")
            print(f"Expected:     2-1-4-3")
            print(f"Match: {'✅ YES' if manual_order == ['2', '1', '4', '3'] else '❌ NO'}")
            
            if manual_order != ['2', '1', '4', '3']:
                print(f"\n❌ PROBLEM IDENTIFIED:")
                print("The real MEC position data does not match the expected ordering!")
                print("This suggests either:")
                print("1. The MEC data is incorrect or corrupted")
                print("2. The expected ordering is for a different time")
                print("3. There's an issue with coordinate system interpretation")
                
                # Check if positions are reasonable
                print(f"\n🔍 POSITION VALIDATION:")
                for probe in ['1', '2', '3', '4']:
                    pos = positions[probe]
                    distance = np.linalg.norm(pos)
                    print(f"   MMS{probe}: Distance from Earth = {distance/6371:.2f} RE")
                
        else:
            print(f"❌ Only found positions for {len(positions)} spacecraft")
            print("Cannot perform formation analysis")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
