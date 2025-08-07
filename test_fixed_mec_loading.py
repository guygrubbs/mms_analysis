#!/usr/bin/env python3
"""
Test Fixed MEC Loading
======================

This script tests the fixed MEC data loading to see if we can now
get real spacecraft positions and determine the correct ordering.
"""

import numpy as np
from datetime import datetime
from mms_mp.data_loader import load_event
from mms_mp.formation_detection import detect_formation_type
import warnings
warnings.filterwarnings('ignore')

def main():
    """Test the fixed MEC data loading"""
    
    print("🔧 TESTING FIXED MEC DATA LOADING")
    print("=" * 45)
    print("Expected spacecraft order: 2-1-4-3")
    print()
    
    # Event parameters
    event_time = '2019-01-27/12:30:50'
    event_dt = datetime(2019, 1, 27, 12, 30, 50)
    trange = ['2019-01-27/12:25:00', '2019-01-27/12:35:00']
    
    print(f"📡 Loading MMS data for: {event_time}")
    print(f"   Time range: {trange[0]} to {trange[1]}")
    
    try:
        # Load data with fixed loader
        data = load_event(
            trange=trange,
            probes=['1', '2', '3', '4'],
            data_rate_fgm='brst',
            data_rate_fpi='brst'
        )
        
        print("✅ Data loaded successfully")
        
        # Check position data quality
        print(f"\n📊 POSITION DATA QUALITY CHECK:")
        print("=" * 35)
        
        positions = {}
        velocities = {}
        
        for probe in ['1', '2', '3', '4']:
            if probe in data:
                probe_data = data[probe]
                
                # Check position data
                if 'POS_gsm' in probe_data:
                    times, pos_data = probe_data['POS_gsm']
                    
                    # Check for valid data
                    valid_mask = ~np.isnan(pos_data).any(axis=1)
                    n_valid = np.sum(valid_mask)
                    
                    print(f"\nMMS{probe} Position Analysis:")
                    print(f"   Total points: {len(pos_data)}")
                    print(f"   Valid points: {n_valid} ({100*n_valid/len(pos_data):.1f}%)")
                    
                    if n_valid > 0:
                        # Use middle position for formation analysis
                        mid_idx = len(pos_data) // 2
                        if valid_mask[mid_idx]:
                            pos_center = pos_data[mid_idx]
                        else:
                            # Find nearest valid position
                            valid_indices = np.where(valid_mask)[0]
                            nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - mid_idx))]
                            pos_center = pos_data[nearest_idx]
                        
                        positions[probe] = pos_center
                        distance = np.linalg.norm(pos_center)
                        
                        print(f"   Position: [{pos_center[0]:.1f}, {pos_center[1]:.1f}, {pos_center[2]:.1f}] km")
                        print(f"   Distance: {distance:.1f} km ({distance/6371:.2f} RE)")
                        print(f"   ✅ Valid position data")
                    else:
                        print(f"   ❌ All position data is NaN")
                
                # Check velocity data
                if 'VEL_gsm' in probe_data:
                    times, vel_data = probe_data['VEL_gsm']
                    valid_mask = ~np.isnan(vel_data).any(axis=1)
                    n_valid = np.sum(valid_mask)
                    
                    if n_valid > 0:
                        mid_idx = len(vel_data) // 2
                        if valid_mask[mid_idx]:
                            vel_center = vel_data[mid_idx]
                        else:
                            valid_indices = np.where(valid_mask)[0]
                            nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - mid_idx))]
                            vel_center = vel_data[nearest_idx]
                        
                        velocities[probe] = vel_center
                        speed = np.linalg.norm(vel_center)
                        print(f"   Velocity: [{vel_center[0]:.3f}, {vel_center[1]:.3f}, {vel_center[2]:.3f}] km/s")
                        print(f"   Speed: {speed:.3f} km/s")
                        print(f"   ✅ Valid velocity data")
                    else:
                        print(f"   ❌ All velocity data is NaN")
        
        # Analyze spacecraft ordering
        print(f"\n🎯 SPACECRAFT ORDERING ANALYSIS:")
        print("=" * 40)
        
        if len(positions) == 4:
            print("✅ Found valid positions for all spacecraft")
            
            # Calculate X-ordering (most relevant for magnetopause crossings)
            x_positions = {probe: positions[probe][0] for probe in ['1', '2', '3', '4']}
            x_ordered = sorted(['1', '2', '3', '4'], key=lambda p: x_positions[p])
            
            print(f"\nX-GSM ordering (most negative to most positive):")
            for probe in x_ordered:
                print(f"   MMS{probe}: X = {x_positions[probe]:8.1f} km")
            
            print(f"\nCalculated order: {'-'.join(x_ordered)}")
            print(f"Expected order:   2-1-4-3")
            
            if x_ordered == ['2', '1', '4', '3']:
                print(f"✅ SUCCESS! Calculated order matches expected order!")
            else:
                print(f"❌ Order mismatch - investigating...")
                
                # Check if it's a different coordinate or time issue
                print(f"\nDifference analysis:")
                expected = ['2', '1', '4', '3']
                for i, (calc, exp) in enumerate(zip(x_ordered, expected)):
                    if calc != exp:
                        print(f"   Position {i+1}: Got MMS{calc}, expected MMS{exp}")
                        print(f"      MMS{calc} X = {x_positions[calc]:.1f} km")
                        print(f"      MMS{exp} X = {x_positions[exp]:.1f} km")
            
            # Try formation detection if we have velocities
            if len(velocities) == 4:
                print(f"\n🔬 FORMATION DETECTION ANALYSIS:")
                print("=" * 40)
                
                try:
                    analysis = detect_formation_type(positions, velocities)
                    
                    print(f"Formation type: {analysis.formation_type}")
                    print(f"Confidence: {analysis.confidence:.3f}")
                    print(f"Linearity: {analysis.linearity:.3f}")
                    print(f"Planarity: {analysis.planarity:.3f}")
                    
                    print(f"\nCalculated orderings:")
                    for ordering_type, order in analysis.spacecraft_ordering.items():
                        order_str = '-'.join(order)
                        is_correct = order == ['2', '1', '4', '3']
                        print(f"   {ordering_type:20}: {order_str} {'✅' if is_correct else '❌'}")
                        
                except Exception as e:
                    print(f"❌ Formation detection failed: {e}")
            else:
                print(f"\n⚠️ Formation detection skipped - missing velocity data")
                print(f"   Found velocities for: {list(velocities.keys())}")
        
        else:
            print(f"❌ Only found valid positions for {len(positions)} spacecraft")
            print(f"   Valid positions: {list(positions.keys())}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
