#!/usr/bin/env python3
"""
Test Direct MEC Loading
=======================

This script tests MEC loading directly without going through the complex
data_loader to isolate the MEC loading issue.
"""

import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pytplot import data_quants
from pyspedas.projects import mms


def test_direct_mec_loading():
    """Test loading MEC data directly"""
    
    print("🔍 Testing Direct MEC Loading")
    print("=" * 50)
    
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    print(f"Time range: {trange[0]} to {trange[1]}")
    
    # Clear data_quants
    data_quants.clear()
    print(f"Cleared data_quants: {len(data_quants)} variables")
    
    try:
        # Load MEC data for MMS1 directly
        print(f"\nLoading MMS1 MEC data directly...")
        
        result = mms.mms_load_mec(
            trange=trange,
            probe='1',
            data_rate='srvy',
            level='l2',
            datatype='epht89q',
            time_clip=True,
            notplot=False
        )
        
        print(f"PySpedas result: {result}")
        print(f"Variables in data_quants after loading: {len(data_quants)}")
        
        # Look for MEC variables
        mec_vars = []
        for var_name in data_quants.keys():
            if 'mms1' in var_name and 'mec' in var_name:
                mec_vars.append(var_name)
        
        print(f"\nMMS1 MEC variables found: {len(mec_vars)}")
        for var in sorted(mec_vars):
            # Use PyTplot get_data function
            from pyspedas import get_data
            times, data = get_data(var)
            print(f"   {var}: {len(times)} points, shape {data.shape}")

            # Show sample data
            if len(data) > 0:
                sample_idx = len(data) // 2
                if len(data.shape) > 1 and data.shape[1] >= 3:
                    print(f"      Sample: [{data[sample_idx][0]:.1f}, {data[sample_idx][1]:.1f}, {data[sample_idx][2]:.1f}]")
                else:
                    print(f"      Sample: {data[sample_idx]}")
        
        # Test position and velocity extraction
        if 'mms1_mec_r_gsm' in data_quants:
            print(f"\n✅ Found mms1_mec_r_gsm - testing position extraction...")
            from pyspedas import get_data
            times, pos_data = get_data('mms1_mec_r_gsm')
            
            # Find closest time to center
            if hasattr(times[0], 'timestamp'):
                time_diffs = [abs((t - center_time).total_seconds()) for t in times]
            else:
                center_timestamp = center_time.timestamp()
                time_diffs = [abs(t - center_timestamp) for t in times]
            
            closest_index = np.argmin(time_diffs)
            position = pos_data[closest_index]
            
            print(f"   Position at {center_time}: [{position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}] km")
            
            # Check if position is reasonable
            distance_re = np.linalg.norm(position) / 6371.0
            if 5.0 < distance_re < 20.0:
                print(f"   ✅ Position is reasonable ({distance_re:.1f} RE)")
                return True
            else:
                print(f"   ⚠️ Position seems unusual ({distance_re:.1f} RE)")
                return False
        else:
            print(f"\n❌ mms1_mec_r_gsm not found in data_quants")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_spacecraft_mec():
    """Test MEC loading for all spacecraft"""
    
    print(f"\n🔍 Testing MEC Loading for All Spacecraft")
    print("=" * 50)
    
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    # Clear data_quants
    data_quants.clear()
    
    positions = {}
    velocities = {}
    
    for probe in ['1', '2', '3', '4']:
        print(f"\nLoading MMS{probe} MEC data...")
        
        try:
            result = mms.mms_load_mec(
                trange=trange,
                probe=probe,
                data_rate='srvy',
                level='l2',
                datatype='epht89q',
                time_clip=True,
                notplot=False
            )
            
            # Check for position data
            pos_var = f'mms{probe}_mec_r_gsm'
            vel_var = f'mms{probe}_mec_v_gsm'
            
            if pos_var in data_quants:
                from pyspedas import get_data
                times, pos_data = get_data(pos_var)
                
                # Find closest time
                if hasattr(times[0], 'timestamp'):
                    time_diffs = [abs((t - center_time).total_seconds()) for t in times]
                else:
                    center_timestamp = center_time.timestamp()
                    time_diffs = [abs(t - center_timestamp) for t in times]
                
                closest_index = np.argmin(time_diffs)
                positions[probe] = pos_data[closest_index]
                
                print(f"   ✅ Position: [{positions[probe][0]:.1f}, {positions[probe][1]:.1f}, {positions[probe][2]:.1f}] km")
            else:
                print(f"   ❌ No position data found")
            
            if vel_var in data_quants:
                times_vel, vel_data = get_data(vel_var)
                closest_index = np.argmin(time_diffs)
                velocities[probe] = vel_data[closest_index]
                
                print(f"   ✅ Velocity: [{velocities[probe][0]:.2f}, {velocities[probe][1]:.2f}, {velocities[probe][2]:.2f}] km/s")
            else:
                print(f"   ⚠️ No velocity data found")
                
        except Exception as e:
            print(f"   ❌ ERROR loading MMS{probe}: {e}")
    
    # Test spacecraft ordering if we have positions
    if len(positions) == 4:
        print(f"\n🔍 Testing Spacecraft Ordering with Real MEC Data")
        print("=" * 50)
        
        # Test different orderings
        orderings = {}
        
        # X_GSM ordering
        orderings['X_GSM'] = sorted(['1', '2', '3', '4'], key=lambda p: positions[p][0])
        
        # Distance from Earth
        distances = {p: np.linalg.norm(positions[p]) for p in ['1', '2', '3', '4']}
        orderings['Distance_from_Earth'] = sorted(['1', '2', '3', '4'], key=lambda p: distances[p])
        
        # Principal component analysis
        formation_center = np.mean([positions[p] for p in ['1', '2', '3', '4']], axis=0)
        pos_array = np.array([positions[p] for p in ['1', '2', '3', '4']])
        centered_positions = pos_array - formation_center
        
        # SVD for principal components
        U, s, Vt = np.linalg.svd(centered_positions)
        projections = {p: np.dot(positions[p] - formation_center, Vt[0]) for p in ['1', '2', '3', '4']}
        orderings['PC1'] = sorted(['1', '2', '3', '4'], key=lambda p: projections[p])
        
        # Print orderings
        independent_source_order = ['2', '1', '4', '3']
        independent_str = ' → '.join([f'MMS{p}' for p in independent_source_order])
        
        print(f"Independent source: {independent_str}")
        print(f"")
        
        matches = 0
        for ordering_name, order in orderings.items():
            order_str = ' → '.join([f'MMS{p}' for p in order])
            print(f"{ordering_name:20s}: {order_str}")
            
            if order == independent_source_order:
                print(f"                     ✅ MATCHES INDEPENDENT SOURCE!")
                matches += 1
            else:
                print(f"                     ⚠️ Different from independent source")
        
        print(f"\nMatches: {matches}/{len(orderings)} orderings match independent source")
        
        if matches > 0:
            print(f"🎉 SUCCESS: Real MEC data matches independent source!")
            return True
        else:
            print(f"⚠️ No orderings match independent source")
            return False
    else:
        print(f"❌ Insufficient position data for ordering analysis")
        return False


def main():
    """Main test function"""
    
    print("DIRECT MEC LOADING TEST")
    print("=" * 80)
    print("Testing MEC loading without complex data_loader")
    print("=" * 80)
    
    # Run tests
    tests = [
        ("Single Spacecraft MEC", test_direct_mec_loading),
        ("All Spacecraft MEC", test_all_spacecraft_mec)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n" + "=" * 80)
        print(f"TEST: {test_name}")
        print("=" * 80)
        
        try:
            if test_func():
                print(f"✅ PASSED: {test_name}")
                passed += 1
            else:
                print(f"❌ FAILED: {test_name}")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
    
    # Final summary
    print(f"\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Direct MEC loading is working correctly")
        print("✅ Real spacecraft positions match independent source")
    else:
        print("⚠️ Some tests failed - MEC loading needs investigation")
    
    return passed == total


if __name__ == "__main__":
    main()
