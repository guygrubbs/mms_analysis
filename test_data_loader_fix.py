#!/usr/bin/env python3
"""
Test Data Loader Fix
====================

Focused test to fix the data loader integration test case.
This test specifically validates that the data loader can load
real MEC ephemeris data without API errors.
"""

import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path to import mms_mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mms_mp


def test_data_loader_api_fix():
    """Test data loader with correct API parameters"""
    
    print("🔍 Testing Data Loader API Fix")
    print("=" * 50)
    
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    print(f"Time range: {trange[0]} to {trange[1]}")
    
    try:
        # Test 1: Load with minimal parameters (just ephemeris)
        print(f"\n📊 Test 1: Loading with minimal parameters...")
        
        evt = mms_mp.load_event(
            trange=trange,
            probes=['1', '2'],  # Just 2 spacecraft to speed up
            include_ephem=True,
            include_edp=False
        )
        
        print(f"   ✅ Event data loaded successfully")
        print(f"   Probes loaded: {list(evt.keys())}")
        
        # Check for position data
        positions_found = 0
        for probe in ['1', '2']:
            if probe in evt and 'POS_gsm' in evt[probe]:
                times, pos_data = evt[probe]['POS_gsm']
                
                if len(pos_data) > 0:
                    # Find closest time
                    if hasattr(times[0], 'timestamp'):
                        time_diffs = [abs((t - center_time).total_seconds()) for t in times]
                    else:
                        target_timestamp = center_time.timestamp()
                        time_diffs = [abs(t - target_timestamp) for t in times]
                    
                    closest_index = np.argmin(time_diffs)
                    position = pos_data[closest_index]
                    
                    # Check if position is real (not NaN)
                    if not np.any(np.isnan(position)) and np.linalg.norm(position) > 10000:
                        positions_found += 1
                        print(f"   ✅ MMS{probe}: Real position data [{position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}] km")
                    else:
                        print(f"   ❌ MMS{probe}: Invalid position (NaN: {np.any(np.isnan(position))}, mag: {np.linalg.norm(position):.1f})")
                else:
                    print(f"   ❌ MMS{probe}: No position data")
            else:
                print(f"   ❌ MMS{probe}: No POS_gsm variable")
        
        if positions_found >= 2:
            print(f"\n✅ TEST 1 PASSED: Data loader provides real MEC positions")
            test1_success = True
        else:
            print(f"\n❌ TEST 1 FAILED: Only {positions_found}/2 real positions")
            test1_success = False
        
        # Test 2: Load with all spacecraft
        print(f"\n📊 Test 2: Loading all spacecraft...")
        
        evt_all = mms_mp.load_event(
            trange=trange,
            probes=['1', '2', '3', '4'],
            include_ephem=True,
            include_edp=False
        )
        
        print(f"   ✅ All spacecraft data loaded successfully")
        
        # Quick validation
        all_positions_found = 0
        for probe in ['1', '2', '3', '4']:
            if probe in evt_all and 'POS_gsm' in evt_all[probe]:
                times, pos_data = evt_all[probe]['POS_gsm']
                if len(pos_data) > 0:
                    closest_index = len(pos_data) // 2  # Use middle point for speed
                    position = pos_data[closest_index]
                    
                    if not np.any(np.isnan(position)) and np.linalg.norm(position) > 10000:
                        all_positions_found += 1
                        print(f"   ✅ MMS{probe}: Real position data")
                    else:
                        print(f"   ❌ MMS{probe}: Invalid position data")
        
        if all_positions_found >= 4:
            print(f"\n✅ TEST 2 PASSED: All spacecraft have real MEC positions")
            test2_success = True
        else:
            print(f"\n❌ TEST 2 FAILED: Only {all_positions_found}/4 real positions")
            test2_success = False
        
        return test1_success and test2_success
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loader_parameters():
    """Test different data loader parameter combinations"""
    
    print(f"\n🔍 Testing Data Loader Parameter Combinations")
    print("=" * 50)
    
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=2)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=2)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    # Test different parameter combinations
    test_configs = [
        {
            'name': 'Ephemeris only',
            'params': {
                'include_ephem': True,
                'include_edp': False
            }
        },
        {
            'name': 'Ephemeris + EDP',
            'params': {
                'include_ephem': True,
                'include_edp': True
            }
        },
        {
            'name': 'Default parameters',
            'params': {
                'include_ephem': True
            }
        }
    ]
    
    successful_configs = 0
    
    for config in test_configs:
        print(f"\n📊 Testing: {config['name']}")
        
        try:
            evt = mms_mp.load_event(
                trange=trange,
                probes=['1'],  # Just one spacecraft for speed
                **config['params']
            )
            
            if '1' in evt and 'POS_gsm' in evt['1']:
                times, pos_data = evt['1']['POS_gsm']
                if len(pos_data) > 0:
                    position = pos_data[len(pos_data) // 2]
                    if not np.any(np.isnan(position)) and np.linalg.norm(position) > 10000:
                        print(f"   ✅ {config['name']}: Real position data loaded")
                        successful_configs += 1
                    else:
                        print(f"   ❌ {config['name']}: Invalid position data")
                else:
                    print(f"   ❌ {config['name']}: No position data")
            else:
                print(f"   ❌ {config['name']}: No POS_gsm variable")
                
        except Exception as e:
            print(f"   ❌ {config['name']}: Error - {e}")
    
    print(f"\nSuccessful configurations: {successful_configs}/{len(test_configs)}")
    
    return successful_configs >= 2  # At least 2 configs should work


def main():
    """Main test function"""
    
    print("DATA LOADER FIX TEST")
    print("=" * 80)
    print("Testing data loader integration with correct API parameters")
    print("=" * 80)
    
    # Run tests
    test1_pass = test_data_loader_api_fix()
    test2_pass = test_data_loader_parameters()
    
    # Summary
    print(f"\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    tests_passed = sum([test1_pass, test2_pass])
    total_tests = 2
    
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 ALL DATA LOADER TESTS PASSED!")
        print("✅ Data loader integration is working correctly")
        print("✅ Real MEC ephemeris data is being loaded")
        print("✅ API parameters are correct")
    else:
        print("⚠️ SOME DATA LOADER TESTS FAILED")
        print("❌ Data loader integration needs more work")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎯 DATA LOADER FIX: SUCCESSFUL")
        print(f"✅ Ready to update main test suite")
    else:
        print(f"\n❌ DATA LOADER FIX: NEEDS MORE WORK")
        print(f"⚠️ Check API parameters and data loading")
