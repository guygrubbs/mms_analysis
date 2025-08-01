#!/usr/bin/env python3
"""
Simple MEC Integration Test
===========================

Focused test to verify MEC ephemeris integration is working correctly.
This test focuses on the core functionality without complex dependencies.
"""

import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path to import mms_mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mms_mp


def test_basic_data_loading():
    """Test basic data loading with MEC ephemeris"""
    
    print("🔍 Testing Basic MEC Data Loading")
    print("=" * 50)
    
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    print(f"Time range: {trange[0]} to {trange[1]}")
    
    try:
        # Load event data
        evt = mms_mp.load_event(
            trange=trange,
            probes=['1', '2', '3', '4'],
            include_ephem=True,
            include_edp=False
        )
        
        print(f"✅ Event data loaded successfully")
        print(f"   Probes loaded: {list(evt.keys())}")
        
        # Check for position data
        positions_found = 0
        velocities_found = 0
        
        for probe in ['1', '2', '3', '4']:
            if probe in evt:
                if 'POS_gsm' in evt[probe]:
                    times, pos_data = evt[probe]['POS_gsm']
                    if len(pos_data) > 0 and not np.all(np.isnan(pos_data[0])):
                        positions_found += 1
                        print(f"   ✅ MMS{probe}: Position data available ({len(pos_data)} points)")
                    else:
                        print(f"   ❌ MMS{probe}: Position data is NaN")
                else:
                    print(f"   ❌ MMS{probe}: No POS_gsm data")
                
                if 'VEL_gsm' in evt[probe]:
                    times, vel_data = evt[probe]['VEL_gsm']
                    if len(vel_data) > 0 and not np.all(np.isnan(vel_data[0])):
                        velocities_found += 1
                        print(f"   ✅ MMS{probe}: Velocity data available ({len(vel_data)} points)")
                    else:
                        print(f"   ⚠️ MMS{probe}: Velocity data is NaN")
                else:
                    print(f"   ⚠️ MMS{probe}: No VEL_gsm data")
        
        print(f"\nSummary:")
        print(f"   Positions found: {positions_found}/4")
        print(f"   Velocities found: {velocities_found}/4")
        
        if positions_found >= 4:
            print(f"   ✅ SUCCESS: All spacecraft positions loaded")
            return True
        else:
            print(f"   ❌ FAILURE: Missing position data")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False


def test_spacecraft_positions_extraction():
    """Test extracting spacecraft positions at specific time"""
    
    print(f"\n🔍 Testing Spacecraft Position Extraction")
    print("=" * 50)
    
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    try:
        evt = mms_mp.load_event(
            trange=trange,
            probes=['1', '2', '3', '4'],
            include_ephem=True
        )
        
        # Extract positions at center time
        positions = {}
        
        for probe in ['1', '2', '3', '4']:
            if probe in evt and 'POS_gsm' in evt[probe]:
                times, pos_data = evt[probe]['POS_gsm']
                
                # Find closest time index
                if hasattr(times[0], 'timestamp'):
                    time_diffs = [abs((t - center_time).total_seconds()) for t in times]
                else:
                    center_timestamp = center_time.timestamp()
                    time_diffs = [abs(t - center_timestamp) for t in times]
                
                closest_index = np.argmin(time_diffs)
                positions[probe] = pos_data[closest_index] / 1000.0  # Convert to km
                
                print(f"   MMS{probe}: [{positions[probe][0]:.1f}, {positions[probe][1]:.1f}, {positions[probe][2]:.1f}] km")
        
        if len(positions) == 4:
            print(f"   ✅ SUCCESS: All positions extracted")
            
            # Test spacecraft ordering
            return test_spacecraft_ordering(positions)
        else:
            print(f"   ❌ FAILURE: Only {len(positions)}/4 positions extracted")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False


def test_spacecraft_ordering(positions):
    """Test spacecraft ordering using real positions"""
    
    print(f"\n🔍 Testing Spacecraft Ordering")
    print("=" * 50)
    
    probes = ['1', '2', '3', '4']
    
    # Test different ordering criteria
    orderings = {}
    
    # X_GSM ordering
    orderings['X_GSM'] = sorted(probes, key=lambda p: positions[p][0])
    
    # Y_GSM ordering  
    orderings['Y_GSM'] = sorted(probes, key=lambda p: positions[p][1])
    
    # Z_GSM ordering
    orderings['Z_GSM'] = sorted(probes, key=lambda p: positions[p][2])
    
    # Distance from Earth
    distances = {p: np.linalg.norm(positions[p]) for p in probes}
    orderings['Distance_from_Earth'] = sorted(probes, key=lambda p: distances[p])
    
    # Principal component analysis
    formation_center = np.mean([positions[p] for p in probes], axis=0)
    pos_array = np.array([positions[p] for p in probes])
    centered_positions = pos_array - formation_center
    
    # SVD for principal components
    U, s, Vt = np.linalg.svd(centered_positions)
    projections = {p: np.dot(positions[p] - formation_center, Vt[0]) for p in probes}
    orderings['PC1'] = sorted(probes, key=lambda p: projections[p])
    
    # Print all orderings
    independent_source_order = ['2', '1', '4', '3']
    independent_str = ' → '.join([f'MMS{p}' for p in independent_source_order])
    
    print(f"Independent source: {independent_str}")
    print(f"")
    
    matches = 0
    total = len(orderings)
    
    for ordering_name, order in orderings.items():
        order_str = ' → '.join([f'MMS{p}' for p in order])
        print(f"{ordering_name:20s}: {order_str}")
        
        if order == independent_source_order:
            print(f"                     ✅ MATCHES INDEPENDENT SOURCE!")
            matches += 1
        else:
            print(f"                     ⚠️ Different from independent source")
    
    print(f"\nMatches: {matches}/{total} orderings match independent source")
    
    if matches > 0:
        print(f"✅ SUCCESS: At least one ordering matches independent source")
        return True
    else:
        print(f"❌ FAILURE: No orderings match independent source")
        return False


def test_formation_detection():
    """Test formation detection with real MEC data"""
    
    print(f"\n🔍 Testing Formation Detection")
    print("=" * 50)
    
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    try:
        evt = mms_mp.load_event(
            trange=trange,
            probes=['1', '2', '3', '4'],
            include_ephem=True
        )
        
        # Extract positions and velocities
        positions = {}
        velocities = {}
        
        for probe in ['1', '2', '3', '4']:
            if probe in evt and 'POS_gsm' in evt[probe]:
                times, pos_data = evt[probe]['POS_gsm']
                
                # Find closest time index
                if hasattr(times[0], 'timestamp'):
                    time_diffs = [abs((t - center_time).total_seconds()) for t in times]
                else:
                    center_timestamp = center_time.timestamp()
                    time_diffs = [abs(t - center_timestamp) for t in times]
                
                closest_index = np.argmin(time_diffs)
                positions[probe] = pos_data[closest_index] / 1000.0  # Convert to km
                
                # Try to get velocity
                if 'VEL_gsm' in evt[probe]:
                    times_vel, vel_data = evt[probe]['VEL_gsm']
                    velocities[probe] = vel_data[closest_index] / 1000.0  # Convert to km/s
        
        if len(positions) == 4:
            # Test formation detection
            try:
                formation_analysis = mms_mp.detect_formation_type(positions, velocities)
                
                print(f"   Formation type: {formation_analysis.formation_type.value}")
                print(f"   Confidence: {formation_analysis.confidence:.3f}")
                
                # Check spacecraft ordering
                if hasattr(formation_analysis, 'spacecraft_ordering'):
                    for ordering_name, order in formation_analysis.spacecraft_ordering.items():
                        order_str = ' → '.join([f'MMS{p}' for p in order])
                        print(f"   {ordering_name}: {order_str}")
                        
                        if order == ['2', '1', '4', '3']:
                            print(f"      ✅ MATCHES INDEPENDENT SOURCE!")
                
                print(f"   ✅ SUCCESS: Formation detection completed")
                return True
                
            except Exception as e:
                print(f"   ❌ Formation detection error: {e}")
                return False
        else:
            print(f"   ❌ FAILURE: Insufficient position data")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False


def main():
    """Main test function"""
    
    print("SIMPLE MEC INTEGRATION TEST")
    print("=" * 80)
    print("Testing core MEC ephemeris functionality")
    print("=" * 80)
    
    # Run tests
    tests = [
        ("Basic Data Loading", test_basic_data_loading),
        ("Position Extraction", test_spacecraft_positions_extraction),
        ("Formation Detection", test_formation_detection)
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
        print("✅ MEC ephemeris integration is working correctly")
    else:
        print("⚠️ Some tests failed - need investigation")
    
    return passed == total


if __name__ == "__main__":
    main()
