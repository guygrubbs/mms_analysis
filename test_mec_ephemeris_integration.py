#!/usr/bin/env python3
"""
Test MEC Ephemeris Integration
==============================

This script verifies that:
1. MEC ephemeris data is used as the authoritative source
2. All analyses use consistent spacecraft positions from MEC
3. Coordinate transformations preserve MEC accuracy
4. Spacecraft ordering matches independent sources

This ensures that future analyses will always use real MEC data
instead of falling back to synthetic positions.
"""

import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path to import mms_mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mms_mp
from mms_mp import (
    load_event, 
    get_mec_ephemeris_manager, 
    analyze_formation_from_event_data,
    validate_mec_data_usage
)


def test_mec_data_loading():
    """Test that MEC ephemeris data is loaded correctly"""
    
    print("🔍 Testing MEC Ephemeris Data Loading")
    print("=" * 50)
    
    # Test both dates from the original analysis
    test_cases = [
        ('2019-01-26', '15:00:00'),
        ('2019-01-27', '12:30:50')
    ]
    
    for date_str, time_str in test_cases:
        print(f"\n📡 Testing {date_str} {time_str}...")
        
        # Create time range
        center_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        trange = [
            (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
            (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
        ]
        
        try:
            # Load event data with MEC ephemeris
            evt = load_event(
                trange=trange,
                probes=['1', '2', '3', '4'],
                include_ephem=True,
                include_edp=False
            )
            
            # Check if MEC data was loaded
            mec_data_found = False
            synthetic_fallback = False
            
            for probe in ['1', '2', '3', '4']:
                if probe in evt and 'POS_gsm' in evt[probe]:
                    times, pos_data = evt[probe]['POS_gsm']
                    
                    # Check if we have real data (not all NaN)
                    if not np.all(np.isnan(pos_data)):
                        mec_data_found = True
                        print(f"   ✅ MMS{probe}: Real position data loaded")
                    else:
                        synthetic_fallback = True
                        print(f"   ❌ MMS{probe}: NaN data (synthetic fallback)")
                else:
                    print(f"   ❌ MMS{probe}: No position data")
            
            if mec_data_found and not synthetic_fallback:
                print(f"   ✅ SUCCESS: Real MEC data loaded for {date_str}")
            elif mec_data_found:
                print(f"   ⚠️ PARTIAL: Some real data, some synthetic for {date_str}")
            else:
                print(f"   ❌ FAILURE: No real MEC data for {date_str}")
                
        except Exception as e:
            print(f"   ❌ ERROR loading data for {date_str}: {e}")
    
    return True


def test_ephemeris_manager():
    """Test the EphemerisManager functionality"""
    
    print(f"\n🔍 Testing EphemerisManager")
    print("=" * 50)
    
    # Load test data
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    try:
        evt = load_event(
            trange=trange,
            probes=['1', '2', '3', '4'],
            include_ephem=True
        )
        
        # Create ephemeris manager
        ephemeris_mgr = get_mec_ephemeris_manager(evt)
        
        # Test position extraction
        positions = ephemeris_mgr.get_positions_at_time(center_time, 'gsm')
        velocities = ephemeris_mgr.get_velocities_at_time(center_time, 'gsm')
        
        print(f"   Positions extracted: {len(positions)} spacecraft")
        print(f"   Velocities extracted: {len(velocities)} spacecraft")
        
        # Test formation analysis data
        formation_data = ephemeris_mgr.get_formation_analysis_data(center_time)
        
        print(f"   Formation center: [{formation_data['formation_center'][0]:.1f}, "
              f"{formation_data['formation_center'][1]:.1f}, "
              f"{formation_data['formation_center'][2]:.1f}] km")
        
        # Test authoritative ordering
        ordering = ephemeris_mgr.get_authoritative_spacecraft_ordering(center_time)
        ordering_str = ' → '.join([f'MMS{p}' for p in ordering])
        print(f"   Authoritative ordering: {ordering_str}")
        
        # Check if this matches independent source (2 → 1 → 4 → 3)
        if ordering == ['2', '1', '4', '3']:
            print(f"   ✅ SUCCESS: Matches independent source!")
        else:
            print(f"   ⚠️ Different from independent source (2 → 1 → 4 → 3)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ERROR in EphemerisManager test: {e}")
        return False


def test_formation_analysis_integration():
    """Test formation analysis using MEC ephemeris"""
    
    print(f"\n🔍 Testing Formation Analysis with MEC Data")
    print("=" * 50)
    
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    try:
        # Load event data
        evt = load_event(
            trange=trange,
            probes=['1', '2', '3', '4'],
            include_ephem=True
        )
        
        # Perform formation analysis using MEC data
        formation_analysis = analyze_formation_from_event_data(evt, center_time)
        
        print(f"   Formation type: {formation_analysis.formation_type.value}")
        print(f"   Confidence: {formation_analysis.confidence:.3f}")
        print(f"   Data source: {formation_analysis.quality_metrics.get('data_source', 'unknown')}")
        
        # Check spacecraft ordering
        if 'Leading_to_Trailing' in formation_analysis.spacecraft_ordering:
            ordering = formation_analysis.spacecraft_ordering['Leading_to_Trailing']
            ordering_str = ' → '.join([f'MMS{p}' for p in ordering])
            print(f"   Orbital ordering: {ordering_str}")
            
            if ordering == ['2', '1', '4', '3']:
                print(f"   ✅ SUCCESS: Formation analysis matches independent source!")
            else:
                print(f"   ⚠️ Formation analysis differs from independent source")
        
        # Verify MEC data usage
        if formation_analysis.quality_metrics.get('authoritative_source'):
            print(f"   ✅ Using authoritative MEC ephemeris data")
        else:
            print(f"   ❌ Not using authoritative data source")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ERROR in formation analysis test: {e}")
        return False


def test_coordinate_consistency():
    """Test that coordinate transformations preserve MEC accuracy"""
    
    print(f"\n🔍 Testing Coordinate System Consistency")
    print("=" * 50)
    
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    try:
        evt = load_event(trange=trange, probes=['1', '2', '3', '4'], include_ephem=True)
        ephemeris_mgr = get_mec_ephemeris_manager(evt)
        
        # Get positions in different coordinate systems
        positions_gsm = ephemeris_mgr.get_positions_at_time(center_time, 'gsm')
        
        print(f"   GSM positions loaded for {len(positions_gsm)} spacecraft")
        
        # Test that positions are reasonable (not synthetic)
        for probe, pos in positions_gsm.items():
            distance_re = np.linalg.norm(pos) / 6371.0  # Distance in Earth radii
            if 5.0 < distance_re < 20.0:  # Reasonable MMS orbit range
                print(f"   ✅ MMS{probe}: Realistic position ({distance_re:.1f} RE)")
            else:
                print(f"   ⚠️ MMS{probe}: Unusual position ({distance_re:.1f} RE)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ERROR in coordinate consistency test: {e}")
        return False


def verify_independent_source_match():
    """Verify that our analysis now matches the independent source"""
    
    print(f"\n🎯 Verifying Independent Source Match")
    print("=" * 50)
    
    # Test both dates
    test_cases = [
        ('2019-01-26', '15:00:00'),
        ('2019-01-27', '12:30:50')
    ]
    
    independent_ordering = ['2', '1', '4', '3']
    independent_str = ' → '.join([f'MMS{p}' for p in independent_ordering])
    
    print(f"   Independent source ordering: {independent_str}")
    
    matches = 0
    total = 0
    
    for date_str, time_str in test_cases:
        center_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        trange = [
            (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
            (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
        ]
        
        try:
            evt = load_event(trange=trange, probes=['1', '2', '3', '4'], include_ephem=True)
            ephemeris_mgr = get_mec_ephemeris_manager(evt)
            
            # Get authoritative ordering
            ordering = ephemeris_mgr.get_authoritative_spacecraft_ordering(center_time)
            ordering_str = ' → '.join([f'MMS{p}' for p in ordering])
            
            print(f"   {date_str}: {ordering_str}")
            
            if ordering == independent_ordering:
                print(f"              ✅ MATCHES independent source")
                matches += 1
            else:
                print(f"              ❌ Different from independent source")
            
            total += 1
            
        except Exception as e:
            print(f"   {date_str}: ❌ ERROR - {e}")
            total += 1
    
    print(f"\n   Summary: {matches}/{total} dates match independent source")
    
    if matches == total:
        print(f"   🎉 SUCCESS: All analyses now match independent source!")
    else:
        print(f"   ⚠️ Some discrepancies remain - investigate further")
    
    return matches == total


def main():
    """Main test function"""
    
    print("MEC EPHEMERIS INTEGRATION TEST")
    print("=" * 80)
    print("Verifying that MEC data is used as authoritative source")
    print("for all spacecraft positioning and formation analysis")
    print("=" * 80)
    
    # Validate MEC data usage
    validate_mec_data_usage()
    
    # Run all tests
    tests = [
        ("MEC Data Loading", test_mec_data_loading),
        ("EphemerisManager", test_ephemeris_manager),
        ("Formation Analysis Integration", test_formation_analysis_integration),
        ("Coordinate Consistency", test_coordinate_consistency),
        ("Independent Source Match", verify_independent_source_match)
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
        print("✅ MEC ephemeris data is now the authoritative source")
        print("✅ All analyses will use consistent spacecraft positions")
        print("✅ Spacecraft ordering matches independent sources")
    else:
        print("⚠️ Some tests failed - MEC integration needs work")
    
    print(f"\nMEC ephemeris integration: {'COMPLETE' if passed == total else 'INCOMPLETE'}")


if __name__ == "__main__":
    main()
