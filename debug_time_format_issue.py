#!/usr/bin/env python3
"""
Debug Time Format Issue
=======================

This script investigates the specific time format and conversion issue
that's causing data to be loaded from 05:00-07:30 UT instead of 11:30-13:30 UT.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path to import mms_mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyspedas.projects import mms
from pytplot import get_data, data_quants


def test_direct_pyspedas_loading():
    """Test direct pyspedas loading with different time formats"""
    
    print("🔍 TESTING DIRECT PYSPEDAS LOADING")
    print("=" * 80)
    
    # Event time and range
    event_time = datetime(2019, 1, 27, 12, 30, 50)
    start_time = event_time - timedelta(hours=1)
    end_time = event_time + timedelta(hours=1)
    
    # Test different time format variations
    time_formats = [
        # Format 1: Standard format used in analysis
        [start_time.strftime('%Y-%m-%d/%H:%M:%S'), end_time.strftime('%Y-%m-%d/%H:%M:%S')],
        
        # Format 2: ISO format
        [start_time.strftime('%Y-%m-%dT%H:%M:%S'), end_time.strftime('%Y-%m-%dT%H:%M:%S')],
        
        # Format 3: Different separator
        [start_time.strftime('%Y-%m-%d %H:%M:%S'), end_time.strftime('%Y-%m-%d %H:%M:%S')],
        
        # Format 4: Explicit UTC
        [start_time.strftime('%Y-%m-%d/%H:%M:%S') + 'Z', end_time.strftime('%Y-%m-%d/%H:%M:%S') + 'Z'],
    ]
    
    format_names = [
        "Standard (%Y-%m-%d/%H:%M:%S)",
        "ISO (%Y-%m-%dT%H:%M:%S)", 
        "Space separator (%Y-%m-%d %H:%M:%S)",
        "UTC explicit (%Y-%m-%d/%H:%M:%SZ)"
    ]
    
    for i, (trange, format_name) in enumerate(zip(time_formats, format_names)):
        print(f"\n📊 Test {i+1}: {format_name}")
        print(f"   trange: {trange}")
        
        try:
            # Clear previous data
            data_quants.clear()
            
            # Load FGM data directly
            result = mms.mms_load_fgm(
                trange=trange,
                probe='1',
                data_rate='srvy',
                level='l2',
                time_clip=True,
                notplot=False
            )
            
            print(f"   Loading result: {result}")
            
            # Check what was loaded
            if 'mms1_fgm_b_gsm_srvy_l2' in data_quants:
                times, b_data = get_data('mms1_fgm_b_gsm_srvy_l2')
                
                if len(times) > 0:
                    # Convert times
                    if hasattr(times[0], 'strftime'):
                        first_time = times[0]
                        last_time = times[-1]
                    else:
                        first_time = datetime.fromtimestamp(times[0])
                        last_time = datetime.fromtimestamp(times[-1])
                    
                    print(f"   ✅ Data loaded: {len(times)} points")
                    print(f"   Time range: {first_time.strftime('%Y-%m-%d %H:%M:%S')} to {last_time.strftime('%Y-%m-%d %H:%M:%S')} UT")
                    
                    # Check if event time is in range
                    if first_time <= event_time <= last_time:
                        print(f"   ✅ Event time IS in range")
                    else:
                        print(f"   ❌ Event time NOT in range!")
                        time_diff = (event_time - first_time).total_seconds() / 3600
                        print(f"      Event time is {time_diff:.1f} hours from start of data")
                else:
                    print(f"   ❌ No data points")
            else:
                print(f"   ❌ No FGM data variable found")
                print(f"   Available variables: {list(data_quants.keys())}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")


def test_time_conversion_issue():
    """Test if there's a time conversion issue"""
    
    print(f"\n🔍 TESTING TIME CONVERSION")
    print("=" * 80)
    
    # Test the exact time range from our analysis
    event_time = datetime(2019, 1, 27, 12, 30, 50)
    start_time = event_time - timedelta(hours=1)
    end_time = event_time + timedelta(hours=1)
    
    trange = [
        start_time.strftime('%Y-%m-%d/%H:%M:%S'),
        end_time.strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    print(f"Requested time range: {trange}")
    print(f"Start time object: {start_time}")
    print(f"End time object: {end_time}")
    print(f"Event time object: {event_time}")
    
    # Test different timezone interpretations
    print(f"\nTimezone tests:")
    print(f"Start time timestamp: {start_time.timestamp()}")
    print(f"End time timestamp: {end_time.timestamp()}")
    print(f"Event time timestamp: {event_time.timestamp()}")
    
    # Check if there's a 7-hour offset (which we observed)
    offset_start = start_time - timedelta(hours=7)
    offset_end = end_time - timedelta(hours=7)
    
    print(f"\n7-hour offset test:")
    print(f"Offset start: {offset_start.strftime('%Y-%m-%d %H:%M:%S')} UT")
    print(f"Offset end: {offset_end.strftime('%Y-%m-%d %H:%M:%S')} UT")
    print(f"This matches observed data range: 05:30-07:30 UT")
    
    # Test if the issue is in the time string parsing
    try:
        data_quants.clear()
        
        print(f"\nTesting with explicit time range...")
        
        # Try loading with the exact observed time range
        observed_trange = ['2019-01-27/05:30:50', '2019-01-27/07:30:50']
        
        result = mms.mms_load_fgm(
            trange=observed_trange,
            probe='1',
            data_rate='srvy',
            level='l2',
            time_clip=True,
            notplot=False
        )
        
        if 'mms1_fgm_b_gsm_srvy_l2' in data_quants:
            times, b_data = get_data('mms1_fgm_b_gsm_srvy_l2')
            
            if len(times) > 0:
                if hasattr(times[0], 'strftime'):
                    first_time = times[0]
                    last_time = times[-1]
                else:
                    first_time = datetime.fromtimestamp(times[0])
                    last_time = datetime.fromtimestamp(times[-1])
                
                print(f"Observed range loading:")
                print(f"   Requested: {observed_trange}")
                print(f"   Actual: {first_time.strftime('%Y-%m-%d %H:%M:%S')} to {last_time.strftime('%Y-%m-%d %H:%M:%S')} UT")
                
                if abs((first_time - datetime(2019, 1, 27, 5, 30, 50)).total_seconds()) < 60:
                    print(f"   ✅ This confirms the time range is being interpreted correctly")
                    print(f"   ❌ The issue is that our requested range is being shifted by 7 hours")
                
    except Exception as e:
        print(f"Error in observed range test: {e}")


def test_timezone_hypothesis():
    """Test if there's a timezone conversion issue"""
    
    print(f"\n🔍 TESTING TIMEZONE HYPOTHESIS")
    print("=" * 80)
    
    # The 7-hour difference suggests a timezone issue
    # UTC-7 would be Pacific Daylight Time (PDT)
    # UTC+7 would be some Asian timezone
    
    event_time = datetime(2019, 1, 27, 12, 30, 50)
    
    # Test different timezone interpretations
    timezone_tests = [
        ("UTC", event_time),
        ("UTC-7 (PDT)", event_time + timedelta(hours=7)),
        ("UTC+7", event_time - timedelta(hours=7)),
        ("Local time issue", event_time - timedelta(hours=7)),
    ]
    
    for tz_name, test_time in timezone_tests:
        start_time = test_time - timedelta(hours=1)
        end_time = test_time + timedelta(hours=1)
        
        print(f"\n{tz_name} interpretation:")
        print(f"   Event time: {test_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Range: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if tz_name == "Local time issue":
            print(f"   ✅ This matches our observed data range!")


def main():
    """Main debug function"""
    
    print("DEBUG TIME FORMAT ISSUE")
    print("=" * 80)
    print("Investigating why data loads from 05:00-07:30 instead of 11:30-13:30")
    print("=" * 80)
    
    # Run tests
    test_direct_pyspedas_loading()
    test_time_conversion_issue()
    test_timezone_hypothesis()
    
    print(f"\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    print("🔍 FINDINGS:")
    print("1. The issue appears to be a 7-hour time offset")
    print("2. Requested: 11:30-13:30 UT → Actual: 05:30-07:30 UT")
    print("3. This suggests a timezone conversion problem")
    print("4. The data loading functions may be interpreting times incorrectly")
    
    print("\n🎯 LIKELY CAUSES:")
    print("1. System timezone affecting time interpretation")
    print("2. PySpedas time parsing issue")
    print("3. Local time vs UTC confusion")
    print("4. Time format parsing error")
    
    print("\n🔧 NEXT STEPS:")
    print("1. Check system timezone settings")
    print("2. Test with explicit UTC timestamps")
    print("3. Investigate PySpedas time handling")
    print("4. Try alternative time formats")


if __name__ == "__main__":
    main()
