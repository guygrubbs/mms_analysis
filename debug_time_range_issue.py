#!/usr/bin/env python3
"""
Debug Time Range Issue
======================

This script investigates why the analysis is showing data from 05:00-07:30 UT
instead of the event time at 12:30:50 UT.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path to import mms_mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mms_mp
from pytplot import get_data, data_quants


def debug_time_range_issue():
    """Debug the time range issue systematically"""
    
    print("🔍 DEBUGGING TIME RANGE ISSUE")
    print("=" * 80)
    
    # Define the event time and range exactly as in the analysis
    event_time = datetime(2019, 1, 27, 12, 30, 50)
    start_time = event_time - timedelta(hours=1)
    end_time = event_time + timedelta(hours=1)
    
    trange = [
        start_time.strftime('%Y-%m-%d/%H:%M:%S'),
        end_time.strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    print(f"📅 REQUESTED TIME RANGE:")
    print(f"   Event time: {event_time.strftime('%Y-%m-%d %H:%M:%S')} UT")
    print(f"   Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')} UT")
    print(f"   End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')} UT")
    print(f"   trange: {trange}")
    
    # Load data with same parameters as analysis
    print(f"\n📊 LOADING DATA...")
    
    try:
        evt = mms_mp.load_event(
            trange=trange,
            probes=['1'],  # Just one spacecraft for debugging
            include_ephem=True,
            include_edp=False,  # Simplify for debugging
            data_rate_fpi='fast',
            data_rate_fgm='srvy'
        )
        
        print(f"✅ Data loaded successfully")
        print(f"   Spacecraft loaded: {list(evt.keys())}")
        
        if '1' in evt:
            print(f"\n📊 MMS1 DATA VARIABLES:")
            for var_name in evt['1'].keys():
                times, data = evt['1'][var_name]
                print(f"   {var_name}: {len(times)} points")
                
                # Check time range for each variable
                if len(times) > 0:
                    # Handle different time formats
                    if hasattr(times[0], 'strftime'):
                        # Already datetime objects
                        first_time = times[0]
                        last_time = times[-1]
                    else:
                        # Convert from timestamp
                        first_time = datetime.fromtimestamp(times[0])
                        last_time = datetime.fromtimestamp(times[-1])
                    
                    print(f"      Time range: {first_time.strftime('%Y-%m-%d %H:%M:%S')} to {last_time.strftime('%Y-%m-%d %H:%M:%S')} UT")
                    
                    # Check if event time is in range
                    if first_time <= event_time <= last_time:
                        print(f"      ✅ Event time IS in range")
                    else:
                        print(f"      ❌ Event time NOT in range!")
                        time_diff = (event_time - first_time).total_seconds() / 3600
                        print(f"         Event time is {time_diff:.1f} hours from start of data")
        
        # Test magnetic field data specifically
        if '1' in evt and 'B_gsm' in evt['1']:
            print(f"\n🔍 DETAILED MAGNETIC FIELD ANALYSIS:")
            times, b_data = evt['1']['B_gsm']
            
            print(f"   Data points: {len(times)}")
            print(f"   Data shape: {b_data.shape}")
            
            # Convert times for analysis
            if hasattr(times[0], 'strftime'):
                time_objects = times
            else:
                time_objects = [datetime.fromtimestamp(t) for t in times]
            
            first_time = time_objects[0]
            last_time = time_objects[-1]
            
            print(f"   First timestamp: {first_time.strftime('%Y-%m-%d %H:%M:%S')} UT")
            print(f"   Last timestamp: {last_time.strftime('%Y-%m-%d %H:%M:%S')} UT")
            print(f"   Duration: {(last_time - first_time).total_seconds() / 3600:.2f} hours")
            
            # Find closest time to event
            time_diffs = [abs((t - event_time).total_seconds()) for t in time_objects]
            closest_index = np.argmin(time_diffs)
            closest_time = time_objects[closest_index]
            closest_diff = min(time_diffs)
            
            print(f"   Closest time to event: {closest_time.strftime('%Y-%m-%d %H:%M:%S')} UT")
            print(f"   Time difference: {closest_diff:.1f} seconds")
            
            if closest_diff < 300:  # Within 5 minutes
                print(f"   ✅ Event time found in data!")
                sample_b = b_data[closest_index]
                print(f"   B-field at event time: [{sample_b[0]:.2f}, {sample_b[1]:.2f}, {sample_b[2]:.2f}] nT")
            else:
                print(f"   ❌ Event time NOT found in data!")
                print(f"   Closest data is {closest_diff/60:.1f} minutes away")
            
            # Create a simple time series plot to visualize the issue
            print(f"\n📊 Creating diagnostic plot...")
            
            fig, ax = plt.subplots(1, 1, figsize=(14, 6))
            
            b_total = np.sqrt(np.sum(b_data**2, axis=1))
            ax.plot(time_objects, b_total, 'b-', linewidth=2, label='|B| (nT)')
            
            # Mark requested time range
            ax.axvline(start_time, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Requested Start')
            ax.axvline(end_time, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Requested End')
            ax.axvline(event_time, color='red', linestyle='-', alpha=0.8, linewidth=3, label='Event Time')
            
            ax.set_xlabel('Time (UT)')
            ax.set_ylabel('|B| (nT)')
            ax.set_title(f'MMS1 Magnetic Field: Actual vs Requested Time Range')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis to show hours
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig('debug_time_range_issue.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Diagnostic plot saved: debug_time_range_issue.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_data_availability():
    """Check what data is actually available for this date"""
    
    print(f"\n🔍 CHECKING DATA AVAILABILITY")
    print("=" * 80)
    
    # Check different time ranges to see what data exists
    test_ranges = [
        # Original requested range
        ['2019-01-27/11:30:50', '2019-01-27/13:30:50'],
        # Early morning (where plots are showing data)
        ['2019-01-27/05:00:00', '2019-01-27/08:00:00'],
        # Full day
        ['2019-01-27/00:00:00', '2019-01-27/23:59:59'],
        # Different day format
        ['2019-01-27T11:30:50', '2019-01-27T13:30:50'],
    ]
    
    for i, trange in enumerate(test_ranges):
        print(f"\n📊 Test {i+1}: {trange[0]} to {trange[1]}")
        
        try:
            # Clear previous data
            data_quants.clear()
            
            # Try to load just magnetic field data
            evt = mms_mp.load_event(
                trange=trange,
                probes=['1'],
                include_ephem=False,
                include_edp=False,
                data_rate_fgm='srvy'
            )
            
            if '1' in evt and 'B_gsm' in evt['1']:
                times, b_data = evt['1']['B_gsm']
                
                if len(times) > 0:
                    if hasattr(times[0], 'strftime'):
                        first_time = times[0]
                        last_time = times[-1]
                    else:
                        first_time = datetime.fromtimestamp(times[0])
                        last_time = datetime.fromtimestamp(times[-1])
                    
                    print(f"   ✅ Data found: {len(times)} points")
                    print(f"   Time range: {first_time.strftime('%Y-%m-%d %H:%M:%S')} to {last_time.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    print(f"   ❌ No data points")
            else:
                print(f"   ❌ No magnetic field data")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")


def main():
    """Main debug function"""
    
    print("DEBUG TIME RANGE ISSUE")
    print("=" * 80)
    print("Investigating why plots show 05:00-07:30 UT instead of 12:30:50 UT")
    print("=" * 80)
    
    # Run debugging
    debug_success = debug_time_range_issue()
    check_data_availability()
    
    print(f"\n" + "=" * 80)
    print("DEBUG SUMMARY")
    print("=" * 80)
    
    if debug_success:
        print("✅ Debug analysis completed")
        print("📊 Check debug_time_range_issue.png for time range visualization")
    else:
        print("❌ Debug analysis failed")
    
    print("\n🔍 NEXT STEPS:")
    print("1. Check the diagnostic plot to see actual vs requested time ranges")
    print("2. Verify data file availability for the correct time period")
    print("3. Check for time format or timezone conversion issues")
    print("4. Investigate data loading and filtering processes")


if __name__ == "__main__":
    main()
