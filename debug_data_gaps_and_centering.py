#!/usr/bin/env python3
"""
Debug Data Gaps and Centering Issue
===================================

This script investigates why there are data gaps and why the plots
are not properly centered on 12:30 UT despite the timezone correction.
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


def investigate_data_availability():
    """Investigate what data is actually available around 12:30 UT"""
    
    print("🔍 INVESTIGATING DATA AVAILABILITY AROUND 12:30 UT")
    print("=" * 80)
    
    # Event time
    event_time = datetime(2019, 1, 27, 12, 30, 50)
    
    # Test different time windows around the event
    test_windows = [
        # Very narrow window around event
        (event_time - timedelta(minutes=30), event_time + timedelta(minutes=30)),
        # 1 hour window
        (event_time - timedelta(hours=1), event_time + timedelta(hours=1)),
        # 2 hour window  
        (event_time - timedelta(hours=2), event_time + timedelta(hours=2)),
        # Full day to see what's available
        (datetime(2019, 1, 27, 0, 0, 0), datetime(2019, 1, 27, 23, 59, 59)),
    ]
    
    window_names = ["30-minute window", "1-hour window", "2-hour window", "Full day"]
    
    for i, ((start_time, end_time), window_name) in enumerate(zip(test_windows, window_names)):
        print(f"\n📊 Test {i+1}: {window_name}")
        print(f"   Time range: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')} UT")
        
        # Use direct time format (no timezone compensation for this test)
        trange = [
            start_time.strftime('%Y-%m-%d/%H:%M:%S'),
            end_time.strftime('%Y-%m-%d/%H:%M:%S')
        ]
        
        try:
            # Clear previous data
            data_quants.clear()
            
            # Load FGM data
            result = mms.mms_load_fgm(
                trange=trange,
                probe='1',
                data_rate='srvy',
                level='l2',
                time_clip=True,
                notplot=False
            )
            
            if 'mms1_fgm_b_gsm_srvy_l2' in data_quants:
                times, b_data = get_data('mms1_fgm_b_gsm_srvy_l2')
                
                if len(times) > 0:
                    # Convert times
                    if hasattr(times[0], 'strftime'):
                        first_time = times[0]
                        last_time = times[-1]
                        time_objects = times
                    else:
                        first_time = datetime.fromtimestamp(times[0])
                        last_time = datetime.fromtimestamp(times[-1])
                        time_objects = [datetime.fromtimestamp(t) for t in times]
                    
                    print(f"   ✅ Data loaded: {len(times)} points")
                    print(f"   Actual range: {first_time.strftime('%Y-%m-%d %H:%M:%S')} to {last_time.strftime('%Y-%m-%d %H:%M:%S')} UT")
                    
                    # Check if event time is in range
                    if first_time <= event_time <= last_time:
                        print(f"   ✅ Event time IS in range")
                        
                        # Find closest time to event
                        time_diffs = [abs((t - event_time).total_seconds()) for t in time_objects]
                        closest_index = np.argmin(time_diffs)
                        closest_time = time_objects[closest_index]
                        closest_diff = min(time_diffs)
                        
                        print(f"   Closest data point: {closest_time.strftime('%Y-%m-%d %H:%M:%S')} UT")
                        print(f"   Time difference: {closest_diff:.1f} seconds")
                        
                        # Check for data gaps
                        time_diffs_between_points = np.diff([t.timestamp() if hasattr(t, 'timestamp') else t for t in time_objects])
                        max_gap = np.max(time_diffs_between_points)
                        avg_cadence = np.mean(time_diffs_between_points)
                        
                        print(f"   Average cadence: {avg_cadence:.1f} seconds")
                        print(f"   Maximum gap: {max_gap:.1f} seconds ({max_gap/60:.1f} minutes)")
                        
                        if max_gap > 300:  # More than 5 minutes
                            print(f"   ⚠️ Large data gap detected!")
                            
                            # Find where the gap is
                            gap_indices = np.where(time_diffs_between_points > 300)[0]
                            for gap_idx in gap_indices:
                                gap_start = time_objects[gap_idx]
                                gap_end = time_objects[gap_idx + 1]
                                gap_duration = (gap_end - gap_start).total_seconds() / 60
                                print(f"      Gap: {gap_start.strftime('%H:%M:%S')} to {gap_end.strftime('%H:%M:%S')} ({gap_duration:.1f} min)")
                        else:
                            print(f"   ✅ No significant data gaps")
                    else:
                        print(f"   ❌ Event time NOT in range")
                        time_diff = (event_time - first_time).total_seconds() / 3600
                        print(f"      Event time is {time_diff:.1f} hours from start of data")
                else:
                    print(f"   ❌ No data points")
            else:
                print(f"   ❌ No FGM data variable found")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")


def test_different_data_rates():
    """Test different data rates to see if higher resolution data is available"""
    
    print(f"\n🔍 TESTING DIFFERENT DATA RATES")
    print("=" * 80)
    
    event_time = datetime(2019, 1, 27, 12, 30, 50)
    start_time = event_time - timedelta(hours=1)
    end_time = event_time + timedelta(hours=1)
    
    trange = [
        start_time.strftime('%Y-%m-%d/%H:%M:%S'),
        end_time.strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    data_rates = ['brst', 'fast', 'srvy']
    
    for data_rate in data_rates:
        print(f"\n📊 Testing {data_rate} mode:")
        
        try:
            data_quants.clear()
            
            result = mms.mms_load_fgm(
                trange=trange,
                probe='1',
                data_rate=data_rate,
                level='l2',
                time_clip=True,
                notplot=False
            )
            
            # Check what variables were loaded
            fgm_vars = [var for var in data_quants.keys() if 'fgm' in var and 'b_gsm' in var]
            
            if fgm_vars:
                var_name = fgm_vars[0]
                times, b_data = get_data(var_name)
                
                if len(times) > 0:
                    if hasattr(times[0], 'strftime'):
                        first_time = times[0]
                        last_time = times[-1]
                    else:
                        first_time = datetime.fromtimestamp(times[0])
                        last_time = datetime.fromtimestamp(times[-1])
                    
                    print(f"   ✅ {data_rate} data: {len(times)} points")
                    print(f"   Range: {first_time.strftime('%Y-%m-%d %H:%M:%S')} to {last_time.strftime('%Y-%m-%d %H:%M:%S')} UT")
                    
                    # Check if event time is in range
                    if first_time <= event_time <= last_time:
                        print(f"   ✅ Event time in range")
                    else:
                        print(f"   ❌ Event time not in range")
                else:
                    print(f"   ❌ No data points")
            else:
                print(f"   ❌ No FGM variables loaded")
                print(f"   Available variables: {list(data_quants.keys())}")
                
        except Exception as e:
            print(f"   ❌ Error with {data_rate}: {e}")


def create_diagnostic_plot():
    """Create a diagnostic plot to visualize the data availability issue"""
    
    print(f"\n🔍 CREATING DIAGNOSTIC PLOT")
    print("=" * 80)
    
    event_time = datetime(2019, 1, 27, 12, 30, 50)
    
    # Load a wider time range to see the full picture
    start_time = datetime(2019, 1, 27, 10, 0, 0)
    end_time = datetime(2019, 1, 27, 16, 0, 0)
    
    trange = [
        start_time.strftime('%Y-%m-%d/%H:%M:%S'),
        end_time.strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    try:
        data_quants.clear()
        
        # Load FGM data for wider range
        result = mms.mms_load_fgm(
            trange=trange,
            probe='1',
            data_rate='srvy',
            level='l2',
            time_clip=True,
            notplot=False
        )
        
        if 'mms1_fgm_b_gsm_srvy_l2' in data_quants:
            times, b_data = get_data('mms1_fgm_b_gsm_srvy_l2')
            
            if len(times) > 0:
                # Convert times
                if hasattr(times[0], 'strftime'):
                    time_objects = times
                else:
                    time_objects = [datetime.fromtimestamp(t) for t in times]
                
                # Create diagnostic plot
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
                
                # Plot 1: Full time range
                b_total = np.sqrt(np.sum(b_data**2, axis=1))
                ax1.plot(time_objects, b_total, 'b-', linewidth=1, alpha=0.7)
                ax1.axvline(event_time, color='red', linestyle='--', linewidth=3, label='Event Time (12:30:50 UT)')
                ax1.set_ylabel('|B| (nT)')
                ax1.set_title('MMS1 Magnetic Field: Full Time Range')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Format x-axis
                import matplotlib.dates as mdates
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                
                # Plot 2: Zoomed around event time
                # Find data within 1 hour of event
                event_mask = [abs((t - event_time).total_seconds()) <= 3600 for t in time_objects]
                
                if any(event_mask):
                    event_times = [t for t, mask in zip(time_objects, event_mask) if mask]
                    event_b = b_total[event_mask]
                    
                    ax2.plot(event_times, event_b, 'b-', linewidth=2)
                    ax2.axvline(event_time, color='red', linestyle='--', linewidth=3, label='Event Time (12:30:50 UT)')
                    ax2.set_ylabel('|B| (nT)')
                    ax2.set_xlabel('Time (UT)')
                    ax2.set_title('MMS1 Magnetic Field: Zoomed Around Event Time')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
                    
                    print(f"   Event time data points: {len(event_times)}")
                else:
                    ax2.text(0.5, 0.5, 'No data available around event time', 
                            transform=ax2.transAxes, ha='center', va='center', fontsize=14)
                    ax2.set_title('No Data Around Event Time')
                
                plt.tight_layout()
                plt.savefig('diagnostic_data_availability.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ Diagnostic plot saved: diagnostic_data_availability.png")
                
                # Print summary statistics
                first_time = time_objects[0]
                last_time = time_objects[-1]
                
                print(f"\n📊 Data Summary:")
                print(f"   Total data points: {len(times)}")
                print(f"   Time range: {first_time.strftime('%Y-%m-%d %H:%M:%S')} to {last_time.strftime('%Y-%m-%d %H:%M:%S')} UT")
                print(f"   Duration: {(last_time - first_time).total_seconds() / 3600:.1f} hours")
                
                # Check if event time is covered
                if first_time <= event_time <= last_time:
                    print(f"   ✅ Event time IS covered by data")
                else:
                    print(f"   ❌ Event time NOT covered by data")
                    time_diff = (event_time - first_time).total_seconds() / 3600
                    print(f"      Event time is {time_diff:.1f} hours from start of data")
            else:
                print(f"   ❌ No data points loaded")
        else:
            print(f"   ❌ No FGM data variable found")
            
    except Exception as e:
        print(f"   ❌ Error creating diagnostic plot: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main debug function"""
    
    print("DEBUG DATA GAPS AND CENTERING ISSUE")
    print("=" * 80)
    print("Investigating data gaps and centering problems around 12:30 UT")
    print("=" * 80)
    
    # Run investigations
    investigate_data_availability()
    test_different_data_rates()
    create_diagnostic_plot()
    
    print(f"\n" + "=" * 80)
    print("INVESTIGATION SUMMARY")
    print("=" * 80)
    
    print("🔍 KEY FINDINGS:")
    print("1. Check if data gaps exist around 12:30 UT")
    print("2. Verify which data rates have coverage")
    print("3. Examine actual vs requested time ranges")
    print("4. Look for file availability issues")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Review diagnostic_data_availability.png")
    print("2. Check for missing data files")
    print("3. Try alternative time ranges")
    print("4. Consider different data products")


if __name__ == "__main__":
    main()
