"""
Debug MMS Time Issue
Event: 2019-01-27 12:30:50 UT

This script investigates the time zone/conversion issue where data appears
at 06:25-06:35 instead of 12:25-12:35 (6 hour offset).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import MMS modules
from mms_mp import data_loader
import glob
import os

def debug_time_loading():
    """
    Debug the time loading issue
    """
    print("🔍 DEBUGGING MMS TIME LOADING ISSUE")
    print("Expected: 2019-01-27 12:25:00 to 12:35:00 UT")
    print("Observed: 2019-01-27 06:25:00 to 06:35:00 UT (6 hour offset)")
    print("=" * 80)
    
    # Define time range for the magnetopause crossing
    event_time = "2019-01-27T12:30:50"
    start_time = "2019-01-27T12:25:00"  # 5 minutes before
    end_time = "2019-01-27T12:35:00"    # 5 minutes after
    
    trange = [start_time, end_time]
    probes = ['1']  # Just test one spacecraft first
    
    print(f"📅 Requested Time Range: {start_time} to {end_time}")
    print(f"🛰️ Testing Spacecraft: MMS{', MMS'.join(probes)}")
    
    # Check what data files are actually available
    print("\n" + "="*80)
    print("1️⃣ CHECKING AVAILABLE DATA FILES")
    print("="*80)
    
    # Check FPI data files
    fpi_pattern = f"pydata/mms1/fpi/fast/l2/dis-moms/2019/01/mms1_fpi_fast_l2_dis-moms_20190127*_v*.cdf"
    fpi_files = glob.glob(fpi_pattern)
    print(f"FPI files found: {len(fpi_files)}")
    for f in fpi_files:
        filename = os.path.basename(f)
        print(f"  {filename}")
        
        # Extract time from filename
        if "20190127" in filename:
            # Look for time part in filename
            parts = filename.split('_')
            for part in parts:
                if len(part) == 14 and part.startswith('20190127'):
                    date_part = part[:8]  # 20190127
                    time_part = part[8:]  # HHMMSS
                    if len(time_part) == 6:
                        hour = time_part[:2]
                        minute = time_part[2:4]
                        second = time_part[4:6]
                        print(f"    File time: {date_part} {hour}:{minute}:{second}")
    
    # Check FGM data files  
    fgm_pattern = f"pydata/mms1/fgm/srvy/l2/2019/01/mms1_fgm_srvy_l2_20190127_v*.cdf"
    fgm_files = glob.glob(fgm_pattern)
    print(f"\nFGM files found: {len(fgm_files)}")
    for f in fgm_files:
        filename = os.path.basename(f)
        print(f"  {filename}")
    
    # Load data using the MMS loader
    print("\n" + "="*80)
    print("2️⃣ LOADING DATA AND CHECKING TIMES")
    print("="*80)
    
    try:
        # Load with different time ranges to see what happens
        test_ranges = [
            ["2019-01-27T12:25:00", "2019-01-27T12:35:00"],  # Original request
            ["2019-01-27T06:25:00", "2019-01-27T06:35:00"],  # Observed time
            ["2019-01-27T00:00:00", "2019-01-27T23:59:59"],  # Full day
        ]
        
        for i, test_range in enumerate(test_ranges):
            print(f"\n--- Test {i+1}: {test_range[0]} to {test_range[1]} ---")
            
            try:
                evt = data_loader.load_event(
                    test_range, probes,
                    data_rate_fgm='srvy',    # Survey mode
                    data_rate_fpi='fast',    # Fast mode
                    include_edp=False,
                    include_ephem=True
                )
                
                if '1' in evt and evt['1']:
                    print(f"✅ Data loaded successfully")
                    
                    # Check each variable's time range
                    for var_name, (t_data, values) in evt['1'].items():
                        if len(t_data) > 0:
                            # Convert timestamps to datetime
                            start_dt = datetime.fromtimestamp(t_data[0])
                            end_dt = datetime.fromtimestamp(t_data[-1])
                            
                            print(f"  {var_name}: {len(t_data)} points")
                            print(f"    Time range: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} to {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                            
                            # Check if this covers our target time
                            target_dt = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
                            if start_dt <= target_dt <= end_dt:
                                print(f"    ✅ Contains target time {target_dt.strftime('%H:%M:%S')}")
                            else:
                                print(f"    ❌ Does NOT contain target time {target_dt.strftime('%H:%M:%S')}")
                else:
                    print(f"❌ No data loaded")
                    
            except Exception as e:
                print(f"❌ Loading failed: {e}")
    
    except Exception as e:
        print(f"❌ Overall loading failed: {e}")
    
    # Test different time zone interpretations
    print("\n" + "="*80)
    print("3️⃣ TESTING TIME ZONE INTERPRETATIONS")
    print("="*80)
    
    # Test if the issue is UTC vs local time
    target_utc = datetime.fromisoformat("2019-01-27T12:30:50+00:00")
    print(f"Target UTC time: {target_utc}")
    print(f"Target UTC timestamp: {target_utc.timestamp()}")
    
    # Check if 6-hour offset suggests a time zone issue
    offset_time = target_utc - timedelta(hours=6)
    print(f"6-hour earlier: {offset_time}")
    print(f"6-hour earlier timestamp: {offset_time.timestamp()}")
    
    # Test loading with the offset time
    print(f"\n--- Testing with 6-hour offset ---")
    offset_start = "2019-01-27T06:25:00"
    offset_end = "2019-01-27T06:35:00"
    
    try:
        evt_offset = data_loader.load_event(
            [offset_start, offset_end], probes,
            data_rate_fgm='srvy',
            data_rate_fpi='fast',
            include_edp=False,
            include_ephem=True
        )
        
        if '1' in evt_offset and evt_offset['1']:
            print(f"✅ Offset time data loaded successfully")
            
            # Check if this data actually corresponds to our target time
            for var_name, (t_data, values) in evt_offset['1'].items():
                if len(t_data) > 0 and var_name == 'N_tot':  # Focus on density
                    start_dt = datetime.fromtimestamp(t_data[0])
                    end_dt = datetime.fromtimestamp(t_data[-1])
                    
                    print(f"  {var_name}: {start_dt.strftime('%H:%M:%S')} to {end_dt.strftime('%H:%M:%S')}")
                    
                    # Check data values to see if they look reasonable
                    valid_data = ~np.isnan(values)
                    if np.any(valid_data):
                        valid_values = values[valid_data]
                        print(f"    Data range: {np.min(valid_values):.2f} to {np.max(valid_values):.2f}")
                        print(f"    Mean: {np.mean(valid_values):.2f}")
                    else:
                        print(f"    All data is NaN")
        else:
            print(f"❌ No offset data loaded")
            
    except Exception as e:
        print(f"❌ Offset loading failed: {e}")


def test_full_day_coverage():
    """
    Test loading a full day to see what time ranges have data
    """
    print("\n" + "="*80)
    print("4️⃣ TESTING FULL DAY COVERAGE")
    print("="*80)
    
    # Load full day
    full_day_range = ["2019-01-27T00:00:00", "2019-01-27T23:59:59"]
    
    try:
        evt_full = data_loader.load_event(
            full_day_range, ['1'],
            data_rate_fgm='srvy',
            data_rate_fpi='fast',
            include_edp=False,
            include_ephem=True
        )
        
        if '1' in evt_full and evt_full['1']:
            print(f"✅ Full day data loaded successfully")
            
            # Check time coverage for each variable
            for var_name, (t_data, values) in evt_full['1'].items():
                if len(t_data) > 0:
                    # Convert all timestamps to datetime
                    times = [datetime.fromtimestamp(t) for t in t_data]
                    
                    start_dt = min(times)
                    end_dt = max(times)
                    
                    print(f"\n  {var_name}: {len(t_data)} points")
                    print(f"    Full range: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} to {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Check for gaps around our target time
                    target_dt = datetime.fromisoformat("2019-01-27T12:30:50")
                    
                    # Find times within 1 hour of target
                    nearby_times = [t for t in times if abs((t - target_dt).total_seconds()) < 3600]
                    
                    if nearby_times:
                        nearby_start = min(nearby_times)
                        nearby_end = max(nearby_times)
                        print(f"    Near target: {nearby_start.strftime('%H:%M:%S')} to {nearby_end.strftime('%H:%M:%S')}")
                        
                        # Check if target time is covered
                        if nearby_start <= target_dt <= nearby_end:
                            print(f"    ✅ Target time {target_dt.strftime('%H:%M:%S')} IS covered")
                        else:
                            print(f"    ❌ Target time {target_dt.strftime('%H:%M:%S')} NOT covered")
                    else:
                        print(f"    ❌ No data within 1 hour of target time")
        else:
            print(f"❌ No full day data loaded")
            
    except Exception as e:
        print(f"❌ Full day loading failed: {e}")


def create_time_coverage_plot():
    """
    Create a plot showing time coverage
    """
    print("\n" + "="*80)
    print("5️⃣ CREATING TIME COVERAGE PLOT")
    print("="*80)
    
    # Load full day
    full_day_range = ["2019-01-27T00:00:00", "2019-01-27T23:59:59"]
    
    try:
        evt_full = data_loader.load_event(
            full_day_range, ['1'],
            data_rate_fgm='srvy',
            data_rate_fpi='fast',
            include_edp=False,
            include_ephem=True
        )
        
        if '1' in evt_full and evt_full['1']:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            y_pos = 0
            
            for i, (var_name, (t_data, values)) in enumerate(evt_full['1'].items()):
                if len(t_data) > 0:
                    # Convert timestamps to datetime
                    times = [datetime.fromtimestamp(t) for t in t_data]
                    
                    # Plot time coverage
                    ax.scatter([t.hour + t.minute/60.0 for t in times], 
                              [y_pos] * len(times), 
                              c=colors[i % len(colors)], 
                              alpha=0.6, s=1, label=var_name)
                    
                    y_pos += 1
            
            # Mark target time
            target_hour = 12 + 30/60.0  # 12:30
            ax.axvline(target_hour, color='red', linestyle='--', linewidth=2, 
                      label='Target Time (12:30 UT)')
            
            # Mark observed time
            observed_hour = 6 + 30/60.0  # 06:30
            ax.axvline(observed_hour, color='orange', linestyle='--', linewidth=2,
                      label='Observed Time (06:30 UT)')
            
            ax.set_xlabel('Hour of Day (UT)')
            ax.set_ylabel('Variable')
            ax.set_title('MMS1 Data Time Coverage - 2019-01-27')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 24)
            
            plt.tight_layout()
            plt.savefig('mms_time_coverage_debug.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print("✅ Time coverage plot saved: mms_time_coverage_debug.png")
        else:
            print("❌ No data for coverage plot")
            
    except Exception as e:
        print(f"❌ Coverage plot failed: {e}")


if __name__ == "__main__":
    print("🔍 MMS TIME DEBUGGING")
    print("Investigating 6-hour offset issue")
    print()
    
    debug_time_loading()
    test_full_day_coverage()
    create_time_coverage_plot()
    
    print("\n🎯 SUMMARY:")
    print("This debug script investigates why data appears at 06:25-06:35")
    print("instead of the requested 12:25-12:35 time range.")
    print("Possible causes:")
    print("  • Time zone conversion issue (UTC vs local)")
    print("  • Epoch time interpretation problem")
    print("  • Data file time range mismatch")
    print("  • MMS data loader time handling bug")
    print("\nCheck the output above and the coverage plot for clues!")
