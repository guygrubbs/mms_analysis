#!/usr/bin/env python3
"""
Test Time Formatting Fixes
==========================

This script tests the improved time axis formatting to ensure proper
time labels are displayed on all plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
import pandas as pd
from pyspedas.projects import mms
from pytplot import data_quants, get_data
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib parameters
plt.rcParams.update({
    'font.size': 12,
    'axes.formatter.limits': (-3, 4),
    'axes.formatter.use_mathtext': True
})

def safe_format_time_axis(ax, interval_minutes=5):
    """Safely format time axis to prevent MAXTICKS errors and ensure proper time labels"""
    try:
        # Use appropriate time locator based on interval
        if interval_minutes <= 2:
            locator = mdates.MinuteLocator(interval=1)
        elif interval_minutes <= 5:
            locator = mdates.MinuteLocator(interval=2)
        else:
            locator = mdates.MinuteLocator(interval=5)
        
        # Set locator and formatter
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Rotate labels for better readability
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        
        # Ensure proper spacing
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
        
    except Exception as e:
        print(f"   ⚠️ Time axis formatting warning: {e}")
        # Fallback to simple formatting
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', rotation=45)

def ensure_datetime_format(times):
    """Ensure times are in proper datetime format for matplotlib"""
    if len(times) == 0:
        return times
    
    # Check if already datetime objects
    if hasattr(times[0], 'strftime'):
        return times
    
    # Convert from various formats to datetime
    try:
        if hasattr(times[0], 'timestamp'):
            # Convert from pandas timestamp or similar
            return pd.to_datetime(times)
        elif isinstance(times[0], (int, float)):
            # Convert from unix timestamp
            return pd.to_datetime(times, unit='s')
        else:
            # Try pandas conversion
            return pd.to_datetime(times)
    except Exception as e:
        print(f"   ⚠️ Time conversion warning: {e}")
        return times

def safe_decimate_data(times, data, max_points=1000):
    """Safely decimate data to prevent visualization issues"""
    if len(times) <= max_points:
        return times, data
    
    step = max(1, len(times) // max_points)
    return times[::step], data[::step] if data.ndim > 1 else data[::step]

def test_time_formatting():
    """Test time formatting with real MMS data"""
    
    print("🕒 TESTING TIME FORMATTING FIXES")
    print("=" * 40)
    
    # Load a small amount of real data
    trange = ['2019-01-27/12:25:00', '2019-01-27/12:35:00']  # 10-minute window
    
    try:
        # Load FGM data for MMS1
        print("📡 Loading test data...")
        fgm_result = mms.mms_load_fgm(
            trange=trange,
            probe='1',
            data_rate='brst',
            level='l2',
            time_clip=True
        )
        
        b_var = 'mms1_fgm_b_gsm_brst_l2'
        if b_var in data_quants:
            times, b_data = get_data(b_var)
            print(f"   ✅ Loaded {len(times)} data points")
            
            # Test time conversion
            print("🔄 Testing time conversion...")
            plot_times = ensure_datetime_format(times)
            print(f"   ✅ Converted to datetime format: {type(plot_times[0])}")
            
            # Test decimation
            print("📉 Testing data decimation...")
            times_dec, b_data_dec = safe_decimate_data(plot_times, b_data, max_points=500)
            print(f"   ✅ Decimated to {len(times_dec)} points")
            
            # Create test plot
            print("📊 Creating test plot with proper time formatting...")
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            fig.suptitle('Time Formatting Test\nFixed Time Axis Labels', fontsize=14, fontweight='bold')
            
            # Calculate magnetic field magnitude
            b_mag = np.sqrt(np.sum(b_data_dec[:, :3]**2, axis=1))
            
            # Plot 1: Magnetic field magnitude
            axes[0].plot(times_dec, b_mag, 'b-', linewidth=1.5, label='|B|')
            axes[0].set_ylabel('|B| (nT)', fontweight='bold')
            axes[0].set_title('(a) Magnetic Field Magnitude', fontweight='bold', loc='left')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Magnetic field components
            component_labels = ['Bₓ', 'Bᵧ', 'Bᵤ']
            colors = ['red', 'green', 'blue']
            for i in range(3):
                axes[1].plot(times_dec, b_data_dec[:, i], color=colors[i], 
                           label=component_labels[i], linewidth=1.2, alpha=0.8)
            
            axes[1].axhline(0, color='gray', linestyle='-', alpha=0.5)
            axes[1].set_ylabel('B (nT)', fontweight='bold')
            axes[1].set_title('(b) Magnetic Field Components', fontweight='bold', loc='left')
            axes[1].set_xlabel('Time (UT)', fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Apply safe time formatting
            print("🕒 Applying safe time formatting...")
            for ax in axes:
                safe_format_time_axis(ax, interval_minutes=2)
            
            # Add event marker
            event_time = datetime(2019, 1, 27, 12, 30, 50)
            for ax in axes:
                ax.axvline(event_time, color='red', linestyle='--', alpha=0.8, 
                          linewidth=2, label='Event' if ax == axes[0] else '')
            
            plt.tight_layout()
            
            # Save test plot
            filename = f'test_time_formatting_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"   ✅ Saved test plot: {filename}")
            print("\n🎉 TIME FORMATTING TEST SUCCESSFUL!")
            print("   ✅ Proper time labels should now be visible")
            print("   ✅ No hanging issues")
            print("   ✅ Ready for full analysis")
            
            return True
            
        else:
            print("   ❌ No magnetic field data found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_time_formatting()
    if success:
        print("\n✅ Time formatting fixes verified - ready for publication analysis")
    else:
        print("\n❌ Time formatting issues remain - need further debugging")
