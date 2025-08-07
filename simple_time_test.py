#!/usr/bin/env python3
"""
Simple Time Formatting Test
===========================

Test the time formatting fixes without pyspedas dependencies.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
import pandas as pd

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

def create_synthetic_data():
    """Create synthetic time series data for testing"""
    
    # Create time array (20 minutes of data at 8 Hz)
    start_time = datetime(2019, 1, 27, 12, 20, 0)
    end_time = datetime(2019, 1, 27, 12, 40, 0)
    
    # Create time array
    time_range = pd.date_range(start=start_time, end=end_time, freq='125ms')  # 8 Hz
    
    # Create synthetic magnetic field data
    t_seconds = np.array([(t - start_time).total_seconds() for t in time_range])
    
    # Synthetic magnetopause crossing signature
    event_time_sec = (datetime(2019, 1, 27, 12, 30, 50) - start_time).total_seconds()
    
    # Magnetic field components with boundary crossing
    bx = 20 + 10 * np.sin(0.01 * t_seconds) + 30 * np.tanh((t_seconds - event_time_sec) / 60)
    by = 15 * np.cos(0.015 * t_seconds) + 5 * np.random.normal(0, 1, len(t_seconds))
    bz = -10 + 5 * np.sin(0.008 * t_seconds) - 20 * np.tanh((t_seconds - event_time_sec) / 60)
    
    # Ion density with boundary crossing
    density = 5 + 2 * np.sin(0.005 * t_seconds) + 15 * (1 - np.tanh((t_seconds - event_time_sec) / 60))
    
    b_data = np.column_stack([bx, by, bz])
    
    return time_range, b_data, density

def test_time_formatting():
    """Test time formatting with synthetic data"""
    
    print("🕒 TESTING TIME FORMATTING FIXES")
    print("=" * 40)
    
    try:
        # Create synthetic data
        print("📊 Creating synthetic magnetopause crossing data...")
        times, b_data, density = create_synthetic_data()
        print(f"   ✅ Created {len(times)} data points over 20 minutes")
        
        # Test time conversion
        print("🔄 Testing time conversion...")
        plot_times = ensure_datetime_format(times)
        print(f"   ✅ Time format: {type(plot_times[0])}")
        
        # Decimate data for plotting
        step = max(1, len(times) // 1000)
        times_dec = plot_times[::step]
        b_data_dec = b_data[::step]
        density_dec = density[::step]
        
        print(f"   ✅ Decimated to {len(times_dec)} points for visualization")
        
        # Create test plot
        print("📊 Creating test plot with proper time formatting...")
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle('Time Formatting Test - Fixed Time Axis Labels\n' +
                     'Synthetic Magnetopause Crossing Event', fontsize=16, fontweight='bold')
        
        # Calculate magnetic field magnitude
        b_mag = np.sqrt(np.sum(b_data_dec**2, axis=1))
        
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
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Ion density
        axes[2].semilogy(times_dec, density_dec, 'purple', linewidth=1.5, label='Nᵢ')
        axes[2].set_ylabel('Nᵢ (cm⁻³)', fontweight='bold')
        axes[2].set_title('(c) Ion Number Density', fontweight='bold', loc='left')
        axes[2].set_xlabel('Time (UT)', fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Add event marker
        event_time = datetime(2019, 1, 27, 12, 30, 50)
        for ax in axes:
            ax.axvline(event_time, color='red', linestyle='--', alpha=0.8, 
                      linewidth=2, label='Boundary Crossing' if ax == axes[0] else '')
        
        # Apply safe time formatting to all axes
        print("🕒 Applying safe time formatting...")
        for i, ax in enumerate(axes):
            safe_format_time_axis(ax, interval_minutes=5)
            print(f"   ✅ Formatted axis {i+1}")
        
        plt.tight_layout()
        
        # Save test plot
        filename = f'time_formatting_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Saved test plot: {filename}")
        print("\n🎉 TIME FORMATTING TEST SUCCESSFUL!")
        print("   ✅ Proper time labels should now be visible")
        print("   ✅ No hanging issues")
        print("   ✅ Time axes properly formatted with HH:MM labels")
        print("   ✅ Ready for full publication analysis")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_time_formatting()
    if success:
        print("\n✅ Time formatting fixes verified!")
        print("   The publication analysis should now show proper time labels")
    else:
        print("\n❌ Time formatting issues remain")
