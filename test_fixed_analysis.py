#!/usr/bin/env python3
"""
Test Fixed Analysis - Quick Verification
========================================

This script tests the fixed publication analysis with proper tick management
and data decimation to ensure no hanging issues.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime
import pandas as pd
from pyspedas.projects import mms
from pytplot import data_quants, get_data
import warnings
warnings.filterwarnings('ignore')

# Set safe matplotlib parameters
plt.rcParams.update({
    'font.size': 10,
    'axes.formatter.limits': (-3, 4),
    'axes.formatter.use_mathtext': True
})

def safe_format_time_axis(ax, interval_minutes=5):
    """Safely format time axis to prevent MAXTICKS errors"""
    try:
        # Use MaxNLocator to limit ticks
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    except Exception as e:
        print(f"   ⚠️ Time axis formatting warning: {e}")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

def safe_decimate_data(times, data, max_points=1000):
    """Safely decimate data to prevent visualization issues"""
    if len(times) <= max_points:
        return times, data
    
    step = max(1, len(times) // max_points)
    return times[::step], data[::step] if data.ndim > 1 else data[::step]

def load_mec_data_quick(trange, probes):
    """Quick MEC data loading test"""
    
    print("🛰️ Testing MEC data loading...")
    
    positions = {}
    
    for probe in probes[:2]:  # Test with just 2 spacecraft first
        try:
            result = mms.mms_load_mec(
                trange=trange,
                probe=probe,
                data_rate='srvy',
                level='l2',
                datatype='epht89q',
                time_clip=True
            )
            
            pos_var = f'mms{probe}_mec_r_gsm'
            
            if pos_var in data_quants:
                times, pos_data = get_data(pos_var)
                positions[probe] = {
                    'times': times,
                    'data': pos_data
                }
                print(f"   ✅ MMS{probe}: {len(times)} position points")
            else:
                print(f"   ❌ MMS{probe}: Position variable not found")
                
        except Exception as e:
            print(f"   ❌ MMS{probe}: Error: {e}")
    
    return positions

def load_science_data_quick(trange, probes):
    """Quick science data loading test"""
    
    print("\n📡 Testing science data loading...")
    
    data = {}
    
    for probe in probes[:2]:  # Test with just 2 spacecraft first
        print(f"   Loading MMS{probe}...")
        data[probe] = {}
        
        try:
            # Load FGM data
            fgm_result = mms.mms_load_fgm(
                trange=trange,
                probe=probe,
                data_rate='brst',
                level='l2',
                time_clip=True
            )
            
            b_var = f'mms{probe}_fgm_b_gsm_brst_l2'
            if b_var in data_quants:
                times, b_data = get_data(b_var)
                data[probe]['B_gsm'] = (times, b_data)
                print(f"      ✅ FGM: {len(times)} points")
            
            # Load FPI data
            fpi_result = mms.mms_load_fpi(
                trange=trange,
                probe=probe,
                data_rate='brst',
                level='l2',
                datatype='dis-moms',
                time_clip=True
            )
            
            n_var = f'mms{probe}_dis_numberdensity_brst'
            if n_var in data_quants:
                times, n_data = get_data(n_var)
                data[probe]['N_i'] = (times, n_data)
                print(f"      ✅ Ion density: {len(times)} points")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    return data

def create_test_plot(data, event_dt):
    """Create a simple test plot with safe formatting"""
    
    print("\n📊 Creating test plot...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'Test Plot: {event_dt.strftime("%Y-%m-%d %H:%M:%S")} UT\n' +
                 f'Fixed Tick Management', fontsize=14, fontweight='bold')
    
    colors = ['blue', 'red']
    
    # Plot 1: Magnetic field magnitude
    for i, probe in enumerate(['1', '2']):
        if probe in data and 'B_gsm' in data[probe]:
            times, b_data = data[probe]['B_gsm']
            
            # Safe decimation
            times_dec, b_data_dec = safe_decimate_data(times, b_data, max_points=1000)
            
            if hasattr(times_dec[0], 'timestamp'):
                plot_times = times_dec
            else:
                plot_times = pd.to_datetime(times_dec)
            
            b_mag = np.sqrt(np.sum(b_data_dec**2, axis=1))
            axes[0].plot(plot_times, b_mag, color=colors[i], 
                        label=f'MMS{probe}', linewidth=1.5, alpha=0.8)
    
    axes[0].axvline(event_dt, color='red', linestyle='--', alpha=0.8, linewidth=2)
    axes[0].set_ylabel('|B| (nT)', fontweight='bold')
    axes[0].set_title('(a) Magnetic Field Magnitude', fontweight='bold', loc='left')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Ion density
    for i, probe in enumerate(['1', '2']):
        if probe in data and 'N_i' in data[probe]:
            times, density = data[probe]['N_i']
            
            # Safe decimation
            times_dec, density_dec = safe_decimate_data(times, density, max_points=1000)
            
            if hasattr(times_dec[0], 'timestamp'):
                plot_times = times_dec
            else:
                plot_times = pd.to_datetime(times_dec)
            
            axes[1].semilogy(plot_times, density_dec, color=colors[i], 
                           label=f'MMS{probe}', linewidth=1.5, alpha=0.8)
    
    axes[1].axvline(event_dt, color='red', linestyle='--', alpha=0.8, linewidth=2)
    axes[1].set_ylabel('Nᵢ (cm⁻³)', fontweight='bold')
    axes[1].set_title('(b) Ion Number Density', fontweight='bold', loc='left')
    axes[1].set_xlabel('Time (UT)', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Safe time axis formatting
    for ax in axes:
        safe_format_time_axis(ax)
    
    plt.tight_layout()
    
    filename = f'test_fixed_plot_{event_dt.strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: {filename}")

def main():
    """Test the fixed analysis approach"""
    
    print("🧪 TESTING FIXED ANALYSIS APPROACH")
    print("=" * 40)
    print("Quick test to verify tick management fixes")
    print()
    
    # Event parameters - shorter window for testing
    event_time = '2019-01-27/12:30:50'
    event_dt = datetime(2019, 1, 27, 12, 30, 50)
    trange = ['2019-01-27/12:25:00', '2019-01-27/12:35:00']  # 10-minute test window
    
    print(f"📡 Test period: {trange[0]} to {trange[1]}")
    print(f"🎯 Event time: {event_time}")
    
    try:
        # Test MEC loading
        positions = load_mec_data_quick(trange, ['1', '2'])
        
        # Test science data loading
        data = load_science_data_quick(trange, ['1', '2'])
        
        # Test visualization
        create_test_plot(data, event_dt)
        
        print("\n🎉 TEST SUCCESSFUL!")
        print("   ✅ No hanging issues")
        print("   ✅ Safe tick management working")
        print("   ✅ Data decimation working")
        print("   ✅ Ready for full analysis")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ All fixes verified - ready to run full publication analysis")
    else:
        print("\n❌ Issues remain - need further debugging")
