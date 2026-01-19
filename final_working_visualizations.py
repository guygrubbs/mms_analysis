#!/usr/bin/env python3
"""
Final Working Event Visualizations with Real MEC Data
=====================================================

This script successfully loads and uses real MEC data by capturing it
BEFORE any other data loading that might clear pytplot.

CONFIRMED: Shows spacecraft order 2-1-4-3 for 2019-01-27 event.
"""

import numpy as np
# NumPy 2.x compatibility for third-party libraries (e.g., pytplot/bokeh)
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
from pyspedas.projects import mms
from pytplot import data_quants, get_data
import warnings
warnings.filterwarnings('ignore')

def load_mec_data_first(trange, probes):
    """Load MEC data FIRST and capture immediately before any other loading"""
    
    print("🛰️ Loading MEC data FIRST (before any other data)...")
    
    all_positions = {}
    all_velocities = {}
    
    for probe in probes:
        try:
            # Load MEC data for this spacecraft ONLY
            result = mms.mms_load_mec(
                trange=trange,
                probe=probe,
                data_rate='srvy',
                level='l2',
                datatype='epht89q',
                time_clip=True
            )
            
            # IMMEDIATELY capture the data before anything else can clear it
            pos_var = f'mms{probe}_mec_r_gsm'
            vel_var = f'mms{probe}_mec_v_gsm'
            
            if pos_var in data_quants and vel_var in data_quants:
                # Get the data immediately
                times_pos, pos_data = get_data(pos_var)
                times_vel, vel_data = get_data(vel_var)
                
                # Store the data in our own dictionaries
                all_positions[probe] = {
                    'times': times_pos,
                    'data': pos_data
                }
                all_velocities[probe] = {
                    'times': times_vel,
                    'data': vel_data
                }
                
                print(f"   ✅ MMS{probe}: {len(times_pos)} position, {len(times_vel)} velocity points")
                
                # Verify data quality
                nan_count_pos = np.isnan(pos_data).sum()
                if nan_count_pos == 0:
                    mid_pos = pos_data[len(pos_data)//2]
                    print(f"      Sample position: [{mid_pos[0]:.1f}, {mid_pos[1]:.1f}, {mid_pos[2]:.1f}] km")
                
            else:
                print(f"   ❌ MMS{probe}: MEC variables not accessible")
                
        except Exception as e:
            print(f"   ❌ MMS{probe}: Error loading MEC data: {e}")
    
    return all_positions, all_velocities

def load_science_data_simple(trange, probes):
    """Load science data using direct pyspedas calls"""
    
    print("\n📡 Loading science data...")
    
    data = {}
    
    for probe in probes:
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
            
            # Get magnetic field data
            b_var = f'mms{probe}_fgm_b_gsm_brst_l2'
            if b_var in data_quants:
                times, b_data = get_data(b_var)
                data[probe]['B_gsm'] = (times, b_data)
                print(f"      ✅ FGM: {len(times)} points")
            
            # Load FPI DIS data
            fpi_result = mms.mms_load_fpi(
                trange=trange,
                probe=probe,
                data_rate='brst',
                level='l2',
                datatype='dis-moms',
                time_clip=True
            )
            
            # Get ion data
            n_var = f'mms{probe}_dis_numberdensity_brst'
            v_var = f'mms{probe}_dis_bulkv_gse_brst'
            t_var = f'mms{probe}_dis_temppara_brst'
            
            if n_var in data_quants:
                times, n_data = get_data(n_var)
                data[probe]['N_tot'] = (times, n_data)
                print(f"      ✅ Ion density: {len(times)} points")
            
            if v_var in data_quants:
                times, v_data = get_data(v_var)
                data[probe]['V_i_gse'] = (times, v_data)
                print(f"      ✅ Ion velocity: {len(times)} points")
            
            if t_var in data_quants:
                times, t_data = get_data(t_var)
                data[probe]['T_i'] = (times, t_data)
                print(f"      ✅ Ion temperature: {len(times)} points")
                
        except Exception as e:
            print(f"      ❌ Error loading science data: {e}")
    
    return data

def get_event_positions(positions, event_dt):
    """Get spacecraft positions at event time"""
    
    event_positions = {}
    
    for probe in ['1', '2', '3', '4']:
        if probe in positions:
            times = positions[probe]['times']
            pos_data = positions[probe]['data']
            
            # Convert event time to timestamp
            event_timestamp = event_dt.timestamp()
            
            # Convert times to timestamps
            if hasattr(times[0], 'timestamp'):
                time_stamps = np.array([t.timestamp() for t in times])
            else:
                time_stamps = times
            
            # Find closest time index
            closest_idx = np.argmin(np.abs(time_stamps - event_timestamp))
            event_pos = pos_data[closest_idx]
            
            event_positions[probe] = event_pos
    
    return event_positions

def verify_spacecraft_ordering(event_positions, event_dt):
    """Verify spacecraft ordering and print results"""
    
    print(f"\n🎯 SPACECRAFT ORDERING VERIFICATION:")
    print("=" * 45)
    
    if len(event_positions) == 4:
        # Print individual positions
        for probe in ['1', '2', '3', '4']:
            pos = event_positions[probe]
            distance = np.linalg.norm(pos)
            print(f"   MMS{probe}: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] km")
            print(f"           Distance: {distance:.1f} km ({distance/6371:.2f} RE)")
        
        # Calculate X-GSM ordering
        x_positions = {probe: event_positions[probe][0] for probe in event_positions.keys()}
        x_ordered = sorted(event_positions.keys(), key=lambda p: x_positions[p])
        order_str = '-'.join(x_ordered)
        
        print(f"\n   X-GSM order: {order_str}")
        print(f"   Expected:    2-1-4-3")
        
        if order_str == '2-1-4-3':
            print(f"   ✅ CORRECT ORDERING CONFIRMED!")
            return True
        else:
            print(f"   ⚠️ Unexpected ordering: {order_str}")
            return False
    else:
        print(f"   ❌ Incomplete position data: {len(event_positions)} spacecraft")
        return False

def create_simple_overview(data, event_dt, event_positions):
    """Create simple overview plot with confirmed spacecraft ordering"""
    print("\n📊 Creating overview plot...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    # Get ordering info for title
    if len(event_positions) == 4:
        x_positions = {probe: event_positions[probe][0] for probe in event_positions.keys()}
        x_ordered = sorted(event_positions.keys(), key=lambda p: x_positions[p])
        order_str = '-'.join(x_ordered)
        order_status = "✅ CONFIRMED" if order_str == '2-1-4-3' else "⚠️ Unexpected"
    else:
        order_str = "Unknown"
        order_status = "❌ Incomplete"
    
    fig.suptitle(f'MMS Event Overview: {event_dt.strftime("%Y-%m-%d %H:%M:%S")} UT\n' +
                 f'Magnetopause Crossing - Real MEC Data - Order: {order_str} ({order_status})', 
                 fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange']
    
    # Plot 1: Magnetic field magnitude
    for i, probe in enumerate(['1', '2', '3', '4']):
        if probe in data and 'B_gsm' in data[probe]:
            times, b_data = data[probe]['B_gsm']
            
            # Smart decimation for performance
            step = max(1, len(times) // 1500)
            times_dec = times[::step]
            b_data_dec = b_data[::step]
            
            if hasattr(times_dec[0], 'timestamp'):
                plot_times = times_dec
            else:
                plot_times = pd.to_datetime(times_dec)
            
            b_mag = np.sqrt(np.sum(b_data_dec**2, axis=1))
            axes[0].plot(plot_times, b_mag, color=colors[i], 
                        label=f'MMS{probe}', alpha=0.8, linewidth=1.5)
    
    axes[0].axvline(event_dt, color='black', linestyle='--', alpha=0.8, label='Event')
    axes[0].set_ylabel('|B| (nT)', fontweight='bold')
    axes[0].set_title('Magnetic Field Magnitude')
    axes[0].legend(ncol=5)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Ion density
    for i, probe in enumerate(['1', '2', '3', '4']):
        if probe in data and 'N_tot' in data[probe]:
            times, density = data[probe]['N_tot']
            
            if hasattr(times[0], 'timestamp'):
                plot_times = times
            else:
                plot_times = pd.to_datetime(times)
            
            axes[1].semilogy(plot_times, density, color=colors[i], 
                           label=f'MMS{probe}', alpha=0.8, linewidth=1.5)
    
    axes[1].axvline(event_dt, color='black', linestyle='--', alpha=0.8)
    axes[1].set_ylabel('Ion Density (cm⁻³)', fontweight='bold')
    axes[1].set_title('Ion Number Density')
    axes[1].legend(ncol=4)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Bz component
    for i, probe in enumerate(['1', '2', '3', '4']):
        if probe in data and 'B_gsm' in data[probe]:
            times, b_data = data[probe]['B_gsm']
            
            step = max(1, len(times) // 1500)
            times_dec = times[::step]
            b_data_dec = b_data[::step]
            
            if hasattr(times_dec[0], 'timestamp'):
                plot_times = times_dec
            else:
                plot_times = pd.to_datetime(times_dec)
            
            axes[2].plot(plot_times, b_data_dec[:, 2], color=colors[i], 
                        label=f'MMS{probe}', alpha=0.8, linewidth=1.5)
    
    axes[2].axvline(event_dt, color='black', linestyle='--', alpha=0.8)
    axes[2].axhline(0, color='gray', linestyle='-', alpha=0.5)
    axes[2].set_ylabel('Bz (nT)', fontweight='bold')
    axes[2].set_title('Magnetic Field Z-Component (GSM)')
    axes[2].set_xlabel('Time (UT)', fontweight='bold')
    axes[2].legend(ncol=4)
    axes[2].grid(True, alpha=0.3)
    
    # Format all axes
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # Skip nbins for log scale axes to avoid matplotlib compatibility issues
        if not ax.get_yscale() == 'log':
            ax.locator_params(axis='y', nbins=6)
    
    plt.tight_layout()
    
    filename = f'mms_final_overview_with_real_mec_{event_dt.strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {filename}")

def main():
    """Create working visualizations with real MEC data"""
    
    print("🎉 FINAL WORKING VISUALIZATIONS WITH REAL MEC DATA")
    print("=" * 60)
    print("Event: 2019-01-27 Magnetopause Crossing")
    print("Expected spacecraft order: 2-1-4-3")
    print()
    
    # Event parameters
    event_time = '2019-01-27/12:30:50'
    event_dt = datetime(2019, 1, 27, 12, 30, 50)
    trange = ['2019-01-27/12:25:00', '2019-01-27/12:35:00']  # Shorter window for testing
    
    print(f"📡 Analysis for: {event_time}")
    print(f"   Time range: {trange[0]} to {trange[1]}")
    
    try:
        # STEP 1: Load MEC data FIRST and capture immediately
        positions, velocities = load_mec_data_first(trange, ['1', '2', '3', '4'])
        
        # STEP 2: Get positions at event time
        event_positions = get_event_positions(positions, event_dt)
        
        # STEP 3: Verify spacecraft ordering
        ordering_correct = verify_spacecraft_ordering(event_positions, event_dt)
        
        # STEP 4: Load science data (this may clear pytplot, but we already have MEC data)
        data = load_science_data_simple(trange, ['1', '2', '3', '4'])
        
        # STEP 5: Create visualization
        create_simple_overview(data, event_dt, event_positions)
        
        if ordering_correct:
            print("\n🎉 SUCCESS! Real MEC data confirms spacecraft order 2-1-4-3")
            print("   Generated publication-quality visualization with real position data")
        else:
            print("\n⚠️ Unexpected spacecraft ordering detected")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
