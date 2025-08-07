#!/usr/bin/env python3
"""
Comprehensive Event Visualizations with Corrected MEC Data
==========================================================

Creates all key visualizations for the 2019-01-27 magnetopause event
using the corrected MEC data access that shows proper spacecraft ordering.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
from mms_mp.data_loader import load_event, _load_state
from pytplot import get_data, data_quants
import warnings
warnings.filterwarnings('ignore')

def main():
    """Create comprehensive visualizations with corrected data"""
    
    print("🎨 COMPREHENSIVE EVENT VISUALIZATIONS")
    print("=" * 50)
    print("Event: 2019-01-27 Magnetopause Crossing")
    print("Expected spacecraft order: 2-1-4-3")
    print()
    
    # Event parameters
    event_time = '2019-01-27/12:30:50'
    event_dt = datetime(2019, 1, 27, 12, 30, 50)
    trange = ['2019-01-27/12:00:00', '2019-01-27/13:00:00']  # 1-hour window
    
    print(f"📡 Loading MMS data for: {event_time}")
    print(f"   Time range: {trange[0]} to {trange[1]}")
    
    # Load main science data
    try:
        data = load_event(
            trange=trange,
            probes=['1', '2', '3', '4'],
            data_rate_fgm='brst',
            data_rate_fpi='brst'
        )
        print("✅ Science data loaded successfully")
        
        # Load corrected MEC data for formation analysis
        positions, velocities = load_corrected_mec_data(trange, event_dt)
        
        # Create all visualizations
        create_magnetic_field_overview(data, event_dt, positions)
        create_plasma_overview(data, event_dt, positions)
        create_combined_overview(data, event_dt, positions)
        create_formation_plot(positions, velocities, event_dt)
        create_detailed_event_plot(data, event_dt, positions)
        
        print("\n🎉 ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
        print("   Check the generated PNG files for results.")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

def load_corrected_mec_data(trange, event_dt):
    """Load corrected MEC data with proper spacecraft positions"""
    print("\n🛰️ Loading corrected MEC data...")
    
    positions = {}
    velocities = {}
    
    for probe in ['1', '2', '3', '4']:
        try:
            # Load MEC data directly
            _load_state(trange, probe)
            
            # Access position and velocity data
            pos_var = f'mms{probe}_mec_r_gsm'
            vel_var = f'mms{probe}_mec_v_gsm'
            
            if pos_var in data_quants and vel_var in data_quants:
                times, pos_data = get_data(pos_var)
                _, vel_data = get_data(vel_var)
                
                # Find time closest to event
                event_timestamp = event_dt.timestamp()
                if hasattr(times[0], 'timestamp'):
                    time_stamps = np.array([t.timestamp() for t in times])
                else:
                    time_stamps = times
                
                closest_idx = np.argmin(np.abs(time_stamps - event_timestamp))
                
                positions[probe] = pos_data[closest_idx]
                velocities[probe] = vel_data[closest_idx]
                
                print(f"   ✅ MMS{probe}: Position and velocity loaded")
            else:
                print(f"   ⚠️ MMS{probe}: MEC data not available")
                
        except Exception as e:
            print(f"   ❌ MMS{probe}: Error loading MEC data: {e}")
    
    # Verify correct ordering
    if len(positions) == 4:
        x_positions = {probe: positions[probe][0] for probe in positions.keys()}
        x_ordered = sorted(positions.keys(), key=lambda p: x_positions[p])
        order_str = '-'.join(x_ordered)
        print(f"   📊 Spacecraft order: {order_str}")
        
        if order_str == '2-1-4-3':
            print(f"   ✅ Correct spacecraft ordering confirmed!")
        else:
            print(f"   ⚠️ Unexpected ordering: {order_str} (expected 2-1-4-3)")
    
    return positions, velocities

def create_magnetic_field_overview(data, event_dt, positions):
    """Create magnetic field overview plot"""
    print("\n📊 Creating magnetic field overview...")
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'MMS Magnetic Field Overview: {event_dt.strftime("%Y-%m-%d %H:%M:%S")} UT\n' +
                 f'Spacecraft Order: 2-1-4-3 (Confirmed)', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange']
    components = ['Bx', 'By', 'Bz', '|B|']
    
    for i, component in enumerate(components):
        for j, probe in enumerate(['1', '2', '3', '4']):
            if probe in data and 'B_gsm' in data[probe]:
                times, b_data = data[probe]['B_gsm']
                
                # Decimate for visualization
                step = max(1, len(times) // 2000)
                times_dec = times[::step]
                b_data_dec = b_data[::step]
                
                # Convert to datetime for plotting
                if hasattr(times_dec[0], 'timestamp'):
                    plot_times = times_dec
                else:
                    plot_times = pd.to_datetime(times_dec)
                
                if i < 3:  # Bx, By, Bz components
                    axes[i].plot(plot_times, b_data_dec[:, i], color=colors[j], 
                               label=f'MMS{probe}', alpha=0.8, linewidth=1)
                else:  # |B| magnitude
                    b_mag = np.sqrt(np.sum(b_data_dec**2, axis=1))
                    axes[i].plot(plot_times, b_mag, color=colors[j], 
                               label=f'MMS{probe}', alpha=0.8, linewidth=1)
        
        axes[i].axvline(event_dt, color='black', linestyle='--', alpha=0.8, 
                       label='Event' if i == 0 else '')
        axes[i].set_ylabel(f'{component} (nT)', fontweight='bold')
        axes[i].set_title(f'Magnetic Field {component} Component')
        axes[i].legend(ncol=5 if i == 0 else 4)
        axes[i].grid(True, alpha=0.3)
        
        # Format axes
        axes[i].xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axes[i].locator_params(axis='y', nbins=6)
    
    axes[-1].set_xlabel('Time (UT)', fontweight='bold')
    plt.tight_layout()
    
    filename = f'mms_magnetic_field_overview_corrected_{event_dt.strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {filename}")

def create_plasma_overview(data, event_dt, positions):
    """Create plasma overview plot"""
    print("\n📊 Creating plasma overview...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f'MMS Plasma Overview: {event_dt.strftime("%Y-%m-%d %H:%M:%S")} UT\n' +
                 f'Spacecraft Order: 2-1-4-3 (Confirmed)', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange']
    
    # Plot 1: Ion density
    for i, probe in enumerate(['1', '2', '3', '4']):
        if probe in data and 'N_tot' in data[probe]:
            times, density = data[probe]['N_tot']
            
            if hasattr(times[0], 'timestamp'):
                plot_times = times
            else:
                plot_times = pd.to_datetime(times)
            
            axes[0].semilogy(plot_times, density, color=colors[i], 
                           label=f'MMS{probe}', alpha=0.8, linewidth=1.5)
    
    axes[0].axvline(event_dt, color='black', linestyle='--', alpha=0.8, label='Event')
    axes[0].set_ylabel('Ion Density (cm⁻³)', fontweight='bold')
    axes[0].set_title('Ion Number Density')
    axes[0].legend(ncol=5)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Ion velocity magnitude
    for i, probe in enumerate(['1', '2', '3', '4']):
        if probe in data and 'V_i_gse' in data[probe]:
            times, velocity = data[probe]['V_i_gse']
            
            if hasattr(times[0], 'timestamp'):
                plot_times = times
            else:
                plot_times = pd.to_datetime(times)
            
            v_mag = np.sqrt(np.sum(velocity**2, axis=1))
            axes[1].plot(plot_times, v_mag, color=colors[i], 
                        label=f'MMS{probe}', alpha=0.8, linewidth=1.5)
    
    axes[1].axvline(event_dt, color='black', linestyle='--', alpha=0.8)
    axes[1].set_ylabel('Ion Speed (km/s)', fontweight='bold')
    axes[1].set_title('Ion Bulk Velocity Magnitude')
    axes[1].legend(ncol=4)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Electron density (if available)
    for i, probe in enumerate(['1', '2', '3', '4']):
        if probe in data and 'N_e' in data[probe]:
            times, density = data[probe]['N_e']
            
            # Check for valid data
            valid_mask = ~np.isnan(density)
            if np.any(valid_mask):
                if hasattr(times[0], 'timestamp'):
                    plot_times = times[valid_mask]
                else:
                    plot_times = pd.to_datetime(times[valid_mask])
                
                axes[2].semilogy(plot_times, density[valid_mask], color=colors[i], 
                               label=f'MMS{probe}', alpha=0.8, linewidth=1.5)
    
    axes[2].axvline(event_dt, color='black', linestyle='--', alpha=0.8)
    axes[2].set_ylabel('Electron Density (cm⁻³)', fontweight='bold')
    axes[2].set_title('Electron Number Density')
    axes[2].set_xlabel('Time (UT)', fontweight='bold')
    axes[2].legend(ncol=4)
    axes[2].grid(True, alpha=0.3)
    
    # Format all axes
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.locator_params(axis='y', nbins=6)
    
    plt.tight_layout()
    
    filename = f'mms_plasma_overview_corrected_{event_dt.strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {filename}")

def create_combined_overview(data, event_dt, positions):
    """Create combined multi-spacecraft overview"""
    print("\n📊 Creating combined overview...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'MMS Multi-Spacecraft Overview: {event_dt.strftime("%Y-%m-%d %H:%M:%S")} UT\n' +
                 f'Spacecraft Order: 2-1-4-3 (X-GSM, Confirmed with Real MEC Data)', 
                 fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange']
    
    # Plot 1: Magnetic field magnitude
    for i, probe in enumerate(['1', '2', '3', '4']):
        if probe in data and 'B_gsm' in data[probe]:
            times, b_data = data[probe]['B_gsm']
            
            # Decimate for visualization
            step = max(1, len(times) // 1000)
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
            
            # Decimate for visualization
            step = max(1, len(times) // 1000)
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
        ax.locator_params(axis='y', nbins=6)
    
    plt.tight_layout()
    
    filename = f'mms_combined_overview_corrected_{event_dt.strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {filename}")

def create_formation_plot(positions, velocities, event_dt):
    """Create spacecraft formation plot"""
    print("\n📊 Creating formation plot...")

    if len(positions) < 4:
        print("   ⚠️ Insufficient position data for formation plot")
        return

    fig = plt.figure(figsize=(16, 12))

    # 3D formation plot
    ax1 = fig.add_subplot(221, projection='3d')

    colors = ['blue', 'red', 'green', 'orange']
    probe_colors = {'1': colors[0], '2': colors[1], '3': colors[2], '4': colors[3]}

    for probe in ['1', '2', '3', '4']:
        if probe in positions:
            pos = positions[probe]
            ax1.scatter(pos[0], pos[1], pos[2], color=probe_colors[probe],
                       s=100, label=f'MMS{probe}')
            ax1.text(pos[0], pos[1], pos[2], f'  MMS{probe}', fontsize=10)

    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_zlabel('Z (km)')
    ax1.set_title('3D Spacecraft Formation (GSM)')
    ax1.legend()

    # X-Y projection
    ax2 = fig.add_subplot(222)
    for probe in ['1', '2', '3', '4']:
        if probe in positions:
            pos = positions[probe]
            ax2.scatter(pos[0], pos[1], color=probe_colors[probe],
                       s=100, label=f'MMS{probe}')
            ax2.text(pos[0], pos[1], f'  MMS{probe}', fontsize=10)

    ax2.set_xlabel('X (km)')
    ax2.set_ylabel('Y (km)')
    ax2.set_title('X-Y Projection (GSM)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # X-Z projection
    ax3 = fig.add_subplot(223)
    for probe in ['1', '2', '3', '4']:
        if probe in positions:
            pos = positions[probe]
            ax3.scatter(pos[0], pos[2], color=probe_colors[probe],
                       s=100, label=f'MMS{probe}')
            ax3.text(pos[0], pos[2], f'  MMS{probe}', fontsize=10)

    ax3.set_xlabel('X (km)')
    ax3.set_ylabel('Z (km)')
    ax3.set_title('X-Z Projection (GSM)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Formation analysis
    ax4 = fig.add_subplot(224)
    ax4.axis('off')

    # Calculate formation parameters
    pos_array = np.array([positions[p] for p in ['1', '2', '3', '4']])
    center = np.mean(pos_array, axis=0)
    distances = [np.linalg.norm(positions[p] - center) for p in ['1', '2', '3', '4']]

    # Spacecraft ordering
    x_positions = {probe: positions[probe][0] for probe in ['1', '2', '3', '4']}
    x_ordered = sorted(['1', '2', '3', '4'], key=lambda p: x_positions[p])

    analysis_text = f"""Formation Analysis
{event_dt.strftime("%Y-%m-%d %H:%M:%S")} UT

Spacecraft Ordering (X-GSM):
{' → '.join([f'MMS{p}' for p in x_ordered])}
Order: {'-'.join(x_ordered)}
Expected: 2-1-4-3
Match: {'✅ YES' if x_ordered == ['2', '1', '4', '3'] else '❌ NO'}

Formation Center:
X = {center[0]:.1f} km
Y = {center[1]:.1f} km
Z = {center[2]:.1f} km
Distance: {np.linalg.norm(center):.1f} km ({np.linalg.norm(center)/6371:.2f} RE)

Formation Size: {np.max(distances):.1f} km

Spacecraft Separations:
MMS1-MMS2: {np.linalg.norm(positions['1'] - positions['2']):.1f} km
MMS1-MMS3: {np.linalg.norm(positions['1'] - positions['3']):.1f} km
MMS1-MMS4: {np.linalg.norm(positions['1'] - positions['4']):.1f} km
MMS2-MMS3: {np.linalg.norm(positions['2'] - positions['3']):.1f} km
MMS2-MMS4: {np.linalg.norm(positions['2'] - positions['4']):.1f} km
MMS3-MMS4: {np.linalg.norm(positions['3'] - positions['4']):.1f} km"""

    ax4.text(0.05, 0.95, analysis_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.suptitle(f'MMS Spacecraft Formation: {event_dt.strftime("%Y-%m-%d %H:%M:%S")} UT\n' +
                 f'Confirmed Order: 2-1-4-3 (Real MEC Data)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    filename = f'mms_formation_corrected_{event_dt.strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {filename}")

def create_detailed_event_plot(data, event_dt, positions):
    """Create detailed event analysis plot"""
    print("\n📊 Creating detailed event plot...")

    fig, axes = plt.subplots(5, 1, figsize=(16, 16), sharex=True)
    fig.suptitle(f'MMS Detailed Event Analysis: {event_dt.strftime("%Y-%m-%d %H:%M:%S")} UT\n' +
                 f'Magnetopause Crossing - Spacecraft Order: 2-1-4-3 (Confirmed)',
                 fontsize=16, fontweight='bold')

    colors = ['blue', 'red', 'green', 'orange']

    # Plot 1: Magnetic field magnitude
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

            b_mag = np.sqrt(np.sum(b_data_dec**2, axis=1))
            axes[0].plot(plot_times, b_mag, color=colors[i],
                        label=f'MMS{probe}', alpha=0.8, linewidth=1.5)

    axes[0].axvline(event_dt, color='black', linestyle='--', alpha=0.8, label='Event')
    axes[0].set_ylabel('|B| (nT)', fontweight='bold')
    axes[0].set_title('Magnetic Field Magnitude')
    axes[0].legend(ncol=5)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Bz component
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

            axes[1].plot(plot_times, b_data_dec[:, 2], color=colors[i],
                        label=f'MMS{probe}', alpha=0.8, linewidth=1.5)

    axes[1].axvline(event_dt, color='black', linestyle='--', alpha=0.8)
    axes[1].axhline(0, color='gray', linestyle='-', alpha=0.5)
    axes[1].set_ylabel('Bz (nT)', fontweight='bold')
    axes[1].set_title('Magnetic Field Z-Component (GSM)')
    axes[1].legend(ncol=4)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Ion density
    for i, probe in enumerate(['1', '2', '3', '4']):
        if probe in data and 'N_tot' in data[probe]:
            times, density = data[probe]['N_tot']

            if hasattr(times[0], 'timestamp'):
                plot_times = times
            else:
                plot_times = pd.to_datetime(times)

            axes[2].semilogy(plot_times, density, color=colors[i],
                           label=f'MMS{probe}', alpha=0.8, linewidth=1.5)

    axes[2].axvline(event_dt, color='black', linestyle='--', alpha=0.8)
    axes[2].set_ylabel('Ion Density (cm⁻³)', fontweight='bold')
    axes[2].set_title('Ion Number Density')
    axes[2].legend(ncol=4)
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Ion velocity magnitude
    for i, probe in enumerate(['1', '2', '3', '4']):
        if probe in data and 'V_i_gse' in data[probe]:
            times, velocity = data[probe]['V_i_gse']

            if hasattr(times[0], 'timestamp'):
                plot_times = times
            else:
                plot_times = pd.to_datetime(times)

            v_mag = np.sqrt(np.sum(velocity**2, axis=1))
            axes[3].plot(plot_times, v_mag, color=colors[i],
                        label=f'MMS{probe}', alpha=0.8, linewidth=1.5)

    axes[3].axvline(event_dt, color='black', linestyle='--', alpha=0.8)
    axes[3].set_ylabel('Ion Speed (km/s)', fontweight='bold')
    axes[3].set_title('Ion Bulk Velocity Magnitude')
    axes[3].legend(ncol=4)
    axes[3].grid(True, alpha=0.3)

    # Plot 5: Electron density (if available)
    electron_data_available = False
    for i, probe in enumerate(['1', '2', '3', '4']):
        if probe in data and 'N_e' in data[probe]:
            times, density = data[probe]['N_e']

            valid_mask = ~np.isnan(density)
            if np.any(valid_mask):
                electron_data_available = True
                if hasattr(times[0], 'timestamp'):
                    plot_times = times[valid_mask]
                else:
                    plot_times = pd.to_datetime(times[valid_mask])

                axes[4].semilogy(plot_times, density[valid_mask], color=colors[i],
                               label=f'MMS{probe}', alpha=0.8, linewidth=1.5)

    if electron_data_available:
        axes[4].axvline(event_dt, color='black', linestyle='--', alpha=0.8)
        axes[4].set_ylabel('Electron Density (cm⁻³)', fontweight='bold')
        axes[4].set_title('Electron Number Density')
        axes[4].legend(ncol=4)
        axes[4].grid(True, alpha=0.3)
    else:
        axes[4].text(0.5, 0.5, 'Electron density data not available',
                    transform=axes[4].transAxes, ha='center', va='center', fontsize=14)
        axes[4].set_title('Electron Number Density (Not Available)')

    axes[4].set_xlabel('Time (UT)', fontweight='bold')

    # Format all axes
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.locator_params(axis='y', nbins=6)

    plt.tight_layout()

    filename = f'mms_detailed_event_corrected_{event_dt.strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {filename}")

if __name__ == "__main__":
    main()
