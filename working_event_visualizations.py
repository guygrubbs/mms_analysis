#!/usr/bin/env python3
"""
Working Event Visualizations for 2019-01-27 Magnetopause Crossing
================================================================

Creates comprehensive visualizations using the available science data.
Note: Spacecraft order 2-1-4-3 is confirmed from literature/analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
from mms_mp.data_loader import load_event
import warnings
warnings.filterwarnings('ignore')

def main():
    """Create working visualizations with available data"""
    
    print("🎨 WORKING EVENT VISUALIZATIONS")
    print("=" * 45)
    print("Event: 2019-01-27 Magnetopause Crossing")
    print("Known spacecraft order: 2-1-4-3 (from analysis)")
    print()
    
    # Event parameters
    event_time = '2019-01-27/12:30:50'
    event_dt = datetime(2019, 1, 27, 12, 30, 50)
    trange = ['2019-01-27/12:00:00', '2019-01-27/13:00:00']
    
    print(f"📡 Loading MMS data for: {event_time}")
    print(f"   Time range: {trange[0]} to {trange[1]}")
    
    try:
        # Load science data
        data = load_event(
            trange=trange,
            probes=['1', '2', '3', '4'],
            data_rate_fgm='brst',
            data_rate_fpi='brst'
        )
        
        print("✅ Science data loaded successfully")
        
        # Report data volume
        for probe in ['1', '2', '3', '4']:
            if probe in data:
                probe_data = data[probe]
                b_points = len(probe_data.get('B_gsm', [[], []])[0])
                n_points = len(probe_data.get('N_tot', [[], []])[0])
                print(f"   MMS{probe}: {b_points:,} B-field, {n_points:,} plasma points")
        
        # Create visualizations
        create_magnetic_field_overview(data, event_dt)
        create_plasma_overview(data, event_dt)
        create_combined_overview(data, event_dt)
        create_detailed_analysis(data, event_dt)
        
        print("\n🎉 ALL VISUALIZATIONS COMPLETED!")
        print("   Generated publication-quality plots for the magnetopause event")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

def create_magnetic_field_overview(data, event_dt):
    """Create magnetic field overview plot"""
    print("\n📊 Creating magnetic field overview...")
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'MMS Magnetic Field Overview: {event_dt.strftime("%Y-%m-%d %H:%M:%S")} UT\n' +
                 f'Magnetopause Crossing - Known Spacecraft Order: 2-1-4-3', 
                 fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange']
    components = ['Bx', 'By', 'Bz', '|B|']
    
    for i, component in enumerate(components):
        for j, probe in enumerate(['1', '2', '3', '4']):
            if probe in data and 'B_gsm' in data[probe]:
                times, b_data = data[probe]['B_gsm']
                
                # Smart decimation
                step = max(1, len(times) // 2000)
                times_dec = times[::step]
                b_data_dec = b_data[::step]
                
                if hasattr(times_dec[0], 'timestamp'):
                    plot_times = times_dec
                else:
                    plot_times = pd.to_datetime(times_dec)
                
                if i < 3:  # Bx, By, Bz components
                    axes[i].plot(plot_times, b_data_dec[:, i], color=colors[j], 
                               label=f'MMS{probe}', alpha=0.8, linewidth=1.2)
                else:  # |B| magnitude
                    b_mag = np.sqrt(np.sum(b_data_dec**2, axis=1))
                    axes[i].plot(plot_times, b_mag, color=colors[j], 
                               label=f'MMS{probe}', alpha=0.8, linewidth=1.2)
        
        axes[i].axvline(event_dt, color='black', linestyle='--', alpha=0.8, 
                       label='Event' if i == 0 else '')
        axes[i].set_ylabel(f'{component} (nT)', fontweight='bold')
        axes[i].set_title(f'Magnetic Field {component} Component')
        axes[i].legend(ncol=5 if i == 0 else 4)
        axes[i].grid(True, alpha=0.3)
        
        # Controlled time formatting
        axes[i].xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axes[i].locator_params(axis='y', nbins=6)
    
    axes[-1].set_xlabel('Time (UT)', fontweight='bold')
    plt.tight_layout()
    
    filename = f'mms_magnetic_field_overview_working_{event_dt.strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {filename}")

def create_plasma_overview(data, event_dt):
    """Create plasma overview plot"""
    print("\n📊 Creating plasma overview...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f'MMS Plasma Overview: {event_dt.strftime("%Y-%m-%d %H:%M:%S")} UT\n' +
                 f'Magnetopause Crossing - Known Spacecraft Order: 2-1-4-3', 
                 fontsize=16, fontweight='bold')
    
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
    
    # Plot 3: Ion temperature
    for i, probe in enumerate(['1', '2', '3', '4']):
        if probe in data and 'T_i' in data[probe]:
            times, temperature = data[probe]['T_i']
            
            if hasattr(times[0], 'timestamp'):
                plot_times = times
            else:
                plot_times = pd.to_datetime(times)
            
            axes[2].semilogy(plot_times, temperature, color=colors[i], 
                           label=f'MMS{probe}', alpha=0.8, linewidth=1.5)
    
    axes[2].axvline(event_dt, color='black', linestyle='--', alpha=0.8)
    axes[2].set_ylabel('Ion Temperature (eV)', fontweight='bold')
    axes[2].set_title('Ion Temperature')
    axes[2].set_xlabel('Time (UT)', fontweight='bold')
    axes[2].legend(ncol=4)
    axes[2].grid(True, alpha=0.3)
    
    # Format all axes
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.locator_params(axis='y', nbins=6)
    
    plt.tight_layout()
    
    filename = f'mms_plasma_overview_working_{event_dt.strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {filename}")

def create_combined_overview(data, event_dt):
    """Create combined multi-spacecraft overview"""
    print("\n📊 Creating combined overview...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'MMS Multi-Spacecraft Overview: {event_dt.strftime("%Y-%m-%d %H:%M:%S")} UT\n' +
                 f'Magnetopause Crossing - Known Order: 2-1-4-3 (X-GSM)', 
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
        ax.locator_params(axis='y', nbins=6)
    
    plt.tight_layout()
    
    filename = f'mms_combined_overview_working_{event_dt.strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {filename}")

def create_detailed_analysis(data, event_dt):
    """Create detailed event analysis plot"""
    print("\n📊 Creating detailed analysis...")
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(f'MMS Detailed Event Analysis: {event_dt.strftime("%Y-%m-%d %H:%M:%S")} UT\n' +
                 f'Magnetopause Crossing - Spacecraft Order: 2-1-4-3 (Confirmed)', 
                 fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange']
    
    # Plot 1: Magnetic field magnitude
    for i, probe in enumerate(['1', '2', '3', '4']):
        if probe in data and 'B_gsm' in data[probe]:
            times, b_data = data[probe]['B_gsm']
            
            step = max(1, len(times) // 2000)
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
            
            step = max(1, len(times) // 2000)
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
    axes[3].set_xlabel('Time (UT)', fontweight='bold')
    axes[3].legend(ncol=4)
    axes[3].grid(True, alpha=0.3)
    
    # Format all axes
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.locator_params(axis='y', nbins=6)
    
    plt.tight_layout()
    
    filename = f'mms_detailed_analysis_working_{event_dt.strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {filename}")

if __name__ == "__main__":
    main()
