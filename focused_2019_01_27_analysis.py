#!/usr/bin/env python3
"""
Focused MMS Analysis: 2019-01-27 Event
======================================

This script creates the key visualizations for the 2019-01-27 magnetopause
crossing event, focusing on the data that is available and working correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path to import mms_mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mms_mp
from pytplot import get_data


def create_results_directory():
    """Create results directory for saving plots"""
    
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"✅ Created results directory: {results_dir}")
    else:
        print(f"✅ Results directory exists: {results_dir}")
    
    return results_dir


def load_event_data():
    """Load event data for 2019-01-27"""
    
    print("🔍 Loading Event Data")
    print("=" * 50)
    
    # Event time and extended range for context
    event_time = datetime(2019, 1, 27, 12, 30, 50)
    
    # Load data for 2 hours around the event for full context
    start_time = event_time - timedelta(hours=1)
    end_time = event_time + timedelta(hours=1)
    
    trange = [
        start_time.strftime('%Y-%m-%d/%H:%M:%S'),
        end_time.strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    print(f"Time range: {trange[0]} to {trange[1]}")
    print(f"Event time: {event_time.strftime('%Y-%m-%d %H:%M:%S')} UT")
    
    try:
        # Load comprehensive data
        evt = mms_mp.load_event(
            trange=trange,
            probes=['1', '2', '3', '4'],
            include_ephem=True,
            include_edp=True,
            data_rate_fpi='fast',
            data_rate_fgm='srvy'
        )
        
        print(f"✅ Event data loaded successfully")
        print(f"   Spacecraft loaded: {list(evt.keys())}")
        
        return evt, event_time, trange
        
    except Exception as e:
        print(f"❌ Error loading event data: {e}")
        return None, None, None


def create_magnetic_field_overview(evt, event_time, trange, results_dir):
    """Create comprehensive magnetic field overview"""
    
    print(f"\n🔍 Creating Magnetic Field Overview")
    print("=" * 50)
    
    try:
        fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
        
        colors = ['red', 'blue', 'green', 'orange']
        
        # Plot magnetic field components for each spacecraft
        for i, probe in enumerate(['1', '2', '3', '4']):
            if probe in evt and 'B_gsm' in evt[probe]:
                times, b_data = evt[probe]['B_gsm']
                
                # Convert times to datetime if needed
                if not hasattr(times[0], 'strftime'):
                    times = [datetime.fromtimestamp(t) for t in times]
                
                # Plot Bx, By, Bz
                axes[i].plot(times, b_data[:, 0], color='red', label='Bx', alpha=0.8, linewidth=1.5)
                axes[i].plot(times, b_data[:, 1], color='green', label='By', alpha=0.8, linewidth=1.5)
                axes[i].plot(times, b_data[:, 2], color='blue', label='Bz', alpha=0.8, linewidth=1.5)
                
                # Plot total field
                b_total = np.sqrt(np.sum(b_data**2, axis=1))
                axes[i].plot(times, b_total, color='black', label='|B|', linewidth=2)
                
                axes[i].set_ylabel(f'MMS{probe}\nB (nT)', fontsize=12)
                axes[i].legend(loc='upper right', fontsize=10)
                axes[i].grid(True, alpha=0.3)
                
                # Mark event time
                axes[i].axvline(event_time, color='red', linestyle='--', alpha=0.8, linewidth=2)
                
                # Set y-axis limits for better visibility
                axes[i].set_ylim(-50, 50)
        
        # Plot magnetic field magnitude comparison
        for i, probe in enumerate(['1', '2', '3', '4']):
            if probe in evt and 'B_gsm' in evt[probe]:
                times, b_data = evt[probe]['B_gsm']
                
                if not hasattr(times[0], 'strftime'):
                    times = [datetime.fromtimestamp(t) for t in times]
                
                b_total = np.sqrt(np.sum(b_data**2, axis=1))
                axes[4].plot(times, b_total, color=colors[i], label=f'MMS{probe}', linewidth=2)
        
        axes[4].set_ylabel('|B| (nT)', fontsize=12)
        axes[4].set_xlabel('Time (UT)', fontsize=12)
        axes[4].legend(fontsize=10)
        axes[4].grid(True, alpha=0.3)
        axes[4].axvline(event_time, color='red', linestyle='--', alpha=0.8, linewidth=2)
        axes[4].set_ylim(0, 50)
        
        plt.suptitle(f'MMS Magnetic Field Data: {event_time.strftime("%Y-%m-%d %H:%M:%S")} UT Event', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/magnetic_field_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Magnetic field overview saved to {results_dir}/magnetic_field_comprehensive.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating magnetic field overview: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_plasma_overview(evt, event_time, trange, results_dir):
    """Create plasma data overview"""
    
    print(f"\n🔍 Creating Plasma Data Overview")
    print("=" * 50)
    
    try:
        fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
        
        colors = ['red', 'blue', 'green', 'orange']
        
        # Plot ion density
        for i, probe in enumerate(['1', '2', '3', '4']):
            if probe in evt and 'N_i' in evt[probe]:
                times, density_data = evt[probe]['N_i']
                
                if not hasattr(times[0], 'strftime'):
                    times = [datetime.fromtimestamp(t) for t in times]
                
                axes[0].plot(times, density_data, color=colors[i], label=f'MMS{probe}', linewidth=2)
        
        axes[0].set_ylabel('Ion Density\n(cm⁻³)', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(event_time, color='red', linestyle='--', alpha=0.8, linewidth=2)
        axes[0].set_yscale('log')
        
        # Plot ion velocity
        for i, probe in enumerate(['1', '2', '3', '4']):
            if probe in evt and 'V_i_gsm' in evt[probe]:
                times, velocity_data = evt[probe]['V_i_gsm']
                
                if not hasattr(times[0], 'strftime'):
                    times = [datetime.fromtimestamp(t) for t in times]
                
                v_magnitude = np.sqrt(np.sum(velocity_data**2, axis=1))
                axes[1].plot(times, v_magnitude, color=colors[i], label=f'MMS{probe}', linewidth=2)
        
        axes[1].set_ylabel('Ion Speed\n(km/s)', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(event_time, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        # Plot electron density
        for i, probe in enumerate(['1', '2', '3', '4']):
            if probe in evt and 'N_e' in evt[probe]:
                times, density_data = evt[probe]['N_e']
                
                if not hasattr(times[0], 'strftime'):
                    times = [datetime.fromtimestamp(t) for t in times]
                
                axes[2].plot(times, density_data, color=colors[i], label=f'MMS{probe}', linewidth=2)
        
        axes[2].set_ylabel('Electron Density\n(cm⁻³)', fontsize=12)
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)
        axes[2].axvline(event_time, color='red', linestyle='--', alpha=0.8, linewidth=2)
        axes[2].set_yscale('log')
        
        # Plot plasma beta (if available)
        beta_plotted = False
        for i, probe in enumerate(['1', '2', '3', '4']):
            if probe in evt:
                # Try to calculate plasma beta from available data
                if 'N_i' in evt[probe] and 'B_gsm' in evt[probe]:
                    times_n, n_data = evt[probe]['N_i']
                    times_b, b_data = evt[probe]['B_gsm']
                    
                    if not hasattr(times_n[0], 'strftime'):
                        times_n = [datetime.fromtimestamp(t) for t in times_n]
                    
                    # Simple plasma beta estimate (assuming T ~ 1 keV)
                    b_total = np.sqrt(np.sum(b_data**2, axis=1))
                    # Interpolate B to density times if needed
                    if len(times_n) == len(b_total):
                        beta = (n_data * 1.38e-23 * 1000) / (b_total * 1e-9)**2 * 2e-7  # Rough estimate
                        axes[3].plot(times_n, beta, color=colors[i], label=f'MMS{probe}', linewidth=2)
                        beta_plotted = True
        
        if beta_plotted:
            axes[3].set_ylabel('Plasma β\n(estimate)', fontsize=12)
            axes[3].legend(fontsize=10)
            axes[3].set_yscale('log')
        else:
            axes[3].text(0.5, 0.5, 'Plasma β calculation not available', 
                        transform=axes[3].transAxes, ha='center', va='center', fontsize=12)
        
        axes[3].grid(True, alpha=0.3)
        axes[3].axvline(event_time, color='red', linestyle='--', alpha=0.8, linewidth=2)
        axes[3].set_xlabel('Time (UT)', fontsize=12)
        
        plt.suptitle(f'MMS Plasma Data Overview: {event_time.strftime("%Y-%m-%d %H:%M:%S")} UT Event', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/plasma_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Plasma overview saved to {results_dir}/plasma_overview.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating plasma overview: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_comprehensive_summary_plot(evt, event_time, trange, results_dir):
    """Create a comprehensive summary plot with all key parameters"""

    print(f"\n🔍 Creating Comprehensive Summary Plot")
    print("=" * 50)

    try:
        fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)

        colors = ['red', 'blue', 'green', 'orange']

        # Plot 1: Magnetic field magnitude
        for i, probe in enumerate(['1', '2', '3', '4']):
            if probe in evt and 'B_gsm' in evt[probe]:
                times, b_data = evt[probe]['B_gsm']

                if not hasattr(times[0], 'strftime'):
                    times = [datetime.fromtimestamp(t) for t in times]

                b_total = np.sqrt(np.sum(b_data**2, axis=1))
                axes[0].plot(times, b_total, color=colors[i], label=f'MMS{probe}', linewidth=2)

        axes[0].set_ylabel('|B| (nT)', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=12, loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(event_time, color='red', linestyle='--', alpha=0.8, linewidth=3, label='Event Time')
        axes[0].set_title('Magnetic Field Magnitude', fontsize=14, fontweight='bold')

        # Plot 2: Ion density
        for i, probe in enumerate(['1', '2', '3', '4']):
            if probe in evt and 'N_i' in evt[probe]:
                times, density_data = evt[probe]['N_i']

                if not hasattr(times[0], 'strftime'):
                    times = [datetime.fromtimestamp(t) for t in times]

                axes[1].plot(times, density_data, color=colors[i], label=f'MMS{probe}', linewidth=2)

        axes[1].set_ylabel('Ion Density (cm⁻³)', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=12, loc='upper right')
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(event_time, color='red', linestyle='--', alpha=0.8, linewidth=3)
        axes[1].set_yscale('log')
        axes[1].set_title('Ion Density', fontsize=14, fontweight='bold')

        # Plot 3: Ion velocity magnitude
        for i, probe in enumerate(['1', '2', '3', '4']):
            if probe in evt and 'V_i_gsm' in evt[probe]:
                times, velocity_data = evt[probe]['V_i_gsm']

                if not hasattr(times[0], 'strftime'):
                    times = [datetime.fromtimestamp(t) for t in times]

                v_magnitude = np.sqrt(np.sum(velocity_data**2, axis=1))
                axes[2].plot(times, v_magnitude, color=colors[i], label=f'MMS{probe}', linewidth=2)

        axes[2].set_ylabel('Ion Speed (km/s)', fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=12, loc='upper right')
        axes[2].grid(True, alpha=0.3)
        axes[2].axvline(event_time, color='red', linestyle='--', alpha=0.8, linewidth=3)
        axes[2].set_title('Ion Velocity Magnitude', fontsize=14, fontweight='bold')

        # Plot 4: Electric field magnitude (if available)
        e_field_plotted = False
        for i, probe in enumerate(['1', '2', '3', '4']):
            if probe in evt and 'E_gse' in evt[probe]:
                times, e_data = evt[probe]['E_gse']

                if not hasattr(times[0], 'strftime'):
                    times = [datetime.fromtimestamp(t) for t in times]

                if len(e_data.shape) == 2 and e_data.shape[1] >= 3:
                    e_total = np.sqrt(np.sum(e_data**2, axis=1))
                    axes[3].plot(times, e_total, color=colors[i], label=f'MMS{probe}', linewidth=2)
                    e_field_plotted = True

        if e_field_plotted:
            axes[3].set_ylabel('|E| (mV/m)', fontsize=14, fontweight='bold')
            axes[3].legend(fontsize=12, loc='upper right')
            axes[3].set_title('Electric Field Magnitude', fontsize=14, fontweight='bold')
        else:
            axes[3].text(0.5, 0.5, 'Electric Field Data Not Available',
                        transform=axes[3].transAxes, ha='center', va='center',
                        fontsize=16, fontweight='bold')
            axes[3].set_ylabel('E-field', fontsize=14, fontweight='bold')

        axes[3].grid(True, alpha=0.3)
        axes[3].axvline(event_time, color='red', linestyle='--', alpha=0.8, linewidth=3)
        axes[3].set_xlabel('Time (UT)', fontsize=14, fontweight='bold')

        plt.suptitle(f'MMS Magnetopause Crossing: {event_time.strftime("%Y-%m-%d %H:%M:%S")} UT',
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{results_dir}/comprehensive_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Comprehensive summary plot saved to {results_dir}/comprehensive_summary.png")

        return True

    except Exception as e:
        print(f"❌ Error creating comprehensive summary plot: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main analysis function"""

    print("FOCUSED MMS ANALYSIS: 2019-01-27 EVENT")
    print("=" * 80)
    print("Creating key visualizations for magnetopause crossing analysis")
    print("=" * 80)

    # Create results directory
    results_dir = create_results_directory()

    # Load event data
    evt, event_time, trange = load_event_data()

    if evt is None:
        print("❌ Failed to load event data - cannot proceed")
        return False

    # List available data
    print(f"\n📊 Available Data Summary:")
    for probe in ['1', '2', '3', '4']:
        if probe in evt:
            print(f"   MMS{probe}: {len(evt[probe])} variables")
            key_vars = ['B_gsm', 'N_i', 'V_i_gsm', 'N_e', 'E_gse', 'POS_gsm']
            available = [var for var in key_vars if var in evt[probe]]
            print(f"      Key variables: {available}")

    # Run analyses
    analyses = [
        ("Magnetic Field Overview", lambda: create_magnetic_field_overview(evt, event_time, trange, results_dir)),
        ("Plasma Data Overview", lambda: create_plasma_overview(evt, event_time, trange, results_dir)),
        ("Comprehensive Summary", lambda: create_comprehensive_summary_plot(evt, event_time, trange, results_dir))
    ]

    successful_analyses = 0

    for analysis_name, analysis_func in analyses:
        print(f"\n" + "=" * 80)
        print(f"RUNNING: {analysis_name}")
        print("=" * 80)

        try:
            result = analysis_func()
            if result:
                successful_analyses += 1
                print(f"✅ COMPLETED: {analysis_name}")
            else:
                print(f"⚠️ PARTIAL: {analysis_name}")
        except Exception as e:
            print(f"❌ FAILED: {analysis_name} - {e}")

    # Final summary
    print(f"\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    total_analyses = len(analyses)

    print(f"Successful analyses: {successful_analyses}/{total_analyses}")
    print(f"Results saved to: {results_dir}/")

    # List generated files
    print(f"\nGenerated files in {results_dir}/:")
    if os.path.exists(results_dir):
        files = [f for f in os.listdir(results_dir) if f.endswith('.png')]
        for file in sorted(files):
            print(f"   📄 {file}")

    if successful_analyses >= total_analyses * 0.8:  # 80% success rate
        print("🎉 ANALYSIS SUCCESSFUL!")
        print("✅ Key MMS visualizations generated")
        print("✅ Magnetopause crossing data analyzed")
        print("✅ Results ready for scientific interpretation")
    else:
        print("⚠️ PARTIAL SUCCESS")
        print("❌ Some analyses failed - check data availability")

    return successful_analyses >= total_analyses * 0.8


if __name__ == "__main__":
    success = main()

    if success:
        print(f"\n🎯 MMS FOCUSED ANALYSIS: COMPLETE SUCCESS")
        print(f"✅ All key visualizations generated for 2019-01-27 event")
    else:
        print(f"\n⚠️ MMS FOCUSED ANALYSIS: PARTIAL SUCCESS")
        print(f"❌ Some components need investigation")
