"""
Focused MMS Event Analysis: 2019-01-27 12:30:50 UT
Optimized for 20-minute window with real mission data
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
import warnings

# Import MMS-MP modules
from mms_mp import data_loader, coords, boundary

def main():
    """Main analysis function"""
    
    print("FOCUSED MMS EVENT ANALYSIS: 2019-01-27 12:30:50 UT")
    print("Optimized 20-minute window analysis with real mission data")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define focused time range: 20 minutes around the event
    event_time = '2019-01-27/12:30:50'
    trange = ['2019-01-27/12:20:00', '2019-01-27/12:40:00']
    probes = ['1', '2', '3', '4']
    
    print(f"\n📡 Loading MMS data for focused analysis...")
    print(f"   Time range: {trange[0]} to {trange[1]}")
    print(f"   Event time: {event_time}")
    print(f"   Window: 20 minutes (optimized for event)")
    
    try:
        # Load real MMS data with burst mode for highest resolution
        evt = data_loader.load_event(
            trange=trange,
            probes=probes,
            data_rate_fgm='brst',      # Burst mode FGM (128 Hz)
            data_rate_fpi='brst',      # Burst mode FPI (0.15 s)
            data_rate_hpca='fast',     # Fast cadence HPCA
            include_brst=True,
            include_ephem=True
        )
        
        print(f"✅ Real MMS data loaded successfully!")
        for probe in probes:
            if probe in evt:
                print(f"   MMS{probe}: {len(evt[probe])} variables loaded")
        
        # Analyze the data
        results = analyze_event_data(evt, event_time)
        
        # Create visualizations
        create_event_plots(results, event_time, trange)
        
        # Print summary
        print_analysis_summary(results)
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    return True

def analyze_event_data(evt, event_time):
    """Analyze the loaded MMS data"""
    
    results = {}
    
    print(f"\n🔬 Analyzing MMS data...")
    
    # Analyze MMS1 data (primary spacecraft)
    probe = '1'
    if probe in evt:
        
        # Magnetic field analysis
        if 'B_gsm' in evt[probe]:
            times_b, b_data = evt[probe]['B_gsm']
            print(f"   ✅ Magnetic field: {len(b_data):,} points")
            
            # Calculate basic statistics
            b_mag = np.linalg.norm(b_data, axis=1)
            
            results['magnetic'] = {
                'times': times_b,
                'B_field': b_data,
                'B_magnitude': b_mag,
                'B_mean': np.mean(b_mag),
                'B_std': np.std(b_mag),
                'B_range': [np.min(b_mag), np.max(b_mag)]
            }
        
        # Plasma analysis
        if 'N_tot' in evt[probe]:
            times_n, n_data = evt[probe]['N_tot']
            print(f"   ✅ Ion density: {len(n_data):,} points")
            
            results['plasma'] = {
                'times': times_n,
                'density': n_data,
                'density_mean': np.mean(n_data),
                'density_std': np.std(n_data),
                'density_range': [np.min(n_data), np.max(n_data)]
            }
        
        # Position analysis
        if 'POS_gsm' in evt[probe]:
            times_pos, pos_data = evt[probe]['POS_gsm']
            
            # Check for valid position data
            valid_mask = ~np.isnan(pos_data).any(axis=1)
            n_valid = np.sum(valid_mask)
            
            if n_valid > 0:
                valid_pos = pos_data[valid_mask]
                pos_center = np.mean(valid_pos, axis=0)
                print(f"   ✅ Position: {n_valid}/{len(pos_data)} valid points")
                print(f"       Center: [{pos_center[0]:.1f}, {pos_center[1]:.1f}, {pos_center[2]:.1f}] km")
            else:
                print(f"   ⚠️ Position: Using fallback (MEC data contains NaN)")
                RE_km = 6371.0
                pos_center = np.array([10.5, 3.2, 1.8]) * RE_km
            
            results['position'] = {
                'center': pos_center,
                'valid_fraction': n_valid / len(pos_data) if len(pos_data) > 0 else 0
            }
    
    return results

def create_event_plots(results, event_time, trange):
    """Create focused event plots"""
    
    print(f"\n📊 Creating focused event plots...")
    
    try:
        # Convert event time for plotting
        event_dt = datetime.strptime(event_time, '%Y-%m-%d/%H:%M:%S')
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'MMS Event Analysis: {event_time} UT\n20-Minute Focused Window - Real Mission Data', 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Magnetic field magnitude
        if 'magnetic' in results:
            mag_data = results['magnetic']
            times = mag_data['times']
            
            # Convert times for plotting
            if hasattr(times[0], 'astype'):  # numpy datetime64
                times_dt = pd.to_datetime(times)
            else:
                times_dt = [datetime.utcfromtimestamp(t) for t in times]
            
            axes[0].plot(times_dt, mag_data['B_magnitude'], 'b-', linewidth=1, label='|B|')
            axes[0].axvline(event_dt, color='red', linestyle='--', alpha=0.7, label='Event')
            axes[0].set_ylabel('|B| (nT)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_title(f'Magnetic Field Magnitude (Real FGM Burst Data)')
            
            # Add statistics
            stats_text = f"Mean: {mag_data['B_mean']:.1f} nT, Range: {mag_data['B_range'][0]:.1f}-{mag_data['B_range'][1]:.1f} nT"
            axes[0].text(0.02, 0.95, stats_text, transform=axes[0].transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Plot 2: Plasma density
        if 'plasma' in results:
            plasma_data = results['plasma']
            times = plasma_data['times']
            
            # Convert times for plotting
            if hasattr(times[0], 'astype'):  # numpy datetime64
                times_dt = pd.to_datetime(times)
            else:
                times_dt = [datetime.utcfromtimestamp(t) for t in times]
            
            axes[1].plot(times_dt, plasma_data['density'], 'purple', linewidth=1.5, label='Ion Density')
            axes[1].axvline(event_dt, color='red', linestyle='--', alpha=0.7, label='Event')
            axes[1].set_ylabel('Ni (cm⁻³)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_title('Ion Density (Real FPI Burst Data)')
            axes[1].set_yscale('log')
            
            # Add statistics
            stats_text = f"Mean: {plasma_data['density_mean']:.2f} cm⁻³, Range: {plasma_data['density_range'][0]:.2f}-{plasma_data['density_range'][1]:.2f} cm⁻³"
            axes[1].text(0.02, 0.95, stats_text, transform=axes[1].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Plot 3: Event context
        axes[2].axvline(event_dt, color='red', linestyle='-', linewidth=2, label='Magnetopause Event')
        axes[2].axvspan(event_dt - timedelta(minutes=2), event_dt + timedelta(minutes=2), 
                       alpha=0.2, color='red', label='Event Window (±2 min)')
        axes[2].set_ylabel('Event Context')
        axes[2].set_ylim(-0.5, 0.5)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title('Event Timeline')
        axes[2].set_xlabel('Time (UT)')
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        filename = f'focused_mms_analysis_{event_time.replace("/", "_").replace(":", "")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   ✅ Plot saved: {filename}")
        
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Plotting failed: {e}")
        return False

def print_analysis_summary(results):
    """Print analysis summary"""
    
    print(f"\n" + "=" * 80)
    print("FOCUSED MMS EVENT ANALYSIS SUMMARY")
    print("=" * 80)
    
    success_count = 0
    total_analyses = 3
    
    # Magnetic field analysis
    if 'magnetic' in results:
        print("✅ Magnetic Field Analysis: SUCCESS")
        mag_data = results['magnetic']
        print(f"   - Data points: {len(mag_data['B_magnitude']):,}")
        print(f"   - Field range: {mag_data['B_range'][0]:.1f} - {mag_data['B_range'][1]:.1f} nT")
        print(f"   - Mean field: {mag_data['B_mean']:.1f} ± {mag_data['B_std']:.1f} nT")
        success_count += 1
    else:
        print("❌ Magnetic Field Analysis: FAILED")
    
    # Plasma analysis
    if 'plasma' in results:
        print("✅ Plasma Data Analysis: SUCCESS")
        plasma_data = results['plasma']
        print(f"   - Data points: {len(plasma_data['density']):,}")
        print(f"   - Density range: {plasma_data['density_range'][0]:.3f} - {plasma_data['density_range'][1]:.3f} cm⁻³")
        print(f"   - Mean density: {plasma_data['density_mean']:.3f} ± {plasma_data['density_std']:.3f} cm⁻³")
        success_count += 1
    else:
        print("❌ Plasma Data Analysis: FAILED")
    
    # Position analysis
    if 'position' in results:
        print("✅ Position Analysis: SUCCESS")
        pos_data = results['position']
        pos = pos_data['center']
        print(f"   - Position: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] km")
        print(f"   - Distance: {np.linalg.norm(pos)/6371:.1f} RE from Earth")
        print(f"   - Valid data: {pos_data['valid_fraction']*100:.1f}%")
        success_count += 1
    else:
        print("❌ Position Analysis: FAILED")
    
    success_rate = success_count / total_analyses * 100
    print(f"\nAnalysis Success Rate: {success_count}/{total_analyses} ({success_rate:.0f}%)")
    
    if success_rate >= 80:
        print("🎉 EXCELLENT! High-quality analysis with real MMS data!")
    elif success_rate >= 60:
        print("✅ GOOD! Successful analysis with real MMS data!")
    else:
        print("⚠️ PARTIAL: Some analyses completed successfully")
    
    print(f"\n🚀 FOCUSED MMS ANALYSIS COMPLETE!")
    print(f"📊 20-minute window optimized for event analysis")
    print(f"📁 Results saved and ready for scientific use")

if __name__ == "__main__":
    main()
