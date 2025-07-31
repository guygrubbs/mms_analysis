"""
Diagnostic Analysis: Why LMN Transformation is Failing
Event: 2019-01-27 12:30:50 UT

This script investigates the root causes of LMN transformation failure
and provides insights into magnetic field conditions during this period.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import MMS modules
from mms_mp import data_loader, coords
from scipy.interpolate import interp1d

def diagnose_lmn_failure():
    """
    Comprehensive diagnosis of LMN transformation failure
    """
    print("🔍 DIAGNOSING LMN TRANSFORMATION FAILURE")
    print("Event: 2019-01-27 12:30:50 UT")
    print("=" * 80)
    
    # Define time ranges for analysis
    event_time = "2019-01-27T12:30:50"
    target_start = "2019-01-27T12:25:00"
    target_end = "2019-01-27T12:35:00"
    
    # Load wider time range for context
    extended_start = "2019-01-27T12:00:00"
    extended_end = "2019-01-27T13:00:00"
    full_day_range = ["2019-01-27T00:00:00", "2019-01-27T23:59:59"]
    
    probes = ['1', '2', '3', '4']
    
    print(f"🎯 Target Event: {event_time}")
    print(f"📊 Analysis Window: {target_start} to {target_end}")
    print(f"🔍 Extended Context: {extended_start} to {extended_end}")
    
    # Load data
    print("\n" + "="*80)
    print("1️⃣ LOADING MAGNETIC FIELD DATA")
    print("="*80)
    
    try:
        evt_full = data_loader.load_event(
            full_day_range, probes,
            data_rate_fgm='srvy',    # Survey mode for FGM (16 Hz)
            data_rate_fpi='fast',    # Fast mode for FPI (4.5s)
            include_edp=False,
            include_ephem=True
        )
        
        print("✅ Data loading successful")
        
        # Convert time ranges
        target_start_dt = datetime.fromisoformat(target_start.replace('Z', '+00:00'))
        target_end_dt = datetime.fromisoformat(target_end.replace('Z', '+00:00'))
        extended_start_dt = datetime.fromisoformat(extended_start.replace('Z', '+00:00'))
        extended_end_dt = datetime.fromisoformat(extended_end.replace('Z', '+00:00'))
        event_dt = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
        
        target_start_ts = target_start_dt.timestamp()
        target_end_ts = target_end_dt.timestamp()
        extended_start_ts = extended_start_dt.timestamp()
        extended_end_ts = extended_end_dt.timestamp()
        
        # Analyze each spacecraft
        print("\n" + "="*80)
        print("2️⃣ MAGNETIC FIELD DATA QUALITY ANALYSIS")
        print("="*80)
        
        spacecraft_analysis = {}
        
        for probe in probes:
            if probe not in evt_full or not evt_full[probe]:
                print(f"\n❌ MMS{probe}: No data available")
                continue
                
            print(f"\n🛰️ ANALYZING MMS{probe}")
            print("-" * 40)
            
            # Extract magnetic field data
            if 'B_gsm' not in evt_full[probe]:
                print(f"❌ No B_gsm data for MMS{probe}")
                continue
                
            t_b_full, b_gsm_full = evt_full[probe]['B_gsm']
            
            # Extract target time range
            target_mask = (t_b_full >= target_start_ts) & (t_b_full <= target_end_ts)
            extended_mask = (t_b_full >= extended_start_ts) & (t_b_full <= extended_end_ts)
            
            if np.sum(target_mask) == 0:
                print(f"❌ No data in target time range")
                continue
                
            t_target = t_b_full[target_mask]
            b_target = b_gsm_full[target_mask, :]
            t_extended = t_b_full[extended_mask]
            b_extended = b_gsm_full[extended_mask, :]
            
            # Convert to datetime
            times_target = [datetime.fromtimestamp(t) for t in t_target]
            times_extended = [datetime.fromtimestamp(t) for t in t_extended]
            
            print(f"📊 Target window: {len(t_target)} data points")
            print(f"📊 Extended window: {len(t_extended)} data points")
            
            # Data quality analysis
            print(f"\n🔍 DATA QUALITY ANALYSIS:")
            
            # Check for NaN/infinite values
            finite_mask_target = np.isfinite(b_target).all(axis=1)
            finite_mask_extended = np.isfinite(b_extended).all(axis=1)
            
            n_finite_target = np.sum(finite_mask_target)
            n_finite_extended = np.sum(finite_mask_extended)
            
            print(f"  • Target window finite samples: {n_finite_target}/{len(t_target)} ({100*n_finite_target/len(t_target):.1f}%)")
            print(f"  • Extended window finite samples: {n_finite_extended}/{len(t_extended)} ({100*n_finite_extended/len(t_extended):.1f}%)")
            
            if n_finite_target < 10:
                print(f"  ⚠️ WARNING: Very few finite samples in target window!")
            
            # Magnetic field statistics
            if n_finite_target > 0:
                b_finite = b_target[finite_mask_target, :]
                b_mag = np.linalg.norm(b_finite, axis=1)
                
                print(f"\n📈 MAGNETIC FIELD STATISTICS (Target Window):")
                print(f"  • |B| range: {np.min(b_mag):.2f} to {np.max(b_mag):.2f} nT")
                print(f"  • |B| mean: {np.mean(b_mag):.2f} ± {np.std(b_mag):.2f} nT")
                print(f"  • Bx range: {np.min(b_finite[:, 0]):.2f} to {np.max(b_finite[:, 0]):.2f} nT")
                print(f"  • By range: {np.min(b_finite[:, 1]):.2f} to {np.max(b_finite[:, 1]):.2f} nT")
                print(f"  • Bz range: {np.min(b_finite[:, 2]):.2f} to {np.max(b_finite[:, 2]):.2f} nT")
                
                # Variance analysis
                var_x = np.var(b_finite[:, 0])
                var_y = np.var(b_finite[:, 1])
                var_z = np.var(b_finite[:, 2])
                total_var = var_x + var_y + var_z
                
                print(f"\n🔄 VARIANCE ANALYSIS:")
                print(f"  • Variance Bx: {var_x:.3f} nT² ({100*var_x/total_var:.1f}%)")
                print(f"  • Variance By: {var_y:.3f} nT² ({100*var_y/total_var:.1f}%)")
                print(f"  • Variance Bz: {var_z:.3f} nT² ({100*var_z/total_var:.1f}%)")
                print(f"  • Total variance: {total_var:.3f} nT²")
                
                # Check if variance is sufficient for MVA
                if total_var < 1.0:  # Threshold for "quiet" conditions
                    print(f"  ⚠️ WARNING: Very low magnetic field variance - insufficient for reliable MVA!")
                elif total_var < 10.0:
                    print(f"  ⚠️ CAUTION: Low magnetic field variance - MVA may be unreliable")
                else:
                    print(f"  ✅ Sufficient variance for MVA analysis")
                
                # Try MVA analysis manually
                print(f"\n🧮 ATTEMPTING MANUAL MVA ANALYSIS:")
                try:
                    # Center the data
                    b_centered = b_finite - np.mean(b_finite, axis=0)
                    
                    # Compute covariance matrix
                    cov_matrix = np.cov(b_centered.T)
                    
                    # Eigenvalue decomposition
                    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                    
                    # Sort by eigenvalue (descending)
                    idx = np.argsort(eigenvals)[::-1]
                    eigenvals = eigenvals[idx]
                    eigenvecs = eigenvecs[:, idx]
                    
                    # Calculate eigenvalue ratios
                    lambda_max = eigenvals[0]
                    lambda_mid = eigenvals[1]
                    lambda_min = eigenvals[2]
                    
                    ratio_max_mid = lambda_max / lambda_mid if lambda_mid > 0 else np.inf
                    ratio_mid_min = lambda_mid / lambda_min if lambda_min > 0 else np.inf
                    
                    print(f"  • Eigenvalues: [{lambda_max:.3f}, {lambda_mid:.3f}, {lambda_min:.3f}]")
                    print(f"  • λmax/λmid ratio: {ratio_max_mid:.2f}")
                    print(f"  • λmid/λmin ratio: {ratio_mid_min:.2f}")
                    
                    # Quality assessment
                    if ratio_max_mid < 2.0:
                        print(f"  ❌ POOR MVA: λmax/λmid < 2.0 - no clear maximum variance direction")
                    elif ratio_mid_min < 2.0:
                        print(f"  ❌ POOR MVA: λmid/λmin < 2.0 - no clear minimum variance direction")
                    elif ratio_max_mid > 10.0 and ratio_mid_min > 10.0:
                        print(f"  ✅ EXCELLENT MVA: Clear separation of variance directions")
                    elif ratio_max_mid > 3.0 and ratio_mid_min > 3.0:
                        print(f"  ✅ GOOD MVA: Adequate separation of variance directions")
                    else:
                        print(f"  ⚠️ MARGINAL MVA: Weak separation of variance directions")
                    
                    # Store results
                    spacecraft_analysis[probe] = {
                        'n_samples': len(t_target),
                        'n_finite': n_finite_target,
                        'b_mag_mean': np.mean(b_mag),
                        'b_mag_std': np.std(b_mag),
                        'total_variance': total_var,
                        'eigenvals': eigenvals,
                        'ratio_max_mid': ratio_max_mid,
                        'ratio_mid_min': ratio_mid_min,
                        'times_target': times_target,
                        'b_target': b_target,
                        'b_finite': b_finite,
                        'finite_mask': finite_mask_target
                    }
                    
                except Exception as e:
                    print(f"  ❌ MVA analysis failed: {e}")
            
            else:
                print(f"❌ No finite magnetic field data in target window")
        
        # Summary and recommendations
        print("\n" + "="*80)
        print("3️⃣ DIAGNOSIS SUMMARY AND RECOMMENDATIONS")
        print("="*80)
        
        if len(spacecraft_analysis) == 0:
            print("❌ CRITICAL: No usable magnetic field data found for any spacecraft")
            print("\nPossible causes:")
            print("  • Data server issues or missing files")
            print("  • Incorrect time range specification")
            print("  • Instrument downtime during this period")
            return
        
        # Analyze overall conditions
        total_samples = sum([sc['n_samples'] for sc in spacecraft_analysis.values()])
        total_finite = sum([sc['n_finite'] for sc in spacecraft_analysis.values()])
        avg_variance = np.mean([sc['total_variance'] for sc in spacecraft_analysis.values()])
        avg_ratio_max_mid = np.mean([sc['ratio_max_mid'] for sc in spacecraft_analysis.values() if np.isfinite(sc['ratio_max_mid'])])
        avg_ratio_mid_min = np.mean([sc['ratio_mid_min'] for sc in spacecraft_analysis.values() if np.isfinite(sc['ratio_mid_min'])])
        
        print(f"\n📊 OVERALL ASSESSMENT:")
        print(f"  • Total data points: {total_samples}")
        print(f"  • Total finite points: {total_finite} ({100*total_finite/total_samples:.1f}%)")
        print(f"  • Average magnetic variance: {avg_variance:.3f} nT²")
        print(f"  • Average λmax/λmid ratio: {avg_ratio_max_mid:.2f}")
        print(f"  • Average λmid/λmin ratio: {avg_ratio_mid_min:.2f}")
        
        print(f"\n🎯 ROOT CAUSE ANALYSIS:")
        
        if total_finite < 40:  # Less than 10 points per spacecraft
            print("❌ PRIMARY ISSUE: Insufficient finite magnetic field data")
            print("   • Likely cause: Data gaps, NaN values, or instrument issues")
            print("   • Solution: Use different time period or check data quality")
        
        elif avg_variance < 1.0:
            print("❌ PRIMARY ISSUE: Extremely quiet magnetic field conditions")
            print("   • Likely cause: No magnetopause crossing occurred")
            print("   • The magnetic field is too stable for boundary analysis")
            print("   • Solution: This may not be a true magnetopause crossing event")
        
        elif avg_ratio_max_mid < 2.0 or avg_ratio_mid_min < 2.0:
            print("❌ PRIMARY ISSUE: Poor eigenvalue separation in MVA")
            print("   • Likely cause: No clear boundary normal structure")
            print("   • The magnetic field variations don't show boundary-like characteristics")
            print("   • Solution: This may not be a magnetopause crossing, or use different analysis method")
        
        else:
            print("✅ Data quality appears adequate - investigating other causes...")
        
        print(f"\n💡 RECOMMENDATIONS:")
        
        if avg_variance < 1.0:
            print("  1. 🔍 Verify this is actually a magnetopause crossing event")
            print("  2. 📅 Check space weather conditions for this date")
            print("  3. 🛰️ Examine spacecraft trajectory and location")
            print("  4. 📊 Use |B| magnitude instead of LMN coordinates")
            print("  5. 🔄 Try a different time period with more magnetic activity")
        
        elif total_finite < 40:
            print("  1. 📡 Check MMS data availability for this time period")
            print("  2. 🔧 Use different data rate (burst vs survey mode)")
            print("  3. ⏰ Expand the time window for analysis")
            print("  4. 🛰️ Focus on spacecraft with better data coverage")
        
        else:
            print("  1. 🔄 Try different time windows for MVA analysis")
            print("  2. 📊 Use hybrid approach with model-based normal direction")
            print("  3. 🧮 Apply data smoothing or filtering")
            print("  4. 📈 Use longer time series for more robust MVA")
        
        # Create diagnostic plots
        create_diagnostic_plots(spacecraft_analysis, event_dt)
        
        return spacecraft_analysis
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return {}


def create_diagnostic_plots(spacecraft_analysis, event_dt):
    """
    Create diagnostic plots showing magnetic field conditions
    """
    print(f"\n📊 Creating diagnostic plots...")
    
    if len(spacecraft_analysis) == 0:
        print("❌ No data available for plotting")
        return
    
    # Create figure with subplots for each spacecraft
    n_spacecraft = len(spacecraft_analysis)
    fig, axes = plt.subplots(n_spacecraft, 2, figsize=(16, 3*n_spacecraft))
    
    if n_spacecraft == 1:
        axes = axes.reshape(1, -1)
    
    for i, (probe, data) in enumerate(spacecraft_analysis.items()):
        # Plot 1: Magnetic field components
        ax1 = axes[i, 0]
        
        times = data['times_target']
        b_data = data['b_target']
        finite_mask = data['finite_mask']
        
        # Plot all components
        ax1.plot(times, b_data[:, 0], 'r-', label='Bx', alpha=0.7)
        ax1.plot(times, b_data[:, 1], 'g-', label='By', alpha=0.7)
        ax1.plot(times, b_data[:, 2], 'b-', label='Bz', alpha=0.7)
        ax1.plot(times, np.linalg.norm(b_data, axis=1), 'k-', label='|B|', linewidth=2)
        
        # Mark non-finite points
        if np.sum(~finite_mask) > 0:
            non_finite_times = [times[j] for j in range(len(times)) if not finite_mask[j]]
            ax1.scatter(non_finite_times, [0]*len(non_finite_times), 
                       color='red', marker='x', s=50, label='Non-finite', zorder=10)
        
        ax1.axvline(event_dt, color='orange', linestyle='--', linewidth=2, label='Event Time')
        ax1.set_ylabel('B (nT)')
        ax1.set_title(f'MMS{probe} Magnetic Field Components')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Variance analysis
        ax2 = axes[i, 1]
        
        if 'eigenvals' in data:
            eigenvals = data['eigenvals']
            labels = ['λmax (L)', 'λmid (M)', 'λmin (N)']
            colors = ['red', 'green', 'blue']
            
            bars = ax2.bar(labels, eigenvals, color=colors, alpha=0.7)
            ax2.set_ylabel('Eigenvalue')
            ax2.set_title(f'MMS{probe} MVA Eigenvalues')
            ax2.set_yscale('log')
            
            # Add ratio annotations
            ratio_max_mid = data['ratio_max_mid']
            ratio_mid_min = data['ratio_mid_min']
            
            ax2.text(0.5, 0.95, f'λmax/λmid = {ratio_max_mid:.2f}', 
                    transform=ax2.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            ax2.text(0.5, 0.85, f'λmid/λmin = {ratio_mid_min:.2f}', 
                    transform=ax2.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
            
            # Quality assessment
            if ratio_max_mid < 2.0 or ratio_mid_min < 2.0:
                quality = "POOR MVA"
                color = "lightcoral"
            elif ratio_max_mid > 10.0 and ratio_mid_min > 10.0:
                quality = "EXCELLENT MVA"
                color = "lightgreen"
            elif ratio_max_mid > 3.0 and ratio_mid_min > 3.0:
                quality = "GOOD MVA"
                color = "lightyellow"
            else:
                quality = "MARGINAL MVA"
                color = "orange"
            
            ax2.text(0.5, 0.75, quality, 
                    transform=ax2.transAxes, ha='center', va='top', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
        else:
            ax2.text(0.5, 0.5, 'MVA Analysis Failed', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=14,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
        
        # Format time axis for bottom row
        if i == n_spacecraft - 1:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            ax1.set_xlabel('Time (UT)')
        else:
            ax1.set_xticklabels([])
    
    plt.suptitle('LMN Transformation Failure Diagnosis\n2019-01-27 12:30:50 UT', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the plot
    plt.savefig('lmn_failure_diagnosis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Diagnostic plots saved: lmn_failure_diagnosis.png")


if __name__ == "__main__":
    print("🔍 MMS LMN TRANSFORMATION FAILURE DIAGNOSIS")
    print("Investigating why boundary normal coordinates cannot be calculated")
    print("Event: 2019-01-27 12:30:50 UT")
    print()
    
    # Run comprehensive diagnosis
    results = diagnose_lmn_failure()
    
    print("\n🎉 DIAGNOSIS COMPLETED!")
    print("\nKey findings will help explain:")
    print("  • Why MVA (Minimum Variance Analysis) is failing")
    print("  • Whether this is actually a magnetopause crossing")
    print("  • What magnetic field conditions exist during this period")
    print("  • How to improve the LMN transformation")
    print("  • Whether to use alternative coordinate systems")
    print("\n📊 Check 'lmn_failure_diagnosis.png' for detailed plots!")
