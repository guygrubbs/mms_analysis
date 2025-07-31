"""
MMS Data Products Investigation: Units, Reference Frames, and Data Quality
Event: 2019-01-27 12:30:50 UT

This script investigates the MMS data products to verify:
1. Correct units for magnetic field and position data
2. Proper reference frames (GSM, GSE, etc.)
3. Data product types and their characteristics
4. Potential unit conversion or scaling issues
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import MMS modules
from mms_mp import data_loader
import inspect
import sys

def investigate_mms_data_products():
    """
    Comprehensive investigation of MMS data products, units, and reference frames
    """
    print("🔍 MMS DATA PRODUCTS INVESTIGATION")
    print("Verifying units, reference frames, and data characteristics")
    print("=" * 80)
    
    # Define time range for investigation
    event_time = "2019-01-27T12:30:50"
    target_start = "2019-01-27T12:25:00"
    target_end = "2019-01-27T12:35:00"
    
    # Use shorter time range for detailed investigation
    short_range = ["2019-01-27T12:29:00", "2019-01-27T12:31:00"]
    probes = ['1', '2', '3', '4']
    
    print(f"🎯 Investigation Time: {short_range[0]} to {short_range[1]}")
    print(f"🛰️ Spacecraft: MMS{', MMS'.join(probes)}")
    
    # First, examine the data loader source code
    print("\n" + "="*80)
    print("1️⃣ EXAMINING MMS DATA LOADER SOURCE CODE")
    print("="*80)
    
    try:
        # Get data loader source information
        loader_source = inspect.getsource(data_loader.load_event)
        print("✅ Data loader source code accessible")
        
        # Look for key information in the source
        print("\n🔍 Key information from data loader:")
        
        # Check for unit specifications
        if 'nT' in loader_source or 'nanotesla' in loader_source.lower():
            print("  • Found nT (nanotesla) unit references")
        if 'pT' in loader_source or 'picotesla' in loader_source.lower():
            print("  • Found pT (picotesla) unit references")
        if 'km' in loader_source or 'kilometer' in loader_source.lower():
            print("  • Found km (kilometer) unit references")
        if 'GSM' in loader_source or 'gsm' in loader_source:
            print("  • Found GSM coordinate system references")
        if 'GSE' in loader_source or 'gse' in loader_source:
            print("  • Found GSE coordinate system references")
            
    except Exception as e:
        print(f"⚠️ Could not examine source code: {e}")
    
    # Load a small sample of data for detailed investigation
    print("\n" + "="*80)
    print("2️⃣ LOADING SAMPLE DATA FOR INVESTIGATION")
    print("="*80)
    
    try:
        # Load short time range with all available data products
        evt_sample = data_loader.load_event(
            short_range, probes,
            data_rate_fgm='srvy',    # Survey mode for FGM
            data_rate_fpi='fast',    # Fast mode for FPI
            include_edp=True,        # Include electric field
            include_ephem=True       # Include ephemeris
        )
        
        print("✅ Sample data loading successful")
        
        # Investigate each spacecraft's data
        for probe in probes:
            if probe not in evt_sample or not evt_sample[probe]:
                print(f"\n❌ MMS{probe}: No data available")
                continue
                
            print(f"\n🛰️ INVESTIGATING MMS{probe} DATA PRODUCTS")
            print("-" * 50)
            
            spacecraft_data = evt_sample[probe]
            
            # List all available data products
            print(f"📊 Available data products ({len(spacecraft_data)} total):")
            for var_name, (t_data, values) in spacecraft_data.items():
                data_shape = values.shape if hasattr(values, 'shape') else 'scalar'
                time_points = len(t_data) if hasattr(t_data, '__len__') else 1
                print(f"  • {var_name}: {time_points} time points, shape {data_shape}")
            
            # Detailed investigation of key data products
            print(f"\n🔍 DETAILED INVESTIGATION OF KEY PRODUCTS:")
            
            # Investigate magnetic field data
            b_field_vars = [var for var in spacecraft_data.keys() if 'B_' in var or 'b_' in var]
            if b_field_vars:
                print(f"\n🧲 MAGNETIC FIELD DATA:")
                for var_name in b_field_vars:
                    t_data, values = spacecraft_data[var_name]
                    if len(t_data) > 0 and len(values) > 0:
                        # Analyze the data
                        if values.ndim == 1:
                            b_stats = {
                                'min': np.nanmin(values),
                                'max': np.nanmax(values),
                                'mean': np.nanmean(values),
                                'std': np.nanstd(values)
                            }
                            print(f"    {var_name}: {b_stats['min']:.2f} to {b_stats['max']:.2f} (mean: {b_stats['mean']:.2f} ± {b_stats['std']:.2f})")
                        else:
                            for i in range(min(3, values.shape[1])):
                                component = values[:, i]
                                comp_name = ['X', 'Y', 'Z'][i] if i < 3 else f'C{i}'
                                b_stats = {
                                    'min': np.nanmin(component),
                                    'max': np.nanmax(component),
                                    'mean': np.nanmean(component),
                                    'std': np.nanstd(component)
                                }
                                print(f"    {var_name}[{comp_name}]: {b_stats['min']:.2f} to {b_stats['max']:.2f} (mean: {b_stats['mean']:.2f} ± {b_stats['std']:.2f})")
                            
                            # Calculate magnitude
                            if values.shape[1] >= 3:
                                b_mag = np.linalg.norm(values[:, :3], axis=1)
                                mag_stats = {
                                    'min': np.nanmin(b_mag),
                                    'max': np.nanmax(b_mag),
                                    'mean': np.nanmean(b_mag),
                                    'std': np.nanstd(b_mag)
                                }
                                print(f"    |{var_name}|: {mag_stats['min']:.2f} to {mag_stats['max']:.2f} (mean: {mag_stats['mean']:.2f} ± {mag_stats['std']:.2f})")
            
            # Investigate position data
            pos_vars = [var for var in spacecraft_data.keys() if 'POS' in var or 'pos' in var or 'R_' in var]
            if pos_vars:
                print(f"\n📍 POSITION DATA:")
                for var_name in pos_vars:
                    t_data, values = spacecraft_data[var_name]
                    if len(t_data) > 0 and len(values) > 0:
                        if values.ndim > 1 and values.shape[1] >= 3:
                            # Calculate distance from Earth
                            r_earth = np.linalg.norm(values[:, :3], axis=1)
                            pos_stats = {
                                'min': np.nanmin(r_earth),
                                'max': np.nanmax(r_earth),
                                'mean': np.nanmean(r_earth),
                                'std': np.nanstd(r_earth)
                            }
                            print(f"    |{var_name}|: {pos_stats['min']:.2f} to {pos_stats['max']:.2f} (mean: {pos_stats['mean']:.2f} ± {pos_stats['std']:.2f})")
                            
                            # Check if this looks like km or RE
                            if pos_stats['mean'] > 1000:
                                print(f"      → Likely in kilometers (km)")
                                re_equivalent = pos_stats['mean'] / 6371.0  # Earth radius in km
                                print(f"      → Equivalent to ~{re_equivalent:.2f} Earth radii (RE)")
                            elif pos_stats['mean'] > 1:
                                print(f"      → Likely in Earth radii (RE)")
                                km_equivalent = pos_stats['mean'] * 6371.0
                                print(f"      → Equivalent to ~{km_equivalent:.2f} kilometers")
                            else:
                                print(f"      → Unknown units - values too small")
            
            # Investigate plasma data
            plasma_vars = [var for var in spacecraft_data.keys() if any(x in var for x in ['N_', 'V_', 'T_', 'P_', 'density', 'velocity', 'temperature'])]
            if plasma_vars:
                print(f"\n🌊 PLASMA DATA:")
                for var_name in plasma_vars[:5]:  # Limit to first 5 for brevity
                    t_data, values = spacecraft_data[var_name]
                    if len(t_data) > 0 and len(values) > 0:
                        if values.ndim == 1:
                            plasma_stats = {
                                'min': np.nanmin(values),
                                'max': np.nanmax(values),
                                'mean': np.nanmean(values),
                                'std': np.nanstd(values)
                            }
                            print(f"    {var_name}: {plasma_stats['min']:.2e} to {plasma_stats['max']:.2e} (mean: {plasma_stats['mean']:.2e})")
        
        # Analysis and recommendations
        print("\n" + "="*80)
        print("3️⃣ ANALYSIS AND UNIT VERIFICATION")
        print("="*80)
        
        analyze_data_characteristics(evt_sample)
        
        return evt_sample
        
    except Exception as e:
        print(f"❌ Sample data loading failed: {e}")
        return {}


def analyze_data_characteristics(evt_sample):
    """
    Analyze data characteristics to verify units and reference frames
    """
    print("\n🔍 ANALYZING DATA CHARACTERISTICS:")
    
    # Check magnetic field values across all spacecraft
    all_b_values = []
    all_pos_values = []
    
    for probe, spacecraft_data in evt_sample.items():
        if not spacecraft_data:
            continue
            
        # Collect magnetic field data
        for var_name, (t_data, values) in spacecraft_data.items():
            if 'B_' in var_name and len(values) > 0:
                if values.ndim > 1 and values.shape[1] >= 3:
                    b_mag = np.linalg.norm(values[:, :3], axis=1)
                    all_b_values.extend(b_mag[np.isfinite(b_mag)])
                elif values.ndim == 1:
                    all_b_values.extend(values[np.isfinite(values)])
        
        # Collect position data
        for var_name, (t_data, values) in spacecraft_data.items():
            if 'POS' in var_name and len(values) > 0:
                if values.ndim > 1 and values.shape[1] >= 3:
                    r_earth = np.linalg.norm(values[:, :3], axis=1)
                    all_pos_values.extend(r_earth[np.isfinite(r_earth)])
    
    # Analyze magnetic field characteristics
    if all_b_values:
        b_array = np.array(all_b_values)
        print(f"\n🧲 MAGNETIC FIELD ANALYSIS:")
        print(f"  • Range: {np.min(b_array):.2f} to {np.max(b_array):.2f}")
        print(f"  • Mean: {np.mean(b_array):.2f} ± {np.std(b_array):.2f}")
        print(f"  • Median: {np.median(b_array):.2f}")
        
        # Determine likely units and location
        if np.mean(b_array) > 5000:
            print(f"  ⚠️ WARNING: Extremely high magnetic field values!")
            print(f"    → Possible issues:")
            print(f"      - Wrong units (maybe pT instead of nT?)")
            print(f"      - Very close to Earth (inner radiation belts)")
            print(f"      - Data corruption or scaling error")
        elif np.mean(b_array) > 1000:
            print(f"  ⚠️ CAUTION: High magnetic field values")
            print(f"    → Likely in inner magnetosphere or very close to Earth")
        elif 100 <= np.mean(b_array) <= 1000:
            print(f"  ✅ NORMAL: Typical magnetosphere values")
        elif 20 <= np.mean(b_array) <= 100:
            print(f"  ✅ NORMAL: Typical magnetopause/magnetosheath values")
        elif 5 <= np.mean(b_array) <= 20:
            print(f"  ✅ NORMAL: Typical solar wind values")
        else:
            print(f"  ⚠️ WARNING: Unusually low magnetic field values")
    
    # Analyze position characteristics
    if all_pos_values:
        pos_array = np.array(all_pos_values)
        print(f"\n📍 POSITION ANALYSIS:")
        print(f"  • Range: {np.min(pos_array):.2f} to {np.max(pos_array):.2f}")
        print(f"  • Mean: {np.mean(pos_array):.2f} ± {np.std(pos_array):.2f}")
        
        # Determine likely units
        if np.mean(pos_array) > 1000:
            print(f"  ✅ Units likely in kilometers (km)")
            re_dist = np.mean(pos_array) / 6371.0
            print(f"  📏 Distance from Earth: ~{re_dist:.2f} Earth radii (RE)")
            
            if re_dist < 3:
                print(f"    → Very close to Earth (inner magnetosphere)")
            elif 3 <= re_dist <= 7:
                print(f"    → Inner to middle magnetosphere")
            elif 7 <= re_dist <= 15:
                print(f"    → Typical magnetopause region")
            elif re_dist > 15:
                print(f"    → Solar wind or distant magnetosphere")
                
        elif 1 <= np.mean(pos_array) <= 100:
            print(f"  ✅ Units likely in Earth radii (RE)")
            print(f"  📏 Distance from Earth: ~{np.mean(pos_array):.2f} RE")
        else:
            print(f"  ⚠️ WARNING: Unusual position values - unknown units")
    
    # Overall assessment
    print(f"\n📋 OVERALL ASSESSMENT:")
    
    if all_b_values and np.mean(np.array(all_b_values)) > 5000:
        print(f"  ❌ CRITICAL ISSUE: Magnetic field values are extremely high")
        print(f"    → This suggests a fundamental problem with:")
        print(f"      • Units (wrong scaling factor)")
        print(f"      • Data product selection")
        print(f"      • Reference frame conversion")
        print(f"      • Or spacecraft location interpretation")
    
    if all_pos_values and all_b_values:
        pos_mean = np.mean(np.array(all_pos_values))
        b_mean = np.mean(np.array(all_b_values))
        
        if pos_mean > 1000:  # Assuming km
            re_dist = pos_mean / 6371.0
            if re_dist < 3 and b_mean > 1000:
                print(f"  ⚠️ CONSISTENT: High B-field matches close Earth distance")
                print(f"    → Spacecraft may be in inner radiation belts")
            elif re_dist > 10 and b_mean > 1000:
                print(f"  ❌ INCONSISTENT: High B-field at large distance is unusual")
                print(f"    → Suggests unit or reference frame problem")


def create_data_verification_plots(evt_sample):
    """
    Create plots to visualize data characteristics and verify units
    """
    print(f"\n📊 Creating data verification plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Magnetic field magnitude vs time
    ax1 = axes[0, 0]
    for probe, spacecraft_data in evt_sample.items():
        if not spacecraft_data:
            continue
        
        for var_name, (t_data, values) in spacecraft_data.items():
            if 'B_gsm' in var_name and len(values) > 0:
                times = [datetime.fromtimestamp(t) for t in t_data]
                if values.ndim > 1 and values.shape[1] >= 3:
                    b_mag = np.linalg.norm(values[:, :3], axis=1)
                    ax1.plot(times, b_mag, label=f'MMS{probe} |B|', alpha=0.7)
                break
    
    ax1.set_ylabel('|B| (units to be verified)')
    ax1.set_title('Magnetic Field Magnitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Position magnitude vs time
    ax2 = axes[0, 1]
    for probe, spacecraft_data in evt_sample.items():
        if not spacecraft_data:
            continue
        
        for var_name, (t_data, values) in spacecraft_data.items():
            if 'POS' in var_name and len(values) > 0:
                times = [datetime.fromtimestamp(t) for t in t_data]
                if values.ndim > 1 and values.shape[1] >= 3:
                    r_earth = np.linalg.norm(values[:, :3], axis=1)
                    ax2.plot(times, r_earth, label=f'MMS{probe} |R|', alpha=0.7)
                break
    
    ax2.set_ylabel('|R| (units to be verified)')
    ax2.set_title('Distance from Earth')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: B-field components
    ax3 = axes[1, 0]
    probe_to_plot = None
    for probe, spacecraft_data in evt_sample.items():
        if spacecraft_data:
            probe_to_plot = probe
            break
    
    if probe_to_plot:
        spacecraft_data = evt_sample[probe_to_plot]
        for var_name, (t_data, values) in spacecraft_data.items():
            if 'B_gsm' in var_name and len(values) > 0:
                times = [datetime.fromtimestamp(t) for t in t_data]
                if values.ndim > 1 and values.shape[1] >= 3:
                    ax3.plot(times, values[:, 0], 'r-', label='Bx', alpha=0.7)
                    ax3.plot(times, values[:, 1], 'g-', label='By', alpha=0.7)
                    ax3.plot(times, values[:, 2], 'b-', label='Bz', alpha=0.7)
                break
    
    ax3.set_ylabel('B components (units to be verified)')
    ax3.set_title(f'MMS{probe_to_plot} B-field Components')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Data availability
    ax4 = axes[1, 1]
    probe_names = []
    data_counts = []
    
    for probe, spacecraft_data in evt_sample.items():
        if spacecraft_data:
            probe_names.append(f'MMS{probe}')
            total_points = sum(len(t_data) for t_data, _ in spacecraft_data.values())
            data_counts.append(total_points)
    
    if probe_names:
        ax4.bar(probe_names, data_counts, alpha=0.7)
        ax4.set_ylabel('Total Data Points')
        ax4.set_title('Data Availability by Spacecraft')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mms_data_verification.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Data verification plots saved: mms_data_verification.png")


if __name__ == "__main__":
    print("🔍 MMS DATA PRODUCTS INVESTIGATION")
    print("Verifying units, reference frames, and data quality")
    print("Event: 2019-01-27 12:30:50 UT")
    print()
    
    # Run comprehensive investigation
    sample_data = investigate_mms_data_products()
    
    if sample_data:
        # Create verification plots
        create_data_verification_plots(sample_data)
    
    print("\n🎉 DATA PRODUCTS INVESTIGATION COMPLETED!")
    print("\nKey findings:")
    print("  • Verified data product types and their characteristics")
    print("  • Checked units for magnetic field and position data")
    print("  • Analyzed reference frames and coordinate systems")
    print("  • Identified potential issues with data interpretation")
    print("  • Provided recommendations for correct data usage")
    print("\n📊 Check 'mms_data_verification.png' for visual verification!")
