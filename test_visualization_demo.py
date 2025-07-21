#!/usr/bin/env python3
"""
MMS-MP Visualization Demo

This script creates synthetic MMS-like data and demonstrates the complete
analysis workflow including visualization of a magnetopause crossing event.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime, timedelta

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mms_mp

def create_synthetic_magnetopause_crossing():
    """Create realistic synthetic data for a magnetopause crossing"""
    print("🔬 Creating synthetic magnetopause crossing data...")
    
    # Time array: 20 minutes of data at 150ms cadence
    n_points = 8000
    t_start = datetime(2019, 1, 27, 12, 20, 0)
    dt = timedelta(milliseconds=150)
    times = np.array([t_start + i * dt for i in range(n_points)])
    t_seconds = np.array([(t - t_start).total_seconds() for t in times])
    
    # Define crossing times (multiple crossings)
    crossing_1_start = 300  # 5 minutes in
    crossing_1_end = 450    # 7.5 minutes in
    crossing_2_start = 900  # 15 minutes in  
    crossing_2_end = 1050   # 17.5 minutes in
    
    print(f"   Data duration: {t_seconds[-1]/60:.1f} minutes")
    print(f"   Sampling rate: {1000/150:.1f} Hz")
    print(f"   Crossing 1: {crossing_1_start}-{crossing_1_end}s")
    print(f"   Crossing 2: {crossing_2_start}-{crossing_2_end}s")
    
    # Synthetic magnetic field with realistic magnetopause signature
    # Magnetosphere: Strong, stable field
    # Magnetosheath: Weaker, more variable field with rotation
    
    B_mag = np.zeros((n_points, 3))
    
    # Magnetosphere baseline
    B_mag[:, 0] = 15 + 2 * np.sin(0.01 * t_seconds)  # Bx: ~15 nT
    B_mag[:, 1] = 8 + np.cos(0.015 * t_seconds)      # By: ~8 nT  
    B_mag[:, 2] = -5 + 0.5 * np.sin(0.008 * t_seconds)  # Bz: ~-5 nT
    
    # Add magnetosheath signatures during crossings
    for start, end in [(crossing_1_start, crossing_1_end), (crossing_2_start, crossing_2_end)]:
        start_idx = int(start / 0.15)
        end_idx = int(end / 0.15)
        
        # Magnetosheath: rotated field, more turbulent
        B_mag[start_idx:end_idx, 0] = 8 + 4 * np.sin(0.05 * t_seconds[start_idx:end_idx])
        B_mag[start_idx:end_idx, 1] = -12 + 3 * np.cos(0.08 * t_seconds[start_idx:end_idx])
        B_mag[start_idx:end_idx, 2] = 6 + 2 * np.sin(0.06 * t_seconds[start_idx:end_idx])
        
        # Add turbulence
        B_mag[start_idx:end_idx, :] += np.random.normal(0, 1.5, (end_idx - start_idx, 3))
    
    # Add general noise
    B_mag += np.random.normal(0, 0.3, B_mag.shape)
    
    # Synthetic plasma densities
    # He+ density enhancement in magnetosheath
    he_density = np.ones(n_points) * 0.08  # Magnetosphere baseline
    
    for start, end in [(crossing_1_start, crossing_1_end), (crossing_2_start, crossing_2_end)]:
        start_idx = int(start / 0.15)
        end_idx = int(end / 0.15)
        
        # Magnetosheath enhancement
        he_density[start_idx:end_idx] = 0.4 + 0.1 * np.sin(0.1 * t_seconds[start_idx:end_idx])
    
    # Add noise
    he_density += np.random.normal(0, 0.02, n_points)
    he_density = np.maximum(he_density, 0.01)  # Keep positive
    
    # Total ion density (correlated with He+)
    ion_density = he_density * 8 + np.random.normal(0, 0.5, n_points)
    ion_density = np.maximum(ion_density, 0.1)
    
    # Electron density (quasi-neutrality)
    electron_density = ion_density + np.random.normal(0, 0.1, n_points)
    
    # Synthetic electric field (for E×B calculation)
    E_field = np.zeros((n_points, 3))
    E_field[:, 0] = 1.5 + 0.5 * np.sin(0.02 * t_seconds)  # Ex: ~1.5 mV/m
    E_field[:, 1] = 0.8 * np.cos(0.025 * t_seconds)       # Ey: varying
    E_field[:, 2] = 0.2 * np.sin(0.03 * t_seconds)        # Ez: small
    
    # Add noise
    E_field += np.random.normal(0, 0.1, E_field.shape)
    
    # Spacecraft position (for LMN calculation)
    pos_gsm = np.array([10500, 4800, -2200])  # km, typical magnetopause location
    
    return {
        'times': times,
        't_seconds': t_seconds,
        'B_gsm': B_mag,
        'E_gse': E_field,
        'he_density': he_density,
        'ion_density': ion_density,
        'electron_density': electron_density,
        'pos_gsm': pos_gsm,
        'crossing_times': [crossing_1_start, crossing_1_end, crossing_2_start, crossing_2_end]
    }

def analyze_synthetic_data(data):
    """Perform complete MMS-MP analysis on synthetic data"""
    print("\n🔍 Performing MMS-MP analysis...")
    
    # Extract data
    t_sec = data['t_seconds']
    B_gsm = data['B_gsm']
    E_gse = data['E_gse']
    he_density = data['he_density']
    pos_gsm = data['pos_gsm']
    
    # Step 1: Coordinate transformation using hybrid LMN
    print("   1. Computing LMN coordinate system...")
    
    # Use middle portion for MVA (avoid boundary effects)
    mid_start = len(B_gsm) // 3
    mid_end = 2 * len(B_gsm) // 3
    B_slice = B_gsm[mid_start:mid_end, :]
    
    lmn = mms_mp.hybrid_lmn(B_slice, pos_gsm_km=pos_gsm)
    print(f"      Eigenvalue ratios: {lmn.r_max_mid:.2f}, {lmn.r_mid_min:.2f}")

    # Determine method used based on eigenvalue ratios
    if lmn.r_max_mid >= 5.0 and lmn.r_mid_min >= 5.0:
        method_used = "MVA"
    else:
        method_used = "Shue model"
    print(f"      Method used: {method_used}")
    
    # Transform magnetic field to LMN coordinates
    B_lmn = lmn.to_lmn(B_gsm)
    
    # Step 2: E×B drift calculation
    print("   2. Computing E×B drift velocity...")
    v_exb = mms_mp.exb_velocity(E_gse, B_gsm, unit_E='mV/m', unit_B='nT')
    v_exb_lmn = lmn.to_lmn(v_exb)
    
    # Step 3: Boundary detection
    print("   3. Detecting boundary crossings...")
    cfg = mms_mp.DetectorCfg(he_in=0.25, he_out=0.15, BN_tol=2.0)
    
    # Create good data mask (assume all data is good for this demo)
    good_mask = np.ones(len(t_sec), dtype=bool)
    
    layers = mms_mp.detect_crossings_multi(
        t_sec, he_density, B_lmn[:, 2], cfg=cfg, good_mask=good_mask
    )
    
    # Extract crossing times
    from mms_mp.boundary import extract_enter_exit
    crossings = extract_enter_exit(layers, t_sec)
    
    print(f"      Detected {len(crossings)} boundary crossings:")
    for i, (t_cross, cross_type) in enumerate(crossings):
        print(f"        {i+1}. {cross_type} at t={t_cross:.1f}s")
    
    # Step 4: Displacement integration
    print("   4. Integrating displacement...")
    vN = v_exb_lmn[:, 2]  # Normal component of E×B velocity
    
    result = mms_mp.integrate_disp(t_sec, vN, scheme='trap')
    
    # Step 5: Layer thickness calculation
    if crossings:
        print("   5. Calculating layer thicknesses...")
        thicknesses = mms_mp.thickness.layer_thicknesses(
            result.t_sec, result.disp_km, crossings
        )
        
        for layer_name, thickness in thicknesses:
            print(f"      {layer_name}: {thickness:.2f} km")
    
    return {
        'lmn': lmn,
        'B_lmn': B_lmn,
        'v_exb': v_exb,
        'v_exb_lmn': v_exb_lmn,
        'layers': layers,
        'crossings': crossings,
        'displacement': result,
        'cfg': cfg
    }

def create_visualization(data, analysis):
    """Create comprehensive visualization of the analysis"""
    print("\n📊 Creating visualization...")
    
    # Extract data for plotting
    t_sec = data['t_seconds']
    t_min = t_sec / 60  # Convert to minutes for plotting
    B_gsm = data['B_gsm']
    B_lmn = analysis['B_lmn']
    he_density = data['he_density']
    ion_density = data['ion_density']
    v_exb_lmn = analysis['v_exb_lmn']
    displacement = analysis['displacement']
    crossings = analysis['crossings']
    
    # Create the plot
    fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('MMS Magnetopause Analysis - Synthetic Data Demo', fontsize=16, fontweight='bold')
    
    # Panel 1: Magnetic field in GSM coordinates
    axes[0].plot(t_min, B_gsm[:, 0], 'b-', label='Bₓ', linewidth=1)
    axes[0].plot(t_min, B_gsm[:, 1], 'g-', label='Bᵧ', linewidth=1)
    axes[0].plot(t_min, B_gsm[:, 2], 'r-', label='Bᵤ', linewidth=1)
    axes[0].set_ylabel('B [nT]\n(GSM)', fontsize=10)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Magnetic Field (GSM Coordinates)', fontsize=11)
    
    # Panel 2: Magnetic field in LMN coordinates
    axes[1].plot(t_min, B_lmn[:, 0], 'b-', label='B_L', linewidth=1)
    axes[1].plot(t_min, B_lmn[:, 1], 'g-', label='B_M', linewidth=1)
    axes[1].plot(t_min, B_lmn[:, 2], 'r-', label='B_N', linewidth=1.5)
    axes[1].set_ylabel('B [nT]\n(LMN)', fontsize=10)
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    # Determine method used
    lmn_obj = analysis["lmn"]
    if lmn_obj.r_max_mid >= 5.0 and lmn_obj.r_mid_min >= 5.0:
        method_used = "MVA"
    else:
        method_used = "Shue model"
    axes[1].set_title(f'Magnetic Field (LMN Coordinates) - Method: {method_used}', fontsize=11)
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
    
    # Panel 3: Plasma densities
    axes[2].semilogy(t_min, ion_density, 'b-', label='N_i', linewidth=1)
    axes[2].semilogy(t_min, he_density, 'orange', label='He⁺', linewidth=1.5)
    axes[2].axhline(analysis['cfg'].he_in, color='red', linestyle='--', alpha=0.7, 
                   label=f'He⁺ in ({analysis["cfg"].he_in})')
    axes[2].axhline(analysis['cfg'].he_out, color='blue', linestyle='--', alpha=0.7,
                   label=f'He⁺ out ({analysis["cfg"].he_out})')
    axes[2].set_ylabel('Density\n[cm⁻³]', fontsize=10)
    axes[2].legend(loc='upper right', fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Plasma Densities with Detection Thresholds', fontsize=11)
    
    # Panel 4: E×B velocity components
    axes[3].plot(t_min, v_exb_lmn[:, 0], 'b-', label='v_L', linewidth=1)
    axes[3].plot(t_min, v_exb_lmn[:, 1], 'g-', label='v_M', linewidth=1)
    axes[3].plot(t_min, v_exb_lmn[:, 2], 'r-', label='v_N', linewidth=1.5)
    axes[3].set_ylabel('E×B Velocity\n[km/s]', fontsize=10)
    axes[3].legend(loc='upper right', fontsize=9)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title('E×B Drift Velocity (LMN Coordinates)', fontsize=11)
    axes[3].axhline(0, color='black', linestyle='--', alpha=0.5)
    
    # Panel 5: Normal magnetic field with boundary layers
    axes[4].plot(t_min, B_lmn[:, 2], 'purple', linewidth=1.5, label='B_N')
    axes[4].axhline(0, color='black', linestyle='-', alpha=0.5)
    axes[4].axhline(analysis['cfg'].BN_tol, color='red', linestyle=':', alpha=0.7, label='B_N tolerance')
    axes[4].axhline(-analysis['cfg'].BN_tol, color='red', linestyle=':', alpha=0.7)
    
    # Note: Skipping layer highlighting for simplicity in demo
    
    axes[4].set_ylabel('B_N [nT]', fontsize=10)
    axes[4].legend(loc='upper right', fontsize=9)
    axes[4].grid(True, alpha=0.3)
    axes[4].set_title('Normal Magnetic Field with Boundary Detection', fontsize=11)
    
    # Panel 6: Displacement
    axes[5].plot(t_min, displacement.disp_km, 'darkgreen', linewidth=2, label='Displacement')
    axes[5].set_ylabel('Displacement\n[km]', fontsize=10)
    axes[5].set_xlabel('Time [minutes from start]', fontsize=10)
    axes[5].legend(loc='upper right', fontsize=9)
    axes[5].grid(True, alpha=0.3)
    axes[5].set_title('Cumulative Displacement from E×B Integration', fontsize=11)
    
    # Mark all crossing times on all panels
    if crossings:
        for t_cross, cross_type in crossings:
            t_cross_min = t_cross / 60
            color = 'red' if 'enter' in cross_type.lower() else 'blue'
            linestyle = '-' if 'enter' in cross_type.lower() else '--'
            
            for ax in axes:
                ax.axvline(t_cross_min, color=color, linestyle=linestyle, alpha=0.8, linewidth=2)
    
    # Add legend for crossing lines
    if crossings:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linestyle='-', label='Magnetosphere Entry'),
            Line2D([0], [0], color='blue', linestyle='--', label='Magnetosheath Entry')
        ]
        axes[0].legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the plot
    filename = 'mms_mp_visualization_demo.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   ✅ Visualization saved as: {filename}")
    
    return fig, filename

def main():
    """Main demo function"""
    print("🚀 MMS Magnetopause Analysis Toolkit - Visualization Demo")
    print("=" * 60)
    
    try:
        # Create synthetic data
        data = create_synthetic_magnetopause_crossing()
        
        # Perform analysis
        analysis = analyze_synthetic_data(data)
        
        # Create visualization
        fig, filename = create_visualization(data, analysis)
        
        # Summary
        print(f"\n📋 Analysis Summary:")
        print(f"   • Data duration: {data['t_seconds'][-1]/60:.1f} minutes")
        # Determine method used
        lmn_obj = analysis['lmn']
        if lmn_obj.r_max_mid >= 5.0 and lmn_obj.r_mid_min >= 5.0:
            method_used = "MVA"
        else:
            method_used = "Shue model"
        print(f"   • LMN method: {method_used}")
        print(f"   • Eigenvalue ratios: {analysis['lmn'].r_max_mid:.2f}, {analysis['lmn'].r_mid_min:.2f}")
        print(f"   • Boundary crossings detected: {len(analysis['crossings'])}")
        print(f"   • Displacement range: {analysis['displacement'].disp_km.min():.1f} to {analysis['displacement'].disp_km.max():.1f} km")
        
        if analysis['crossings']:
            thicknesses = mms_mp.thickness.layer_thicknesses(
                analysis['displacement'].t_sec, 
                analysis['displacement'].disp_km, 
                analysis['crossings']
            )
            print(f"   • Layer thicknesses:")
            for layer_name, thickness in thicknesses:
                print(f"     - {layer_name}: {thickness:.2f} km")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📊 Visualization saved as: {filename}")
        print(f"📖 The plot shows a complete magnetopause analysis workflow:")
        print(f"   1. Magnetic field in GSM and LMN coordinates")
        print(f"   2. Plasma densities with detection thresholds")
        print(f"   3. E×B drift velocities")
        print(f"   4. Boundary detection results")
        print(f"   5. Displacement integration")
        
        # Show the plot
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
