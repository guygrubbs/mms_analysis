#!/usr/bin/env python3
"""
Simple MMS-MP Visualization Demo

This script demonstrates the core functionality of the MMS-MP toolkit
with a focus on visualization of key results.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mms_mp

def create_simple_demo():
    """Create and analyze synthetic magnetopause data with visualization"""
    print("🚀 MMS-MP Simple Visualization Demo")
    print("=" * 40)
    
    # Create synthetic data
    print("📊 Creating synthetic data...")
    n_points = 2000
    t = np.linspace(0, 600, n_points)  # 10 minutes
    
    # Synthetic magnetic field with boundary crossing
    B_mag = np.zeros((n_points, 3))
    
    # Magnetosphere field
    B_mag[:, 0] = 15 + 2 * np.sin(0.01 * t)  # Bx
    B_mag[:, 1] = 8 + np.cos(0.015 * t)      # By
    B_mag[:, 2] = -5 + 0.5 * np.sin(0.008 * t)  # Bz
    
    # Add magnetosheath crossing (middle section)
    crossing_start = n_points // 3
    crossing_end = 2 * n_points // 3
    
    B_mag[crossing_start:crossing_end, 0] = 8 + 3 * np.sin(0.05 * t[crossing_start:crossing_end])
    B_mag[crossing_start:crossing_end, 1] = -10 + 2 * np.cos(0.08 * t[crossing_start:crossing_end])
    B_mag[crossing_start:crossing_end, 2] = 6 + np.sin(0.06 * t[crossing_start:crossing_end])
    
    # Add noise
    B_mag += np.random.normal(0, 0.5, B_mag.shape)
    
    # Synthetic He+ density
    he_density = np.ones(n_points) * 0.1
    he_density[crossing_start:crossing_end] = 0.4
    he_density += np.random.normal(0, 0.02, n_points)
    he_density = np.maximum(he_density, 0.01)
    
    # Synthetic electric field
    E_field = np.ones((n_points, 3)) * [1.5, 0.8, 0.2]  # mV/m
    E_field += np.random.normal(0, 0.1, E_field.shape)
    
    print(f"   ✓ Created {n_points} data points over {t[-1]/60:.1f} minutes")
    
    # Analysis
    print("🔍 Performing analysis...")
    
    # 1. LMN coordinate transformation
    pos_gsm = np.array([10000, 5000, -2000])  # km
    B_slice = B_mag[800:1200, :]  # Use middle section for MVA
    
    lmn = mms_mp.hybrid_lmn(B_slice, pos_gsm_km=pos_gsm)
    B_lmn = lmn.to_lmn(B_mag)
    
    print(f"   ✓ LMN transformation complete")
    print(f"     Eigenvalue ratios: {lmn.r_max_mid:.2f}, {lmn.r_mid_min:.2f}")
    
    # 2. E×B drift calculation
    v_exb = mms_mp.exb_velocity(E_field, B_mag, unit_E='mV/m', unit_B='nT')
    v_exb_lmn = lmn.to_lmn(v_exb)
    
    print(f"   ✓ E×B drift calculated")
    print(f"     Average drift speed: {np.mean(np.linalg.norm(v_exb, axis=1)):.1f} km/s")
    
    # 3. Displacement integration
    vN = v_exb_lmn[:, 2]  # Normal component
    result = mms_mp.integrate_disp(t, vN, scheme='trap')
    
    print(f"   ✓ Displacement integrated")
    print(f"     Total displacement: {result.disp_km[-1] - result.disp_km[0]:.1f} km")
    
    # 4. Simple boundary detection
    cfg = mms_mp.DetectorCfg(he_in=0.25, he_out=0.15, BN_tol=2.0)
    layers = mms_mp.detect_crossings_multi(t, he_density, B_lmn[:, 2], cfg=cfg)
    
    from mms_mp.boundary import extract_enter_exit
    crossings = extract_enter_exit(layers, t)
    
    print(f"   ✓ Boundary detection complete")
    print(f"     Detected {len(crossings)} crossings")
    
    # Create visualization
    print("📈 Creating visualization...")
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('MMS Magnetopause Analysis - Simple Demo', fontsize=14, fontweight='bold')
    
    t_min = t / 60  # Convert to minutes
    
    # Panel 1: Magnetic field (GSM)
    axes[0].plot(t_min, B_mag[:, 0], 'b-', label='Bₓ', linewidth=1)
    axes[0].plot(t_min, B_mag[:, 1], 'g-', label='Bᵧ', linewidth=1)
    axes[0].plot(t_min, B_mag[:, 2], 'r-', label='Bᵤ', linewidth=1)
    axes[0].set_ylabel('B [nT]\n(GSM)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Magnetic Field (GSM Coordinates)')
    
    # Panel 2: Magnetic field (LMN)
    axes[1].plot(t_min, B_lmn[:, 0], 'b-', label='B_L', linewidth=1)
    axes[1].plot(t_min, B_lmn[:, 1], 'g-', label='B_M', linewidth=1)
    axes[1].plot(t_min, B_lmn[:, 2], 'r-', label='B_N', linewidth=1.5)
    axes[1].set_ylabel('B [nT]\n(LMN)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Magnetic Field (LMN Coordinates)')
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
    
    # Panel 3: He+ density
    axes[2].plot(t_min, he_density, 'orange', linewidth=1.5, label='He⁺')
    axes[2].axhline(cfg.he_in, color='red', linestyle='--', alpha=0.7, label=f'In threshold ({cfg.he_in})')
    axes[2].axhline(cfg.he_out, color='blue', linestyle='--', alpha=0.7, label=f'Out threshold ({cfg.he_out})')
    axes[2].set_ylabel('He⁺ density\n[cm⁻³]')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('He⁺ Density with Detection Thresholds')
    
    # Panel 4: E×B velocity
    axes[3].plot(t_min, v_exb_lmn[:, 0], 'b-', label='v_L', linewidth=1)
    axes[3].plot(t_min, v_exb_lmn[:, 1], 'g-', label='v_M', linewidth=1)
    axes[3].plot(t_min, v_exb_lmn[:, 2], 'r-', label='v_N', linewidth=1.5)
    axes[3].set_ylabel('E×B Velocity\n[km/s]')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title('E×B Drift Velocity (LMN Coordinates)')
    axes[3].axhline(0, color='black', linestyle='--', alpha=0.5)
    
    # Panel 5: Displacement
    axes[4].plot(t_min, result.disp_km, 'darkgreen', linewidth=2, label='Displacement')
    axes[4].set_ylabel('Displacement\n[km]')
    axes[4].set_xlabel('Time [minutes]')
    axes[4].legend(loc='upper right')
    axes[4].grid(True, alpha=0.3)
    axes[4].set_title('Cumulative Displacement')
    
    # Mark crossings
    if crossings:
        for t_cross, cross_type in crossings[:4]:  # Show first 4 crossings only
            t_cross_min = t_cross / 60
            color = 'red' if 'enter' in cross_type.lower() else 'blue'
            for ax in axes:
                ax.axvline(t_cross_min, color=color, linestyle=':', alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save plot
    filename = 'mms_mp_simple_demo.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   ✓ Plot saved as: {filename}")
    
    # Summary
    print(f"\n📋 Analysis Summary:")
    print(f"   • Data duration: {t[-1]/60:.1f} minutes")
    print(f"   • Eigenvalue ratios: {lmn.r_max_mid:.2f}, {lmn.r_mid_min:.2f}")
    print(f"   • Average E×B speed: {np.mean(np.linalg.norm(v_exb, axis=1)):.1f} km/s")
    print(f"   • Total displacement: {result.disp_km[-1] - result.disp_km[0]:.1f} km")
    print(f"   • Boundary crossings: {len(crossings)}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"📊 The plot shows:")
    print(f"   1. Magnetic field in GSM and LMN coordinates")
    print(f"   2. He⁺ density with detection thresholds")
    print(f"   3. E×B drift velocities")
    print(f"   4. Integrated displacement")
    print(f"   5. Detected boundary crossings (vertical lines)")
    
    # Show plot
    plt.show()
    
    return True

if __name__ == "__main__":
    try:
        success = create_simple_demo()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
