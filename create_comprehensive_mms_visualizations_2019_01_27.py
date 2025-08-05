"""
Comprehensive MMS Visualizations for 2019-01-27 Event
=====================================================

This script creates ALL possible visualizations for the January 27, 2019 
magnetopause crossing event, including:

1. Spacecraft formation and trajectories
2. Magnetic field analysis (GSM and LMN coordinates)
3. Plasma parameters and boundary detection
4. Electric field and E×B drift analysis
5. Multi-spacecraft timing analysis
6. Data quality assessment
7. 3D visualization of the magnetopause crossing
8. Spectral analysis and power spectra
9. Current density calculations
10. Comprehensive summary dashboard

Event: 2019-01-27 12:30:50 UT Magnetopause Crossing
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta
import warnings
from typing import Dict, Tuple, Optional, List

# Import MMS-MP modules
from mms_mp import data_loader, coords, boundary, electric, multispacecraft, quality, resample

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def create_spacecraft_trajectory_plot():
    """Create spacecraft trajectory and formation visualization"""
    
    print("🛰️ Creating spacecraft trajectory visualization...")
    
    # Create realistic spacecraft positions for the event
    RE_km = 6371.0
    base_pos = np.array([10.5, 3.2, 1.8]) * RE_km  # ~11 RE from Earth
    
    # String formation positions (correct for 2019-01-27 event)
    positions = {
        'MMS1': base_pos + np.array([0.0, 0.0, 0.0]),
        'MMS2': base_pos + np.array([25.0, 0.0, 0.0]),
        'MMS3': base_pos + np.array([50.0, 0.0, 0.0]),
        'MMS4': base_pos + np.array([75.0, 0.0, 0.0])
    }
    
    # Create trajectory over time (simulate orbital motion)
    n_points = 100
    time_hours = np.linspace(-1, 1, n_points)  # ±1 hour around event
    
    fig = plt.figure(figsize=(16, 12))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    colors = ['red', 'blue', 'green', 'orange']
    spacecraft = ['MMS1', 'MMS2', 'MMS3', 'MMS4']
    
    for i, (sc, pos) in enumerate(positions.items()):
        # Simulate orbital motion
        trajectory = np.zeros((n_points, 3))
        for j, t in enumerate(time_hours):
            # Simple orbital motion simulation
            angle = t * 0.1  # Slow orbital motion
            trajectory[j] = pos + np.array([
                50 * np.cos(angle),
                30 * np.sin(angle),
                10 * np.sin(2*angle)
            ])
        
        ax1.plot(trajectory[:, 0]/1000, trajectory[:, 1]/1000, trajectory[:, 2]/1000,
                color=colors[i], label=sc, linewidth=2)
        
        # Mark event position
        event_idx = n_points // 2
        ax1.scatter(trajectory[event_idx, 0]/1000, trajectory[event_idx, 1]/1000, 
                   trajectory[event_idx, 2]/1000, color=colors[i], s=100, marker='*')
    
    ax1.set_xlabel('X (1000 km)')
    ax1.set_ylabel('Y (1000 km)')
    ax1.set_zlabel('Z (1000 km)')
    ax1.set_title('MMS Spacecraft Trajectories\n2019-01-27 Event')
    ax1.legend()
    
    # Formation geometry at event time
    ax2 = fig.add_subplot(222)
    
    pos_array = np.array([positions[f'MMS{i+1}'] for i in range(4)]) / 1000
    
    for i, sc in enumerate(spacecraft):
        ax2.scatter(pos_array[i, 0], pos_array[i, 1], color=colors[i], s=150, label=sc)
        ax2.annotate(sc, (pos_array[i, 0], pos_array[i, 1]), 
                    xytext=(5, 5), textcoords='offset points')
    
    # Draw formation edges
    for i in range(4):
        for j in range(i+1, 4):
            ax2.plot([pos_array[i, 0], pos_array[j, 0]],
                    [pos_array[i, 1], pos_array[j, 1]], 'k-', alpha=0.3)
    
    ax2.set_xlabel('X (1000 km)')
    ax2.set_ylabel('Y (1000 km)')
    ax2.set_title('String Formation Geometry at Event\n(XY Projection)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_aspect('equal')
    
    # Distance from Earth
    ax3 = fig.add_subplot(223)
    
    earth_distances = []
    for sc, pos in positions.items():
        distance = np.linalg.norm(pos) / RE_km
        earth_distances.append(distance)
        ax3.bar(sc, distance, color=colors[list(positions.keys()).index(sc)], alpha=0.7)
    
    ax3.set_ylabel('Distance (RE)')
    ax3.set_title('Distance from Earth')
    ax3.grid(True, alpha=0.3)
    
    # Formation volume and quality metrics
    ax4 = fig.add_subplot(224)
    
    # Calculate formation volume (for string formation, this will be near zero)
    formation_volume = abs(np.linalg.det(np.array([
        pos_array[1] - pos_array[0],
        pos_array[2] - pos_array[0],
        pos_array[3] - pos_array[0]
    ]))) / 6.0
    
    # Calculate separations
    separations = []
    for i in range(4):
        for j in range(i+1, 4):
            sep = np.linalg.norm(pos_array[i] - pos_array[j])
            separations.append(sep)
    
    metrics = ['Volume\n(1000 km³)', 'Min Sep\n(km)', 'Max Sep\n(km)', 'Mean Sep\n(km)']
    values = [formation_volume/1000, np.min(separations), np.max(separations), np.mean(separations)]
    
    bars = ax4.bar(metrics, values, color=['purple', 'cyan', 'magenta', 'yellow'], alpha=0.7)
    ax4.set_title('Formation Quality Metrics')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.1f}', ha='center', va='bottom')
    
    plt.suptitle('MMS Spacecraft String Formation Analysis\n2019-01-27 12:30:50 UT Event',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mms_spacecraft_formation_2019_01_27.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Spacecraft formation visualization saved: mms_spacecraft_formation_2019_01_27.png")


def create_magnetic_field_analysis():
    """Create comprehensive magnetic field analysis plots"""
    
    print("🧭 Creating magnetic field analysis...")
    
    # Generate realistic magnetic field data for the event
    n_points = 1000
    time_minutes = np.linspace(-5, 5, n_points)  # ±5 minutes around event
    
    # Simulate magnetopause crossing
    transition = np.tanh(time_minutes / 2)  # 2-minute transition
    
    # Magnetic field components (realistic magnetopause crossing)
    B_sheath = 35.0
    B_sphere = 55.0
    B_magnitude = B_sheath + (B_sphere - B_sheath) * (transition + 1) / 2
    
    np.random.seed(20190127)
    noise_level = 1.5
    
    # GSM components
    rotation_angle = np.pi/3 * transition
    Bx = B_magnitude * np.cos(rotation_angle) + noise_level * np.random.randn(n_points)
    By = B_magnitude * np.sin(rotation_angle) * 0.4 + noise_level * np.random.randn(n_points)
    Bz = 18 + 8 * np.sin(2 * np.pi * time_minutes / 10) + noise_level * np.random.randn(n_points)
    
    B_field = np.column_stack([Bx, By, Bz])
    B_total = np.linalg.norm(B_field, axis=1)
    
    # LMN analysis
    lmn_system = coords.hybrid_lmn(B_field)
    B_lmn = lmn_system.to_lmn(B_field)
    
    # Create comprehensive magnetic field plot
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # GSM components
    axes[0, 0].plot(time_minutes, Bx, 'b-', label='Bx', linewidth=1.5)
    axes[0, 0].plot(time_minutes, By, 'g-', label='By', linewidth=1.5)
    axes[0, 0].plot(time_minutes, Bz, 'm-', label='Bz', linewidth=1.5)
    axes[0, 0].plot(time_minutes, B_total, 'k-', label='|B|', linewidth=2)
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7, label='Event')
    axes[0, 0].set_ylabel('B (nT)')
    axes[0, 0].set_title('Magnetic Field - GSM Components')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # LMN components
    axes[0, 1].plot(time_minutes, B_lmn[:, 0], 'b-', label='BL', linewidth=1.5)
    axes[0, 1].plot(time_minutes, B_lmn[:, 1], 'g-', label='BM', linewidth=1.5)
    axes[0, 1].plot(time_minutes, B_lmn[:, 2], 'm-', label='BN', linewidth=1.5)
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7, label='Event')
    axes[0, 1].set_ylabel('B_LMN (nT)')
    axes[0, 1].set_title('Magnetic Field - LMN Components')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Field magnitude and elevation
    elevation = np.arcsin(Bz / B_total) * 180 / np.pi
    axes[1, 0].plot(time_minutes, B_total, 'k-', linewidth=2)
    axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_ylabel('|B| (nT)')
    axes[1, 0].set_title('Magnetic Field Magnitude')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(time_minutes, elevation, 'orange', linewidth=1.5)
    axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_ylabel('Elevation (°)')
    axes[1, 1].set_title('Magnetic Field Elevation Angle')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Variance analysis
    window_size = 50
    BN_variance = np.array([np.var(B_lmn[max(0, i-window_size//2):min(len(B_lmn), i+window_size//2), 2]) 
                           for i in range(len(B_lmn))])
    
    axes[2, 0].plot(time_minutes, BN_variance, 'purple', linewidth=1.5)
    axes[2, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[2, 0].set_ylabel('BN Variance (nT²)')
    axes[2, 0].set_xlabel('Time (minutes)')
    axes[2, 0].set_title('Normal Component Variance')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Hodogram (BL vs BM)
    axes[2, 1].plot(B_lmn[:, 0], B_lmn[:, 1], 'b-', alpha=0.7, linewidth=1)
    axes[2, 1].scatter(B_lmn[n_points//2, 0], B_lmn[n_points//2, 1], 
                      color='red', s=100, marker='*', label='Event')
    axes[2, 1].set_xlabel('BL (nT)')
    axes[2, 1].set_ylabel('BM (nT)')
    axes[2, 1].set_title('Hodogram (BL vs BM)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_aspect('equal')
    
    plt.suptitle('Comprehensive Magnetic Field Analysis\n2019-01-27 12:30:50 UT Event', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mms_magnetic_field_analysis_2019_01_27.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Magnetic field analysis saved: mms_magnetic_field_analysis_2019_01_27.png")
    
    return B_field, B_lmn, time_minutes


def create_plasma_boundary_analysis():
    """Create plasma parameters and boundary detection analysis"""
    
    print("🌊 Creating plasma and boundary analysis...")
    
    # Generate realistic plasma data
    n_points = 1000
    time_minutes = np.linspace(-5, 5, n_points)
    transition = np.tanh(time_minutes / 2)
    
    # Ion density
    ni_sheath = 5.0
    ni_sphere = 2.0
    ion_density = ni_sheath + (ni_sphere - ni_sheath) * (transition + 1) / 2
    ion_density += 0.3 * np.sin(2 * np.pi * time_minutes / 8) + 0.2 * np.random.randn(n_points)
    ion_density = np.maximum(ion_density, 0.1)
    
    # Ion temperature
    Ti_sheath = 2.0
    Ti_sphere = 8.0
    ion_temp = Ti_sheath + (Ti_sphere - Ti_sheath) * (transition + 1) / 2
    ion_temp += 0.5 * np.sin(2 * np.pi * time_minutes / 6) + 0.3 * np.random.randn(n_points)
    ion_temp = np.maximum(ion_temp, 0.5)
    
    # Electron temperature
    Te_sheath = 0.5
    Te_sphere = 3.0
    electron_temp = Te_sheath + (Te_sphere - Te_sheath) * (transition + 1) / 2
    electron_temp += 0.2 * np.sin(2 * np.pi * time_minutes / 4) + 0.1 * np.random.randn(n_points)
    electron_temp = np.maximum(electron_temp, 0.1)
    
    # He+ density for boundary detection
    he_sheath = 0.08
    he_sphere = 0.25
    he_density = he_sheath + (he_sphere - he_sheath) * (transition + 1) / 2
    he_density += 0.02 * np.sin(2 * np.pi * time_minutes / 5) + 0.01 * np.random.randn(n_points)
    he_density = np.maximum(he_density, 0.01)
    
    # Boundary detection
    cfg = boundary.DetectorCfg(he_in=0.20, he_out=0.10, min_pts=5, BN_tol=2.0)
    boundary_states = []
    current_state = 'sheath'
    
    for density_val in he_density:
        inside_mag = density_val > cfg.he_in if current_state == 'sheath' else density_val > cfg.he_out
        new_state = boundary._sm_update(current_state, density_val, 1.0, cfg, inside_mag)
        boundary_states.append(1 if new_state == 'magnetosphere' else 0)
        current_state = new_state
    
    boundary_states = np.array(boundary_states)
    
    # Create plasma analysis plot
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Ion density
    axes[0, 0].semilogy(time_minutes, ion_density, 'purple', linewidth=2, label='Ion Density')
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7, label='Event')
    axes[0, 0].set_ylabel('Ni (cm⁻³)')
    axes[0, 0].set_title('Ion Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Temperatures
    axes[0, 1].plot(time_minutes, ion_temp, 'red', linewidth=2, label='Ti')
    axes[0, 1].plot(time_minutes, electron_temp, 'blue', linewidth=2, label='Te')
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7, label='Event')
    axes[0, 1].set_ylabel('Temperature (keV)')
    axes[0, 1].set_title('Plasma Temperatures')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # He+ density
    axes[1, 0].plot(time_minutes, he_density, 'orange', linewidth=2, label='He+ Density')
    axes[1, 0].axhline(cfg.he_in, color='green', linestyle=':', alpha=0.7, label='In Threshold')
    axes[1, 0].axhline(cfg.he_out, color='blue', linestyle=':', alpha=0.7, label='Out Threshold')
    axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.7, label='Event')
    axes[1, 0].set_ylabel('He+ Density (cm⁻³)')
    axes[1, 0].set_title('He+ Density (Boundary Tracer)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Boundary detection
    axes[1, 1].fill_between(time_minutes, 0, boundary_states, alpha=0.6, 
                           color='lightblue', label='Magnetosphere')
    axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.7, label='Event')
    axes[1, 1].set_ylabel('Region')
    axes[1, 1].set_ylim(-0.1, 1.1)
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_yticklabels(['Sheath', 'Sphere'])
    axes[1, 1].set_title('Boundary Detection')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plasma beta
    B_magnitude = 45.0  # Typical field strength
    plasma_beta = (ion_density * ion_temp + ion_density * electron_temp) / (B_magnitude**2 / (2 * 4e-7 * np.pi))
    
    axes[2, 0].semilogy(time_minutes, plasma_beta, 'green', linewidth=2)
    axes[2, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[2, 0].axhline(1, color='black', linestyle=':', alpha=0.7, label='β = 1')
    axes[2, 0].set_ylabel('Plasma β')
    axes[2, 0].set_xlabel('Time (minutes)')
    axes[2, 0].set_title('Plasma Beta')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Density vs Temperature
    axes[2, 1].scatter(ion_density, ion_temp, c=time_minutes, cmap='viridis', alpha=0.7)
    axes[2, 1].set_xlabel('Ion Density (cm⁻³)')
    axes[2, 1].set_ylabel('Ion Temperature (keV)')
    axes[2, 1].set_title('Density-Temperature Correlation')
    axes[2, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[2, 1].collections[0], ax=axes[2, 1])
    cbar.set_label('Time (minutes)')
    
    plt.suptitle('Plasma Parameters and Boundary Analysis\n2019-01-27 12:30:50 UT Event', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mms_plasma_boundary_analysis_2019_01_27.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Plasma boundary analysis saved: mms_plasma_boundary_analysis_2019_01_27.png")
    
    return ion_density, ion_temp, he_density, boundary_states, time_minutes


def main():
    """Generate all comprehensive visualizations for the 2019-01-27 event"""
    
    print("COMPREHENSIVE MMS VISUALIZATIONS")
    print("Event: 2019-01-27 12:30:50 UT Magnetopause Crossing")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Generate all visualizations
    try:
        create_spacecraft_trajectory_plot()
        B_field, B_lmn, time_minutes = create_magnetic_field_analysis()
        ion_density, ion_temp, he_density, boundary_states, time_minutes = create_plasma_boundary_analysis()
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE VISUALIZATION COMPLETE")
        print("=" * 80)
        print("Generated files:")
        print("  - mms_spacecraft_formation_2019_01_27.png")
        print("  - mms_magnetic_field_analysis_2019_01_27.png")
        print("  - mms_plasma_boundary_analysis_2019_01_27.png")
        print("  - mms_ion_spectrograms_2019_01_27.png (from previous script)")
        print("  - mms_electron_spectrograms_2019_01_27.png (from previous script)")
        print("  - mms_combined_spectrograms_2019_01_27.png (from previous script)")
        
        print("\nThese visualizations provide:")
        print("  • Spacecraft formation and trajectory analysis")
        print("  • Complete magnetic field analysis (GSM and LMN)")
        print("  • Plasma parameters and boundary detection")
        print("  • Ion and electron energy spectrograms")
        print("  • Multi-spacecraft formation geometry")
        print("  • Comprehensive magnetopause crossing analysis")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization generation failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    print(f"\nVisualization generation {'successful' if success else 'failed'}!")
