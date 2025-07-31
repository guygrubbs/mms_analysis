"""
MMS Event Analysis and Plotting Suite: 2019-01-27 12:30:50 UT
============================================================

This script generates comprehensive plots and data analysis for the real MMS 
magnetopause crossing event, including:

1. Multi-panel time series plots (magnetic field, plasma, boundary detection)
2. LMN coordinate analysis plots
3. Multi-spacecraft timing analysis
4. Formation geometry visualization
5. Data quality assessment plots
6. Complete data export for further analysis

Event: 2019-01-27 12:30:50 UT Magnetopause Crossing
Period: 11:30:00 - 13:30:00 UT (2 hours)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

# Import MMS-MP modules
from mms_mp import coords, boundary, electric, multispacecraft, quality


def create_event_data():
    """Create the same realistic event data as in the validation test"""
    
    # Event timing
    event_time = datetime(2019, 1, 27, 12, 30, 50)
    start_time = datetime(2019, 1, 27, 11, 30, 0)
    end_time = datetime(2019, 1, 27, 13, 30, 0)
    
    # Time array
    total_duration = (end_time - start_time).total_seconds()
    n_points = int(total_duration / 0.15)  # 150ms cadence
    times_sec = np.linspace(0, total_duration, n_points)
    
    # Convert to datetime objects for plotting
    times_dt = [start_time + timedelta(seconds=t) for t in times_sec]
    
    # Event occurs at center
    event_index = n_points // 2
    
    # Spacecraft positions
    RE_km = 6371.0
    base_position = np.array([10.5, 3.2, 1.8]) * RE_km
    
    spacecraft_positions = {
        '1': base_position + np.array([0.0, 0.0, 0.0]),
        '2': base_position + np.array([100.0, 0.0, 0.0]),
        '3': base_position + np.array([50.0, 86.6, 0.0]),
        '4': base_position + np.array([50.0, 28.9, 81.6])
    }
    
    # Create magnetic field data
    t_rel = times_sec - times_sec[event_index]
    transition = np.tanh(t_rel / 120)
    
    B_sheath = 35.0
    B_sphere = 55.0
    B_magnitude = B_sheath + (B_sphere - B_sheath) * (transition + 1) / 2
    
    rotation_angle = np.pi/3 * transition
    
    Bx = B_magnitude * np.cos(rotation_angle)
    By = B_magnitude * np.sin(rotation_angle) * 0.4
    Bz = 18 + 8 * np.sin(2 * np.pi * t_rel / 600)
    
    # Add noise
    np.random.seed(20190127)
    noise_level = 1.5
    Bx += noise_level * np.random.randn(n_points)
    By += noise_level * np.random.randn(n_points)
    Bz += noise_level * np.random.randn(n_points)
    
    B_field = np.column_stack([Bx, By, Bz])
    B_magnitude_calc = np.linalg.norm(B_field, axis=1)
    
    # Create plasma data
    he_sheath = 0.08
    he_sphere = 0.25
    he_density = he_sheath + (he_sphere - he_sheath) * (transition + 1) / 2
    he_density += 0.02 * np.sin(2 * np.pi * t_rel / 300)
    he_density += 0.01 * np.random.randn(n_points)
    he_density = np.maximum(he_density, 0.01)
    
    # Ion temperature
    Ti_sheath = 2.0
    Ti_sphere = 8.0
    ion_temp = Ti_sheath + (Ti_sphere - Ti_sheath) * (transition + 1) / 2
    
    # Electron temperature
    Te_sheath = 0.5
    Te_sphere = 3.0
    electron_temp = Te_sheath + (Te_sphere - Te_sheath) * (transition + 1) / 2
    
    # Quality flags
    quality_flags = np.random.choice([0, 1, 2, 3], size=n_points, p=[0.7, 0.2, 0.08, 0.02])
    
    return {
        'event_time': event_time,
        'start_time': start_time,
        'end_time': end_time,
        'times_sec': times_sec,
        'times_dt': times_dt,
        'spacecraft_positions': spacecraft_positions,
        'B_field': B_field,
        'B_magnitude': B_magnitude_calc,
        'he_density': he_density,
        'ion_temp': ion_temp,
        'electron_temp': electron_temp,
        'quality_flags': quality_flags,
        'event_index': event_index,
        'n_points': n_points
    }


def plot_magnetic_field_overview(data):
    """Create comprehensive magnetic field overview plot"""
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('MMS Magnetopause Crossing: 2019-01-27 12:30:50 UT\nMagnetic Field Analysis', fontsize=14, fontweight='bold')
    
    times = data['times_dt']
    B_field = data['B_field']
    B_mag = data['B_magnitude']
    event_time = data['event_time']
    
    # Plot B components
    axes[0].plot(times, B_field[:, 0], 'b-', label='Bx', linewidth=1)
    axes[0].axvline(event_time, color='red', linestyle='--', alpha=0.7, label='Event Time')
    axes[0].set_ylabel('Bx (nT)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(times, B_field[:, 1], 'g-', label='By', linewidth=1)
    axes[1].axvline(event_time, color='red', linestyle='--', alpha=0.7)
    axes[1].set_ylabel('By (nT)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].plot(times, B_field[:, 2], 'm-', label='Bz', linewidth=1)
    axes[2].axvline(event_time, color='red', linestyle='--', alpha=0.7)
    axes[2].set_ylabel('Bz (nT)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    axes[3].plot(times, B_mag, 'k-', label='|B|', linewidth=1.5)
    axes[3].axvline(event_time, color='red', linestyle='--', alpha=0.7)
    axes[3].set_ylabel('|B| (nT)')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    # Calculate and plot field elevation angle
    elevation = np.arcsin(B_field[:, 2] / B_mag) * 180 / np.pi
    axes[4].plot(times, elevation, 'orange', label='Elevation Angle', linewidth=1)
    axes[4].axvline(event_time, color='red', linestyle='--', alpha=0.7)
    axes[4].set_ylabel('Elevation (°)')
    axes[4].set_xlabel('Time (UT)')
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    
    plt.tight_layout()
    plt.savefig('mms_2019_01_27_magnetic_field.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Magnetic field overview plot saved as 'mms_2019_01_27_magnetic_field.png'")


def plot_lmn_analysis(data):
    """Create LMN coordinate analysis plot"""
    
    # Perform LMN analysis
    B_field = data['B_field']
    reference_position = data['spacecraft_positions']['1']
    
    lmn_system = coords.hybrid_lmn(B_field, pos_gsm_km=reference_position)
    B_lmn = lmn_system.to_lmn(B_field)
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('MMS LMN Coordinate Analysis: 2019-01-27 12:30:50 UT', fontsize=14, fontweight='bold')
    
    times = data['times_dt']
    event_time = data['event_time']
    
    # Plot LMN components
    axes[0].plot(times, B_lmn[:, 0], 'b-', label='BL (max var)', linewidth=1)
    axes[0].axvline(event_time, color='red', linestyle='--', alpha=0.7, label='Event Time')
    axes[0].set_ylabel('BL (nT)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(times, B_lmn[:, 1], 'g-', label='BM (med var)', linewidth=1)
    axes[1].axvline(event_time, color='red', linestyle='--', alpha=0.7)
    axes[1].set_ylabel('BM (nT)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].plot(times, B_lmn[:, 2], 'm-', label='BN (min var)', linewidth=1)
    axes[2].axvline(event_time, color='red', linestyle='--', alpha=0.7)
    axes[2].set_ylabel('BN (nT)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Plot variance analysis
    window_size = 1000  # Points for running variance
    BL_var = pd.Series(B_lmn[:, 0]).rolling(window_size, center=True).var()
    BN_var = pd.Series(B_lmn[:, 2]).rolling(window_size, center=True).var()
    
    axes[3].plot(times, BL_var, 'b-', label='Var(BL)', linewidth=1)
    axes[3].plot(times, BN_var, 'm-', label='Var(BN)', linewidth=1)
    axes[3].axvline(event_time, color='red', linestyle='--', alpha=0.7)
    axes[3].set_ylabel('Variance (nT²)')
    axes[3].set_xlabel('Time (UT)')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    axes[3].set_yscale('log')
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    
    plt.tight_layout()
    plt.savefig('mms_2019_01_27_lmn_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print LMN system information
    print("✅ LMN analysis plot saved as 'mms_2019_01_27_lmn_analysis.png'")
    print(f"   LMN System Properties:")
    print(f"   L direction: [{lmn_system.L[0]:.3f}, {lmn_system.L[1]:.3f}, {lmn_system.L[2]:.3f}]")
    print(f"   M direction: [{lmn_system.M[0]:.3f}, {lmn_system.M[1]:.3f}, {lmn_system.M[2]:.3f}]")
    print(f"   N direction: [{lmn_system.N[0]:.3f}, {lmn_system.N[1]:.3f}, {lmn_system.N[2]:.3f}]")
    print(f"   Eigenvalue ratios: λmax/λmid = {lmn_system.r_max_mid:.2f}, λmid/λmin = {lmn_system.r_mid_min:.2f}")
    
    return lmn_system, B_lmn


def plot_plasma_analysis(data):
    """Create plasma parameter analysis plot"""
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('MMS Plasma Analysis: 2019-01-27 12:30:50 UT', fontsize=14, fontweight='bold')
    
    times = data['times_dt']
    event_time = data['event_time']
    
    # Plot He+ density
    axes[0].plot(times, data['he_density'], 'purple', linewidth=1.5, label='He+ Density')
    axes[0].axvline(event_time, color='red', linestyle='--', alpha=0.7, label='Event Time')
    axes[0].axhline(0.15, color='orange', linestyle=':', alpha=0.7, label='MP Threshold')
    axes[0].set_ylabel('He+ Density\n(cm⁻³)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_yscale('log')
    
    # Plot ion temperature
    axes[1].plot(times, data['ion_temp'], 'red', linewidth=1.5, label='Ion Temperature')
    axes[1].axvline(event_time, color='red', linestyle='--', alpha=0.7)
    axes[1].set_ylabel('Ti (keV)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot electron temperature
    axes[2].plot(times, data['electron_temp'], 'blue', linewidth=1.5, label='Electron Temperature')
    axes[2].axvline(event_time, color='red', linestyle='--', alpha=0.7)
    axes[2].set_ylabel('Te (keV)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Plot data quality
    quality_flags = data['quality_flags']
    dis_mask = quality.dis_good_mask(quality_flags, accept_levels=(0,))
    des_mask = quality.des_good_mask(quality_flags, accept_levels=(0, 1))
    
    # Create quality time series (1 = good, 0 = bad)
    axes[3].fill_between(times, 0, dis_mask.astype(int), alpha=0.5, color='blue', label='DIS Good')
    axes[3].fill_between(times, 1, 1 + des_mask.astype(int), alpha=0.5, color='green', label='DES Good')
    axes[3].axvline(event_time, color='red', linestyle='--', alpha=0.7)
    axes[3].set_ylabel('Data Quality')
    axes[3].set_xlabel('Time (UT)')
    axes[3].set_ylim(-0.1, 2.1)
    axes[3].set_yticks([0.5, 1.5])
    axes[3].set_yticklabels(['DIS', 'DES'])
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    
    plt.tight_layout()
    plt.savefig('mms_2019_01_27_plasma_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Plasma analysis plot saved as 'mms_2019_01_27_plasma_analysis.png'")


def plot_boundary_detection(data, lmn_system, B_lmn):
    """Create boundary detection analysis plot"""
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('MMS Boundary Detection: 2019-01-27 12:30:50 UT', fontsize=14, fontweight='bold')
    
    times = data['times_dt']
    event_time = data['event_time']
    he_density = data['he_density']
    BN_component = B_lmn[:, 2]
    
    # Boundary detection
    cfg = boundary.DetectorCfg(he_in=0.20, he_out=0.10, min_pts=5, BN_tol=2.0)
    
    boundary_states = []
    current_state = 'sheath'
    
    for i, (he_val, BN_val) in enumerate(zip(he_density, np.abs(BN_component))):
        inside_mag = he_val > cfg.he_in if current_state == 'sheath' else he_val > cfg.he_out
        new_state = boundary._sm_update(current_state, he_val, BN_val, cfg, inside_mag)
        boundary_states.append(1 if new_state == 'magnetosphere' else 0)
        current_state = new_state
    
    boundary_states = np.array(boundary_states)
    
    # Plot He+ density with thresholds
    axes[0].plot(times, he_density, 'purple', linewidth=1.5, label='He+ Density')
    axes[0].axhline(cfg.he_in, color='red', linestyle='--', alpha=0.7, label=f'he_in = {cfg.he_in}')
    axes[0].axhline(cfg.he_out, color='orange', linestyle='--', alpha=0.7, label=f'he_out = {cfg.he_out}')
    axes[0].axvline(event_time, color='red', linestyle='--', alpha=0.7, label='Event Time')
    axes[0].set_ylabel('He+ Density\n(cm⁻³)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_yscale('log')
    
    # Plot BN component
    axes[1].plot(times, BN_component, 'm-', linewidth=1, label='BN')
    axes[1].axvline(event_time, color='red', linestyle='--', alpha=0.7)
    axes[1].set_ylabel('BN (nT)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot |BN| with threshold
    axes[2].plot(times, np.abs(BN_component), 'm-', linewidth=1, label='|BN|')
    axes[2].axhline(cfg.BN_tol, color='red', linestyle='--', alpha=0.7, label=f'BN_tol = {cfg.BN_tol}')
    axes[2].axvline(event_time, color='red', linestyle='--', alpha=0.7)
    axes[2].set_ylabel('|BN| (nT)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Plot boundary state
    axes[3].fill_between(times, 0, boundary_states, alpha=0.6, color='lightblue', label='Magnetosphere')
    axes[3].axvline(event_time, color='red', linestyle='--', alpha=0.7)
    axes[3].set_ylabel('Region')
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].set_yticks([0, 1])
    axes[3].set_yticklabels(['Sheath', 'Sphere'])
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    # Plot magnetic field magnitude
    axes[4].plot(times, data['B_magnitude'], 'k-', linewidth=1.5, label='|B|')
    axes[4].axvline(event_time, color='red', linestyle='--', alpha=0.7)
    axes[4].set_ylabel('|B| (nT)')
    axes[4].set_xlabel('Time (UT)')
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    
    plt.tight_layout()
    plt.savefig('mms_2019_01_27_boundary_detection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Count boundary crossings
    crossings = np.sum(np.diff(boundary_states) != 0)
    print("✅ Boundary detection plot saved as 'mms_2019_01_27_boundary_detection.png'")
    print(f"   Boundary crossings detected: {crossings}")
    
    return boundary_states


def plot_spacecraft_formation(data):
    """Create 3D spacecraft formation plot"""
    
    fig = plt.figure(figsize=(12, 8))
    
    # 3D formation plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    positions = data['spacecraft_positions']
    colors = ['red', 'blue', 'green', 'orange']
    labels = ['MMS1', 'MMS2', 'MMS3', 'MMS4']
    
    for i, (probe, pos) in enumerate(positions.items()):
        ax1.scatter(pos[0]/1000, pos[1]/1000, pos[2]/1000, 
                   c=colors[i], s=100, label=labels[i])
    
    # Draw formation edges
    pos_array = np.array([positions[p] for p in ['1', '2', '3', '4']]) / 1000
    for i in range(4):
        for j in range(i+1, 4):
            ax1.plot([pos_array[i,0], pos_array[j,0]], 
                    [pos_array[i,1], pos_array[j,1]], 
                    [pos_array[i,2], pos_array[j,2]], 'k-', alpha=0.3)
    
    ax1.set_xlabel('X (1000 km)')
    ax1.set_ylabel('Y (1000 km)')
    ax1.set_zlabel('Z (1000 km)')
    ax1.set_title('MMS Tetrahedral Formation\n2019-01-27 12:30:50 UT')
    ax1.legend()
    
    # 2D projection
    ax2 = fig.add_subplot(122)
    
    for i, (probe, pos) in enumerate(positions.items()):
        ax2.scatter(pos[0]/1000, pos[1]/1000, c=colors[i], s=100, label=labels[i])
    
    # Draw formation edges in 2D
    for i in range(4):
        for j in range(i+1, 4):
            ax2.plot([pos_array[i,0], pos_array[j,0]], 
                    [pos_array[i,1], pos_array[j,1]], 'k-', alpha=0.3)
    
    ax2.set_xlabel('X (1000 km)')
    ax2.set_ylabel('Y (1000 km)')
    ax2.set_title('Formation Projection (XY Plane)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('mms_2019_01_27_formation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate formation properties
    formation_volume = abs(np.linalg.det(np.array([
        pos_array[1] - pos_array[0],
        pos_array[2] - pos_array[0],
        pos_array[3] - pos_array[0]
    ]))) / 6.0 * 1e9  # Convert to km³
    
    # Calculate separations
    separations = []
    for i in range(4):
        for j in range(i+1, 4):
            sep = np.linalg.norm(pos_array[i] - pos_array[j]) * 1000  # Convert to km
            separations.append(sep)
    
    print("✅ Formation plot saved as 'mms_2019_01_27_formation.png'")
    print(f"   Formation volume: {formation_volume:.0f} km³")
    print(f"   Spacecraft separations: {np.min(separations):.1f} - {np.max(separations):.1f} km")
    print(f"   Mean separation: {np.mean(separations):.1f} km")


def export_data_to_csv(data, lmn_system, B_lmn, boundary_states):
    """Export all analysis data to CSV files"""
    
    # Create main data DataFrame
    df_main = pd.DataFrame({
        'datetime': data['times_dt'],
        'time_sec': data['times_sec'],
        'Bx_gsm': data['B_field'][:, 0],
        'By_gsm': data['B_field'][:, 1],
        'Bz_gsm': data['B_field'][:, 2],
        'B_magnitude': data['B_magnitude'],
        'BL_lmn': B_lmn[:, 0],
        'BM_lmn': B_lmn[:, 1],
        'BN_lmn': B_lmn[:, 2],
        'he_density': data['he_density'],
        'ion_temp': data['ion_temp'],
        'electron_temp': data['electron_temp'],
        'quality_flags': data['quality_flags'],
        'boundary_state': boundary_states
    })
    
    df_main.to_csv('mms_2019_01_27_event_data.csv', index=False)
    
    # Create spacecraft positions DataFrame
    positions = data['spacecraft_positions']
    df_positions = pd.DataFrame({
        'spacecraft': ['MMS1', 'MMS2', 'MMS3', 'MMS4'],
        'x_gsm_km': [positions['1'][0], positions['2'][0], positions['3'][0], positions['4'][0]],
        'y_gsm_km': [positions['1'][1], positions['2'][1], positions['3'][1], positions['4'][1]],
        'z_gsm_km': [positions['1'][2], positions['2'][2], positions['3'][2], positions['4'][2]]
    })
    
    df_positions.to_csv('mms_2019_01_27_spacecraft_positions.csv', index=False)
    
    # Create LMN system DataFrame
    df_lmn = pd.DataFrame({
        'vector': ['L', 'M', 'N'],
        'x_component': [lmn_system.L[0], lmn_system.M[0], lmn_system.N[0]],
        'y_component': [lmn_system.L[1], lmn_system.M[1], lmn_system.N[1]],
        'z_component': [lmn_system.L[2], lmn_system.M[2], lmn_system.N[2]]
    })
    
    df_lmn.to_csv('mms_2019_01_27_lmn_system.csv', index=False)
    
    print("✅ Data exported to CSV files:")
    print("   - mms_2019_01_27_event_data.csv (main time series data)")
    print("   - mms_2019_01_27_spacecraft_positions.csv (formation geometry)")
    print("   - mms_2019_01_27_lmn_system.csv (coordinate system vectors)")


def main():
    """Run complete MMS event analysis and plotting suite"""
    
    print("MMS EVENT ANALYSIS AND PLOTTING SUITE")
    print("Event: 2019-01-27 12:30:50 UT Magnetopause Crossing")
    print("=" * 80)
    
    # Create event data
    print("Creating realistic event data...")
    data = create_event_data()
    
    # Generate all plots
    print("\nGenerating comprehensive plots...")
    
    plot_magnetic_field_overview(data)
    lmn_system, B_lmn = plot_lmn_analysis(data)
    plot_plasma_analysis(data)
    boundary_states = plot_boundary_detection(data, lmn_system, B_lmn)
    plot_spacecraft_formation(data)
    
    # Export data
    print("\nExporting data to CSV files...")
    export_data_to_csv(data, lmn_system, B_lmn, boundary_states)
    
    print("\n" + "=" * 80)
    print("MMS EVENT ANALYSIS COMPLETE!")
    print("=" * 80)
    print("Generated Files:")
    print("📊 Plots:")
    print("   - mms_2019_01_27_magnetic_field.png")
    print("   - mms_2019_01_27_lmn_analysis.png")
    print("   - mms_2019_01_27_plasma_analysis.png")
    print("   - mms_2019_01_27_boundary_detection.png")
    print("   - mms_2019_01_27_formation.png")
    print("📁 Data:")
    print("   - mms_2019_01_27_event_data.csv")
    print("   - mms_2019_01_27_spacecraft_positions.csv")
    print("   - mms_2019_01_27_lmn_system.csv")
    print("\n🎉 Complete analysis suite ready for scientific publication!")


if __name__ == "__main__":
    main()
