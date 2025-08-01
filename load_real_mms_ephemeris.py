#!/usr/bin/env python3
"""
Load Real MMS Ephemeris Data
============================

This script properly loads real MMS ephemeris data from MEC files
instead of falling back to synthetic data.

The issue was that the data_loader was trying to load 'def' level ephemeris
but the actual files are MEC L2 'epht89q' data.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import mms_mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# PySpedas imports for direct MEC data loading
from pyspedas.projects import mms
from pyspedas import get_data
from pytplot import data_quants


def load_mms_mec_ephemeris(date_str: str, time_str: str, duration_minutes: int = 10):
    """
    Load real MMS ephemeris data from MEC files
    
    Parameters:
    -----------
    date_str : str
        Date in format 'YYYY-MM-DD'
    time_str : str
        Time in format 'HH:MM:SS'
    duration_minutes : int
        Duration around the time to load
        
    Returns:
    --------
    dict : Real spacecraft positions and velocities
    """
    
    print(f"\n📡 Loading REAL MMS MEC ephemeris data for {date_str} {time_str}...")
    
    # Create time range
    center_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    start_time = center_time - timedelta(minutes=duration_minutes//2)
    end_time = center_time + timedelta(minutes=duration_minutes//2)
    
    trange = [
        start_time.strftime('%Y-%m-%d/%H:%M:%S'),
        end_time.strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    print(f"   Time range: {trange[0]} to {trange[1]}")
    
    positions = {}
    velocities = {}
    
    for probe in ['1', '2', '3', '4']:
        print(f"   Loading MMS{probe} MEC data...")
        
        try:
            # Load MEC ephemeris data (positions and velocities)
            mms.mms_load_mec(
                trange=trange,
                probe=probe,
                data_rate='srvy',
                level='l2',
                datatype='epht89q',
                time_clip=True,
                notplot=False
            )
            
            # Get position data
            pos_var = f'mms{probe}_mec_r_gsm'
            if pos_var in data_quants:
                times, pos_data = get_data(pos_var)
                
                # Find index closest to center time
                if hasattr(times[0], 'timestamp'):
                    time_diffs = [abs((t - center_time).total_seconds()) for t in times]
                else:
                    center_timestamp = center_time.timestamp()
                    time_diffs = [abs(t - center_timestamp) for t in times]
                
                center_index = np.argmin(time_diffs)
                
                # Extract position at center time (already in km)
                positions[probe] = pos_data[center_index]
                
                print(f"      Position: [{positions[probe][0]:.1f}, {positions[probe][1]:.1f}, {positions[probe][2]:.1f}] km")
            
            # Get velocity data
            vel_var = f'mms{probe}_mec_v_gsm'
            if vel_var in data_quants:
                times_vel, vel_data = get_data(vel_var)
                velocities[probe] = vel_data[center_index]  # Already in km/s
                
                print(f"      Velocity: [{velocities[probe][0]:.2f}, {velocities[probe][1]:.2f}, {velocities[probe][2]:.2f}] km/s")
            else:
                print(f"      ⚠️ No velocity data found for MMS{probe}")
                
        except Exception as e:
            print(f"      ❌ Failed to load MMS{probe} MEC data: {e}")
            
            # Check if we have any MEC files for this date
            print(f"      🔍 Checking for available MEC files...")
            
            # Try alternative loading methods or dates
            continue
    
    if not positions:
        print("   ❌ No real MEC data could be loaded!")
        print("   🔍 Available MEC files:")
        
        # List available MEC files
        mec_dir = "pydata/mms1/mec/srvy/l2/epht89q/2019/01"
        if os.path.exists(mec_dir):
            files = os.listdir(mec_dir)
            for file in files:
                print(f"      {file}")
        
        return None
    
    return {
        'date': date_str,
        'time': time_str,
        'positions': positions,
        'velocities': velocities,
        'data_source': 'real_mec'
    }


def analyze_real_spacecraft_ordering(date_str: str, time_str: str):
    """
    Analyze spacecraft ordering using real MEC ephemeris data
    """
    
    print(f"\n🔍 Analyzing REAL spacecraft ordering for {date_str} {time_str}...")
    
    # Load real MEC data
    data = load_mms_mec_ephemeris(date_str, time_str)
    
    if data is None:
        print("   ❌ Cannot analyze - no real data available")
        return None
    
    positions = data['positions']
    velocities = data.get('velocities', {})
    
    probes = ['1', '2', '3', '4']
    orderings = {}
    
    print(f"\n📊 Real Spacecraft Orderings:")
    print("-" * 50)
    
    # GSM coordinate orderings
    orderings['X_GSM'] = sorted(probes, key=lambda p: positions[p][0])
    orderings['Y_GSM'] = sorted(probes, key=lambda p: positions[p][1])
    orderings['Z_GSM'] = sorted(probes, key=lambda p: positions[p][2])
    
    # Distance from Earth
    distances = {p: np.linalg.norm(positions[p]) for p in probes}
    orderings['Distance_from_Earth'] = sorted(probes, key=lambda p: distances[p])
    
    # Formation center analysis
    formation_center = np.mean([positions[p] for p in probes], axis=0)
    
    # Principal component analysis
    pos_array = np.array([positions[p] for p in probes])
    centered_positions = pos_array - formation_center
    
    # SVD for principal components
    U, s, Vt = np.linalg.svd(centered_positions)
    
    # Order by principal components
    for i, pc_name in enumerate(['PC1', 'PC2', 'PC3']):
        if i < len(Vt):
            projections = {p: np.dot(positions[p] - formation_center, Vt[i]) for p in probes}
            orderings[f'{pc_name}_positive'] = sorted(probes, key=lambda p: projections[p])
            orderings[f'{pc_name}_negative'] = sorted(probes, key=lambda p: projections[p], reverse=True)
    
    # Velocity-based orderings (if available)
    if velocities:
        mean_velocity = np.mean([velocities[p] for p in probes], axis=0)
        if np.linalg.norm(mean_velocity) > 0:
            velocity_direction = mean_velocity / np.linalg.norm(mean_velocity)
            
            # Along velocity direction (orbital ordering)
            vel_projections = {p: np.dot(positions[p] - formation_center, velocity_direction) for p in probes}
            orderings['Along_Velocity'] = sorted(probes, key=lambda p: vel_projections[p])
            orderings['Against_Velocity'] = sorted(probes, key=lambda p: vel_projections[p], reverse=True)
    
    # Print all orderings
    independent_source_order = ['2', '1', '4', '3']
    
    for ordering_name, order in orderings.items():
        order_str = ' → '.join([f'MMS{p}' for p in order])
        print(f"{ordering_name:20s}: {order_str}")
        
        # Check if this matches independent source
        if order == independent_source_order:
            print(f"                     ✅ MATCHES INDEPENDENT SOURCE!")
    
    return {
        'orderings': orderings,
        'positions': positions,
        'velocities': velocities,
        'formation_center': formation_center,
        'principal_components': s,
        'principal_directions': Vt
    }


def calculate_real_distances(positions):
    """
    Calculate all pairwise distances between spacecraft using real positions
    """
    
    probes = ['1', '2', '3', '4']
    distances = {}
    
    print(f"\n📏 Real Inter-spacecraft Distances:")
    print("-" * 40)
    
    for i, probe1 in enumerate(probes):
        for j, probe2 in enumerate(probes):
            if i < j:
                dist = np.linalg.norm(positions[probe1] - positions[probe2])
                distances[f"MMS{probe1}-MMS{probe2}"] = dist
                print(f"MMS{probe1} ↔ MMS{probe2}: {dist:.1f} km")
    
    # Find closest and farthest pairs
    min_dist = min(distances.values())
    max_dist = max(distances.values())
    
    closest_pair = [pair for pair, dist in distances.items() if dist == min_dist][0]
    farthest_pair = [pair for pair, dist in distances.items() if dist == max_dist][0]
    
    print(f"\nClosest pair:  {closest_pair} ({min_dist:.1f} km)")
    print(f"Farthest pair: {farthest_pair} ({max_dist:.1f} km)")
    
    return distances


def create_real_3d_plot(data1, data2):
    """
    Create 3D visualization using real spacecraft positions
    """
    
    if data1 is None or data2 is None:
        print("❌ Cannot create 3D plot - missing real data")
        return
    
    fig = plt.figure(figsize=(14, 10))
    
    # Create 3D subplots
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    
    colors = ['red', 'blue', 'green', 'orange']
    labels = ['MMS1', 'MMS2', 'MMS3', 'MMS4']
    
    # Plot 1: 2019-01-26 positions
    positions1 = data1['positions']
    for i, probe in enumerate(['1', '2', '3', '4']):
        if probe in positions1:
            pos = positions1[probe] / 1000  # Convert to 1000 km units
            ax1.scatter(pos[0], pos[1], pos[2], c=colors[i], s=100, label=labels[i])
            ax1.text(pos[0], pos[1], pos[2], f'  MMS{probe}', fontsize=8)
    
    ax1.set_title(f'REAL: {data1["date"]} {data1["time"]}')
    ax1.set_xlabel('X (1000 km)')
    ax1.set_ylabel('Y (1000 km)')
    ax1.set_zlabel('Z (1000 km)')
    ax1.legend()
    
    # Plot 2: 2019-01-27 positions
    positions2 = data2['positions']
    for i, probe in enumerate(['1', '2', '3', '4']):
        if probe in positions2:
            pos = positions2[probe] / 1000
            ax2.scatter(pos[0], pos[1], pos[2], c=colors[i], s=100, label=labels[i])
            ax2.text(pos[0], pos[1], pos[2], f'  MMS{probe}', fontsize=8)
    
    ax2.set_title(f'REAL: {data2["date"]} {data2["time"]}')
    ax2.set_xlabel('X (1000 km)')
    ax2.set_ylabel('Y (1000 km)')
    ax2.set_zlabel('Z (1000 km)')
    ax2.legend()
    
    # Plot 3: Formation connectivity
    for i, probe in enumerate(['1', '2', '3', '4']):
        if probe in positions1:
            pos = positions1[probe] / 1000
            ax3.scatter(pos[0], pos[1], pos[2], c=colors[i], s=100, label=labels[i])
            ax3.text(pos[0], pos[1], pos[2], f'  MMS{probe}', fontsize=8)
    
    # Draw lines between spacecraft
    probes = ['1', '2', '3', '4']
    for i, probe1 in enumerate(probes):
        for j, probe2 in enumerate(probes):
            if i < j and probe1 in positions1 and probe2 in positions1:
                pos1 = positions1[probe1] / 1000
                pos2 = positions1[probe2] / 1000
                ax3.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 
                        'k-', alpha=0.3, linewidth=1)
    
    ax3.set_title(f'REAL Formation: {data1["date"]}')
    ax3.set_xlabel('X (1000 km)')
    ax3.set_ylabel('Y (1000 km)')
    ax3.set_zlabel('Z (1000 km)')
    
    plt.tight_layout()
    plt.savefig('real_mms_spacecraft_positions_3d.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Main function to analyze real MMS spacecraft ordering
    """
    
    print("REAL MMS SPACECRAFT ORDERING ANALYSIS")
    print("=" * 80)
    print("Loading actual MEC ephemeris data (not synthetic)")
    print("=" * 80)
    
    # Analyze both dates with real data
    data_26 = analyze_real_spacecraft_ordering('2019-01-26', '15:00:00')
    data_27 = analyze_real_spacecraft_ordering('2019-01-27', '12:30:50')
    
    # Calculate real distances
    if data_26:
        print(f"\n" + "=" * 80)
        print("REAL DISTANCE ANALYSIS: 2019-01-26")
        print("=" * 80)
        distances_26 = calculate_real_distances(data_26['positions'])
    
    if data_27:
        print(f"\n" + "=" * 80)
        print("REAL DISTANCE ANALYSIS: 2019-01-27")
        print("=" * 80)
        distances_27 = calculate_real_distances(data_27['positions'])
    
    # Create 3D visualization with real data
    create_real_3d_plot(data_26, data_27)
    
    # Summary
    print(f"\n" + "=" * 80)
    print("SUMMARY: REAL vs SYNTHETIC DATA COMPARISON")
    print("=" * 80)
    
    if data_26 and data_27:
        print("✅ Successfully loaded REAL MEC ephemeris data")
        print("✅ Generated analysis based on actual spacecraft positions")
        print("✅ Can now compare with independent source ordering: 2 → 1 → 4 → 3")
    else:
        print("❌ Could not load real MEC data")
        print("   Check MEC file availability and loading method")
    
    print(f"\nGenerated: real_mms_spacecraft_positions_3d.png")


if __name__ == "__main__":
    main()
