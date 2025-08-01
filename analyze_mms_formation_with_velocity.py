#!/usr/bin/env python3
"""
MMS Formation Analysis with Velocity-Aware Ordering
==================================================

This script properly analyzes MMS spacecraft formations by considering both
position AND velocity to determine orbital ordering (who is leading/trailing).

Key Features:
- Loads real MMS ephemeris data (positions and velocities)
- Determines formation type (string-of-pearls, tetrahedral, etc.)
- Calculates velocity-aware spacecraft ordering
- Identifies which spacecraft is leading in orbit
- Explains ordering differences between dates

For string-of-pearls formations, the ordering should be based on orbital motion:
- Leading spacecraft: ahead in orbital path
- Trailing spacecraft: behind in orbital path
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import mms_mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mms_mp
from mms_mp.formation_detection import detect_formation_type, print_formation_analysis
from mms_mp import data_loader


def load_mms_ephemeris_data(date_str: str, time_str: str, duration_minutes: int = 10):
    """
    Load MMS ephemeris data including positions and velocities
    
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
    dict : Spacecraft positions, velocities, and metadata
    """
    
    print(f"\n📡 Loading MMS ephemeris data for {date_str} {time_str}...")
    
    # Create time range
    center_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    start_time = center_time - timedelta(minutes=duration_minutes//2)
    end_time = center_time + timedelta(minutes=duration_minutes//2)
    
    trange = [
        start_time.strftime('%Y-%m-%d/%H:%M:%S'),
        end_time.strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    print(f"   Time range: {trange[0]} to {trange[1]}")
    
    try:
        # Load MMS data including ephemeris
        evt = data_loader.load_event(
            trange=trange,
            probes=['1', '2', '3', '4'],
            include_ephem=True,
            include_edp=False,
            include_fpi=False,
            include_hpca=False
        )
        
        # Extract spacecraft positions and velocities at center time
        positions = {}
        velocities = {}
        center_index = None
        
        for probe in ['1', '2', '3', '4']:
            if probe in evt:
                # Position data
                if 'POS_gsm' in evt[probe]:
                    times, pos_data = evt[probe]['POS_gsm']
                    
                    # Find index closest to center time
                    if center_index is None:
                        if hasattr(times[0], 'timestamp'):
                            time_diffs = [abs((t - center_time).total_seconds()) for t in times]
                        else:
                            center_timestamp = center_time.timestamp()
                            time_diffs = [abs(t - center_timestamp) for t in times]
                        center_index = np.argmin(time_diffs)
                    
                    # Extract position at center time (convert to km)
                    positions[probe] = pos_data[center_index] / 1000.0
                    
                # Velocity data
                if 'VEL_gsm' in evt[probe]:
                    times, vel_data = evt[probe]['VEL_gsm']
                    velocities[probe] = vel_data[center_index] / 1000.0  # Convert to km/s
                
                print(f"   MMS{probe}: Pos=[{positions[probe][0]:.1f}, {positions[probe][1]:.1f}, {positions[probe][2]:.1f}] km")
                if probe in velocities:
                    print(f"           Vel=[{velocities[probe][0]:.2f}, {velocities[probe][1]:.2f}, {velocities[probe][2]:.2f}] km/s")
        
        return {
            'date': date_str,
            'time': time_str,
            'center_time': center_time,
            'positions': positions,
            'velocities': velocities,
            'data_loaded': True
        }
        
    except Exception as e:
        print(f"   ❌ Failed to load real data: {e}")
        
        # Create synthetic data with realistic orbital motion
        print("   🔄 Creating synthetic data with orbital motion...")
        return create_synthetic_orbital_data(date_str, time_str)


def create_synthetic_orbital_data(date_str: str, time_str: str):
    """Create synthetic spacecraft data with realistic orbital motion"""
    
    center_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    
    # Base position near magnetopause (typical MMS orbit)
    RE_km = 6371.0
    base_position = np.array([10.5, 3.2, 1.8]) * RE_km  # ~11.5 RE
    
    # Orbital velocity (typical MMS orbital speed ~2-4 km/s)
    orbital_speed = 3.0  # km/s
    # Velocity direction (roughly tangential to orbit)
    velocity_direction = np.array([-0.3, 0.8, 0.5])  # Normalized below
    velocity_direction = velocity_direction / np.linalg.norm(velocity_direction)
    base_velocity = orbital_speed * velocity_direction
    
    # String-of-pearls formation: spacecraft arranged along orbital path
    # with small separations (~100-200 km)
    along_track_separations = [0, 150, 300, 450]  # km along orbit
    
    positions = {}
    velocities = {}
    
    for i, probe in enumerate(['1', '2', '3', '4']):
        # Position: base + offset along velocity direction
        offset = along_track_separations[i] * velocity_direction
        positions[probe] = base_position + offset
        
        # Velocity: all spacecraft have similar orbital velocity
        # Small variations due to orbital mechanics
        vel_variation = np.random.normal(0, 0.1, 3)  # Small random variations
        velocities[probe] = base_velocity + vel_variation
        
        print(f"   MMS{probe}: Pos=[{positions[probe][0]:.1f}, {positions[probe][1]:.1f}, {positions[probe][2]:.1f}] km (synthetic)")
        print(f"           Vel=[{velocities[probe][0]:.2f}, {velocities[probe][1]:.2f}, {velocities[probe][2]:.2f}] km/s")
    
    return {
        'date': date_str,
        'time': time_str,
        'center_time': center_time,
        'positions': positions,
        'velocities': velocities,
        'data_loaded': False
    }


def analyze_formation_with_velocity(data):
    """Analyze formation using both position and velocity information"""
    
    print(f"\n🔍 Analyzing formation with velocity for {data['date']} {data['time']}...")
    
    positions = data['positions']
    velocities = data.get('velocities', None)
    
    # Perform velocity-aware formation analysis
    formation_analysis = detect_formation_type(positions, velocities)
    
    # Print detailed analysis
    print_formation_analysis(formation_analysis, verbose=True)
    
    # Additional velocity-specific analysis
    if velocities is not None:
        print(f"\n🚀 Velocity-Specific Analysis:")
        
        # Calculate mean orbital velocity
        mean_velocity = np.mean([velocities[p] for p in ['1', '2', '3', '4']], axis=0)
        orbital_speed = np.linalg.norm(mean_velocity)
        print(f"   Mean orbital speed: {orbital_speed:.2f} km/s")
        
        # Velocity direction
        velocity_direction = mean_velocity / orbital_speed
        print(f"   Velocity direction: [{velocity_direction[0]:.3f}, {velocity_direction[1]:.3f}, {velocity_direction[2]:.3f}]")
        
        # Formation center
        formation_center = np.mean([positions[p] for p in ['1', '2', '3', '4']], axis=0)
        
        # Calculate along-track positions (projection along velocity)
        along_track_positions = {}
        for probe in ['1', '2', '3', '4']:
            relative_pos = positions[probe] - formation_center
            along_track_pos = np.dot(relative_pos, velocity_direction)
            along_track_positions[probe] = along_track_pos
        
        # Order spacecraft by along-track position (leading to trailing)
        orbital_order = sorted(['1', '2', '3', '4'], 
                              key=lambda p: along_track_positions[p], 
                              reverse=True)
        
        print(f"\n   Orbital Ordering (Leading → Trailing):")
        for i, probe in enumerate(orbital_order):
            status = "LEADING" if i == 0 else "TRAILING" if i == 3 else "MIDDLE"
            print(f"      {i+1}. MMS{probe} ({status}) - Along-track: {along_track_positions[probe]:+.1f} km")
        
        # Check if this matches the detected formation ordering
        if 'Leading_to_Trailing' in formation_analysis.spacecraft_ordering:
            detected_order = formation_analysis.spacecraft_ordering['Leading_to_Trailing']
            if orbital_order == detected_order:
                print(f"   ✅ Orbital ordering matches formation analysis")
            else:
                print(f"   ⚠️ Orbital ordering differs from formation analysis")
                print(f"      Calculated: {' → '.join([f'MMS{p}' for p in orbital_order])}")
                print(f"      Detected:   {' → '.join([f'MMS{p}' for p in detected_order])}")
    
    return formation_analysis


def compare_formations_with_velocity(data1, data2):
    """Compare formations between two dates using velocity-aware analysis"""
    
    print(f"\n🔄 Comparing velocity-aware formations: {data1['date']} vs {data2['date']}...")
    
    analysis1 = analyze_formation_with_velocity(data1)
    analysis2 = analyze_formation_with_velocity(data2)
    
    # Compare formation types
    print(f"\n📊 Formation Type Comparison:")
    print(f"   {data1['date']}: {analysis1.formation_type.value}")
    print(f"   {data2['date']}: {analysis2.formation_type.value}")
    
    if analysis1.formation_type == analysis2.formation_type:
        print(f"   ✅ Formation types are CONSISTENT")
    else:
        print(f"   ⚠️ Formation types are DIFFERENT")
        print(f"      This explains why ordering analysis might differ!")
    
    # Compare velocity-aware orderings
    if ('Leading_to_Trailing' in analysis1.spacecraft_ordering and 
        'Leading_to_Trailing' in analysis2.spacecraft_ordering):
        
        order1 = analysis1.spacecraft_ordering['Leading_to_Trailing']
        order2 = analysis2.spacecraft_ordering['Leading_to_Trailing']
        
        print(f"\n📊 Orbital Ordering Comparison:")
        print(f"   {data1['date']}: {' → '.join([f'MMS{p}' for p in order1])}")
        print(f"   {data2['date']}: {' → '.join([f'MMS{p}' for p in order2])}")
        
        if order1 == order2:
            print(f"   ✅ Orbital orderings are CONSISTENT")
        else:
            print(f"   ❌ Orbital orderings are DIFFERENT")
            print(f"      This is the source of the ordering discrepancy!")
    
    return analysis1, analysis2


def main():
    """Main function to analyze MMS formations with velocity awareness"""
    
    print("MMS FORMATION ANALYSIS WITH VELOCITY-AWARE ORDERING")
    print("=" * 80)
    print("This analysis considers orbital motion to determine spacecraft ordering")
    print("for string-of-pearls formations (who is leading vs trailing in orbit)")
    print("=" * 80)
    
    # Load data for both dates
    data_26 = load_mms_ephemeris_data('2019-01-26', '15:00:00')
    data_27 = load_mms_ephemeris_data('2019-01-27', '12:30:50')
    
    # Compare formations with velocity awareness
    analysis1, analysis2 = compare_formations_with_velocity(data_26, data_27)
    
    # Summary and recommendations
    print(f"\n" + "=" * 80)
    print("SUMMARY: WHY SPACECRAFT ORDERING DIFFERS")
    print("=" * 80)
    
    print("The key insight is that spacecraft ordering in string-of-pearls formations")
    print("should be based on ORBITAL MOTION (who is leading/trailing), not just")
    print("static position coordinates.")
    print()
    print("Previous analysis issues:")
    print("  ❌ Used only static positions (X, Y, Z coordinates)")
    print("  ❌ Ignored orbital velocity and direction of motion")
    print("  ❌ Assumed tetrahedral analysis for string-of-pearls formation")
    print()
    print("Corrected analysis:")
    print("  ✅ Considers orbital velocity direction")
    print("  ✅ Orders spacecraft by along-track position (leading → trailing)")
    print("  ✅ Uses formation-specific analysis methods")
    print("  ✅ Accounts for orbital mechanics")


if __name__ == "__main__":
    main()
