#!/usr/bin/env python3
"""
Debug Spacecraft Ordering: 2019-01-26 vs 2019-01-27
==================================================

This script investigates why spacecraft ordering appears different between
2019-01-26 15:00 UT and 2019-01-27 12:30:50 UT when the physical formation
should be similar on consecutive days.

Potential issues to investigate:
1. Coordinate system differences (GSE vs GSM vs LMN)
2. Reference frame or origin differences
3. Time-dependent coordinate transformations
4. Spacecraft position loading inconsistencies
5. Formation analysis algorithm differences
6. LMN coordinate system calculation differences
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import mms_mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mms_mp
from mms_mp import data_loader, coords, multispacecraft
from mms_mp.formation_detection import detect_formation_type, print_formation_analysis


def load_spacecraft_positions_for_date(date_str, time_str, duration_minutes=10):
    """
    Load actual spacecraft positions for a specific date and time
    
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
    dict : Spacecraft positions and metadata
    """
    
    print(f"\n📡 Loading spacecraft positions for {date_str} {time_str}...")
    
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
            include_edp=False
        )
        
        # Extract spacecraft positions at the center time
        positions = {}
        center_index = None
        
        for probe in ['1', '2', '3', '4']:
            if probe in evt and 'POS_gsm' in evt[probe]:
                times, pos_data = evt[probe]['POS_gsm']
                
                # Find index closest to center time
                if center_index is None:
                    # Convert times to datetime for comparison
                    if hasattr(times[0], 'timestamp'):
                        time_diffs = [abs((t - center_time).total_seconds()) for t in times]
                    else:
                        # Assume times are in seconds since epoch
                        center_timestamp = center_time.timestamp()
                        time_diffs = [abs(t - center_timestamp) for t in times]
                    
                    center_index = np.argmin(time_diffs)
                
                # Extract position at center time
                positions[probe] = pos_data[center_index] / 1000.0  # Convert to km
                
                print(f"   MMS{probe}: [{positions[probe][0]:.1f}, {positions[probe][1]:.1f}, {positions[probe][2]:.1f}] km")
        
        return {
            'date': date_str,
            'time': time_str,
            'center_time': center_time,
            'positions': positions,
            'data_loaded': True
        }
        
    except Exception as e:
        print(f"   ❌ Failed to load real data: {e}")
        
        # Create synthetic positions based on typical MMS formation
        print("   🔄 Creating synthetic positions...")
        return create_synthetic_positions(date_str, time_str)


def create_synthetic_positions(date_str, time_str):
    """Create synthetic but realistic spacecraft positions"""
    
    center_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    
    # Base position near magnetopause
    RE_km = 6371.0
    base_position = np.array([10.5, 3.2, 1.8]) * RE_km  # ~11.5 RE
    
    # Tetrahedral formation with ~100 km separations
    positions = {
        '1': base_position + np.array([0.0, 0.0, 0.0]),        # Reference
        '2': base_position + np.array([100.0, 0.0, 0.0]),      # 100 km X
        '3': base_position + np.array([50.0, 86.6, 0.0]),      # 60° in XY
        '4': base_position + np.array([50.0, 28.9, 81.6])      # Above plane
    }
    
    for probe in ['1', '2', '3', '4']:
        print(f"   MMS{probe}: [{positions[probe][0]:.1f}, {positions[probe][1]:.1f}, {positions[probe][2]:.1f}] km (synthetic)")
    
    return {
        'date': date_str,
        'time': time_str,
        'center_time': center_time,
        'positions': positions,
        'data_loaded': False
    }


def analyze_formation_geometry(positions_data):
    """Analyze formation geometry using automatic formation detection"""

    print(f"\n🔍 Analyzing formation geometry for {positions_data['date']} {positions_data['time']}...")

    positions = positions_data['positions']

    # Use automatic formation detection
    formation_analysis = detect_formation_type(positions)

    # Print detailed analysis
    print_formation_analysis(formation_analysis, verbose=False)

    # Legacy calculations for compatibility
    pos_array = np.array([positions[p] for p in ['1', '2', '3', '4']])
    formation_center = np.mean(pos_array, axis=0)
    distance_from_earth = np.linalg.norm(formation_center) / 6371.0  # RE
    
    # Use orderings from formation analysis
    orderings = formation_analysis.spacecraft_ordering

    print(f"\n   Spacecraft orderings:")
    for coord_sys, order in orderings.items():
        print(f"      {coord_sys:8s}: {' → '.join([f'MMS{p}' for p in order])}")

    return {
        'formation_analysis': formation_analysis,
        'formation_volume': formation_analysis.volume,
        'distance_from_earth': distance_from_earth,
        'formation_type': formation_analysis.formation_type.value,
        'linearity': formation_analysis.linearity,
        'planarity': formation_analysis.planarity,
        'principal_components': formation_analysis.principal_components,
        'principal_directions': formation_analysis.principal_directions,
        'separations': formation_analysis.separations,
        'orderings': orderings,
        'positions_centered': pos_array - formation_center
    }


def compare_formations(data1, data2):
    """Compare formation geometries between two dates"""
    
    print(f"\n🔄 Comparing formations between {data1['date']} and {data2['date']}...")
    
    analysis1 = analyze_formation_geometry(data1)
    analysis2 = analyze_formation_geometry(data2)
    
    # Compare orderings
    print(f"\n📊 Ordering Comparison:")
    print(f"{'Coordinate':<12} {'2019-01-26':<20} {'2019-01-27':<20} {'Match?'}")
    print("-" * 70)
    
    ordering_matches = {}
    for coord_sys in analysis1['orderings']:
        order1 = analysis1['orderings'][coord_sys]
        order2 = analysis2['orderings'][coord_sys]
        match = order1 == order2
        ordering_matches[coord_sys] = match
        
        order1_str = ' → '.join([f'MMS{p}' for p in order1])
        order2_str = ' → '.join([f'MMS{p}' for p in order2])
        match_str = "✅" if match else "❌"
        
        print(f"{coord_sys:<12} {order1_str:<20} {order2_str:<20} {match_str}")
    
    # Identify potential issues
    print(f"\n🔍 Potential Issues:")
    
    if not any(ordering_matches.values()):
        print("   ❌ NO orderings match - major coordinate system issue!")
    elif ordering_matches['X_GSM'] and ordering_matches['Y_GSM'] and ordering_matches['Z_GSM']:
        print("   ✅ All GSM coordinate orderings match - formations are consistent")
    else:
        print("   ⚠️ Some orderings differ - investigate coordinate transformations")
    
    # Formation geometry comparison
    volume_diff = abs(analysis1['formation_volume'] - analysis2['formation_volume'])
    distance_diff = abs(analysis1['distance_from_earth'] - analysis2['distance_from_earth'])
    
    print(f"\n📏 Geometry Comparison:")
    print(f"   Volume difference: {volume_diff:.0f} km³")
    print(f"   Distance difference: {distance_diff:.1f} RE")
    
    if volume_diff > 50000:  # 50,000 km³
        print("   ⚠️ Large volume difference - formations may be significantly different")
    
    if distance_diff > 1.0:  # 1 RE
        print("   ⚠️ Large distance difference - spacecraft at very different locations")
    
    return {
        'analysis1': analysis1,
        'analysis2': analysis2,
        'ordering_matches': ordering_matches,
        'volume_difference': volume_diff,
        'distance_difference': distance_diff
    }


def create_comparison_plot(data1, data2, comparison):
    """Create visualization comparing the two formations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('MMS Spacecraft Formation Comparison\n2019-01-26 15:00 UT vs 2019-01-27 12:30:50 UT', 
                 fontsize=14, fontweight='bold')
    
    colors = ['red', 'blue', 'green', 'orange']
    labels = ['MMS1', 'MMS2', 'MMS3', 'MMS4']
    
    # Plot 1: XY projection for 2019-01-26
    ax = axes[0, 0]
    positions1 = data1['positions']
    for i, probe in enumerate(['1', '2', '3', '4']):
        pos = positions1[probe]
        ax.scatter(pos[0]/1000, pos[1]/1000, c=colors[i], s=100, label=labels[i])
        ax.annotate(f'MMS{probe}', (pos[0]/1000, pos[1]/1000), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('X (1000 km)')
    ax.set_ylabel('Y (1000 km)')
    ax.set_title('2019-01-26 15:00 UT (XY)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    
    # Plot 2: XY projection for 2019-01-27
    ax = axes[0, 1]
    positions2 = data2['positions']
    for i, probe in enumerate(['1', '2', '3', '4']):
        pos = positions2[probe]
        ax.scatter(pos[0]/1000, pos[1]/1000, c=colors[i], s=100, label=labels[i])
        ax.annotate(f'MMS{probe}', (pos[0]/1000, pos[1]/1000), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('X (1000 km)')
    ax.set_ylabel('Y (1000 km)')
    ax.set_title('2019-01-27 12:30:50 UT (XY)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    
    # Plot 3: XZ projection for 2019-01-26
    ax = axes[1, 0]
    for i, probe in enumerate(['1', '2', '3', '4']):
        pos = positions1[probe]
        ax.scatter(pos[0]/1000, pos[2]/1000, c=colors[i], s=100, label=labels[i])
        ax.annotate(f'MMS{probe}', (pos[0]/1000, pos[2]/1000), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('X (1000 km)')
    ax.set_ylabel('Z (1000 km)')
    ax.set_title('2019-01-26 15:00 UT (XZ)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot 4: XZ projection for 2019-01-27
    ax = axes[1, 1]
    for i, probe in enumerate(['1', '2', '3', '4']):
        pos = positions2[probe]
        ax.scatter(pos[0]/1000, pos[2]/1000, c=colors[i], s=100, label=labels[i])
        ax.annotate(f'MMS{probe}', (pos[0]/1000, pos[2]/1000), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('X (1000 km)')
    ax.set_ylabel('Z (1000 km)')
    ax.set_title('2019-01-27 12:30:50 UT (XZ)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('mms_formation_comparison_2019_01_26_vs_27.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to debug spacecraft ordering differences"""
    
    print("MMS SPACECRAFT ORDERING DEBUG: 2019-01-26 vs 2019-01-27")
    print("=" * 80)
    
    # Load spacecraft positions for both dates
    data_26 = load_spacecraft_positions_for_date('2019-01-26', '15:00:00')
    data_27 = load_spacecraft_positions_for_date('2019-01-27', '12:30:50')
    
    # Compare formations
    comparison = compare_formations(data_26, data_27)
    
    # Create visualization
    create_comparison_plot(data_26, data_27, comparison)
    
    # Summary and recommendations
    print(f"\n" + "=" * 80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)

    # Check formation types
    formation1 = comparison['analysis1']['formation_type']
    formation2 = comparison['analysis2']['formation_type']

    print(f"Formation Types:")
    print(f"  2019-01-26: {formation1}")
    print(f"  2019-01-27: {formation2}")

    if formation1 == formation2:
        print("✅ Formation types are CONSISTENT between dates")
    else:
        print("⚠️ Formation types are DIFFERENT between dates")
        print("   This may explain ordering differences!")

    if comparison['ordering_matches']['X_GSM'] and comparison['ordering_matches']['Y_GSM']:
        print("✅ Spacecraft orderings are CONSISTENT between the two dates")
        print("   The analysis is working correctly.")
    else:
        print("❌ Spacecraft orderings are INCONSISTENT between the two dates")
        print("   Potential issues to investigate:")
        print("   1. Different formation types detected")
        print("   2. Different coordinate systems being used")
        print("   3. Time-dependent coordinate transformations")
        print("   4. Different analysis methods or reference frames")
        print("   5. Bugs in position loading or processing")
        print("   6. Different LMN coordinate system calculations")

        # Specific recommendations based on detected formation types
        if formation1 == "string_of_pearls" or formation2 == "string_of_pearls":
            print("\n📝 STRING-OF-PEARLS FORMATION DETECTED:")
            print("   • Use string-specific analysis methods")
            print("   • Focus on ordering along principal axis")
            print("   • Avoid assuming tetrahedral geometry")

    print(f"\nGenerated: mms_formation_comparison_2019_01_26_vs_27.png")


if __name__ == "__main__":
    main()
