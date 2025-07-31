"""
Test MMS Formation Configurations
=================================

This script tests both tetrahedral and string formations used by MMS
during different mission phases and science campaigns.

Formation Types:
1. Tetrahedral: For magnetopause boundary studies, reconnection regions
2. String: For reconnection diffusion regions, current sheet studies
"""

import numpy as np
import sys
import traceback
from datetime import datetime

# Import MMS-MP modules
from mms_mp import coords, multispacecraft


def create_formation_positions(formation_type="tetrahedral", separation_km=100):
    """
    Create spacecraft positions for different MMS formations
    
    Parameters:
    -----------
    formation_type : str
        "tetrahedral" or "string"
    separation_km : float
        Characteristic separation distance in km
        
    Returns:
    --------
    dict : Spacecraft positions in GSM coordinates (km)
    """
    
    if formation_type.lower() == "tetrahedral":
        # Tetrahedral formation for 3D gradient measurements
        # Optimized for magnetopause boundary analysis
        positions = {
            '1': np.array([0.0, 0.0, 0.0]),                    # Reference
            '2': np.array([separation_km, 0.0, 0.0]),          # X direction
            '3': np.array([separation_km/2, separation_km*0.866, 0.0]),  # 60° in XY plane
            '4': np.array([separation_km/2, separation_km*0.289, separation_km*0.816])  # Above plane
        }
        
    elif formation_type.lower() == "string":
        # String formation for reconnection studies
        # Optimized for temporal vs spatial disambiguation
        positions = {
            '1': np.array([0.0, 0.0, 0.0]),                    # Reference
            '2': np.array([separation_km*0.5, 0.0, 0.0]),      # 0.5 separation
            '3': np.array([separation_km*1.0, 0.0, 0.0]),      # 1.0 separation
            '4': np.array([separation_km*1.5, 0.0, 0.0])       # 1.5 separation
        }
        
    else:
        raise ValueError(f"Unknown formation type: {formation_type}")
    
    return positions


def test_formation_characteristics():
    """Test characteristics of different MMS formations"""
    print("Testing MMS formation characteristics...")
    
    formations_to_test = [
        ("tetrahedral", 100),  # 100 km tetrahedral
        ("tetrahedral", 50),   # 50 km tetrahedral (closer)
        ("string", 100),       # 100 km string
        ("string", 200),       # 200 km string (wider)
    ]
    
    results = {}
    
    for formation_type, separation in formations_to_test:
        print(f"\n   Testing {formation_type} formation ({separation} km separation):")
        
        try:
            positions = create_formation_positions(formation_type, separation)
            
            # Calculate formation metrics
            separations = []
            for i, probe1 in enumerate(['1', '2', '3', '4']):
                for j, probe2 in enumerate(['1', '2', '3', '4']):
                    if i < j:
                        sep = np.linalg.norm(positions[probe1] - positions[probe2])
                        separations.append(sep)
            
            max_sep = np.max(separations)
            min_sep = np.min(separations)
            mean_sep = np.mean(separations)
            
            # Calculate formation volume
            r1, r2, r3, r4 = [positions[p] for p in ['1', '2', '3', '4']]
            matrix = np.array([r2-r1, r3-r1, r4-r1])
            volume = abs(np.linalg.det(matrix)) / 6.0
            
            # Formation quality metrics
            if formation_type == "tetrahedral":
                # Tetrahedrality: how close to perfect tetrahedron
                edge_std = np.std(separations)
                quality = 1.0 / (1.0 + edge_std / mean_sep)
                quality_name = "Tetrahedrality"
                
            else:  # string
                # Linearity: how well spacecraft lie on a line
                positions_array = np.array([positions[p] for p in ['1', '2', '3', '4']])
                centroid = np.mean(positions_array, axis=0)
                centered = positions_array - centroid
                _, s, _ = np.linalg.svd(centered)
                quality = s[0] / np.sum(s)  # Ratio of first singular value
                quality_name = "Linearity"
            
            results[f"{formation_type}_{separation}"] = {
                'max_separation': max_sep,
                'min_separation': min_sep,
                'mean_separation': mean_sep,
                'volume': volume,
                'quality': quality,
                'quality_name': quality_name
            }
            
            print(f"      Max separation: {max_sep:.1f} km")
            print(f"      Min separation: {min_sep:.1f} km")
            print(f"      Formation volume: {volume:.0f} km³")
            print(f"      {quality_name}: {quality:.3f}")
            
            # Validate formation
            assert max_sep > 0, "Invalid formation - zero separation"
            assert volume >= 0, "Invalid formation - negative volume"
            assert 0 <= quality <= 1, f"Invalid quality metric: {quality}"
            
            if formation_type == "tetrahedral":
                assert volume > 1000, f"Tetrahedral volume too small: {volume:.0f} km³"
                assert quality > 0.5, f"Poor tetrahedrality: {quality:.3f}"
            else:  # string
                assert volume < 10000, f"String volume too large: {volume:.0f} km³"
                assert quality > 0.8, f"Poor linearity: {quality:.3f}"
            
        except Exception as e:
            print(f"      FAIL: {e}")
            return False
    
    print(f"\n   OK: All formation configurations validated")
    return True


def test_timing_analysis_by_formation():
    """Test timing analysis performance for different formations"""
    print("\nTesting timing analysis by formation type...")
    
    formations = [
        ("tetrahedral", 100),
        ("string", 150)
    ]
    
    for formation_type, separation in formations:
        print(f"\n   Testing {formation_type} formation timing analysis:")
        
        try:
            positions = create_formation_positions(formation_type, separation)
            
            # Test different boundary orientations
            if formation_type == "tetrahedral":
                # For tetrahedral, test multiple orientations
                test_normals = [
                    np.array([1.0, 0.0, 0.0]),      # X-normal
                    np.array([0.0, 1.0, 0.0]),      # Y-normal
                    np.array([0.577, 0.577, 0.577]) # Diagonal
                ]
            else:  # string
                # For string, test orientations that provide good timing resolution
                # String is along X, so use oblique angles to get time delays
                test_normals = [
                    np.array([0.866, 0.5, 0.0]),    # 30° from X in XY plane
                    np.array([0.707, 0.0, 0.707]),  # 45° from X in XZ plane
                    np.array([0.577, 0.577, 0.577]) # Diagonal (equal components)
                ]
            
            timing_results = []
            
            for i, boundary_normal in enumerate(test_normals):
                boundary_velocity = 50.0  # km/s
                base_time = 1000.0
                
                # Calculate crossing times
                crossing_times = {}
                for probe, pos in positions.items():
                    projection = np.dot(pos, boundary_normal)
                    delay = projection / boundary_velocity
                    crossing_times[probe] = base_time + delay
                
                # Check time delay spread
                delays = [crossing_times[p] - base_time for p in ['1', '2', '3', '4']]
                delay_spread = max(delays) - min(delays)
                
                if delay_spread < 0.01:  # Less than 0.01 second
                    print(f"      Orientation {i+1}: Poor timing resolution ({delay_spread:.4f}s)")
                    continue
                
                # Perform timing analysis
                normal, velocity, quality = multispacecraft.timing_normal(positions, crossing_times)
                
                # Calculate errors
                normal_error = np.linalg.norm(normal - boundary_normal)
                velocity_error = abs(velocity - boundary_velocity) / boundary_velocity
                
                timing_results.append({
                    'orientation': i+1,
                    'delay_spread': delay_spread,
                    'normal_error': normal_error,
                    'velocity_error': velocity_error,
                    'quality': quality
                })
                
                print(f"      Orientation {i+1}: delay_spread={delay_spread:.3f}s, "
                      f"normal_error={normal_error:.3f}, velocity_error={velocity_error:.3f}")
            
            # Validate that at least one orientation works well
            good_orientations = [r for r in timing_results if r['normal_error'] < 0.5 and r['velocity_error'] < 0.5]
            
            assert len(good_orientations) > 0, f"No good timing orientations for {formation_type}"
            
            best_result = min(good_orientations, key=lambda x: x['normal_error'])
            print(f"      Best orientation: {best_result['orientation']} "
                  f"(normal_error={best_result['normal_error']:.3f})")
            
        except Exception as e:
            print(f"      FAIL: {e}")
            traceback.print_exc()
            return False
    
    print(f"\n   OK: Timing analysis validated for all formations")
    return True


def test_science_applications():
    """Test formation suitability for different science applications"""
    print("\nTesting formation suitability for science applications...")
    
    science_cases = [
        {
            'name': 'Magnetopause Boundary Analysis',
            'preferred_formation': 'tetrahedral',
            'min_separation': 50,   # km
            'max_separation': 200,  # km
            'requires_3d': True
        },
        {
            'name': 'Reconnection Diffusion Region',
            'preferred_formation': 'string',
            'min_separation': 100,  # km
            'max_separation': 500,  # km
            'requires_3d': False
        },
        {
            'name': 'Current Sheet Structure',
            'preferred_formation': 'string',
            'min_separation': 50,   # km
            'max_separation': 300,  # km
            'requires_3d': False
        }
    ]
    
    for case in science_cases:
        print(f"\n   Testing: {case['name']}")
        
        formation_type = case['preferred_formation']
        test_separation = (case['min_separation'] + case['max_separation']) / 2
        
        try:
            positions = create_formation_positions(formation_type, test_separation)
            
            # Calculate formation volume
            r1, r2, r3, r4 = [positions[p] for p in ['1', '2', '3', '4']]
            matrix = np.array([r2-r1, r3-r1, r4-r1])
            volume = abs(np.linalg.det(matrix)) / 6.0
            
            # Check 3D capability
            has_3d_capability = volume > 1000  # km³
            
            if case['requires_3d']:
                assert has_3d_capability, f"Insufficient 3D capability for {case['name']}"
                print(f"      3D capability: YES (volume={volume:.0f} km³)")
            else:
                print(f"      3D capability: {'YES' if has_3d_capability else 'NO'} (volume={volume:.0f} km³)")
            
            # Check separation range
            separations = []
            for i, probe1 in enumerate(['1', '2', '3', '4']):
                for j, probe2 in enumerate(['1', '2', '3', '4']):
                    if i < j:
                        sep = np.linalg.norm(positions[probe1] - positions[probe2])
                        separations.append(sep)
            
            mean_sep = np.mean(separations)
            assert case['min_separation'] <= mean_sep <= case['max_separation'], \
                   f"Separation {mean_sep:.1f} km outside range [{case['min_separation']}-{case['max_separation']}]"
            
            print(f"      Formation: {formation_type.upper()}")
            print(f"      Mean separation: {mean_sep:.1f} km")
            print(f"      Suitable: YES")
            
        except Exception as e:
            print(f"      FAIL: {e}")
            return False
    
    print(f"\n   OK: All science applications validated")
    return True


def main():
    """Run MMS formation configuration tests"""
    
    print("MMS FORMATION CONFIGURATION VALIDATION")
    print("Testing tetrahedral and string formations")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    tests = [
        ("Formation Characteristics", test_formation_characteristics),
        ("Timing Analysis by Formation", test_timing_analysis_by_formation),
        ("Science Applications", test_science_applications)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 60)
        
        try:
            if test_func():
                passed_tests += 1
                print(f"RESULT: PASSED")
            else:
                print(f"RESULT: FAILED")
        except Exception as e:
            print(f"RESULT: ERROR - {e}")
            traceback.print_exc()
    
    # Final assessment
    print("\n" + "=" * 80)
    print("MMS FORMATION VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {100*passed_tests/total_tests:.1f}%")
    
    if passed_tests == total_tests:
        print("\nALL MMS FORMATION TESTS PASSED!")
        print("Tetrahedral formation validated for magnetopause studies")
        print("String formation validated for reconnection studies")
        print("Timing analysis optimized for both configurations")
        print("Science applications properly matched to formations")
        print("\nMMS FORMATION ANALYSIS READY FOR ALL MISSION PHASES!")
        return True
    else:
        print(f"\nFORMATION VALIDATION ISSUES DETECTED")
        print(f"Success rate: {100*passed_tests/total_tests:.1f}%")
        print("Review failed tests and address issues")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
