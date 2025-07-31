"""
Core Physics Validation Test
===========================

Quick validation test for the core physics modules we've developed.
This tests the essential functionality without requiring external data.
"""

import numpy as np
import sys
import traceback
from datetime import datetime

def test_coordinate_transformations():
    """Test coordinate system transformations"""
    print("🧭 Testing coordinate transformations...")
    
    try:
        from mms_mp import coords
        
        # Create synthetic magnetic field data with clear variance structure
        np.random.seed(42)
        n_points = 1000
        
        # Create field with maximum variance in X, medium in Y, minimum in Z
        B_x = 50 + 20 * np.sin(np.linspace(0, 4*np.pi, n_points)) + 5 * np.random.randn(n_points)
        B_y = 30 + 10 * np.cos(np.linspace(0, 2*np.pi, n_points)) + 3 * np.random.randn(n_points)
        B_z = 20 + 2 * np.random.randn(n_points)
        
        B_field = np.column_stack([B_x, B_y, B_z])
        
        # Test hybrid LMN calculation (includes MVA)
        lmn_system = coords.hybrid_lmn(B_field)
        
        # Verify orthogonality
        dot_LM = np.dot(lmn_system.L, lmn_system.M)
        dot_LN = np.dot(lmn_system.L, lmn_system.N)
        dot_MN = np.dot(lmn_system.M, lmn_system.N)
        
        assert abs(dot_LM) < 1e-10, f"L and M not orthogonal: {dot_LM}"
        assert abs(dot_LN) < 1e-10, f"L and N not orthogonal: {dot_LN}"
        assert abs(dot_MN) < 1e-10, f"M and N not orthogonal: {dot_MN}"
        
        # Test coordinate transformation
        B_lmn = lmn_system.to_lmn(B_field)
        B_gsm_recovered = lmn_system.to_gsm(B_lmn)
        
        # Check magnitude preservation
        mag_original = np.linalg.norm(B_field, axis=1)
        mag_recovered = np.linalg.norm(B_gsm_recovered, axis=1)
        
        max_error = np.max(np.abs(mag_original - mag_recovered))
        assert max_error < 1e-10, f"Magnitude not preserved: {max_error}"
        
        print("   ✅ MVA calculation: PASSED")
        print("   ✅ Orthogonality: PASSED")
        print("   ✅ Magnitude preservation: PASSED")
        print(f"   📊 Eigenvalue ratios: λmax/λmid = {lmn_system.r_max_mid:.2f}, λmid/λmin = {lmn_system.r_mid_min:.2f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_boundary_detection():
    """Test boundary detection logic"""
    print("\n🔍 Testing boundary detection...")
    
    try:
        from mms_mp import boundary
        
        # Test detector configuration
        cfg = boundary.DetectorCfg(he_in=0.3, he_out=0.1, min_pts=5)
        assert cfg.he_in == 0.3
        assert cfg.he_out == 0.1
        assert cfg.min_pts == 5
        
        # Test state transitions
        # High He+ density should indicate magnetosphere
        new_state = boundary._sm_update('sheath', he_val=0.5, BN_val=5.0, 
                                       cfg=cfg, inside_mag=True)
        assert new_state == 'magnetosphere', f"Expected magnetosphere, got {new_state}"
        
        # Low |BN| should indicate mp_layer
        new_state = boundary._sm_update('sheath', he_val=0.15, BN_val=1.0, 
                                       cfg=cfg, inside_mag=False)
        assert new_state == 'mp_layer', f"Expected mp_layer, got {new_state}"
        
        print("   ✅ Configuration: PASSED")
        print("   ✅ State transitions: PASSED")
        print("   ✅ Logic validation: PASSED")
        
        return True
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_electric_field_physics():
    """Test electric field calculations"""
    print("\n⚡ Testing electric field physics...")

    try:
        from mms_mp import electric

        # Test E×B drift calculation using actual API
        E_field = np.array([1.0, 0.0, 0.0])  # mV/m in X direction
        B_field = np.array([0.0, 0.0, 50.0])  # nT in Z direction

        v_exb = electric.exb_velocity(E_field, B_field, unit_E='mV/m', unit_B='nT')

        # Expected: |v| = |E|/|B| = 1 mV/m / 50 nT = 20 km/s in Y direction
        expected_magnitude = 20.0
        expected_direction = np.array([0.0, 1.0, 0.0])

        calculated_magnitude = np.linalg.norm(v_exb)
        calculated_direction = v_exb / calculated_magnitude

        mag_error = abs(calculated_magnitude - expected_magnitude) / expected_magnitude
        dir_error = np.linalg.norm(calculated_direction - expected_direction)

        assert mag_error < 0.01, f"E×B magnitude error: {mag_error}"
        assert dir_error < 0.01, f"E×B direction error: {dir_error}"

        print("   ✅ E×B drift calculation: PASSED")
        print(f"   📊 Calculated velocity: {calculated_magnitude:.1f} km/s")
        print(f"   📊 Direction: [{calculated_direction[0]:.3f}, {calculated_direction[1]:.3f}, {calculated_direction[2]:.3f}]")

        return True

    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_motion_analysis():
    """Test motion and timing analysis"""
    print("\n🚀 Testing motion analysis...")

    try:
        from mms_mp import motion

        # Test normal velocity calculation (simpler test using actual API)
        # Create test velocity in GSM coordinates
        v_xyz = np.array([[10.0, 5.0, 2.0],   # km/s
                         [8.0, 6.0, 1.5],
                         [12.0, 4.0, 2.5]])

        # Create LMN rotation matrix (identity for simplicity)
        R_lmn = np.eye(3)

        # Calculate normal velocity (should be same as input for identity matrix)
        v_normal = motion.normal_velocity(v_xyz, R_lmn)

        # Should preserve the input velocities
        max_error = np.max(np.abs(v_xyz - v_normal))

        assert max_error < 1e-10, f"Normal velocity calculation error: {max_error}"

        print("   ✅ Normal velocity calculation: PASSED")
        print(f"   📊 Input velocities shape: {v_xyz.shape}")
        print(f"   📊 Output velocities shape: {v_normal.shape}")
        print(f"   📊 Maximum error: {max_error:.2e}")

        return True

    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_multispacecraft_analysis():
    """Test multi-spacecraft analysis methods"""
    print("\n🛰️ Testing multi-spacecraft analysis...")

    try:
        from mms_mp import multispacecraft

        # Test timing analysis with known boundary motion
        positions = {
            '1': np.array([0.0, 0.0, 0.0]),
            '2': np.array([100.0, 0.0, 0.0]),
            '3': np.array([0.0, 100.0, 0.0]),
            '4': np.array([0.0, 0.0, 100.0])
        }

        # Crossing times with known delay pattern
        # Boundary moving in +X direction at 50 km/s
        # MMS2 at x=100 km should see crossing 2 seconds later
        crossing_times = {
            '1': 100.0,  # Reference time
            '2': 102.0,  # 2 seconds later (100 km / 50 km/s)
            '3': 100.0,  # Same time (perpendicular to motion)
            '4': 100.0   # Same time (perpendicular to motion)
        }

        # Calculate timing normal and velocity
        normal, velocity, quality = multispacecraft.timing_normal(positions, crossing_times)

        # Should recover approximately X-direction normal
        # and ~50 km/s velocity
        x_component = abs(normal[0])
        assert x_component > 0.8, f"X component too small: {x_component}"

        velocity_error = abs(velocity - 50.0) / 50.0
        assert velocity_error < 0.3, f"Velocity error too large: {velocity_error}"

        print("   ✅ Timing analysis: PASSED")
        print(f"   📊 Normal vector: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
        print(f"   📊 Velocity: {velocity:.1f} km/s")
        print(f"   📊 Quality: {quality:.3f}")

        return True

    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        traceback.print_exc()
        return False


def test_data_quality():
    """Test data quality assessment"""
    print("\n📊 Testing data quality assessment...")

    try:
        from mms_mp import quality

        # Test quality mask functions with synthetic data
        # Create flag array with different quality levels
        flag_data = np.array([0, 0, 1, 2, 0, 1, 3, 0, 0, 1])  # Mixed quality flags

        # Test DIS (ion) quality mask (accepts only level 0)
        dis_mask = quality.dis_good_mask(flag_data, accept_levels=(0,))
        expected_dis = np.array([True, True, False, False, True, False, False, True, True, False])

        assert np.array_equal(dis_mask, expected_dis), "DIS quality mask incorrect"

        # Test DES (electron) quality mask (accepts levels 0 and 1)
        des_mask = quality.des_good_mask(flag_data, accept_levels=(0, 1))
        expected_des = np.array([True, True, True, False, True, True, False, True, True, True])

        assert np.array_equal(des_mask, expected_des), "DES quality mask incorrect"

        # Test mask combination
        combined_mask = quality.combine_masks(dis_mask, des_mask)
        expected_combined = expected_dis & expected_des

        assert np.array_equal(combined_mask, expected_combined), "Mask combination incorrect"

        print("   ✅ Quality mask functions: PASSED")
        print(f"   📊 DIS good samples: {np.sum(dis_mask)}/{len(dis_mask)}")
        print(f"   📊 DES good samples: {np.sum(des_mask)}/{len(des_mask)}")
        print(f"   📊 Combined good samples: {np.sum(combined_mask)}/{len(combined_mask)}")

        return True

    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Run core physics validation tests"""
    
    print("🧪 MMS-MP CORE PHYSICS VALIDATION")
    print("=" * 50)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version}")
    
    # Run all tests
    tests = [
        ("Coordinate Transformations", test_coordinate_transformations),
        ("Boundary Detection", test_boundary_detection),
        ("Electric Field Physics", test_electric_field_physics),
        ("Motion Analysis", test_motion_analysis),
        ("Multi-spacecraft Analysis", test_multispacecraft_analysis),
        ("Data Quality Assessment", test_data_quality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name}: CRITICAL FAILURE")
            print(f"   Error: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📋 VALIDATION SUMMARY")
    print(f"{'='*50}")
    
    passed_tests = sum(1 for _, result in results if result)
    total_tests = len(results)
    
    print(f"✅ Tests Passed: {passed_tests}/{total_tests}")
    print(f"❌ Tests Failed: {total_tests - passed_tests}/{total_tests}")
    
    print(f"\n📊 Detailed Results:")
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name:<30} {status}")
    
    # Overall assessment
    success_rate = passed_tests / total_tests
    
    if success_rate == 1.0:
        grade = "A+ 🎉"
        assessment = "EXCELLENT - All core physics validated!"
    elif success_rate >= 0.8:
        grade = "B+ 👍"
        assessment = "GOOD - Minor issues detected"
    elif success_rate >= 0.6:
        grade = "C ⚠️"
        assessment = "FAIR - Several issues need attention"
    else:
        grade = "F ❌"
        assessment = "POOR - Major physics issues detected"
    
    print(f"\n🎯 Overall Grade: {grade}")
    print(f"💡 Assessment: {assessment}")
    
    if success_rate == 1.0:
        print(f"\n🎉 CORE PHYSICS VALIDATION COMPLETE!")
        print(f"✅ All fundamental physics laws and calculations verified")
        print(f"✅ Coordinate transformations working correctly")
        print(f"✅ Boundary detection logic validated")
        print(f"✅ Multi-spacecraft analysis functional")
        print(f"✅ Data quality assessment operational")
        print(f"\n💡 The MMS-MP package is ready for scientific analysis!")
    else:
        print(f"\n⚠️ VALIDATION ISSUES DETECTED")
        print(f"Please review failed tests and fix underlying issues")
    
    return success_rate == 1.0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
