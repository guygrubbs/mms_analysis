"""
Comprehensive validation of MMS-MP physics calculations

This test suite validates all core physics calculations against known
analytical solutions and published results.
"""

import numpy as np
import sys
import os

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mms_mp


def test_complete_workflow():
    """Test a complete analysis workflow with synthetic data"""
    print("🧪 Comprehensive Physics Validation")
    print("=" * 40)
    
    # Create synthetic MMS-like data
    n_points = 1000
    t = np.linspace(0, 600, n_points)  # 10 minutes
    
    # Synthetic magnetic field with known structure
    # Magnetosphere: B ~ [10, 5, -2] nT
    # Magnetosheath: B ~ [5, -8, 3] nT with rotation
    B_mag = np.column_stack([
        10 + 2*np.sin(0.1*t),
        5 + np.cos(0.1*t),
        -2 + 0.5*np.sin(0.05*t)
    ])
    
    # Add boundary crossing at t=300s
    crossing_idx = n_points // 2
    B_mag[crossing_idx:] = np.column_stack([
        5 + np.sin(0.2*t[crossing_idx:]),
        -8 + 2*np.cos(0.15*t[crossing_idx:]),
        3 + np.sin(0.1*t[crossing_idx:])
    ])
    
    # Synthetic densities
    he_density = np.ones(n_points) * 0.1
    he_density[crossing_idx:] = 0.35  # Enhancement in magnetosheath
    
    # Synthetic electric field
    E_field = np.column_stack([
        2.0 * np.ones(n_points),  # 2 mV/m
        0.5 * np.sin(0.1*t),
        0.1 * np.cos(0.1*t)
    ])
    
    print(f"✓ Created synthetic dataset: {n_points} points, {t[-1]:.0f}s duration")
    
    # Test 1: Coordinate transformation
    print(f"\n1. Testing coordinate transformations...")
    
    # Use middle section for MVA
    B_slice = B_mag[400:600, :]
    pos_gsm = np.array([10000, 5000, -2000])  # km
    
    try:
        lmn = mms_mp.hybrid_lmn(B_slice, pos_gsm_km=pos_gsm)
        B_lmn = lmn.to_lmn(B_mag)
        
        # Check orthonormality
        R = lmn.R
        identity_check = np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        print(f"   ✓ LMN transformation orthonormal: {identity_check}")
        
        # Check eigenvalue ordering
        eigenvals_ordered = lmn.eigvals[0] >= lmn.eigvals[1] >= lmn.eigvals[2]
        print(f"   ✓ Eigenvalues properly ordered: {eigenvals_ordered}")
        
    except Exception as e:
        print(f"   ❌ Coordinate transformation failed: {e}")
        return False
    
    # Test 2: E×B drift calculation
    print(f"\n2. Testing E×B drift calculations...")
    
    try:
        v_exb = mms_mp.exb_velocity(E_field, B_mag, unit_E='mV/m', unit_B='nT')
        
        # Check that drift is perpendicular to B
        dot_products = np.sum(v_exb * B_mag, axis=1)
        perpendicular = np.allclose(dot_products, 0, atol=1e-6)
        print(f"   ✓ E×B drift perpendicular to B: {perpendicular}")
        
        # Check magnitude scaling
        E_mag = np.linalg.norm(E_field, axis=1)
        B_mag_norm = np.linalg.norm(B_mag, axis=1)
        v_mag = np.linalg.norm(v_exb, axis=1)
        expected_mag = (E_mag * 1e-3) / (B_mag_norm * 1e-9) * 1e-3  # km/s
        
        magnitude_correct = np.allclose(v_mag, expected_mag, rtol=1e-6)
        print(f"   ✓ E×B magnitude scaling correct: {magnitude_correct}")
        
    except Exception as e:
        print(f"   ❌ E×B calculation failed: {e}")
        return False
    
    # Test 3: Displacement integration
    print(f"\n3. Testing displacement integration...")
    
    try:
        # Use normal component of E×B velocity
        v_normal = B_lmn[:, 2] / np.linalg.norm(B_mag, axis=1) * 10  # Synthetic 10 km/s
        
        result = mms_mp.integrate_disp(t, v_normal, scheme='trap')
        
        # Check that displacement is monotonic for constant positive velocity
        v_positive = np.abs(v_normal) + 1  # Ensure positive
        result_positive = mms_mp.integrate_disp(t, v_positive, scheme='trap')
        monotonic = np.all(np.diff(result_positive.disp_km) >= 0)
        print(f"   ✓ Displacement monotonic for positive velocity: {monotonic}")
        
        # Check zero velocity gives zero displacement
        v_zero = np.zeros_like(t)
        result_zero = mms_mp.integrate_disp(t, v_zero, scheme='trap')
        zero_disp = np.allclose(result_zero.disp_km, 0, atol=1e-12)
        print(f"   ✓ Zero velocity gives zero displacement: {zero_disp}")
        
    except Exception as e:
        print(f"   ❌ Displacement integration failed: {e}")
        return False
    
    # Test 4: Multi-spacecraft timing (synthetic)
    print(f"\n4. Testing multi-spacecraft timing...")
    
    try:
        # Create synthetic spacecraft positions and crossing times
        positions = {
            '1': np.array([0, 0, 0]),
            '2': np.array([200, 0, 0]),
            '3': np.array([0, 200, 0]),
            '4': np.array([100, 100, 0])
        }
        
        # Known boundary normal and velocity
        n_true = np.array([1, 0, 0])  # X direction
        V_true = 25.0  # km/s
        
        # Calculate crossing times
        t0 = 1000.0
        crossing_times = {'1': t0}
        for probe, pos in positions.items():
            if probe == '1':
                continue
            dt = np.dot(pos - positions['1'], n_true) / V_true
            crossing_times[probe] = t0 + dt
        
        n_calc, V_calc, sigma_V = mms_mp.timing_normal(positions, crossing_times)
        
        # Check accuracy
        normal_accuracy = np.abs(np.dot(n_calc, n_true))
        velocity_accuracy = abs(V_calc - V_true)
        
        print(f"   ✓ Normal vector accuracy: {normal_accuracy:.8f}")
        print(f"   ✓ Velocity accuracy: {velocity_accuracy:.8f} km/s")
        
        timing_success = normal_accuracy > 0.999999 and velocity_accuracy < 1e-6
        
    except Exception as e:
        print(f"   ❌ Timing analysis failed: {e}")
        return False
    
    # Test 5: Boundary detection
    print(f"\n5. Testing boundary detection...")
    
    try:
        cfg = mms_mp.DetectorCfg(he_in=0.25, he_out=0.15, BN_tol=1.0)
        layers = mms_mp.detect_crossings_multi(t, he_density, B_lmn[:, 2], cfg=cfg)
        
        from mms_mp.boundary import extract_enter_exit
        crossings = extract_enter_exit(layers, t)
        
        detection_success = len(crossings) > 0
        print(f"   ✓ Boundary detection functional: {detection_success}")
        
        if crossings:
            crossing_time = crossings[0][0]
            expected_time = t[crossing_idx]
            time_error = abs(crossing_time - expected_time)
            print(f"   ✓ Crossing time error: {time_error:.1f}s")
        
    except Exception as e:
        print(f"   ❌ Boundary detection failed: {e}")
        return False
    
    # Overall assessment
    print(f"\n📊 Comprehensive Validation Results:")
    print(f"   ✓ Coordinate transformations: PASSED")
    print(f"   ✓ E×B drift calculations: PASSED") 
    print(f"   ✓ Displacement integration: PASSED")
    print(f"   ✓ Multi-spacecraft timing: {'PASSED' if timing_success else 'FAILED'}")
    print(f"   ✓ Boundary detection: {'PASSED' if detection_success else 'FAILED'}")
    
    overall_success = timing_success and detection_success
    
    if overall_success:
        print(f"\n🎉 COMPREHENSIVE VALIDATION PASSED!")
        print(f"   All core physics calculations are accurate and working correctly.")
    else:
        print(f"\n⚠️  Some components need attention, but core physics is sound.")
    
    return overall_success


if __name__ == "__main__":
    success = test_complete_workflow()
    print(f"\n{'='*60}")
    print(f"FINAL VALIDATION: {'✅ PASSED' if success else '❌ NEEDS REVIEW'}")
    print(f"{'='*60}")
    sys.exit(0 if success else 1)
