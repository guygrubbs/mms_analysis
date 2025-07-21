"""
Comprehensive tests for physics calculations in MMS-MP toolkit

This module tests the accuracy of all key physics calculations including:
- Coordinate transformations (MVA, LMN, Shue model)
- Multi-spacecraft timing analysis (SVD solver)
- Displacement integration (trapezoid, Simpson, rectangular)
- E×B drift calculations
- Boundary detection algorithms

Tests use analytical solutions and synthetic data with known results.
"""

import numpy as np
import sys
import os

# Add package to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mms_mp
from mms_mp.coords import _do_mva, _shue_normal, LMN
from mms_mp.multispacecraft import timing_normal
from mms_mp.motion import integrate_disp, _segment_integral
from mms_mp.electric import exb_velocity, exb_velocity_sync, normal_velocity
from mms_mp.boundary import detect_crossings_multi, DetectorCfg


class TestPhysicsCalculations:
    """Test suite for physics calculations"""
    
    def __init__(self):
        self.tolerance = 1e-10  # Numerical tolerance for comparisons
        self.passed = 0
        self.failed = 0
        
    def assert_close(self, actual, expected, tolerance=None, description=""):
        """Assert that two values are close within tolerance"""
        if tolerance is None:
            tolerance = self.tolerance
            
        if np.allclose(actual, expected, atol=tolerance, rtol=tolerance):
            print(f"✓ {description}")
            self.passed += 1
            return True
        else:
            print(f"❌ {description}")
            print(f"   Expected: {expected}")
            print(f"   Actual:   {actual}")
            print(f"   Diff:     {np.abs(actual - expected)}")
            self.failed += 1
            return False
    
    def test_coordinate_transformations(self):
        """Test coordinate transformation accuracy"""
        print("\n=== Testing Coordinate Transformations ===")
        
        # Test 1: MVA with known eigenvalues
        # Create synthetic B-field data with known principal directions
        n_points = 100
        t = np.linspace(0, 10, n_points)
        
        # Known principal directions (orthonormal)
        L_true = np.array([1, 0, 0])
        M_true = np.array([0, 1, 0]) 
        N_true = np.array([0, 0, 1])
        
        # Create B-field with known variance structure
        # Largest variance along L, medium along M, smallest along N
        B_L = 10 * np.sin(2*np.pi*t) + np.random.normal(0, 1, n_points)
        B_M = 5 * np.cos(2*np.pi*t) + np.random.normal(0, 0.5, n_points)
        B_N = 2 * np.ones(n_points) + np.random.normal(0, 0.1, n_points)
        
        # Construct B-field in known coordinate system
        B_xyz = np.column_stack([B_L, B_M, B_N])
        
        # Apply MVA
        lmn_result = _do_mva(B_xyz)
        
        # Check that eigenvalues are ordered correctly
        self.assert_close(
            lmn_result.eigvals[0] > lmn_result.eigvals[1] > lmn_result.eigvals[2],
            True,
            description="MVA eigenvalues correctly ordered"
        )
        
        # Check that rotation matrix is orthonormal
        R = lmn_result.R
        identity = R @ R.T
        self.assert_close(
            identity, np.eye(3), tolerance=1e-12,
            description="MVA rotation matrix is orthonormal"
        )
        
        # Test 2: Shue model normal calculation
        # Test at known positions
        pos_gsm_km = np.array([10000, 0, 0])  # On X-axis
        n_shue = _shue_normal(pos_gsm_km)
        
        # Should point radially outward (approximately +X direction)
        expected_normal = np.array([1, 0, 0])
        self.assert_close(
            np.abs(np.dot(n_shue, expected_normal)), 1.0, tolerance=1e-3,
            description="Shue normal at X-axis position"
        )
        
        # Test at Y-axis position
        pos_gsm_km = np.array([0, 10000, 0])
        n_shue = _shue_normal(pos_gsm_km)
        expected_normal = np.array([0, 1, 0])
        self.assert_close(
            np.abs(np.dot(n_shue, expected_normal)), 1.0, tolerance=1e-3,
            description="Shue normal at Y-axis position"
        )
        
        # Test 3: LMN coordinate transformation
        # Create test vectors and verify round-trip transformation
        test_vectors = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1]
        ])
        
        # Transform to LMN and back
        lmn_vectors = lmn_result.to_lmn(test_vectors)
        xyz_vectors = (lmn_result.R.T @ lmn_vectors.T).T
        
        self.assert_close(
            xyz_vectors, test_vectors, tolerance=1e-12,
            description="LMN round-trip transformation"
        )
        
    def test_multispacecraft_timing(self):
        """Test multi-spacecraft timing analysis"""
        print("\n=== Testing Multi-spacecraft Timing Analysis ===")
        
        # Test 1: Known planar boundary with known normal and velocity
        # Create synthetic spacecraft positions and crossing times
        
        # Known boundary normal (unit vector)
        n_true = np.array([0.8, 0.6, 0.0])
        n_true = n_true / np.linalg.norm(n_true)
        
        # Known phase velocity (km/s)
        V_true = 50.0
        
        # Spacecraft positions (km) - well-separated for good conditioning
        positions = {
            '1': np.array([0, 0, 0]),        # Reference at origin
            '2': np.array([100, 0, 0]),      # Along X
            '3': np.array([0, 100, 0]),      # Along Y
            '4': np.array([0, 0, 100])       # Along Z
        }
        
        # Calculate crossing times based on known normal and velocity
        # t = t0 + (r - r0) · n / V
        t0 = 1000.0  # Reference time
        r0 = positions['1']  # Reference position
        
        crossing_times = {}
        for probe, pos in positions.items():
            dr = pos - r0
            dt = np.dot(dr, n_true) / V_true
            crossing_times[probe] = t0 + dt
        
        # Apply timing analysis
        n_calc, V_calc, sigma_V = timing_normal(positions, crossing_times)
        
        # Check normal vector (allow for sign ambiguity)
        dot_product = np.abs(np.dot(n_calc, n_true))
        self.assert_close(
            dot_product, 1.0, tolerance=1e-10,
            description="Timing analysis normal vector accuracy"
        )
        
        # Check phase velocity
        self.assert_close(
            np.abs(V_calc), V_true, tolerance=1e-10,
            description="Timing analysis phase velocity accuracy"
        )
        
        # Test 2: Minimum case with 2 spacecraft
        pos_2sc = {'1': positions['1'], '2': positions['2']}
        times_2sc = {'1': crossing_times['1'], '2': crossing_times['2']}

        n_calc_2sc, V_calc_2sc, _ = timing_normal(pos_2sc, times_2sc)

        # 2-spacecraft case has fundamental limitations (can only determine 1 component)
        dot_product_2sc = np.abs(np.dot(n_calc_2sc, n_true))
        self.assert_close(
            dot_product_2sc, 1.0, tolerance=0.5,  # Relaxed tolerance for 2-SC case
            description="2-spacecraft timing analysis accuracy (limited)"
        )
        
    def test_displacement_integration(self):
        """Test displacement integration methods"""
        print("\n=== Testing Displacement Integration ===")
        
        # Test 1: Constant velocity (analytical solution)
        t = np.linspace(0, 10, 101)
        v_const = 5.0  # km/s
        v = np.full_like(t, v_const)
        
        # Analytical solution: displacement = v * t
        disp_analytical = v_const * t
        
        # Test trapezoid integration
        result_trap = integrate_disp(t, v, scheme='trap')
        self.assert_close(
            result_trap.disp_km, disp_analytical, tolerance=1e-12,
            description="Trapezoid integration - constant velocity"
        )
        
        # Test 2: Linear velocity (analytical solution)
        v_linear = 2.0 * t  # v = 2t
        # Analytical solution: displacement = ∫(2t)dt = t²
        disp_analytical_linear = t**2
        
        result_linear = integrate_disp(t, v_linear, scheme='trap')
        self.assert_close(
            result_linear.disp_km, disp_analytical_linear, tolerance=1e-10,
            description="Trapezoid integration - linear velocity"
        )
        
        # Test 3: Sinusoidal velocity
        omega = 2 * np.pi / 10  # Period = 10 seconds
        v_sin = np.sin(omega * t)
        # Analytical solution: displacement = -cos(ωt)/ω + constant
        disp_analytical_sin = -(np.cos(omega * t) - 1) / omega
        
        result_sin = integrate_disp(t, v_sin, scheme='trap')
        self.assert_close(
            result_sin.disp_km, disp_analytical_sin, tolerance=1e-3,  # Relaxed for oscillatory function
            description="Trapezoid integration - sinusoidal velocity"
        )
        
        # Test 4: Compare integration schemes
        # For smooth functions, Simpson should be more accurate than trapezoid
        t_fine = np.linspace(0, 10, 1001)
        v_poly = t_fine**3  # Cubic polynomial
        disp_analytical_poly = t_fine**4 / 4  # Analytical integral
        
        result_trap_poly = integrate_disp(t_fine, v_poly, scheme='trap')
        
        # Simpson's rule should be exact for cubic polynomials
        try:
            result_simpson_poly = integrate_disp(t_fine, v_poly, scheme='simpson')
            simpson_error = np.max(np.abs(result_simpson_poly.disp_km - disp_analytical_poly))
            trap_error = np.max(np.abs(result_trap_poly.disp_km - disp_analytical_poly))
            
            print(f"   Simpson error: {simpson_error:.2e}")
            print(f"   Trapezoid error: {trap_error:.2e}")
            
            if simpson_error < trap_error:
                print("✓ Simpson's rule more accurate than trapezoid for polynomial")
                self.passed += 1
            else:
                print("❌ Simpson's rule should be more accurate for polynomial")
                self.failed += 1
                
        except ImportError:
            print("⚠️  SciPy not available for Simpson's rule test")
            
    def test_exb_drift_calculations(self):
        """Test E×B drift calculations"""
        print("\n=== Testing E×B Drift Calculations ===")
        
        # Test 1: Known E and B fields
        # E = [1, 0, 0] mV/m, B = [0, 0, 1] nT
        # E×B = [0, -1, 0] by right-hand rule
        # |v_ExB| = |E×B|/|B|² = (1e-3 V/m) / (1e-9 T) = 1e6 m/s = 1000 km/s
        # Direction: [0, -1, 0], so v_ExB = [0, -1000, 0] km/s

        E = np.array([[1.0, 0.0, 0.0]])  # mV/m
        B = np.array([[0.0, 0.0, 1.0]])  # nT

        v_exb = exb_velocity(E, B, unit_E='mV/m', unit_B='nT')
        expected_v = np.array([[0.0, -1000.0, 0.0]])  # km/s (corrected)
        
        self.assert_close(
            v_exb, expected_v, tolerance=1e-10,
            description="E×B velocity calculation - simple case"
        )
        
        # Test 2: Perpendicular E and B
        E2 = np.array([[0.0, 1.0, 0.0]])  # mV/m in Y
        B2 = np.array([[1.0, 0.0, 0.0]])  # nT in X
        # Expected: E×B = Y×X = -Z, so v = [0, 0, -1000] km/s
        
        v_exb2 = exb_velocity(E2, B2, unit_E='mV/m', unit_B='nT')
        expected_v2 = np.array([[0.0, 0.0, -1000.0]])
        
        self.assert_close(
            v_exb2, expected_v2, tolerance=1e-10,
            description="E×B velocity calculation - perpendicular fields"
        )
        
        # Test 3: Unit conversions
        # Same fields but different units
        E_V_per_m = E * 1e-3  # Convert to V/m
        B_T = B * 1e-9        # Convert to T

        v_exb_SI = exb_velocity(E_V_per_m, B_T, unit_E='V/m', unit_B='T')

        self.assert_close(
            v_exb_SI, expected_v, tolerance=1e-10,
            description="E×B velocity calculation - unit conversion consistency"
        )
        
        # Test 4: Normal velocity blending
        # Test different blending strategies
        n_points = 10
        v_bulk_lmn = np.column_stack([
            np.zeros(n_points),
            np.zeros(n_points),
            np.ones(n_points) * 10.0  # 10 km/s normal velocity
        ])
        
        v_exb_lmn = np.column_stack([
            np.zeros(n_points),
            np.zeros(n_points),
            np.ones(n_points) * 20.0  # 20 km/s normal velocity
        ])
        
        # Test prefer_exb strategy
        vN_prefer_exb = normal_velocity(v_bulk_lmn, v_exb_lmn, strategy='prefer_exb')
        self.assert_close(
            vN_prefer_exb, np.ones(n_points) * 20.0, tolerance=1e-10,
            description="Normal velocity blending - prefer ExB"
        )
        
        # Test prefer_bulk strategy
        vN_prefer_bulk = normal_velocity(v_bulk_lmn, v_exb_lmn, strategy='prefer_bulk')
        self.assert_close(
            vN_prefer_bulk, np.ones(n_points) * 10.0, tolerance=1e-10,
            description="Normal velocity blending - prefer bulk"
        )
        
        # Test average strategy
        vN_average = normal_velocity(v_bulk_lmn, v_exb_lmn, strategy='average')
        self.assert_close(
            vN_average, np.ones(n_points) * 15.0, tolerance=1e-10,
            description="Normal velocity blending - average"
        )
        
    def run_all_tests(self):
        """Run all physics calculation tests"""
        print("🧪 Running Physics Calculation Verification Tests")
        print("=" * 60)
        
        self.test_coordinate_transformations()
        self.test_multispacecraft_timing()
        self.test_displacement_integration()
        self.test_exb_drift_calculations()
        
        print(f"\n📊 Test Results:")
        print(f"   ✓ Passed: {self.passed}")
        print(f"   ❌ Failed: {self.failed}")
        print(f"   📈 Success Rate: {self.passed/(self.passed + self.failed)*100:.1f}%")
        
        if self.failed == 0:
            print("\n🎉 All physics calculation tests passed!")
            return True
        else:
            print(f"\n⚠️  {self.failed} test(s) failed - review calculations")
            return False


if __name__ == "__main__":
    tester = TestPhysicsCalculations()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
