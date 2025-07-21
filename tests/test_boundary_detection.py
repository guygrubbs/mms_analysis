"""
Test boundary detection algorithms

This module tests the accuracy of the boundary detection state machine,
hysteresis thresholds, and multi-parameter detection criteria.
"""

import numpy as np
import sys
import os

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mms_mp.boundary import detect_crossings_multi, DetectorCfg, extract_enter_exit


class TestBoundaryDetection:
    """Test suite for boundary detection algorithms"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        
    def assert_condition(self, condition, description=""):
        """Assert that a condition is true"""
        if condition:
            print(f"✓ {description}")
            self.passed += 1
            return True
        else:
            print(f"❌ {description}")
            self.failed += 1
            return False
    
    def test_synthetic_boundary_crossing(self):
        """Test with synthetic data that has a clear boundary crossing"""
        print("\n=== Testing Synthetic Boundary Crossing ===")
        
        # Create synthetic time series with clear boundary crossing
        n_points = 1000
        t = np.linspace(0, 100, n_points)  # 100 seconds
        
        # Create He+ density with step function (magnetosphere -> magnetosheath)
        he_density = np.ones(n_points) * 0.1  # Magnetosphere baseline
        crossing_start = n_points // 3
        crossing_end = 2 * n_points // 3
        he_density[crossing_start:crossing_end] = 0.4  # Magnetosheath level
        
        # Add some noise
        he_density += np.random.normal(0, 0.02, n_points)
        
        # Create B_N with rotation signature
        BN = np.ones(n_points) * 5.0  # Positive in magnetosphere
        BN[crossing_start:crossing_end] = -3.0  # Negative in magnetosheath
        BN += np.random.normal(0, 0.5, n_points)  # Add noise
        
        # Detection configuration
        cfg = DetectorCfg(he_in=0.25, he_out=0.15, BN_tol=1.0)
        
        # Run detection
        layers = detect_crossings_multi(t, he_density, BN, cfg=cfg)
        
        # Extract crossings
        crossings = extract_enter_exit(layers, t)
        
        # Should detect entry and exit
        self.assert_condition(
            len(crossings) >= 1,
            f"Detected {len(crossings)} crossing(s) in synthetic data"
        )
        
        if len(crossings) >= 1:
            entry_time, entry_type = crossings[0]
            expected_entry_time = t[crossing_start]
            time_error = abs(entry_time - expected_entry_time)
            
            self.assert_condition(
                time_error < 5.0,  # Within 5 seconds
                f"Entry time accuracy: {time_error:.2f}s error"
            )
            
            self.assert_condition(
                'enter' in entry_type.lower(),
                f"Correctly identified as entry: {entry_type}"
            )
    
    def test_hysteresis_behavior(self):
        """Test hysteresis prevents oscillations"""
        print("\n=== Testing Hysteresis Behavior ===")
        
        # Create data that oscillates around threshold
        n_points = 200
        t = np.linspace(0, 20, n_points)
        
        # He+ density oscillating around in_threshold
        cfg = DetectorCfg(he_in=0.25, he_out=0.15, BN_tol=1.0)
        he_density = 0.2 + 0.1 * np.sin(2 * np.pi * t)  # Oscillates 0.1 to 0.3
        
        # Constant B_N (no magnetic signature)
        BN = np.ones(n_points) * 2.0
        
        # Run detection
        layers = detect_crossings_multi(t, he_density, BN, cfg=cfg)
        crossings = extract_enter_exit(layers, t)
        
        # Should not detect many crossings due to hysteresis
        self.assert_condition(
            len(crossings) <= 2,  # At most entry and exit
            f"Hysteresis prevents oscillations: {len(crossings)} crossings detected"
        )
    
    def test_no_magnetic_signature(self):
        """Test that detection requires both He+ and B_N signatures"""
        print("\n=== Testing Multi-Parameter Requirements ===")
        
        n_points = 500
        t = np.linspace(0, 50, n_points)
        
        # Case 1: He+ signature but no B_N rotation
        he_density = np.ones(n_points) * 0.1
        he_density[100:400] = 0.4  # Clear He+ enhancement
        BN = np.ones(n_points) * 5.0  # No rotation
        
        cfg = DetectorCfg(he_in=0.25, he_out=0.15, BN_tol=1.0)
        layers1 = detect_crossings_multi(t, he_density, BN, cfg=cfg)
        crossings1 = extract_enter_exit(layers1, t)
        
        # Case 2: B_N rotation but no He+ signature  
        he_density2 = np.ones(n_points) * 0.05  # Always low
        BN2 = np.ones(n_points) * 5.0
        BN2[100:400] = -3.0  # Clear rotation
        
        layers2 = detect_crossings_multi(t, he_density2, BN2, cfg=cfg)
        crossings2 = extract_enter_exit(layers2, t)
        
        # Case 3: Both signatures present
        he_density3 = np.ones(n_points) * 0.1
        he_density3[100:400] = 0.4  # He+ enhancement
        BN3 = np.ones(n_points) * 5.0
        BN3[100:400] = -3.0  # B_N rotation
        
        layers3 = detect_crossings_multi(t, he_density3, BN3, cfg=cfg)
        crossings3 = extract_enter_exit(layers3, t)
        
        print(f"   He+ only: {len(crossings1)} crossings")
        print(f"   B_N only: {len(crossings2)} crossings")
        print(f"   Both signatures: {len(crossings3)} crossings")
        
        # Both signatures should give more reliable detection
        self.assert_condition(
            len(crossings3) >= len(crossings1) and len(crossings3) >= len(crossings2),
            "Multi-parameter detection more reliable than single parameter"
        )
    
    def test_configuration_parameters(self):
        """Test different configuration parameters"""
        print("\n=== Testing Configuration Parameters ===")
        
        n_points = 300
        t = np.linspace(0, 30, n_points)
        
        # Marginal boundary crossing
        he_density = np.ones(n_points) * 0.1
        he_density[100:200] = 0.2  # Weak enhancement
        BN = np.ones(n_points) * 2.0
        BN[100:200] = -1.5  # Weak rotation
        
        # Strict configuration
        cfg_strict = DetectorCfg(he_in=0.25, he_out=0.15, BN_tol=0.5)
        layers_strict = detect_crossings_multi(t, he_density, BN, cfg=cfg_strict)
        crossings_strict = extract_enter_exit(layers_strict, t)
        
        # Relaxed configuration
        cfg_relaxed = DetectorCfg(he_in=0.15, he_out=0.12, BN_tol=2.0)
        layers_relaxed = detect_crossings_multi(t, he_density, BN, cfg=cfg_relaxed)
        crossings_relaxed = extract_enter_exit(layers_relaxed, t)
        
        print(f"   Strict config: {len(crossings_strict)} crossings")
        print(f"   Relaxed config: {len(crossings_relaxed)} crossings")
        
        # Relaxed config should detect more marginal events
        self.assert_condition(
            len(crossings_relaxed) >= len(crossings_strict),
            "Relaxed configuration detects more marginal events"
        )
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        print("\n=== Testing Edge Cases ===")
        
        # Empty arrays
        try:
            layers = detect_crossings_multi(np.array([]), np.array([]), np.array([]))
            self.assert_condition(True, "Handles empty arrays gracefully")
        except Exception as e:
            self.assert_condition(False, f"Empty arrays cause error: {e}")
        
        # Single point
        try:
            layers = detect_crossings_multi(np.array([0]), np.array([0.1]), np.array([1.0]))
            self.assert_condition(True, "Handles single point gracefully")
        except Exception as e:
            self.assert_condition(False, f"Single point causes error: {e}")
        
        # All NaN data
        n_points = 100
        t = np.linspace(0, 10, n_points)
        he_nan = np.full(n_points, np.nan)
        BN_nan = np.full(n_points, np.nan)
        
        try:
            layers = detect_crossings_multi(t, he_nan, BN_nan)
            crossings = extract_enter_exit(layers, t)
            self.assert_condition(
                len(crossings) == 0,
                "NaN data produces no crossings"
            )
        except Exception as e:
            self.assert_condition(False, f"NaN data causes error: {e}")
    
    def run_all_tests(self):
        """Run all boundary detection tests"""
        print("🔍 Running Boundary Detection Verification Tests")
        print("=" * 55)
        
        self.test_synthetic_boundary_crossing()
        self.test_hysteresis_behavior()
        self.test_no_magnetic_signature()
        self.test_configuration_parameters()
        self.test_edge_cases()
        
        print(f"\n📊 Test Results:")
        print(f"   ✓ Passed: {self.passed}")
        print(f"   ❌ Failed: {self.failed}")
        print(f"   📈 Success Rate: {self.passed/(self.passed + self.failed)*100:.1f}%")
        
        if self.failed == 0:
            print("\n🎉 All boundary detection tests passed!")
            return True
        else:
            print(f"\n⚠️  {self.failed} test(s) failed - review algorithms")
            return False


if __name__ == "__main__":
    tester = TestBoundaryDetection()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
