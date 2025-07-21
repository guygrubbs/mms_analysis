"""
Test and verify E×B drift calculation

The E×B drift velocity is: v⃗ = (E⃗ × B⃗) / |B⃗|²

Let's verify this step by step.
"""

import numpy as np

def test_cross_product():
    """Test basic cross product"""
    print("Testing Cross Product")
    print("=" * 20)
    
    # Test case: E = [1, 0, 0], B = [0, 0, 1]
    # E × B should be [0, 1, 0] (right-hand rule)
    E = np.array([1, 0, 0])
    B = np.array([0, 0, 1])
    
    cross = np.cross(E, B)
    print(f"E = {E}")
    print(f"B = {B}")
    print(f"E × B = {cross}")
    print(f"Expected: [0, 1, 0]")
    
    expected = np.array([0, 1, 0])
    if np.allclose(cross, expected):
        print("✅ Cross product correct")
    else:
        print("❌ Cross product incorrect")
    
    # Test case 2: E = [0, 1, 0], B = [1, 0, 0]
    # E × B should be [0, 0, -1]
    E2 = np.array([0, 1, 0])
    B2 = np.array([1, 0, 0])
    cross2 = np.cross(E2, B2)
    expected2 = np.array([0, 0, -1])
    
    print(f"\nE = {E2}")
    print(f"B = {B2}")
    print(f"E × B = {cross2}")
    print(f"Expected: [0, 0, -1]")
    
    if np.allclose(cross2, expected2):
        print("✅ Cross product correct")
    else:
        print("❌ Cross product incorrect")


def test_exb_physics():
    """Test E×B physics calculation"""
    print("\nTesting E×B Physics")
    print("=" * 20)
    
    # Test case from the failed test:
    # E = [1, 0, 0] mV/m, B = [0, 0, 1] nT
    # Expected v_ExB = [0, 1000, 0] km/s
    
    E_mV_per_m = np.array([[1.0, 0.0, 0.0]])
    B_nT = np.array([[0.0, 0.0, 1.0]])
    
    # Convert to SI units
    E_V_per_m = E_mV_per_m * 1e-3  # mV/m → V/m
    B_T = B_nT * 1e-9              # nT → T
    
    print(f"E = {E_V_per_m[0]} V/m")
    print(f"B = {B_T[0]} T")
    
    # Calculate E × B
    ExB = np.cross(E_V_per_m, B_T)
    print(f"E × B = {ExB[0]} V·T/m")
    
    # Calculate |B|²
    B_squared = np.sum(B_T**2, axis=1, keepdims=True)
    print(f"|B|² = {B_squared[0]} T²")
    
    # Calculate v = (E × B) / |B|²
    v_m_per_s = ExB / B_squared
    print(f"v = {v_m_per_s[0]} m/s")
    
    # Convert to km/s
    v_km_per_s = v_m_per_s * 1e-3
    print(f"v = {v_km_per_s[0]} km/s")
    
    expected = np.array([0.0, 1000.0, 0.0])
    print(f"Expected: {expected} km/s")
    
    if np.allclose(v_km_per_s[0], expected):
        print("✅ E×B calculation correct")
        return True
    else:
        print("❌ E×B calculation incorrect")
        return False


def test_current_implementation():
    """Test the current implementation in the codebase"""
    print("\nTesting Current Implementation")
    print("=" * 30)
    
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from mms_mp.electric import exb_velocity
    
    E = np.array([[1.0, 0.0, 0.0]])  # mV/m
    B = np.array([[0.0, 0.0, 1.0]])  # nT
    
    v_exb = exb_velocity(E, B, unit_E='mV/m', unit_B='nT')
    print(f"Current implementation result: {v_exb[0]} km/s")
    print(f"Expected: [0, 1000, 0] km/s")
    
    expected = np.array([[0.0, 1000.0, 0.0]])
    if np.allclose(v_exb, expected):
        print("✅ Current implementation correct")
        return True
    else:
        print("❌ Current implementation incorrect")
        print(f"Difference: {v_exb - expected}")
        return False


if __name__ == "__main__":
    test_cross_product()
    physics_ok = test_exb_physics()
    impl_ok = test_current_implementation()
    
    if physics_ok and impl_ok:
        print("\n🎉 All E×B tests passed!")
    else:
        print("\n⚠️ Some E×B tests failed")
