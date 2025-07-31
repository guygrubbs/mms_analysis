"""
Test Coordinate Handedness Fix
=============================

Quick test to verify the coordinate handedness fix works.
"""

import numpy as np
import traceback

def test_coordinate_handedness():
    """Test that LMN coordinate system is right-handed"""
    
    try:
        from mms_mp.coords import hybrid_lmn
        
        print("Testing coordinate handedness fix...")
        
        # Create synthetic field with clear variance structure
        np.random.seed(42)
        n_points = 500
        
        # Field with systematic variation to create variance structure
        t = np.linspace(0, 2*np.pi, n_points)
        B_field = np.zeros((n_points, 3))
        B_field[:, 0] = 50 + 20 * np.sin(t) + 3 * np.random.randn(n_points)      # Max variance
        B_field[:, 1] = 30 + 10 * np.cos(t/2) + 2 * np.random.randn(n_points)   # Med variance  
        B_field[:, 2] = 20 + 5 * np.sin(t/3) + 1 * np.random.randn(n_points)    # Min variance
        
        print(f"B_field shape: {B_field.shape}")
        print(f"B_field sample: {B_field[0]}")
        
        # Get LMN system
        lmn_system = hybrid_lmn(B_field)
        
        print(f"L vector: {lmn_system.L}")
        print(f"M vector: {lmn_system.M}")
        print(f"N vector: {lmn_system.N}")
        
        # Test orthogonality
        dot_LM = np.dot(lmn_system.L, lmn_system.M)
        dot_LN = np.dot(lmn_system.L, lmn_system.N)
        dot_MN = np.dot(lmn_system.M, lmn_system.N)
        
        print(f"L·M = {dot_LM:.2e}")
        print(f"L·N = {dot_LN:.2e}")
        print(f"M·N = {dot_MN:.2e}")
        
        # Test unit vectors
        L_norm = np.linalg.norm(lmn_system.L)
        M_norm = np.linalg.norm(lmn_system.M)
        N_norm = np.linalg.norm(lmn_system.N)
        
        print(f"|L| = {L_norm:.6f}")
        print(f"|M| = {M_norm:.6f}")
        print(f"|N| = {N_norm:.6f}")
        
        # Test right-handedness: L × M should point in N direction
        cross_LM = np.cross(lmn_system.L, lmn_system.M)
        handedness = np.dot(cross_LM, lmn_system.N)
        
        print(f"L×M·N = {handedness:.6f}")
        
        # Validate results
        tolerance = 1e-10
        
        assert abs(dot_LM) < tolerance, f"L·M = {dot_LM:.2e} (should be ~0)"
        assert abs(dot_LN) < tolerance, f"L·N = {dot_LN:.2e} (should be ~0)"
        assert abs(dot_MN) < tolerance, f"M·N = {dot_MN:.2e} (should be ~0)"
        
        assert abs(L_norm - 1.0) < tolerance, f"|L| = {L_norm:.6f} (should be 1)"
        assert abs(M_norm - 1.0) < tolerance, f"|M| = {M_norm:.6f} (should be 1)"
        assert abs(N_norm - 1.0) < tolerance, f"|N| = {N_norm:.6f} (should be 1)"
        
        assert handedness > 0.99, f"Not right-handed: L×M·N = {handedness:.6f}"
        
        print("\n✅ All coordinate tests PASSED!")
        print("✅ Orthogonality verified")
        print("✅ Unit vectors verified")
        print("✅ Right-handed system verified")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Coordinate test FAILED: {e}")
        traceback.print_exc()
        return False


def test_coordinate_transformation():
    """Test coordinate transformation round-trip"""
    
    try:
        from mms_mp.coords import hybrid_lmn
        
        print("\nTesting coordinate transformation...")
        
        # Create test data
        np.random.seed(123)
        B_gsm = np.random.randn(100, 3) * 20 + np.array([40, 25, 15])
        
        lmn_system = hybrid_lmn(B_gsm)
        
        # Transform to LMN and back
        B_lmn = lmn_system.to_lmn(B_gsm)
        B_gsm_recovered = lmn_system.to_gsm(B_lmn)
        
        # Test magnitude preservation
        mag_original = np.linalg.norm(B_gsm, axis=1)
        mag_lmn = np.linalg.norm(B_lmn, axis=1)
        mag_recovered = np.linalg.norm(B_gsm_recovered, axis=1)
        
        mag_error_lmn = np.max(np.abs(mag_original - mag_lmn))
        mag_error_recovered = np.max(np.abs(mag_original - mag_recovered))
        
        # Test round-trip accuracy
        roundtrip_error = np.max(np.abs(B_gsm - B_gsm_recovered))
        
        print(f"Magnitude error (LMN): {mag_error_lmn:.2e}")
        print(f"Magnitude error (recovered): {mag_error_recovered:.2e}")
        print(f"Round-trip error: {roundtrip_error:.2e}")
        
        assert mag_error_lmn < 1e-12, f"Magnitude not preserved in LMN: {mag_error_lmn:.2e}"
        assert mag_error_recovered < 1e-12, f"Magnitude not preserved in recovery: {mag_error_recovered:.2e}"
        assert roundtrip_error < 1e-12, f"Round-trip error: {roundtrip_error:.2e}"
        
        print("✅ Coordinate transformation PASSED!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Coordinate transformation FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Run coordinate tests"""
    
    print("🔧 TESTING COORDINATE HANDEDNESS FIX")
    print("=" * 50)
    
    # Test handedness
    handedness_ok = test_coordinate_handedness()
    
    # Test transformation
    transform_ok = test_coordinate_transformation()
    
    print("\n" + "=" * 50)
    print("COORDINATE TEST SUMMARY")
    print("=" * 50)
    print(f"Handedness: {'✅ FIXED' if handedness_ok else '❌ STILL BROKEN'}")
    print(f"Transformation: {'✅ OK' if transform_ok else '❌ FAILED'}")
    
    if handedness_ok and transform_ok:
        print("\n🎉 ALL COORDINATE ISSUES FIXED!")
        return True
    else:
        print("\n⚠️ Coordinate issues remain")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
