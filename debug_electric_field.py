"""
Debug Electric Field Function
============================

Simple test to debug the electric field calculation issue.
"""

import numpy as np
import traceback

def test_electric_field_debug():
    """Debug the electric field function"""
    
    try:
        from mms_mp.electric import exb_velocity
        
        print("Testing electric field function...")
        
        # Test case 1: Simple 1D arrays
        print("\nTest 1: 1D arrays")
        E_field = np.array([1.0, 0.0, 0.0])  # mV/m
        B_field = np.array([0.0, 0.0, 50.0])  # nT
        
        print(f"E_field shape: {E_field.shape}")
        print(f"B_field shape: {B_field.shape}")
        print(f"E_field: {E_field}")
        print(f"B_field: {B_field}")
        
        v_exb = exb_velocity(E_field, B_field, unit_E='mV/m', unit_B='nT')
        
        print(f"Result: {v_exb}")
        print(f"Result shape: {v_exb.shape}")
        print(f"Result magnitude: {np.linalg.norm(v_exb):.2f} km/s")
        
        # Test case 2: 2D arrays
        print("\nTest 2: 2D arrays")
        E_field_2d = np.array([[1.0, 0.0, 0.0], [0.5, 0.2, 0.0]])  # mV/m
        B_field_2d = np.array([[0.0, 0.0, 50.0], [30.0, 20.0, 40.0]])  # nT
        
        print(f"E_field_2d shape: {E_field_2d.shape}")
        print(f"B_field_2d shape: {B_field_2d.shape}")
        
        v_exb_2d = exb_velocity(E_field_2d, B_field_2d, unit_E='mV/m', unit_B='nT')
        
        print(f"Result 2D: {v_exb_2d}")
        print(f"Result 2D shape: {v_exb_2d.shape}")
        
        print("\n✅ Electric field function working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Electric field function failed: {e}")
        traceback.print_exc()
        return False


def test_coordinate_transform_debug():
    """Debug the coordinate transformation"""
    
    try:
        from mms_mp.coords import hybrid_lmn
        
        print("\nTesting coordinate transformation...")
        
        # Create simple test data
        np.random.seed(42)
        B_field = np.random.randn(100, 3) * 30 + np.array([40, 20, 15])
        
        print(f"B_field shape: {B_field.shape}")
        print(f"B_field sample: {B_field[0]}")
        
        lmn_system = hybrid_lmn(B_field)
        
        print(f"LMN system created successfully")
        print(f"L vector: {lmn_system.L}")
        print(f"M vector: {lmn_system.M}")
        print(f"N vector: {lmn_system.N}")
        
        # Test to_lmn
        B_lmn = lmn_system.to_lmn(B_field)
        print(f"B_lmn shape: {B_lmn.shape}")
        print(f"B_lmn sample: {B_lmn[0]}")
        
        # Test to_gsm
        B_gsm_recovered = lmn_system.to_gsm(B_lmn)
        print(f"B_gsm_recovered shape: {B_gsm_recovered.shape}")
        print(f"B_gsm_recovered sample: {B_gsm_recovered[0]}")
        
        # Check round-trip error
        error = np.max(np.abs(B_field - B_gsm_recovered))
        print(f"Round-trip error: {error:.2e}")
        
        print("\n✅ Coordinate transformation working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Coordinate transformation failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run debug tests"""
    
    print("🔧 DEBUGGING MMS-MP MODULES")
    print("=" * 50)
    
    # Test electric field
    electric_ok = test_electric_field_debug()
    
    # Test coordinate transformation
    coords_ok = test_coordinate_transform_debug()
    
    print("\n" + "=" * 50)
    print("DEBUG SUMMARY")
    print("=" * 50)
    print(f"Electric field: {'✅ OK' if electric_ok else '❌ FAILED'}")
    print(f"Coordinates: {'✅ OK' if coords_ok else '❌ FAILED'}")
    
    if electric_ok and coords_ok:
        print("\n🎉 All core functions working!")
        return True
    else:
        print("\n⚠️ Issues detected - need fixes")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
