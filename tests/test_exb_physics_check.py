"""
Check E×B physics and determine correct convention

Let's verify the physics from first principles and check what the 
correct expectation should be.
"""

import numpy as np

def check_exb_physics():
    """Check E×B physics from first principles"""
    print("E×B Drift Physics Check")
    print("=" * 25)
    
    # Case 1: E pointing in +X, B pointing in +Z
    # This represents: E field pointing East, B field pointing up (North pole)
    # The E×B drift should point in +Y direction (North)
    
    E = np.array([1, 0, 0])  # +X direction
    B = np.array([0, 0, 1])  # +Z direction
    
    print(f"E⃗ = {E} (pointing +X)")
    print(f"B⃗ = {B} (pointing +Z)")
    
    # Right-hand rule: fingers point along E (+X), curl toward B (+Z)
    # Thumb points in +Y direction
    ExB = np.cross(E, B)
    print(f"E⃗ × B⃗ = {ExB}")
    print("Expected by right-hand rule: [0, 1, 0] ❌")
    print("Actual result: [0, -1, 0] ✓")
    
    print("\n🤔 Wait, let me check the right-hand rule again...")
    print("Right-hand rule for cross product A × B:")
    print("1. Point fingers in direction of A (E⃗)")
    print("2. Curl fingers toward B (B⃗)")  
    print("3. Thumb points in direction of A × B")
    
    print(f"\nFor E⃗ = [1,0,0] and B⃗ = [0,0,1]:")
    print("1. Fingers point in +X")
    print("2. Curl from +X toward +Z (90° rotation about Y-axis)")
    print("3. Thumb points in -Y direction")
    print("Therefore E⃗ × B⃗ = [0, -1, 0] ✓")
    
    # Let's check the opposite case
    print(f"\nCase 2: E⃗ = [0,1,0], B⃗ = [1,0,0]")
    E2 = np.array([0, 1, 0])
    B2 = np.array([1, 0, 0])
    ExB2 = np.cross(E2, B2)
    print(f"E⃗ × B⃗ = {ExB2}")
    print("Right-hand rule: fingers +Y, curl toward +X, thumb points -Z")
    print("Expected: [0, 0, -1] ✓")
    
    # So the cross product is working correctly
    # The question is: what should the test expect?
    
    print(f"\n📚 Physics Literature Check:")
    print("The E×B drift velocity is: v⃗ = (E⃗ × B⃗) / B²")
    print("This is the standard formula in plasma physics.")
    print("The direction follows the right-hand rule.")
    
    # Let's think about the physical situation:
    # If E points East (+X) and B points up (+Z), 
    # particles should drift North (+Y) or South (-Y)?
    
    print(f"\n🧭 Physical Interpretation:")
    print("For a positive charge in crossed E and B fields:")
    print("- E⃗ = [1,0,0] (East): Force F⃗_E = qE⃗ points East")
    print("- B⃗ = [0,0,1] (Up): Lorentz force F⃗_B = q(v⃗ × B⃗)")
    print("- The particle will drift perpendicular to both E and B")
    print("- E×B drift is independent of charge and mass")
    print("- Direction: E⃗ × B⃗ = [0,-1,0] (South)")
    
    print(f"\n🔍 Conclusion:")
    print("The current implementation is CORRECT.")
    print("The test expectation [0, 1000, 0] is WRONG.")
    print("The correct result should be [0, -1000, 0].")
    
    return True

def check_literature_examples():
    """Check against known literature examples"""
    print(f"\n📖 Literature Examples")
    print("=" * 20)
    
    # Example from Baumjohann & Treumann "Basic Space Plasma Physics"
    # E = 1 mV/m in X direction, B = 50 nT in Z direction
    # Expected drift ~ 20 km/s in -Y direction
    
    E_mV_m = 1.0  # mV/m
    B_nT = 50.0   # nT
    
    # Convert to SI
    E_SI = E_mV_m * 1e-3  # V/m
    B_SI = B_nT * 1e-9    # T
    
    # Calculate drift speed
    v_drift = E_SI / B_SI  # m/s
    v_drift_km_s = v_drift * 1e-3  # km/s
    
    print(f"Literature example:")
    print(f"E = {E_mV_m} mV/m, B = {B_nT} nT")
    print(f"Expected drift speed: |v| = E/B = {v_drift_km_s:.1f} km/s")
    print(f"Direction: E⃗ × B⃗ / |B⃗|² gives the direction")
    
    # Our test case
    print(f"\nOur test case:")
    print(f"E = 1 mV/m, B = 1 nT")
    print(f"Expected speed: |v| = 1e-3 / 1e-9 = 1e6 m/s = 1000 km/s ✓")
    print(f"Direction: [1,0,0] × [0,0,1] = [0,-1,0] ✓")
    print(f"Result: [0, -1000, 0] km/s ✓")

if __name__ == "__main__":
    check_exb_physics()
    check_literature_examples()
    print(f"\n✅ Physics check complete: Current implementation is CORRECT")
    print(f"❌ Test expectation needs to be fixed")
