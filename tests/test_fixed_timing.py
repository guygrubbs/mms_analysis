"""
Test the fixed timing analysis in the main codebase
"""

import numpy as np
import sys
import os

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mms_mp.multispacecraft import timing_normal

def test_fixed_timing():
    """Test the fixed timing analysis"""
    print("Testing Fixed Timing Analysis in Main Codebase")
    print("=" * 50)
    
    # Test case: Known planar boundary
    n_true = np.array([0.8, 0.6, 0.0])
    n_true = n_true / np.linalg.norm(n_true)
    V_true = 50.0
    
    # Well-separated spacecraft positions
    positions = {
        '1': np.array([0, 0, 0]),
        '2': np.array([100, 0, 0]),
        '3': np.array([0, 100, 0]),
        '4': np.array([0, 0, 100])
    }
    
    # Calculate crossing times
    t0 = 1000.0
    r0 = positions['1']
    
    crossing_times = {'1': t0}
    for probe, pos in positions.items():
        if probe == '1':
            continue
        dr = pos - r0
        dt = np.dot(n_true, dr) / V_true
        crossing_times[probe] = t0 + dt
    
    print(f"True normal: {n_true}")
    print(f"True velocity: {V_true} km/s")
    
    # Test fixed algorithm
    n_calc, V_calc, sigma_V = timing_normal(positions, crossing_times)
    
    print(f"Calculated normal: {n_calc}")
    print(f"Calculated velocity: {V_calc} km/s")
    print(f"Velocity uncertainty: {sigma_V} km/s")
    
    # Check accuracy
    dot_product = np.abs(np.dot(n_calc, n_true))
    print(f"\nNormal accuracy: {dot_product:.10f}")
    print(f"Velocity accuracy: {abs(V_calc - V_true):.10f}")
    
    # Verify physics equation
    print(f"\nVerifying physics equation:")
    for probe, pos in positions.items():
        if probe == '1':
            continue
        dr = pos - r0
        dt = crossing_times[probe] - t0
        lhs = np.dot(n_calc, dr)
        rhs = V_calc * dt
        print(f"  {probe}: {lhs:.6f} = {rhs:.6f} (diff: {abs(lhs-rhs):.2e})")
    
    success = dot_product > 0.999999 and abs(V_calc - V_true) < 1e-8
    
    if success:
        print("\n✅ Fixed timing analysis PASSED")
    else:
        print("\n❌ Fixed timing analysis FAILED")
    
    return success

if __name__ == "__main__":
    success = test_fixed_timing()
    sys.exit(0 if success else 1)
