"""
Test and fix for multi-spacecraft timing analysis

The timing analysis should solve: n⃗ · (r⃗ᵢ - r⃗₀) = V(tᵢ - t₀)

This can be written as a matrix equation:
[dr₁ₓ dr₁ᵧ dr₁ᵤ -dt₁] [nₓ]   [0]
[dr₂ₓ dr₂ᵧ dr₂ᵤ -dt₂] [nᵧ] = [0]
[dr₃ₓ dr₃ᵧ dr₃ᵤ -dt₃] [nᵤ]   [0]
                        [V ]

Where drᵢ = rᵢ - r₀ and dtᵢ = tᵢ - t₀
"""

import numpy as np
import sys
import os

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def timing_normal_corrected(pos_gsm, t_cross, normalise=True, min_sc=2):
    """
    Corrected multi-spacecraft timing analysis
    
    The physics equation is: n⃗ · (r⃗ᵢ - r⃗₀) = V(tᵢ - t₀)
    
    This gives us the matrix equation M·x = 0 where:
    M = [dr₁ₓ dr₁ᵧ dr₁ᵤ -dt₁]
        [dr₂ₓ dr₂ᵧ dr₂ᵤ -dt₂]
        [...]
    x = [nₓ, nᵧ, nᵤ, V]ᵀ
    """
    probes = sorted(set(pos_gsm) & set(t_cross))
    if len(probes) < min_sc:
        raise ValueError(f"Need ≥{min_sc} spacecraft; only have {len(probes)}.")
    
    # Reference spacecraft = first in sorted list
    p0 = probes[0]
    r0 = pos_gsm[p0]
    t0 = t_cross[p0]
    
    # Build matrix M where M·x = 0 and x = [nₓ, nᵧ, nᵤ, V]ᵀ
    rows = []
    for p in probes[1:]:
        dr = pos_gsm[p] - r0          # km
        dt = t_cross[p] - t0          # s
        # The equation is: n⃗ · dr⃗ = V·dt
        # Rearranged: n⃗ · dr⃗ - V·dt = 0
        # So the row is: [drₓ, drᵧ, drᵤ, -dt]
        rows.append(np.hstack((dr, -dt)))
    
    M = np.vstack(rows)  # shape (N-1, 4)
    
    # SVD to find null space
    U, S, VT = np.linalg.svd(M)
    x = VT[-1]  # Right singular vector corresponding to smallest singular value
    
    n = x[:3]
    V = x[3]  # Note: no negative sign here since we already have -dt in matrix
    
    if normalise:
        n_norm = np.linalg.norm(n)
        if n_norm == 0:
            raise RuntimeError("Degenerate solution: |n|=0")
        n_hat = n / n_norm
        V = V / n_norm  # Scale velocity consistently
    else:
        n_hat = n
    
    # Uncertainty estimate
    if len(S) >= 2:
        sigma = S[-1]  # Smallest singular value
        rel = sigma / S[0] if S[0] > 0 else np.inf  # Relative to largest
        sigma_V = abs(V) * rel
    else:
        sigma_V = np.nan
    
    return n_hat, V, sigma_V


def test_timing_analysis():
    """Test the corrected timing analysis"""
    print("Testing Corrected Multi-spacecraft Timing Analysis")
    print("=" * 50)
    
    # Test case: Known planar boundary
    # Boundary normal (unit vector)
    n_true = np.array([0.8, 0.6, 0.0])
    n_true = n_true / np.linalg.norm(n_true)
    
    # Phase velocity (km/s)
    V_true = 50.0
    
    # Spacecraft positions (km) - well-separated for good conditioning
    positions = {
        '1': np.array([0, 0, 0]),        # Reference at origin
        '2': np.array([100, 0, 0]),      # Along X
        '3': np.array([0, 100, 0]),      # Along Y
        '4': np.array([0, 0, 100])       # Along Z
    }
    
    # Calculate crossing times based on physics equation
    # n⃗ · (r⃗ᵢ - r⃗₀) = V(tᵢ - t₀)
    # So: tᵢ = t₀ + n⃗ · (r⃗ᵢ - r⃗₀) / V
    t0 = 1000.0  # Reference time
    r0 = positions['1']  # Reference position
    
    crossing_times = {}
    crossing_times['1'] = t0  # Reference
    
    for probe, pos in positions.items():
        if probe == '1':
            continue
        dr = pos - r0
        dt = np.dot(n_true, dr) / V_true
        crossing_times[probe] = t0 + dt
    
    print(f"True normal: {n_true}")
    print(f"True velocity: {V_true} km/s")
    print(f"Crossing times: {crossing_times}")
    
    # Test corrected algorithm
    n_calc, V_calc, sigma_V = timing_normal_corrected(positions, crossing_times)
    
    print(f"\nCalculated normal: {n_calc}")
    print(f"Calculated velocity: {V_calc} km/s")
    print(f"Velocity uncertainty: {sigma_V} km/s")
    
    # Check accuracy (allow for sign ambiguity in normal)
    dot_product = np.abs(np.dot(n_calc, n_true))
    print(f"\nNormal vector accuracy: |n_calc · n_true| = {dot_product:.10f}")
    print(f"Velocity accuracy: |V_calc - V_true| = {abs(V_calc - V_true):.10f}")
    
    # Verify the physics equation is satisfied
    print(f"\nVerifying physics equation n⃗ · (r⃗ᵢ - r⃗₀) = V(tᵢ - t₀):")
    for probe, pos in positions.items():
        if probe == '1':
            continue
        dr = pos - r0
        dt = crossing_times[probe] - t0
        lhs = np.dot(n_calc, dr)
        rhs = V_calc * dt
        print(f"  {probe}: {lhs:.6f} = {rhs:.6f} (diff: {abs(lhs-rhs):.2e})")
    
    # Test with minimum 2 spacecraft
    pos_2sc = {'1': positions['1'], '2': positions['2']}
    times_2sc = {'1': crossing_times['1'], '2': crossing_times['2']}
    
    n_calc_2sc, V_calc_2sc, _ = timing_normal_corrected(pos_2sc, times_2sc)
    dot_product_2sc = np.abs(np.dot(n_calc_2sc, n_true))
    
    print(f"\n2-spacecraft case:")
    print(f"Normal accuracy: {dot_product_2sc:.10f}")
    print(f"Velocity accuracy: {abs(V_calc_2sc - V_true):.10f}")
    
    # Success criteria
    # 4-spacecraft case should be very accurate
    # 2-spacecraft case has fundamental limitations (can only determine 1 component)
    success = (
        dot_product > 0.999999 and
        abs(V_calc - V_true) < 1e-8 and
        dot_product_2sc > 0.5  # 2-SC case has limited accuracy
    )
    
    if success:
        print("\n✅ Timing analysis test PASSED")
    else:
        print("\n❌ Timing analysis test FAILED")
    
    return success


if __name__ == "__main__":
    success = test_timing_analysis()
    sys.exit(0 if success else 1)
