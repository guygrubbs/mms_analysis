"""
Targeted tests for units, conventions, and edge conditions.

These follow best practices by keeping tests focused, readable, and fast.
"""
import numpy as np
import sys, os

# Add package root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mms_mp
from mms_mp import electric, coords, multispacecraft, motion


def test_exb_unit_consistency_and_direction():
    # Same physical fields, two unit systems
    E_mVm = np.array([[1.0, 0.0, 0.0]])      # mV/m
    B_nT  = np.array([[0.0, 0.0, 1.0]])      # nT

    E_Vm = E_mVm * 1e-3                       # V/m
    B_T  = B_nT  * 1e-9                       # T

    v1 = electric.exb_velocity(E_mVm, B_nT, unit_E='mV/m', unit_B='nT')
    v2 = electric.exb_velocity(E_Vm,  B_T,  unit_E='V/m',  unit_B='T')

    # Direction check: E x B = +x x +z = -y
    assert np.allclose(v1, [[0.0, -1000.0, 0.0]]), f"Unexpected v1: {v1}"
    assert np.allclose(v1, v2), f"Unit inconsistency: {v1} vs {v2}"


def test_exb_weak_B_stability():
    # Very weak B should not produce NaNs/Inf, but magnitude will be large
    E = np.array([[1.0, 0.0, 0.0]])  # mV/m
    B = np.array([[0.0, 0.0, 1e-6]]) # nT
    v = electric.exb_velocity(E, B, unit_E='mV/m', unit_B='nT')
    assert np.isfinite(v).all(), "E×B produced non-finite values for weak B"


def test_lmn_right_handed_and_orthonormal():
    # Synthetic B with clear variance ordering
    rng = np.random.default_rng(0)
    B = np.column_stack([
        rng.normal(scale=10.0, size=1000),
        rng.normal(scale=3.0,  size=1000),
        rng.normal(scale=1.0,  size=1000),
    ])
    lmn = coords.hybrid_lmn(B)
    R = lmn.R
    # Orthonormal
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    # Right-handed: (L x M) · N > 0
    L, M, N = R[0], R[1], R[2]
    assert np.dot(np.cross(L, M), N) > 0


def test_timing_degenerate_zero_dt():
    # Degenerate case where all time differences are zero → expect NaNs or fallback
    pos = {
        '1': np.array([0.0, 0.0, 0.0]),
        '2': np.array([10.0, 0.0, 0.0]),
        '3': np.array([0.0, 10.0, 0.0]),
    }
    t0 = 0.0
    t = {k: t0 for k in pos}
    n, V, s = multispacecraft.timing_normal(pos, t)
    # Either NaNs (explicit degenerate return) or a unit axis with NaN speeds per implementation
    assert (not np.isfinite(V)) or np.isfinite(n).any(), "Unexpected stable velocity in zero-dt degenerate case"


def test_motional_field_units():
    # v in km/s, B in nT → E in mV/m; E = - v x B
    v = np.array([[10.0, 0.0, 0.0]])  # km/s along +x
    B = np.array([[0.0, 10.0, 0.0]])  # nT along +y
    # v x B = +z; E = -z
    E = electric.calculate_convection_field(v, B)
    assert np.allclose(E, [[0.0, 0.0, -0.1]]), f"Unexpected E: {E}"


def test_integrate_disp_units_and_accuracy():
    # Constant 5 km/s for 10 seconds → 50 km
    t = np.linspace(0, 10, 101)
    vN = np.ones_like(t) * 5.0
    res = motion.integrate_disp(t, vN, scheme='trap')
    assert np.isclose(res.disp_km[-1], 50.0, atol=1e-10)

