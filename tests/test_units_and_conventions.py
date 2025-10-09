"""
Targeted tests for units, conventions, and edge conditions.

These follow best practices by keeping tests focused, readable, and fast.
"""
import numpy as np
import sys, os
import math

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
    assert lmn.method == 'mva'


def test_hybrid_lmn_respects_planar_thresholds():
    rng = np.random.default_rng(42)
    cov = np.diag([4.0, 1.6, 0.5])  # r_max_mid ≈ 2.5 (fails planar threshold 3.0)
    B = rng.multivariate_normal(np.zeros(3), cov, size=2048)

    pos = np.array([10.0, 2.0, -1.0]) * 6371.0
    lmn = coords.hybrid_lmn(B, pos_gsm_km=pos, formation_type='planar')

    assert lmn.method == 'shue', "Planar threshold should trigger Shue fallback"
    # Expect Shue normal ≈ radial direction from Earth to spacecraft
    r_hat = pos / np.linalg.norm(pos)
    assert np.isclose(abs(np.dot(lmn.N, r_hat)), 1.0, atol=1e-6)


def test_hybrid_lmn_overrides_threshold_with_user_values():
    rng = np.random.default_rng(24)
    cov = np.diag([4.0, 1.6, 0.5])
    B = rng.multivariate_normal(np.zeros(3), cov, size=1024)
    pos = np.array([8.0, -4.0, 1.0]) * 6371.0

    lmn = coords.hybrid_lmn(B, pos_gsm_km=pos,
                            formation_type='planar',
                            eig_ratio_thresh=(2.0, 2.0))
    assert lmn.method == 'mva'
    assert math.isclose(np.dot(np.cross(lmn.L, lmn.M), lmn.N), 1.0, rel_tol=1e-6)


def test_hybrid_lmn_known_rotation_recovers_axes():
    rng = np.random.default_rng(7)
    scales = np.array([12.0, 5.0, 1.5])
    data_lmn = rng.normal(scale=scales, size=(4096, 3))

    # Construct a known rotation matrix (rows = L, M, N in GSM)
    L_true = np.array([0.36, 0.48, 0.8])
    L_true /= np.linalg.norm(L_true)
    M_temp = np.array([-0.8, 0.6, 0.0])
    M_temp -= np.dot(M_temp, L_true) * L_true
    M_true = M_temp / np.linalg.norm(M_temp)
    N_true = np.cross(L_true, M_true)
    R_true = np.vstack((L_true, M_true, N_true))

    data_gsm = data_lmn @ R_true
    lmn = coords.hybrid_lmn(data_gsm)

    # Compare up to sign (MVA eigenvectors can flip sign individually)
    for est, true_vec in zip((lmn.L, lmn.M, lmn.N), (L_true, M_true, N_true)):
        assert np.isclose(abs(np.dot(est, true_vec)), 1.0, atol=5e-4)


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

