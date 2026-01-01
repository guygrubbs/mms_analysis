"""Unit tests for the physics-driven algorithmic LMN builder.

These tests focus specifically on ``mms_mp.coords.algorithmic_lmn`` using a
simple synthetic multi-spacecraft configuration with a known boundary normal.
"""

import numpy as np

from mms_mp import coords


def test_algorithmic_lmn_basic_properties():
    """Algorithmic LMN should recover a shared normal close to the true normal.

    We construct a synthetic planar boundary with normal along +X, a constant
    phase speed, and four spacecraft arranged along the X axis. Magnetic
    fields are purely tangential (large variance in Y/Z, small in X), so that
    both MVA and timing information favour an N direction close to +X.
    """

    np.random.seed(1234)
    n_true = np.array([1.0, 0.0, 0.0])  # boundary normal (GSM X)
    V_boundary = 50.0  # km/s
    t0 = 1000.0

    # Shared magnetic-field time axis around the crossing
    t = np.linspace(t0 - 60.0, t0 + 60.0, 200)

    # Tangential field: large variance in Y/Z, small in X -> N ~ X
    By = 10.0 * np.sin(2 * np.pi * (t - t0) / 60.0)
    Bz = 5.0 * np.cos(2 * np.pi * (t - t0) / 60.0)
    Bx_noise = 0.1 * np.random.randn(t.size)
    B_template = np.column_stack([Bx_noise, By, Bz])

    probes = ["1", "2", "3", "4"]
    b_times = {}
    b_gsm = {}
    pos_times = {}
    pos_gsm_km = {}
    t_cross = {}

    for i, p in enumerate(probes):
        x = 100.0 * i  # km separation along X
        pos_times[p] = np.array([t0])
        pos_gsm_km[p] = np.array([[x, 0.0, 0.0]])
        # Crossing times consistent with boundary motion along +X
        t_cross[p] = t0 + x / V_boundary
        # Re-use same B template for all probes
        b_times[p] = t.copy()
        b_gsm[p] = B_template.copy()

    lmn_map = coords.algorithmic_lmn(
        b_times=b_times,
        b_gsm=b_gsm,
        pos_times=pos_times,
        pos_gsm_km=pos_gsm_km,
        t_cross=t_cross,
        window_half_width_s=30.0,
    )

    assert set(lmn_map.keys()) == set(probes)

    # All probes should share (within numerical precision) a common N
    N_ref = lmn_map["1"].N / np.linalg.norm(lmn_map["1"].N)
    for p in probes[1:]:
        Np = lmn_map[p].N / np.linalg.norm(lmn_map[p].N)
        cos_angle = float(np.clip(np.abs(np.dot(N_ref, Np)), -1.0, 1.0))
        assert cos_angle > 0.999, f"N not shared across probes: dot={cos_angle} for probe {p}"

    # Shared normal should align with the true normal (up to sign)
    cos_to_true = float(np.clip(np.abs(np.dot(N_ref, n_true)), -1.0, 1.0))
    assert cos_to_true > 0.9, f"Algorithmic normal far from expected: cos={cos_to_true}"

    # Each LMN should be orthonormal and right-handed, with proper metadata
    for p in probes:
        lm = lmn_map[p]
        assert lm.method == "algorithmic"
        assert lm.meta is not None and lm.meta.get("source") == "algorithmic_lmn"
        # Orthonormality
        assert abs(np.dot(lm.L, lm.M)) < 1e-10
        assert abs(np.dot(lm.L, lm.N)) < 1e-10
        assert abs(np.dot(lm.M, lm.N)) < 1e-10
        assert abs(np.linalg.norm(lm.L) - 1.0) < 1e-10
        assert abs(np.linalg.norm(lm.M) - 1.0) < 1e-10
        assert abs(np.linalg.norm(lm.N) - 1.0) < 1e-10
        # Right-handedness
        cross_LM = np.cross(lm.L, lm.M)
        assert np.dot(cross_LM, lm.N) > 0.0


def test_algorithmic_lmn_tangential_vi_alignment():
    """With ``tangential_strategy='Vi'``, L should align with mean Vi in the plane.

    We reuse the synthetic boundary configuration but add a constant ion
    velocity vector that lies entirely in the tangential (Y) direction.
    The resulting L directions should closely follow the projected Vi
    vector in the plane perpendicular to the shared N.
    """

    np.random.seed(1234)
    n_true = np.array([1.0, 0.0, 0.0])
    V_boundary = 50.0
    t0 = 1000.0

    t = np.linspace(t0 - 60.0, t0 + 60.0, 200)
    By = 10.0 * np.sin(2 * np.pi * (t - t0) / 60.0)
    Bz = 5.0 * np.cos(2 * np.pi * (t - t0) / 60.0)
    Bx_noise = 0.1 * np.random.randn(t.size)
    B_template = np.column_stack([Bx_noise, By, Bz])

    probes = ["1", "2", "3", "4"]
    b_times = {}
    b_gsm = {}
    pos_times = {}
    pos_gsm_km = {}
    vi_times = {}
    vi_gsm = {}
    t_cross = {}

    vi_vec = np.array([0.0, 1.0, 0.0])  # purely tangential (Y) direction

    for i, p in enumerate(probes):
        x = 100.0 * i  # km separation along X
        pos_times[p] = np.array([t0])
        pos_gsm_km[p] = np.array([[x, 0.0, 0.0]])
        # Crossing times consistent with boundary motion along +X
        t_cross[p] = t0 + x / V_boundary
        # Shared B/Vi time series
        b_times[p] = t.copy()
        b_gsm[p] = B_template.copy()
        vi_times[p] = t.copy()
        vi_gsm[p] = np.tile(vi_vec, (t.size, 1))

    lmn_map = coords.algorithmic_lmn(
        b_times=b_times,
        b_gsm=b_gsm,
        pos_times=pos_times,
        pos_gsm_km=pos_gsm_km,
        t_cross=t_cross,
        window_half_width_s=30.0,
        tangential_strategy="Vi",
        vi_times=vi_times,
        vi_gsm=vi_gsm,
    )

    N_ref = lmn_map["1"].N / np.linalg.norm(lmn_map["1"].N)
    for p in probes:
        lm = lmn_map[p]
        v_proj = vi_vec - np.dot(vi_vec, N_ref) * N_ref
        if np.linalg.norm(v_proj) < 1e-6:
            continue
        v_proj /= np.linalg.norm(v_proj)
        L = lm.L / np.linalg.norm(lm.L)
        cos_angle = float(np.clip(np.abs(np.dot(L, v_proj)), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
        assert angle_deg < 5.0, (
            f"L not aligned with Vi_tan for probe {p}: angle={angle_deg:.2f} deg"
        )
        assert lm.meta.get("tangential_strategy") in {"Vi", "vi"}


def test_algorithmic_lmn_tangential_timing_alignment():
    """With ``tangential_strategy='timing'``, L follows position offsets.

    We construct a configuration where spacecraft are offset in Y while the
    boundary normal is along +X. The tangential direction implied by the
    position offsets (relative to the formation centre) should be captured
    by L in the plane perpendicular to N.
    """

    np.random.seed(1234)
    n_true = np.array([1.0, 0.0, 0.0])
    V_boundary = 50.0
    t0 = 1000.0

    t = np.linspace(t0 - 60.0, t0 + 60.0, 200)
    By = 10.0 * np.sin(2 * np.pi * (t - t0) / 60.0)
    Bz = 5.0 * np.cos(2 * np.pi * (t - t0) / 60.0)
    Bx_noise = 0.1 * np.random.randn(t.size)
    B_template = np.column_stack([Bx_noise, By, Bz])

    probes = ["1", "2", "3", "4"]
    b_times = {}
    b_gsm = {}
    pos_times = {}
    pos_gsm_km = {}
    t_cross = {}
    pos_for_center = {}

    y_offsets = [0.0, 50.0, -50.0, 100.0]

    for i, (p, y) in enumerate(zip(probes, y_offsets)):
        x = 100.0 * i
        pos = np.array([[x, y, 0.0]])
        pos_times[p] = np.array([t0])
        pos_gsm_km[p] = pos
        pos_for_center[p] = pos[0]
        t_cross[p] = t0 + x / V_boundary
        b_times[p] = t.copy()
        b_gsm[p] = B_template.copy()

    lmn_map = coords.algorithmic_lmn(
        b_times=b_times,
        b_gsm=b_gsm,
        pos_times=pos_times,
        pos_gsm_km=pos_gsm_km,
        t_cross=t_cross,
        window_half_width_s=30.0,
        tangential_strategy="timing",
    )

    N_ref = lmn_map["1"].N / np.linalg.norm(lmn_map["1"].N)
    center = sum(pos_for_center.values()) / float(len(pos_for_center))

    for p in probes:
        lm = lmn_map[p]
        pos = pos_for_center[p]
        delta = pos - center
        delta_tan = delta - np.dot(delta, N_ref) * N_ref
        if np.linalg.norm(delta_tan) < 1e-6:
            continue
        delta_tan /= np.linalg.norm(delta_tan)
        L = lm.L / np.linalg.norm(lm.L)
        cos_angle = float(np.clip(np.abs(np.dot(L, delta_tan)), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
        assert angle_deg < 5.0, (
            f"L not aligned with timing_tan for probe {p}: angle={angle_deg:.2f} deg"
        )
        assert lm.meta.get("tangential_strategy").lower().startswith("timing")


def test_algorithmic_lmn_single_spacecraft_basic():
    """Single-spacecraft configuration should fall back to MVA+Shue blend.

    In this synthetic test only one probe has valid inputs. The boundary
    normal is along +X, the spacecraft is located on the +X axis, and the
    magnetic field is predominantly tangential. The resulting N should align
    reasonably with +X, the LMN triad must be orthonormal and right-handed,
    and no timing-normal specific logic should be required.
    """

    np.random.seed(42)
    n_true = np.array([1.0, 0.0, 0.0])
    t0 = 1000.0

    # Time axis and tangential magnetic field
    t = np.linspace(t0 - 60.0, t0 + 60.0, 200)
    By = 10.0 * np.sin(2 * np.pi * (t - t0) / 60.0)
    Bz = 5.0 * np.cos(2 * np.pi * (t - t0) / 60.0)
    Bx_noise = 0.1 * np.random.randn(t.size)
    B = np.column_stack([Bx_noise, By, Bz])

    probe = "1"
    b_times = {probe: t}
    b_gsm = {probe: B}
    pos_times = {probe: np.array([t0])}
    # Place spacecraft on +X axis so Shue normal roughly matches +X as well.
    pos_gsm_km = {probe: np.array([[1000.0, 0.0, 0.0]])}
    t_cross = {probe: t0}

    lmn_map = coords.algorithmic_lmn(
        b_times=b_times,
        b_gsm=b_gsm,
        pos_times=pos_times,
        pos_gsm_km=pos_gsm_km,
        t_cross=t_cross,
        window_half_width_s=30.0,
        # Use non-zero timing weight to verify it is silently ignored when
        # timing normals are unavailable and MVA+Shue are renormalised.
        normal_weights=(0.8, 0.15, 0.05),
    )

    assert set(lmn_map.keys()) == {probe}
    lm = lmn_map[probe]

    # N should align with the true normal (up to sign)
    N_unit = lm.N / np.linalg.norm(lm.N)
    cos_to_true = float(np.clip(np.abs(np.dot(N_unit, n_true)), -1.0, 1.0))
    assert cos_to_true > 0.9, f"Single-spacecraft normal far from expected: cos={cos_to_true}"

    # LMN must be orthonormal and right-handed
    assert abs(np.dot(lm.L, lm.M)) < 1e-10
    assert abs(np.dot(lm.L, lm.N)) < 1e-10
    assert abs(np.dot(lm.M, lm.N)) < 1e-10
    assert abs(np.linalg.norm(lm.L) - 1.0) < 1e-10
    assert abs(np.linalg.norm(lm.M) - 1.0) < 1e-10
    assert abs(np.linalg.norm(lm.N) - 1.0) < 1e-10
    cross_LM = np.cross(lm.L, lm.M)
    assert np.dot(cross_LM, lm.N) > 0.0
