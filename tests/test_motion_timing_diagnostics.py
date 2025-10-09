"""Targeted tests for motion integration and timing diagnostics."""

from __future__ import annotations

import numpy as np

from mms_mp.motion import integrate_disp
from mms_mp.multispacecraft import timing_normal


def test_integrate_disp_adaptive_step_refines_sparse_sampling():
    """Adaptive densification should recover high-frequency structure."""

    # Construct a rapidly varying velocity profile sampled coarsely
    t = np.linspace(0.0, 10.0, 201)  # 0.05 s cadence reference
    v_true = 15.0 * np.exp(-t / 3.0)  # decaying flow, km/s
    analytical = 15.0 * 3.0 * (1.0 - np.exp(-t / 3.0))

    # Down-sample to 2 s cadence to mimic sparse survey coverage
    t_sparse = np.linspace(0.0, 10.0, 6)
    v_sparse = 15.0 * np.exp(-t_sparse / 3.0)
    analytical_sparse = 15.0 * 3.0 * (1.0 - np.exp(-t_sparse / 3.0))

    # Baseline rectangular integration under-resolves curvature
    result_sparse = integrate_disp(t_sparse, v_sparse, scheme='rect', max_step_s=None)
    error_sparse = np.max(np.abs(result_sparse.disp_km - analytical_sparse))

    # Enforce 0.5 s sub-steps to recover the curvature
    result_refined = integrate_disp(t_sparse, v_sparse, scheme='rect', max_step_s=0.5)
    error_refined = np.max(np.abs(result_refined.disp_km - analytical_sparse))

    assert error_refined < error_sparse * 0.5, "Adaptive densification should reduce integration error"
    assert result_refined.max_step_s == 0.5


def test_integrate_disp_reports_gap_metadata():
    """Integration metadata should expose filled gaps and segment counts."""

    t = np.arange(0.0, 10.0, 1.0)
    v = np.linspace(5.0, 14.0, t.size)
    v[4] = np.nan  # isolated NaN should be filled
    v[7:9] = np.nan  # multi-sample gap should remain broken

    result = integrate_disp(t, v, fill_gaps=True, max_step_s=None)

    # Only the single-sample gap is interpolated
    assert result.n_gaps_filled == 1
    # Two contiguous NaNs create a second integration segment
    assert result.segment_count == 2


def test_timing_normal_returns_diagnostics():
    """Diagnostics should flag ill-conditioned constellations."""

    positions = {
        '1': np.array([0.0, 0.0, 0.0]),
        '2': np.array([150.0, 0.0, 0.0]),
        '3': np.array([0.0, 150.0, 0.0]),
        '4': np.array([50.0, 50.0, 100.0]),
    }
    normal = np.array([0.4, -0.3, 1.0])
    normal = normal / np.linalg.norm(normal)
    velocity = 25.0
    base_time = 1_546_567_200.0
    crossing_times = {
        '1': base_time,
        '2': base_time + np.dot(positions['2'] - positions['1'], normal) / velocity,
        '3': base_time + np.dot(positions['3'] - positions['1'], normal) / velocity,
        '4': base_time + np.dot(positions['4'] - positions['1'], normal) / velocity,
    }

    n_hat, V, sigma_V, diag = timing_normal(positions, crossing_times, return_diagnostics=True)
    assert diag['points_used'] == 4
    assert 'condition_number' in diag and diag['condition_number'] > 1.0
    assert not diag['degenerate']

    # Collapse constellation along X to force degeneracy
    skinny_positions = {
        '1': np.array([0.0, 0.0, 0.0]),
        '2': np.array([0.0, 120.0, 0.0]),
        '3': np.array([0.0, 240.0, 0.0]),
    }
    skinny_times = {
        '1': base_time,
        '2': base_time + 1.5,
        '3': base_time + 3.0,
    }
    _, _, _, skinny_diag = timing_normal(skinny_positions, skinny_times, return_diagnostics=True)
    assert skinny_diag['degenerate']
