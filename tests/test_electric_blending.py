from __future__ import annotations

import numpy as np

from mms_mp import electric


def test_exb_velocity_quality_mask_with_minimum_b():
    E = np.array([[1.0, 0.0, 0.0], [0.2, 0.1, 0.0]])  # mV/m
    B = np.array([[0.0, 0.0, 50.0], [0.0, 0.0, 0.01]])  # nT

    v, quality = electric.exb_velocity(E, B, unit_E='mV/m', unit_B='nT', min_b=1.0, return_quality=True)

    assert quality.shape == (2,)
    assert quality[0]
    assert not quality[1]
    np.testing.assert_allclose(v[0], [0.0, -20.0, 0.0], atol=1e-9)
    assert np.isnan(v[1]).all()


def test_normal_velocity_metadata_tracks_sources():
    n_points = 5
    v_bulk_lmn = np.column_stack([
        np.zeros(n_points),
        np.zeros(n_points),
        np.linspace(5.0, 15.0, n_points),
    ])
    v_exb_lmn = np.column_stack([
        np.zeros(n_points),
        np.zeros(n_points),
        np.linspace(40.0, 0.0, n_points),
    ])

    b_mag = np.array([20.0, 8.0, 0.2, 15.0, np.nan])

    result = electric.normal_velocity(
        v_bulk_lmn,
        v_exb_lmn,
        strategy='prefer_exb',
        b_mag_nT=b_mag,
        min_b_nT=5.0,
        return_metadata=True,
    )

    assert isinstance(result.vn, np.ndarray)
    assert result.source.tolist() == ['exb', 'exb', 'bulk', 'exb', 'bulk']
    assert result.exb_valid.tolist() == [True, True, False, True, False]
    assert result.bulk_valid.tolist() == [True, True, True, True, True]


def test_normal_velocity_average_only_uses_valid_values():
    v_bulk_lmn = np.array([[0.0, 0.0, np.nan], [0.0, 0.0, 10.0]])
    v_exb_lmn = np.array([[0.0, 0.0, 20.0], [0.0, 0.0, np.nan]])

    result = electric.normal_velocity(
        v_bulk_lmn,
        v_exb_lmn,
        strategy='average',
        return_metadata=True,
    )

    np.testing.assert_allclose(result.vn[0], 20.0)
    np.testing.assert_allclose(result.vn[1], 10.0)
    assert result.source.tolist() == ['exb', 'bulk']
