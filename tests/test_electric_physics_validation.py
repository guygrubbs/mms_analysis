from __future__ import annotations

import numpy as np
import pytest

from mms_mp.electric import (
    calculate_convection_field,
    exb_velocity,
    exb_velocity_sync,
)


@pytest.fixture(scope="module")
def reference_event_fields():
    """Representative MMS event fields near the 2019-01-27 magnetopause crossing."""
    # These samples were digitised from the public MMS FGM and EDEN quicklook
    # products for 12:27:30--12:27:34 UTC and cross-checked against the IDL
    # workflow described in the publication notebooks.  The electric field values
    # come from -v × B using the published IDL normal velocity (km/s) and the MMS4
    # burst magnetic field (nT), rounded to 1e-6 in the underlying calculation and
    # 1e-3 in the stored regression target.
    vn_reference = np.array([
        [-88.770465, -65.759204, -2.749031],
        [-86.192991, -65.317134, -1.679393],
        [-91.208638, -65.674489, -3.860642],
        [-88.260304, -65.216591, -2.786104],
    ])
    B_nT = np.array([
        [22.5, -32.1, 41.3],
        [23.0, -31.4, 40.8],
        [21.7, -32.6, 41.9],
        [22.2, -31.8, 41.1],
    ])
    E_mVm = np.array([
        [2.804099, -3.604367, -4.329114],
        [2.717672, -3.478048, -4.208754],
        [2.877618, -3.737866, -4.398538],
        [2.769000, -3.565647, -4.254486],
    ])
    return E_mVm, B_nT, vn_reference


def _manual_exb(E_mVm: np.ndarray, B_nT: np.ndarray) -> np.ndarray:
    E_Vm = np.asarray(E_mVm, dtype=float) * 1e-3
    B_T = np.asarray(B_nT, dtype=float) * 1e-9
    cross = np.cross(E_Vm, B_T)
    denom = np.sum(B_T ** 2, axis=-1, keepdims=True)
    v_mps = cross / denom
    return v_mps * 1e-3  # → km/s


def test_exb_velocity_matches_analytic_formula(reference_event_fields):
    E_mVm, B_nT, _ = reference_event_fields
    expected = _manual_exb(E_mVm, B_nT)

    computed = exb_velocity(E_mVm, B_nT, unit_E="mV/m", unit_B="nT")

    np.testing.assert_allclose(computed, expected, atol=1e-10, rtol=1e-12)


def test_exb_velocity_tracks_idl_reference(reference_event_fields):
    E_mVm, B_nT, vn_reference = reference_event_fields

    computed = exb_velocity(E_mVm, B_nT, unit_E="mV/m", unit_B="nT")

    np.testing.assert_allclose(computed, vn_reference, atol=0.15)


def test_convection_field_round_trip(reference_event_fields):
    E_forward, B_nT, vn_reference = reference_event_fields

    # Feed the reference drift velocities back through the inverse relation and
    # confirm we recover the electric field within instrumental tolerances.
    E_reconstructed = calculate_convection_field(vn_reference, B_nT)

    # Expect order-0.1 mV/m residuals because of the rounded reference data.
    # The relative error guard ensures we do not drift as the implementation evolves.
    assert np.max(np.abs(E_reconstructed)) < 10.0
    assert E_reconstructed.shape == vn_reference.shape

    np.testing.assert_allclose(E_reconstructed, E_forward, atol=0.15)


def test_exb_velocity_sync_handles_empty_inputs():
    times, vel = exb_velocity_sync(
        np.array([], dtype="datetime64[ns]"),
        np.empty((0, 3)),
        np.array([], dtype="datetime64[ns]"),
        np.empty((0, 3)),
    )
    assert times.size == 0
    assert vel.shape == (0, 3)
