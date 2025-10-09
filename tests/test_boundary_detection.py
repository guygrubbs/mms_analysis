"""Regression tests for the multi-parameter boundary detector."""

from __future__ import annotations

import numpy as np

from mms_mp.boundary import DetectorCfg, detect_crossings_multi, extract_enter_exit


def _build_transition_series(n_points: int = 600, cadence_s: float = 0.5):
    """Construct a physically-motivated sheath → boundary → magnetosphere profile."""

    times = np.array(
        [np.datetime64("2019-01-27T12:00:00") + np.timedelta64(int(i * cadence_s * 1e3), "ms") for i in range(n_points)]
    )

    # Magnetosheath segment (dense, low He⁺, strong |Bₙ|)
    sheath_len = n_points // 3
    boundary_len = n_points // 10
    magnetosphere_len = n_points - sheath_len - boundary_len

    he_sheath = np.full(sheath_len, 0.08)  # cm⁻³
    he_boundary = np.linspace(0.12, 0.2, boundary_len)
    he_magnetosphere = np.full(magnetosphere_len, 0.32)
    he = np.concatenate([he_sheath, he_boundary, he_magnetosphere])

    ni_sheath = np.full(sheath_len, 12.0)
    ni_boundary = np.linspace(9.0, 6.5, boundary_len)
    ni_magnetosphere = np.full(magnetosphere_len, 4.5)
    ni = np.concatenate([ni_sheath, ni_boundary, ni_magnetosphere])

    bn_sheath = np.full(sheath_len, -11.0)
    bn_boundary = np.linspace(-3.5, 3.5, boundary_len)
    bn_magnetosphere = np.full(magnetosphere_len, 5.0)
    bn = np.concatenate([bn_sheath, bn_boundary, bn_magnetosphere])

    return times, he, ni, bn


def test_detect_crossings_multi_segments_physical_transition():
    times, he, ni, bn = _build_transition_series()

    layers = detect_crossings_multi(times, he, bn, ni=ni)
    labels = [layer[0] for layer in layers]

    # Expect the canonical ordering: sheath → boundary layer → magnetosphere
    assert labels == ["sheath", "mp_layer", "magnetosphere"]

    # Ensure the boundary layer spans the constructed transition window
    _, start, end = layers[1]
    assert end - start + 1 >= 10


def test_detector_resists_noise_without_bn_rotation():
    times, he, ni, bn = _build_transition_series()

    # Remove the Bn rotation so only density signatures remain.
    bn[:] = 2.0
    cfg = DetectorCfg(min_layer_pts=12, min_mp_layer_pts=6)
    layers = detect_crossings_multi(times, he, bn, ni=ni, cfg=cfg)

    # Without a magnetic rotation the detector should not create a boundary layer.
    assert all(layer[0] != "mp_layer" for layer in layers)


def test_extract_enter_exit_orders_events():
    times, he, ni, bn = _build_transition_series()
    layers = detect_crossings_multi(times, he, bn, ni=ni)

    events = extract_enter_exit(layers, times)
    # Two events (enter + exit) for the magnetosphere interval
    assert len(events) == 2
    assert events[0][0] < events[1][0]
    assert events[0][1] == "enter"
    assert events[1][1] == "exit"


def test_nan_segments_are_skipped_cleanly():
    times, he, ni, bn = _build_transition_series()
    he[50:70] = np.nan
    ni[50:70] = np.nan

    layers = detect_crossings_multi(times, he, bn, ni=ni)
    assert layers, "Detector should keep valid segments despite NaNs"

