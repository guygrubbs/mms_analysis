"""Physics-informed magnetopause boundary identification utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np


@dataclass
class DetectorCfg:
    """Configuration controlling the multi-parameter boundary detector.

    Parameters
    ----------
    he_in, he_out:
        Hysteresis thresholds (cm⁻³) on the smoothed He⁺ density.  Typical
        magnetosphere values for the 2019-01-27 event are ≳0.25 cm⁻³ while the
        sheath drops below ≈0.12 cm⁻³ once the spacecraft leaves the enclosed
        flux tubes.
    he_sigma:
        Softening width (cm⁻³) for the logistic response around the thresholds.
    he_frac_in, he_frac_out, he_frac_sigma:
        Equivalent thresholds for the He⁺ fraction (He⁺/total ion density).
    BN_tol:
        Magnetic-neutral tolerance (nT).  Values with |Bₙ| ≲ BN_tol are treated
        as part of the current sheet / boundary layer.
    bn_sigma:
        Softening width (nT) for the |Bₙ| logistic response.
    bn_flip:
        Magnitude (nT) at which |Bₙ| is considered decisively outside the
        boundary region (typical sheath rotation levels during the event).
    bn_grad_min, bn_grad_sigma:
        Minimum rotation rate (nT/s) and softening width required to promote a
        boundary-layer classification.  Sharp reversals in Bₙ are key
        signatures of the magnetopause.
    tot_jump, tot_grad_sigma:
        Expected density contrast across the magnetopause and the softening
        width for the log-gradient response.  The default factor of 1.8 comes
        from empirical fits to MMS and Cluster crossings (Paschmann & Daly,
        1998; Fuselier et al., 2014).
    smooth_pts:
        Width of the NaN-aware running mean applied to the inputs prior to
        scoring.  Odd values maintain symmetry.
    min_pts:
        Minimum number of consecutive samples required to keep any state.
    min_layer_pts:
        Minimum length for magnetosheath/magnetosphere plateaus (in samples).
    min_mp_layer_pts:
        Minimum length for the boundary-layer state (shorter because the current
        sheet is typically thinner).
    transition_window_s:
        Characteristic timescale (s) used when converting the density
        log-gradient to a contrast ratio.
    """

    he_in: float = 0.25
    he_out: float = 0.12
    he_sigma: float = 0.03
    he_frac_in: float = 0.07
    he_frac_out: float = 0.03
    he_frac_sigma: float = 0.01
    BN_tol: float = 3.0
    bn_sigma: float = 1.2
    bn_flip: float = 6.0
    bn_grad_min: float = 0.06
    bn_grad_sigma: float = 0.02
    tot_jump: float = 1.8
    tot_grad_sigma: float = 0.12
    smooth_pts: int = 9
    min_pts: int = 5
    min_layer_pts: int = 18
    min_mp_layer_pts: int = 5
    transition_window_s: float = 20.0

    def __post_init__(self) -> None:
        if self.smooth_pts < 1:
            raise ValueError("smooth_pts must be ≥1")
        if self.min_pts < 1:
            raise ValueError("min_pts must be ≥1")
        if self.min_layer_pts < self.min_pts:
            raise ValueError("min_layer_pts must be ≥ min_pts")
        if self.min_mp_layer_pts < 1:
            raise ValueError("min_mp_layer_pts must be ≥1")

def _sm_update(
    state: Literal["sheath", "mp_layer", "magnetosphere"],
    he_val: float,
    BN_val: float,
    cfg: DetectorCfg,
    inside_mag: bool,
    ni_val: Optional[float] = None,
) -> Literal["sheath", "mp_layer", "magnetosphere"]:
    """Compatibility helper mirroring the logistic scoring on a single sample."""

    he_inside = _sigmoid((he_val - cfg.he_in) / cfg.he_sigma)
    he_outside = _sigmoid((cfg.he_out - he_val) / cfg.he_sigma)

    if ni_val is not None and ni_val > 0:
        he_frac = he_val / ni_val
        frac_inside = _sigmoid((he_frac - cfg.he_frac_in) / cfg.he_frac_sigma)
        frac_outside = _sigmoid((cfg.he_frac_out - he_frac) / cfg.he_frac_sigma)
    else:
        frac_inside = frac_outside = 0.5

    bn_neutral = _sigmoid((cfg.BN_tol - abs(BN_val)) / cfg.bn_sigma)
    bn_sheath = _sigmoid((abs(BN_val) - cfg.bn_flip) / cfg.bn_sigma)

    magnetosphere_score = (
        0.45 * he_inside + 0.25 * frac_inside + 0.20 * (1.0 - bn_sheath) + 0.10
    )
    sheath_score = 0.45 * he_outside + 0.25 * frac_outside + 0.30 * bn_sheath
    layer_score = (
        0.40 * bn_neutral + 0.20 * (1.0 - bn_sheath) + 0.40 * min(he_inside, he_outside)
    )

    scores = {
        "sheath": sheath_score,
        "mp_layer": layer_score,
        "magnetosphere": magnetosphere_score,
    }

    best_state = max(scores, key=scores.get)
    if scores[best_state] - scores[state] < 0.05:
        return state
    return best_state


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable logistic function."""

    clipped = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _nan_running_mean(arr: np.ndarray, width: int) -> np.ndarray:
    """NaN-aware moving average with symmetric window."""

    if width <= 1:
        return arr
    if width % 2 == 0:
        width += 1
    pad = width // 2
    padded = np.pad(arr, pad_width=pad, mode="constant", constant_values=np.nan)
    kernel = np.ones(width, dtype=float)

    values = np.convolve(np.nan_to_num(padded, nan=0.0), kernel, mode="valid")
    counts = np.convolve((~np.isnan(padded)).astype(float), kernel, mode="valid")

    out = np.empty_like(arr, dtype=float)
    with np.errstate(invalid="ignore"):
        out = np.divide(values, counts, out=np.full_like(arr, np.nan), where=counts > 0)
    return out


def _to_seconds(t: Sequence) -> np.ndarray:
    """Convert a time-like sequence to floating-point seconds."""

    arr = np.asarray(t)
    if arr.size == 0:
        return arr.astype(float)
    if np.issubdtype(arr.dtype, np.datetime64):
        ns = arr.astype("datetime64[ns]").astype(np.int64)
        return ns.astype(float) * 1e-9
    if np.issubdtype(arr.dtype, np.timedelta64):
        ns = arr.astype("timedelta64[ns]").astype(np.int64)
        return ns.astype(float) * 1e-9
    return arr.astype(float)


def detect_crossings_multi(
    t: Sequence,
    he: Sequence,
    BN: Sequence,
    *,
    ni: Optional[Sequence] = None,
    cfg: DetectorCfg = DetectorCfg(),
    good_mask: Optional[Sequence[bool]] = None,
) -> List[Tuple[str, int, int]]:
    """Identify magnetopause regions using He⁺, total density, and Bₙ physics.

    The detector fuses He⁺ density, the He⁺ fraction, the magnitude and
    rotation rate of the normal magnetic-field component, and the total ion
    density contrast to yield a three-state classification:

    - ``"sheath"`` (magnetosheath),
    - ``"mp_layer"`` (boundary / current layer),
    - ``"magnetosphere"``.

    Each channel is smoothed and mapped to a logistic response so that small
    perturbations around the literature thresholds do not trigger spurious
    toggles.  This mirrors the behaviour of the 2019-01-27 IDL-based
    classifier but is implemented in a fully transparent, Pythonic way.

    Parameters
    ----------
    t
        Time axis, convertible to seconds via :func:`numpy.asarray`.  May be a
        ``datetime64`` array, floats in seconds, or similar.

    he
        He⁺ number density (cm⁻³), typically from HPCA omni moments.

    BN
        Normal component of the magnetic field (nT) in the chosen LMN frame.

    ni
        Optional total ion density (cm⁻³), usually FPI-DIS ``N_tot``.  When
        provided, the He⁺ fraction (He⁺/N_tot) is used as an additional
        discriminator between magnetosphere and sheath.

    cfg
        :class:`DetectorCfg` instance controlling thresholds and smoothing
        scales.  Defaults are tuned to the 2019-01-27 event but are broadly
        applicable to similar magnetopause crossings.

    good_mask
        Optional boolean array marking samples that should be considered in
        the classification.  NaNs in ``he``, ``BN`` or ``ni`` are excluded
        automatically regardless of this mask.

    Returns
    -------
    intervals : list of (state, i_start, i_end)
        A list of contiguous index intervals labelled with one of
        ``"sheath"``, ``"mp_layer"``, or ``"magnetosphere"``.  Indices refer to
        the original ``t``/``he`` arrays.

    Notes
    -----
    - The classification is performed in index space and does not enforce any
      absolute duration thresholds beyond those encoded in :class:`DetectorCfg`.
    - Results are typically visualised alongside BN, density, and He⁺ series
      to verify that identified layers match the expected magnetopause
      structure.
    """

    t_arr = np.asarray(t)
    he_arr = np.asarray(he, dtype=float)
    bn_arr = np.asarray(BN, dtype=float)
    if he_arr.shape != bn_arr.shape:
        raise ValueError("he and BN must share the same shape")
    if t_arr.shape[0] != he_arr.shape[0]:
        raise ValueError("time array must match data length")

    if t_arr.size == 0:
        return []

    ni_arr: Optional[np.ndarray]
    if ni is None:
        ni_arr = None
    else:
        ni_arr = np.asarray(ni, dtype=float)
        if ni_arr.shape != he_arr.shape:
            raise ValueError("ni must match he shape")

    if good_mask is None:
        mask = np.ones_like(he_arr, dtype=bool)
    else:
        mask = np.asarray(good_mask, dtype=bool)
        if mask.shape != he_arr.shape:
            raise ValueError("good_mask must match he shape")

    finite_mask = mask & np.isfinite(he_arr) & np.isfinite(bn_arr)
    if ni_arr is not None:
        finite_mask &= np.isfinite(ni_arr)

    if not np.any(finite_mask):
        return []

    he_smooth = _nan_running_mean(np.where(finite_mask, he_arr, np.nan), cfg.smooth_pts)
    bn_smooth = _nan_running_mean(np.where(finite_mask, bn_arr, np.nan), cfg.smooth_pts)

    if ni_arr is not None:
        ni_smooth = _nan_running_mean(np.where(finite_mask, ni_arr, np.nan), cfg.smooth_pts)
        with np.errstate(divide="ignore", invalid="ignore"):
            he_frac = np.divide(
                he_smooth,
                ni_smooth,
                out=np.zeros_like(he_smooth),
                where=np.isfinite(ni_smooth) & (ni_smooth > 0),
            )
    else:
        ni_smooth = None
        he_frac = None

    t_seconds = _to_seconds(t_arr)
    if t_seconds.size > 1:
        bn_grad = np.gradient(bn_smooth, t_seconds, edge_order=2)
    else:
        bn_grad = np.zeros_like(bn_smooth)

    if ni_smooth is not None and t_seconds.size > 1:
        log_ni = np.log(np.clip(ni_smooth, 1e-3, None))
        log_grad = np.abs(np.gradient(log_ni, t_seconds, edge_order=2))
        density_contrast = log_grad * cfg.transition_window_s
    else:
        density_contrast = np.zeros_like(he_smooth)

    he_inside = _sigmoid((he_smooth - cfg.he_in) / cfg.he_sigma)
    he_outside = _sigmoid((cfg.he_out - he_smooth) / cfg.he_sigma)

    if he_frac is not None:
        frac_inside = _sigmoid((he_frac - cfg.he_frac_in) / cfg.he_frac_sigma)
        frac_outside = _sigmoid((cfg.he_frac_out - he_frac) / cfg.he_frac_sigma)
    else:
        frac_inside = np.full_like(he_inside, 0.5)
        frac_outside = np.full_like(he_inside, 0.5)

    bn_neutral = _sigmoid((cfg.BN_tol - np.abs(bn_smooth)) / cfg.bn_sigma)
    bn_sheath = _sigmoid((np.abs(bn_smooth) - cfg.bn_flip) / cfg.bn_sigma)
    bn_rotation = _sigmoid((np.abs(bn_grad) - cfg.bn_grad_min) / cfg.bn_grad_sigma)
    density_jump = _sigmoid((density_contrast - np.log(cfg.tot_jump)) / cfg.tot_grad_sigma)

    def _clean(feature: np.ndarray, fill: float = 0.0) -> np.ndarray:
        out = np.array(feature, dtype=float)
        out[~np.isfinite(out)] = fill
        return out

    he_inside = _clean(he_inside, fill=0.0)
    he_outside = _clean(he_outside, fill=0.0)
    frac_inside = _clean(frac_inside, fill=0.5)
    frac_outside = _clean(frac_outside, fill=0.5)
    bn_neutral = _clean(bn_neutral, fill=0.0)
    bn_sheath = _clean(bn_sheath, fill=0.0)
    bn_rotation = _clean(bn_rotation, fill=0.0)
    density_jump = _clean(density_jump, fill=0.0)

    magnetosphere_score = (
        0.45 * he_inside
        + 0.25 * frac_inside
        + 0.20 * (1.0 - bn_sheath)
        + 0.10 * (1.0 - density_jump)
    )

    sheath_score = (
        0.45 * he_outside + 0.25 * frac_outside + 0.20 * bn_sheath + 0.10 * density_jump
    )

    layer_score = (
        0.40 * bn_neutral
        + 0.30 * bn_rotation
        + 0.30 * np.minimum(he_inside, he_outside)
    )

    scores = np.stack([sheath_score, layer_score, magnetosphere_score], axis=1)
    labels = np.array(["sheath", "mp_layer", "magnetosphere"], dtype=object)
    class_idx = np.argmax(scores, axis=1)
    classifications = labels[class_idx]
    classifications[~finite_mask] = "invalid"

    layers: List[Tuple[str, int, int]] = []
    i = 0
    n = classifications.shape[0]
    while i < n:
        state = classifications[i]
        if state == "invalid":
            i += 1
            continue
        j = i
        while j + 1 < n and classifications[j + 1] == state:
            j += 1
        length = j - i + 1
        if state == "mp_layer":
            if length >= cfg.min_mp_layer_pts:
                layers.append((state, i, j))
        else:
            if length >= cfg.min_layer_pts:
                layers.append((state, i, j))
        i = j + 1

    # merge adjacent layers of the same type to avoid fragmentation
    merged: List[Tuple[str, int, int]] = []
    for layer in layers:
        if merged and merged[-1][0] == layer[0] and merged[-1][2] + 1 >= layer[1]:
            prev = merged.pop()
            merged.append((prev[0], prev[1], layer[2]))
        else:
            merged.append(layer)

    return merged


def extract_enter_exit(
    layers: List[Tuple[str, int, int]], t: Sequence
) -> List[Tuple[float, str]]:
    """Convert layer intervals to entry/exit events in epoch seconds."""

    if not layers:
        return []

    t_arr = np.asarray(t)
    if t_arr.size == 0:
        return []

    if np.issubdtype(t_arr.dtype, np.datetime64):
        seconds = t_arr.astype("datetime64[ns]").astype(np.int64) * 1e-9
    elif np.issubdtype(t_arr.dtype, np.timedelta64):
        seconds = t_arr.astype("timedelta64[ns]").astype(np.int64) * 1e-9
    else:
        seconds = t_arr.astype(float)

    events: List[Tuple[float, str]] = []
    for typ, start, end in layers:
        if typ == "magnetosphere":
            events.append((float(seconds[start]), "enter"))
            events.append((float(seconds[end]), "exit"))

    return sorted(events, key=lambda item: item[0])
