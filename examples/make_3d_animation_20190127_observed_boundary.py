"""Time-evolving 3D magnetopause visualisation using an observed boundary.

This companion script to ``make_3d_animation_20190127.py`` replaces the
static Shue (1997) surface with a boundary surface reconstructed from the
MMS constellation itself. The orientation is set by the multi-spacecraft
algorithmic LMN normal and, at each animation frame, the along-normal
position is inferred from where the spacecraft are sampling the
magnetosphere, boundary layer, or magnetosheath.

The output MP4 is written to::

    results/events_pub/2019-01-27_1215-1255/mms_mp_3d_animation_20190127_observed.mp4

The data-loading and LMN construction are delegated to the existing Shue
animation script and overview helper so both animations use the same
inputs and time base for a fair side-by-side comparison.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Ensure repo root (which contains the ``mms_mp`` package) is importable when
# this script is executed via ``python examples/...``.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import examples.make_overview_20190127_algorithmic as ov  # type: ignore
from examples.make_3d_animation_20190127 import (
    _load_event,
    _build_algorithmic_lmn_map,
    _rotate_B,
    _extract_pos_re,
    _build_scores,
    _detect_threshold_crossings,
    _build_region_labels,
)

EVENT_DIR = pathlib.Path("results/events_pub/2019-01-27_1215-1255")
EVENT_DIR.mkdir(parents=True, exist_ok=True)

# Size control for the observed-boundary surface patch in the 3D panel.
#
# The patch half-width ``extent`` (in R_E) is computed as
#
#     extent = max(PATCH_SIZE_FACTOR * span, PATCH_MIN_EXTENT_RE)
#
# where ``span`` is the maximum spatial range of the MMS constellation over the
# event. Increasing ``PATCH_SIZE_FACTOR`` makes the patch extend further beyond
# the spacecraft trajectories, improving visual context relative to the Shue
# surface; ``PATCH_MIN_EXTENT_RE`` enforces a floor so the patch remains
# clearly visible even for very compact constellations.
PATCH_SIZE_FACTOR: float = 2.5
PATCH_MIN_EXTENT_RE: float = 3.0


def _set_equal_aspect(ax) -> None:
    """Force equal scaling on a 3D axes.

    Matplotlib's default 3D projection uses independent x/y/z scales, which can
    visually skew planes so that a surface truly perpendicular to a given
    vector *appears* tilted or nearly parallel to nearby trajectories. Applying
    this helper after setting axis limits makes the spatial orientation of the
    observed-boundary patch easier to interpret.
    """

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    x_mid = 0.5 * (xlim[0] + xlim[1])
    y_mid = 0.5 * (ylim[0] + ylim[1])
    z_mid = 0.5 * (zlim[0] + zlim[1])

    max_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    half = 0.5 * max_range

    ax.set_xlim3d(x_mid - half, x_mid + half)
    ax.set_ylim3d(y_mid - half, y_mid + half)
    ax.set_zlim3d(z_mid - half, z_mid + half)


def _shared_normal(lmn_map: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
    """Estimate a shared magnetopause normal from the algorithmic LMN triads.

    For this event the optimised ``algorithmic_lmn`` configuration yields
    almost identical N for MMS1–4 (mean separation < 10 deg; see
    tests/test_algorithmic_lmn.py). We exploit this by averaging the
    per-probe N vectors to obtain a single boundary normal used to define
    the observed magnetopause surface.
    """

    n_list = []
    for triad in lmn_map.values():
        Np = np.asarray(triad.get("N"), float)
        if Np.shape == (3,) and np.all(np.isfinite(Np)):
            n_list.append(Np / (np.linalg.norm(Np) + 1e-12))
    if not n_list:
        return np.array([1.0, 0.0, 0.0])
    vec = np.vstack(n_list).mean(axis=0)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0])
    return vec / norm


def main() -> None:  # pragma: no cover - orchestration only
    # 1. Load event, build LMN + B_LMN, positions on the 1 s master grid
    evt = _load_event()
    lmn_map = _build_algorithmic_lmn_map(evt)
    B_lmn = _rotate_B(evt, lmn_map)
    pos_re = _extract_pos_re(evt)

    # Use MMS1 B_LMN grid as the master animation time base so that both the
    # Shue-based and observation-based animations are synchronised.
    t_master, *_ = B_lmn["1"]

    # 2. Composite boundary scores and discrete crossings (unchanged)
    score_times, score_vals = _build_scores(evt)
    crossings = _detect_threshold_crossings(
        score_times, score_vals, threshold=0.4, min_sep_s=10.0
    )
    # Simple diagnostic to show how many discrete crossings are available to
    # anchor the observed boundary position.
    n_cross_total = sum(len(v) for v in crossings.values())
    print(f"[animation/observed] total crossing anchors: {n_cross_total}")

    # 3. Continuous region classification on t_master grid
    layers, region_labels = _build_region_labels(t_master, evt, B_lmn)

    # Crossing-direction diagnostics. For each probe, record the time and
    # approximate direction (outbound/inbound) of its last threshold crossing
    # based on the region_labels sequence around that time. This helps
    # interpret whether spacecraft found inside or outside the observed sphere
    # near the end of the interval are consistent with the boundary history.
    last_crossing_time: dict[str, np.datetime64] = {}
    last_crossing_dir: dict[str, str] = {}
    for p, ct_list in crossings.items():
        if not ct_list:
            print(f"[animation/observed] MMS{p}: no discrete crossings above threshold")
            last_crossing_dir[p] = "none"
            continue
        last_ct = ct_list[-1]
        last_crossing_time[p] = last_ct

        ct_arr = np.array(last_ct, dtype=t_master.dtype)
        idx = np.searchsorted(t_master, ct_arr, side="left")
        if idx <= 0:
            k_cross = 0
        elif idx >= t_master.size:
            k_cross = t_master.size - 1
        else:
            if (ct_arr - t_master[idx - 1]) <= (t_master[idx] - ct_arr):
                k_cross = idx - 1
            else:
                k_cross = idx

        direction = "unknown"
        lab_arr = region_labels.get(p)
        if lab_arr is not None and lab_arr.size >= 3:
            k_before = max(k_cross - 1, 0)
            k_after = min(k_cross + 1, lab_arr.size - 1)
            before = str(lab_arr[k_before])
            after = str(lab_arr[k_after])
            if before == "magnetosphere" and after in ("mp_layer", "sheath"):
                direction = "outbound (magnetosphere→sheath)"
            elif before in ("mp_layer", "sheath") and after == "magnetosphere":
                direction = "inbound (sheath→magnetosphere)"

        last_crossing_dir[p] = direction
        print(
            f"[animation/observed] MMS{p} last crossing at {last_ct} classified as {direction}"
        )

    # 4. Build an observation-based boundary model.
    #
    #    Orientation: shared N from the algorithmic LMN triads (physics-driven
    #    timing + MVA + Shue blend), with a small **visual correction** below
    #    to ensure that the rendered plane actually follows the magnetopause
    #    normal rather than its tangent when the LMN triad happens to be
    #    rotated by ~90 degrees relative to the Shue model.
    #
    #    Position: for each time sample along t_master we project each MMS
    #    position onto the final visual normal N_sh and choose the along-normal
    #    offset where either (a) one or more probes are in the boundary layer
    #    (mp_layer) or (b) magnetosphere and sheath populations coexist and we
    #    can place the surface halfway between their mean positions. Times with
    #    insufficient information simply omit the surface.
    N_sh_raw = _shared_normal(lmn_map)
    N_sh = N_sh_raw.copy()
    print(f"[animation/observed] N_sh_raw (GSM, unit) = {N_sh_raw}")

    # Diagnostic: compare N_sh_raw to a Shue-like normal and to the local
    # spacecraft trajectory direction at a representative crossing.
    def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, float).reshape(3)
        b = np.asarray(b, float).reshape(3)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return float("nan")
        c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    # Optional visual override for the boundary normal. If diagnostics reveal
    # that the shared N from the LMN triads is nearly **tangential** to the
    # Shue magnetopause normal (≈90° offset) while one of the tangential LMN
    # axes is instead closely aligned with the Shue normal, we will use that
    # better-aligned axis as the **visual** normal for the rendered plane.
    N_sh_override: Optional[np.ndarray] = None
    override_source: Optional[str] = None

    sample_probe: Optional[str] = None
    for cand in ("1", "2", "3", "4"):
        if crossings.get(cand):
            sample_probe = cand
            break

    if sample_probe is not None and sample_probe in pos_re:
        # Local import to avoid circular dependencies at module import time.
        from mms_mp.coords import _shue_normal

        ct_list = crossings.get(sample_probe, [])
        if ct_list:
            ct = ct_list[len(ct_list) // 2]  # pick a mid-event crossing

            # Position in R_E on the resampled POS grid for the diagnostic probe
            t_pos, x_pos, y_pos, z_pos = pos_re[sample_probe]
            t_ct = np.array(ct, dtype=t_pos.dtype)
            idx = np.searchsorted(t_pos, t_ct, side="left")
            if idx <= 0:
                j = 0
            elif idx >= t_pos.size:
                j = t_pos.size - 1
            else:
                # choose the closer of the neighbours
                if (t_ct - t_pos[idx - 1]) <= (t_pos[idx] - t_ct):
                    j = idx - 1
                else:
                    j = idx

            r = np.array([x_pos[j], y_pos[j], z_pos[j]], float)
            rhat = None
            if np.all(np.isfinite(r)) and np.linalg.norm(r) > 0.0:
                R_obs = float(np.linalg.norm(r))
                rhat = r / R_obs
                print(
                    f"[animation/observed] unit radial (RE) at crossing MMS{sample_probe} = {rhat}"
                )
                print(
                    f"[animation/observed] observed radial distance at crossing MMS{sample_probe} = "
                    f"{R_obs:.2f} R_E"
                )
                print(
                    f"[animation/observed] angle(N_sh_raw, radial@MMS{sample_probe}) = "
                    f"{_angle_deg(N_sh_raw, rhat):.2f} deg"
                )

            # Compute Shue model normal at the same spacecraft location using the
            # high-cadence POS_gsm in km from the event structure.
            tpos_km, pos_km = evt[sample_probe]["POS_gsm"]
            tpos_km = np.asarray(tpos_km, dtype="datetime64[ns]")
            t_ct_ns = t_ct.astype("datetime64[ns]")
            j2 = int(np.argmin(np.abs(tpos_km - t_ct_ns)))
            r_km = np.asarray(pos_km[j2, :3], float)
            n_shue = None
            if np.all(np.isfinite(r_km)) and np.linalg.norm(r_km) > 0.0:
                n_shue = _shue_normal(r_km)
                print(f"[animation/observed] Shue normal at MMS{sample_probe} = {n_shue}")
                print(
                    f"[animation/observed] angle(N_sh_raw, Shue_normal) = "
                    f"{_angle_deg(N_sh_raw, n_shue):.2f} deg"
                )
                if rhat is not None:
                    print(
                        f"[animation/observed] angle(Shue_normal, radial) = "
                        f"{_angle_deg(n_shue, rhat):.2f} deg"
                    )

                    # Compare observed radial distance with the Shue (1997) model
                    # stand-off distance along the same direction. The Shue
                    # surface is axisymmetric around X_GSM; its radius depends
                    # only on the polar angle theta from +X.
                    rho_dir = float(np.sqrt(rhat[1] ** 2 + rhat[2] ** 2))
                    theta_dir = float(np.arctan2(rho_dir, rhat[0]))
                    r0_shue = 11.4 * 2.0 ** (-1.0 / 6.6)
                    alpha_shue = 0.58  # dipole tilt ~ 0 for this event
                    R_shue = float(
                        r0_shue * (2.0 / (1.0 + np.cos(theta_dir))) ** alpha_shue
                    )
                    print(
                        f"[animation/observed] Shue model radius along MMS{sample_probe} direction = "
                        f"{R_shue:.2f} R_E"
                    )

            # Approximate spacecraft trajectory direction near the crossing
            j0 = max(j - 3, 0)
            j1 = min(j + 3, x_pos.size - 1)
            if j1 > j0:
                r0 = np.array([x_pos[j0], y_pos[j0], z_pos[j0]], float)
                r1 = np.array([x_pos[j1], y_pos[j1], z_pos[j1]], float)
                v_vec = r1 - r0
                if np.linalg.norm(v_vec) > 0.0:
                    print(
                        f"[animation/observed] approx V_sc direction (RE, unscaled) = {v_vec}"
                    )
                    print(
                        f"[animation/observed] angle(N_sh_raw, approx V_sc@MMS{sample_probe}) = "
                        f"{_angle_deg(N_sh_raw, v_vec):.2f} deg"
                    )

            triad = lmn_map.get(sample_probe)
            if triad is not None:
                L_vec = np.asarray(triad.get("L"), float)
                M_vec = np.asarray(triad.get("M"), float)
                N_vec = np.asarray(triad.get("N"), float)
                print(f"[animation/observed] LMN for MMS{sample_probe}:")
                print(f"  L = {L_vec}")
                print(f"  M = {M_vec}")
                print(f"  N = {N_vec}")
                print(
                    f"[animation/observed] angle(N_sh_raw, N_LMN) = {_angle_deg(N_sh_raw, N_vec):.2f} deg"
                )
                print(
                    f"[animation/observed] angle(N_sh_raw, L_LMN) = {_angle_deg(N_sh_raw, L_vec):.2f} deg"
                )
                print(
                    f"[animation/observed] angle(N_sh_raw, M_LMN) = {_angle_deg(N_sh_raw, M_vec):.2f} deg"
                )
                if rhat is not None:
                    print(
                        f"[animation/observed] angle(N_LMN, radial) = "
                        f"{_angle_deg(N_vec, rhat):.2f} deg"
                    )
                if n_shue is not None:
                    print(
                        f"[animation/observed] angle(N_LMN, Shue_normal) = "
                        f"{_angle_deg(N_vec, n_shue):.2f} deg"
                    )

                    # Choose the LMN axis that best matches the Shue normal. We
                    # treat ±L, ±M, ±N as equivalent for alignment purposes and
                    # keep the sign that makes the candidate co-directed with
                    # the Shue normal.
                    sh_unit = n_shue / (np.linalg.norm(n_shue) + 1e-12)

                    def _best_aligned_axis() -> Tuple[str, np.ndarray, float]:
                        best_name = ""
                        best_vec = np.zeros(3, float)
                        best_angle = 180.0
                        for name, vec in ("N_LMN", N_vec), ("L_LMN", L_vec), ("M_LMN", M_vec):
                            v = vec / (np.linalg.norm(vec) + 1e-12)
                            cos = float(np.dot(v, sh_unit))
                            angle = float(
                                np.degrees(
                                    np.arccos(np.clip(abs(cos), -1.0, 1.0))
                                )
                            )
                            if angle < best_angle:
                                best_angle = angle
                                # Orient candidate to be co-directed with Shue
                                best_vec = v * (1.0 if cos >= 0.0 else -1.0)
                                best_name = name
                        return best_name, best_vec, best_angle

                    best_name, best_vec, best_angle = _best_aligned_axis()
                    current_angle = _angle_deg(N_sh_raw, n_shue)
                    print(
                        f"[animation/observed] best-aligned LMN axis vs Shue = {best_name} "
                        f"(angle = {best_angle:.2f} deg, current N angle = {current_angle:.2f} deg)"
                    )

                    # Only override if a *different* axis is substantially
                    # closer to the Shue normal (improvement > 30 deg). This
                    # avoids perturbing well-behaved events where N is already
                    # close to the Shue/physical normal.
                    if best_name != "N_LMN" and (current_angle - best_angle) > 30.0:
                        N_sh_override = best_vec / (np.linalg.norm(best_vec) + 1e-12)
                        override_source = best_name

    # Finalise the visual normal to be used for the plane construction.
    if N_sh_override is not None:
        N_sh = N_sh_override
        print(
            f"[animation/observed] Using {override_source} as visual boundary normal; "
            f"N_sh (GSM, unit) = {N_sh}"
        )
    else:
        print(
            f"[animation/observed] Using algorithmic shared normal as visual N_sh; "
            f"N_sh (GSM, unit) = {N_sh}"
        )

    t_master_ns = t_master.astype("datetime64[ns]").astype("int64")
    pos_master: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for p, (t_pos, x, y, z) in pos_re.items():
        t_pos_ns = t_pos.astype("datetime64[ns]").astype("int64")
        idx = np.searchsorted(t_pos_ns, t_master_ns, side="left")
        idx = np.clip(idx, 0, len(t_pos_ns) - 1)
        pos_master[p] = (
            np.asarray(x)[idx],
            np.asarray(y)[idx],
            np.asarray(z)[idx],
        )

    # Build boundary position time series that are always defined over the
    # full animation interval.
    #
    #  (1) plane_offsets[k]: along-normal position N_sh·r, appropriate for the
    #      original planar patch visualisation.
    #  (2) R_offsets[k]: true radial distance |r| from the Earth, which we use
    #      as the radius of the observed spherical magnetopause surface.
    #
    # Both series are anchored at the discrete boundary crossings detected
    # from the composite score ("crossings"). For each such crossing we take
    # the spacecraft position r in R_E, forming anchor pairs
    #
    #     (t_cross, s_cross = N_sh·r, R_cross = |r|).
    #
    # We then linearly interpolate these anchors in time to obtain smooth
    # plane_offsets[k] and R_offsets[k] for every t_master[k], using
    # np.interp endpoint behaviour to "maintain the last known boundary"
    # outside the explicit crossing interval.
    #
    # If, for some reason, no usable crossings exist for this event, we fall
    # back to a Shue-like subsolar stand-off distance r0 for both series.
    n_samples = t_master.size
    plane_offsets = np.full(n_samples, np.nan, dtype=float)
    R_offsets = np.full(n_samples, np.nan, dtype=float)

    anchor_idx: list[float] = []
    anchor_offsets: list[float] = []
    anchor_radii: list[float] = []

    for p, ct_list in crossings.items():
        pos = pos_master.get(p)
        if pos is None:
            continue
        x_arr, y_arr, z_arr = pos
        if x_arr.size == 0:
            continue
        for ct in ct_list:
            # Find the nearest index in the 1 s master grid to this
            # crossing time.
            ct_arr = np.array(ct, dtype=t_master.dtype)
            idx = np.searchsorted(t_master, ct_arr, side="left")
            if idx <= 0:
                k = 0
            elif idx >= n_samples:
                k = n_samples - 1
            else:
                # Choose the closer of the two neighbouring samples.
                if (ct_arr - t_master[idx - 1]) <= (t_master[idx] - ct_arr):
                    k = idx - 1
                else:
                    k = idx
            if k >= x_arr.size:
                continue
            r = np.array([x_arr[k], y_arr[k], z_arr[k]], float)
            s = float(np.dot(N_sh, r))
            R_mag = float(np.linalg.norm(r))
            anchor_idx.append(float(k))
            anchor_offsets.append(s)
            anchor_radii.append(R_mag)

    if anchor_offsets:
        # Sort anchors in time and interpolate across the full index range.
        idxs = np.asarray(anchor_idx, dtype=float)
        vals_s = np.asarray(anchor_offsets, dtype=float)
        vals_R = np.asarray(anchor_radii, dtype=float)
        order = np.argsort(idxs)
        idxs = idxs[order]
        vals_s = vals_s[order]
        vals_R = vals_R[order]
        x = np.arange(n_samples, dtype=float)
        plane_offsets = np.interp(x, idxs, vals_s)
        R_offsets = np.interp(x, idxs, vals_R)
    else:
        # No usable crossing anchors: fall back to a Shue-like subsolar
        # stand-off distance. Shue (1997) gives the subsolar radius as
        #
        #     r0 = 11.4 * P_sw^(-1/6.6)
        #
        # Using the same nominal dynamic pressure P_sw = 2 nPa as the Shue
        # surface helper and projecting the nose point (r0, 0, 0) onto N_sh
        # yields a reasonable constant along-normal offset for this event; we
        # also use r0 itself as the default radial stand-off.
        r0 = 11.4 * 2.0 ** (-1.0 / 6.6)
        plane_offsets.fill(float(N_sh[0] * r0))
        R_offsets.fill(float(r0))

    # Diagnostics: summarise the distribution of offsets. This is a quick check
    # that the surface is not static and that it spans a reasonable range in
    # R_E.
    valid_plane = plane_offsets[np.isfinite(plane_offsets)]
    if valid_plane.size:
        po_min = float(valid_plane.min())
        po_max = float(valid_plane.max())
        po_std = float(valid_plane.std())
        print(
            f"[animation/observed] plane_offsets stats: n={valid_plane.size}, "
            f"min={po_min:.2f}, max={po_max:.2f}, std={po_std:.2f}"
        )

    valid_R = R_offsets[np.isfinite(R_offsets)]
    if valid_R.size:
        R_min = float(valid_R.min())
        R_max = float(valid_R.max())
        R_std = float(valid_R.std())
        print(
            f"[animation/observed] R_offsets (radial) stats: n={valid_R.size}, "
            f"min={R_min:.2f}, max={R_max:.2f}, std={R_std:.2f}"
        )

    # 5. Prepare animation frame indices (Δt = 4 s)
    t0 = t_master[0].astype("datetime64[s]")
    t1 = t_master[-1].astype("datetime64[s]")
    n_frames = int((t1 - t0).astype(int) / 4) + 1
    print(f"[animation/observed] t0={t0} t1={t1} n_frames={n_frames}")
    t_frames = t0 + np.arange(n_frames).astype("timedelta64[s]") * 4

    t_master_s = t_master.astype("datetime64[s]")
    master_ns = t_master_s.astype("int64")
    frame_ns = t_frames.astype("int64")
    frame_idx = np.searchsorted(master_ns, frame_ns, side="left")
    frame_idx = np.clip(frame_idx, 0, len(master_ns) - 1)

    # 6. Build figure layout (BL/BM/BN panel unchanged from Shue script)
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.0, 1.6], height_ratios=[1, 1, 1], hspace=0.05, wspace=0.1)

    ax_BL = fig.add_subplot(gs[0, 0])
    ax_BM = fig.add_subplot(gs[1, 0], sharex=ax_BL)
    ax_BN = fig.add_subplot(gs[2, 0], sharex=ax_BL)
    ax_3d = fig.add_subplot(gs[:, 1], projection="3d")

    colors = {"1": "C0", "2": "C1", "3": "C2", "4": "C3"}
    region_colors = {"magnetosphere": "#b0d5ff", "mp_layer": "#d3b0ff", "sheath": "#d0d0d0"}

    for p, (t, BL, BM, BN) in B_lmn.items():
        c = colors.get(p, None)
        ax_BL.plot(t, BL, label=f"MMS{p}", color=c, linewidth=1.0)
        ax_BM.plot(t, BM, label=f"MMS{p}", color=c, linewidth=1.0)
        ax_BN.plot(t, BN, label=f"MMS{p}", color=c, linewidth=1.0)

    ax_BL.set_ylabel("B_L (nT)")
    ax_BM.set_ylabel("B_M (nT)")
    ax_BN.set_ylabel("B_N (nT)")
    ax_BN.set_xlabel("Time (UT)")
    ax_BL.legend(loc="upper right", fontsize=8, ncol=2)

    if "1" in layers:
        t_bn, *_ = B_lmn["1"]
        for state, i1, i2 in layers["1"]:
            ax_BN.axvspan(t_bn[i1], t_bn[i2], color=region_colors.get(state, "0.9"), alpha=0.25, linewidth=0)

    for p, ct_list in crossings.items():
        c = colors.get(p, None)
        for ct in ct_list:
            for ax in (ax_BL, ax_BM, ax_BN):
                ax.axvline(ct, color=c, linestyle="--", linewidth=0.8, alpha=0.7)

    for ax in (ax_BL, ax_BM, ax_BN):
        ax.grid(True, alpha=0.3)
        ax.set_xlim(t_master[0], t_master[-1])

    # Only show x-axis tick labels on the bottom BN panel. BL and BM share
    # the same time axis but suppressing their labels avoids overlapping
    # date strings while keeping the cursor and crossings synchronised.
    for ax in (ax_BL, ax_BM):
        ax.tick_params(labelbottom=False)

    fig.autofmt_xdate()

    # region_text summarises current layer classifications (magnetosphere /
    # mp_layer / sheath) for probes with valid labels. When no classification
    # is available we leave it blank; this does *not* indicate missing
    # ephemeris or field data.
    region_text = ax_BN.text(0.01, 1.02, "", transform=ax_BN.transAxes, va="bottom", ha="left", fontsize=9)
    time_text = fig.text(0.02, 0.93, "", fontsize=10, weight="bold")
    # Display the instantaneous observed radial stand-off distance R_offsets[k]
    # used for the spherical boundary surface.
    R_text = ax_3d.text2D(
        0.02,
        0.95,
        "",
        transform=ax_3d.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )
    cursor_BL = ax_BL.axvline(t_master[0], color="k", linestyle="-", linewidth=1.2)
    cursor_BM = ax_BM.axvline(t_master[0], color="k", linestyle="-", linewidth=1.2)
    cursor_BN = ax_BN.axvline(t_master[0], color="k", linestyle="-", linewidth=1.2)

    # 7. 3D static content: Earth sphere, trajectories, and axis limits
    #
    # Reuse a standard spherical parameterisation (theta, phi) both for the
    # Earth (R=1 R_E) and, below, for the observed magnetopause surface whose
    # radius R(t) will be driven by the data-derived R_offsets time series
    # (radial distance of the boundary from the Earth).
    phi = np.linspace(0, 2 * np.pi, 60)
    theta_s = np.linspace(0, np.pi, 30)
    th_s, ph_s = np.meshgrid(theta_s, phi)

    r_e = 1.0
    xe = r_e * np.cos(th_s)
    ye = r_e * np.sin(th_s) * np.cos(ph_s)
    ze = r_e * np.sin(th_s) * np.sin(ph_s)
    ax_3d.plot_surface(xe, ye, ze, color="0.8", alpha=0.9, linewidth=0.0)

    all_xyz = []
    for p, (t, x, y, z) in pos_re.items():
        c = colors.get(p, None)
        ax_3d.plot(x, y, z, color=c, linewidth=1.2, label=f"MMS{p}")
        all_xyz.append(np.vstack([x, y, z]).T)
    if all_xyz:
        xyz = np.vstack(all_xyz)
        xmin, ymin, zmin = np.min(xyz, axis=0)
        xmax, ymax, zmax = np.max(xyz, axis=0)
        margin = 1.5
        ax_3d.set_xlim(xmin - margin, xmax + margin)
        ax_3d.set_ylim(ymin - margin, ymax + margin)
        ax_3d.set_zlim(zmin - margin, zmax + margin)
        span = max(xmax - xmin, ymax - ymin, zmax - zmin)
        # Use the constellation barycentre as a reference point for positioning
        # the observed boundary patch so it stays spatially co-located with the
        # spacecraft ensemble as it moves.
        center_ref = xyz.mean(axis=0)
    else:
        span = 20.0
        center_ref = np.zeros(3, dtype=float)

    # Use equal aspect so that the orientation of the observed boundary patch
    # relative to the MMS trajectories is not distorted by unequal x/y/z
    # scaling.
    _set_equal_aspect(ax_3d)

    ax_3d.set_xlabel("X_GSM (R_E)")
    ax_3d.set_ylabel("Y_GSM (R_E)")
    ax_3d.set_zlabel("Z_GSM (R_E)")
    ax_3d.view_init(elev=20.0, azim=-60.0)
    ax_3d.legend(loc="upper right", fontsize=8)

    # Observed-boundary surface geometry
    # ---------------------------------
    # Instead of a flat planar patch, we now render the magnetopause boundary as
    # a sphere centred on the Earth. The radius R(t) of this sphere is taken
    # from the R_offsets(t) time series, which explicitly tracks the radial
    # distance |r| of the boundary as inferred from the spacecraft crossings.

    def _sphere_coords(radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return Cartesian coordinates for a sphere of given radius (in R_E).

        The mesh is centred at the origin (Earth) and uses the same (th_s,
        ph_s) angular grid as the Earth sphere above, so that the observed
        magnetopause can be compared directly to the Shue model surface in the
        companion animation.
        """

        r = float(radius)
        if not np.isfinite(r) or r <= 0.0:
            r = 1.0
        xs = r * np.cos(th_s)
        ys = r * np.sin(th_s) * np.cos(ph_s)
        zs = r * np.sin(th_s) * np.sin(ph_s)
        return xs, ys, zs

    boundary = {"surf": None}

    # Animated 3D markers per probe (positions evaluated on t_master via pos_master)
    scatters = {}
    for p in ov.PROBES:
        pos = pos_master.get(p)
        if pos is None:
            continue
        x_arr, y_arr, z_arr = pos
        if x_arr.size == 0:
            continue
        c = colors.get(p, None)
        s = ax_3d.scatter([x_arr[0]], [y_arr[0]], [z_arr[0]], s=35, color=c, edgecolor="k", zorder=5)
        scatters[p] = (s, x_arr, y_arr, z_arr)

    def init():
        artists = [
            cursor_BL,
            cursor_BM,
            cursor_BN,
            *(s for s, *_ in scatters.values()),
            region_text,
            time_text,
            R_text,
        ]
        k0 = int(frame_idx[0])
        d0 = R_offsets[k0]
        if np.isfinite(d0):
            XB, YB, ZB = _sphere_coords(float(d0))
            boundary["surf"] = ax_3d.plot_surface(
                XB, YB, ZB, rstride=2, cstride=2, color="#c5b0e6", alpha=0.35, linewidth=0.0
            )
            artists.append(boundary["surf"])
        return tuple(artists)

    def update(frame: int):
        k = int(frame_idx[frame])
        t = t_master[k]

        for cursor in (cursor_BL, cursor_BM, cursor_BN):
            cursor.set_xdata([t, t])

        summary = []
        for p in ov.PROBES:
            lab_arr = region_labels.get(p)
            if lab_arr is None or k >= lab_arr.size:
                continue
            region = str(lab_arr[k])
            if region == "invalid":
                continue
            summary.append(f"MMS{p}: {region}")
        if summary:
            region_text.set_text("   ".join(summary))
        else:
            # No valid region labels at this time; leave the summary blank to
            # avoid suggesting that the underlying data are missing.
            region_text.set_text("")

        time_text.set_text(str(t))

        R_val = R_offsets[k]
        if np.isfinite(R_val):
            R_text.set_text(f"R_obs = {R_val:.2f} R_E")
        else:
            R_text.set_text("R_obs = NaN")

        for p, (scat, x_arr, y_arr, z_arr) in scatters.items():
            if k < x_arr.size:
                scat._offsets3d = (
                    np.array([x_arr[k]]),
                    np.array([y_arr[k]]),
                    np.array([z_arr[k]]),
                )

        # End-of-interval diagnostics: at the final animation frame, compare
        # spacecraft radial distances |r| to the observed boundary radius
        # R_offsets[k] and to the composite score / region labels.
        if frame == n_frames - 1:
            R_end = float(R_offsets[k]) if np.isfinite(R_offsets[k]) else float("nan")
            print(
                f"[animation/observed] End-frame diagnostics at t={t}: "
                f"R_obs={R_end:.2f} R_E"
            )
            for p in ov.PROBES:
                pos = pos_master.get(p)
                rmag = float("nan")
                delta = float("nan")
                loc = "unknown"
                if pos is not None:
                    x_arr_p, y_arr_p, z_arr_p = pos
                    if k < x_arr_p.size:
                        r_vec = np.array(
                            [x_arr_p[k], y_arr_p[k], z_arr_p[k]], float
                        )
                        rmag = float(np.linalg.norm(r_vec))
                        if np.isfinite(R_end):
                            delta = rmag - R_end
                            if delta >= 0.0:
                                loc = "outside (sheath-side, |r| > R_obs)"
                            else:
                                loc = "inside (magnetosphere-side, |r| < R_obs)"

                region_label = "N/A"
                lab_arr = region_labels.get(p)
                if lab_arr is not None and k < lab_arr.size:
                    region_label = str(lab_arr[k])

                score_val = float("nan")
                score_state = "n/a"
                t_sc = score_times.get(p)
                sc_arr = score_vals.get(p)
                if t_sc is not None and sc_arr is not None and sc_arr.size:
                    t_sc_s = t_sc.astype("datetime64[s]")
                    t_end_s = t.astype("datetime64[s]")
                    ts_int = t_sc_s.astype("int64")
                    te_int = t_end_s.astype("int64")
                    j = np.searchsorted(ts_int, te_int, side="left")
                    if j <= 0:
                        idx_sc = 0
                    elif j >= ts_int.size:
                        idx_sc = ts_int.size - 1
                    else:
                        if (te_int - ts_int[j - 1]) <= (ts_int[j] - te_int):
                            idx_sc = j - 1
                        else:
                            idx_sc = j
                    score_val = float(sc_arr[idx_sc])
                    score_state = ">=0.4" if score_val >= 0.4 else "<0.4"

                last_ct = last_crossing_time.get(p)
                last_dir = last_crossing_dir.get(p, "none")

                print(
                    f"[animation/observed]   MMS{p}: |r|={rmag:.2f} R_E, "
                    f"Delta=|r|-R_obs={delta:.2f} R_E, loc={loc}, "
                    f"region={region_label}, score={score_val:.2f} ({score_state}), "
                    f"last_cross={last_ct} ({last_dir})"
                )

        d = R_offsets[k]
        if np.isfinite(d):
            XB, YB, ZB = _sphere_coords(float(d))
            if boundary["surf"] is not None:
                boundary["surf"].remove()
            boundary["surf"] = ax_3d.plot_surface(
                XB, YB, ZB, rstride=2, cstride=2, color="#c5b0e6", alpha=0.35, linewidth=0.0
            )
        else:
            if boundary["surf"] is not None:
                boundary["surf"].remove()
                boundary["surf"] = None

        artists = [
            cursor_BL,
            cursor_BM,
            cursor_BN,
            *(s for s, *_ in scatters.values()),
            region_text,
            time_text,
            R_text,
        ]
        if boundary["surf"] is not None:
            artists.append(boundary["surf"])
        return tuple(artists)

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=False)

    out_path = EVENT_DIR / "mms_mp_3d_animation_20190127_observed.mp4"
    writer = FFMpegWriter(
        fps=15,
        bitrate=2000,
        extra_args=["-minrate", "2000k", "-maxrate", "2000k", "-bufsize", "4000k"],
    )
    anim.save(out_path, writer=writer, dpi=100)
    plt.close(fig)

    print(f"[animation/observed] Wrote {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()

