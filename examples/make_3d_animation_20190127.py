"""Time-evolving 3D magnetopause visualisation for 2019-01-27.

This script builds a multi-panel animation showing

- B_L, B_M, B_N (algorithmic LMN) at 1 s cadence for MMS1–4
- Continuous region classification (magnetosphere/mp_layer/sheath)
- Discrete boundary crossings from the composite ≥0.4 score
- 3D GSM view with a Shue (1997) magnetopause surface
- MMS1–4 trajectories and instantaneous positions

The output MP4 is written to::

    results/events_pub/2019-01-27_1215-1255/mms_mp_3d_animation_20190127.mp4

The script assumes strict local caching for MMS data; it does not trigger any
re-downloads and relies on mms_mp.data_loader + existing local CDFs.

To convert the MP4 to a GIF externally using ``ffmpeg``, run for example::

    ffmpeg -i mms_mp_3d_animation_20190127.mp4 \
           -vf "fps=10,scale=800:-1:flags=lanczos" \
           -loop 0 mms_mp_3d_animation_20190127.gif
"""

from __future__ import annotations

import pathlib
import sys
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Ensure repo root (which contains the ``mms_mp`` package) is importable when
# this script is executed via ``python examples/...``.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mms_mp as mp
from mms_mp import data_loader
from mms_mp.boundary import detect_crossings_multi, DetectorCfg

import examples.make_overview_20190127_algorithmic as ov  # type: ignore


RE_KM = 6371.0
EVENT_DIR = pathlib.Path("results/events_pub/2019-01-27_1215-1255")
EVENT_DIR.mkdir(parents=True, exist_ok=True)


def _load_event():
    """Load FGM/FPI/MEC for the canonical event with strict local caching."""

    return data_loader.load_event(
        list(ov.TRANGE),
        probes=list(ov.PROBES),
        data_rate_fgm="srvy",
        data_rate_fpi="fast",
        include_hpca=False,
        include_edp=False,
        include_ephem=True,
    )


def _build_algorithmic_lmn_map(evt):
    """Delegate to overview script helper for LMN triads."""

    return ov._build_algorithmic_lmn_map(evt)


def _rotate_B(evt, lmn_map):
    """Rotate B_gsm to (BL, BM, BN) on a 1 s grid per probe."""

    return ov._rotate_B(evt, lmn_map)


def _extract_pos_re(evt) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Resample POS_gsm to 1 s and convert to R_E.

    Returns
    -------
    pos : dict
        pos[p] = (t, x_re, y_re, z_re).
    """

    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for p in ov.PROBES:
        if "POS_gsm" not in evt.get(p, {}):
            continue
        t_pos, pos = evt[p]["POS_gsm"]
        df = data_loader.to_dataframe(t_pos, pos, cols=["x", "y", "z"])
        df = data_loader.resample(df, "1s")
        t = df.index.values.astype("datetime64[ns]")
        arr = df[["x", "y", "z"]].values / RE_KM
        out[p] = (t, arr[:, 0], arr[:, 1], arr[:, 2])
    return out


def _composite_score_single(t_b, b_xyz, t_n, n_i):
    """Lightweight composite boundary score replicating test logic (no BN)."""

    from test_boundary_threshold_case import composite_boundary_score  # type: ignore

    times, score = composite_boundary_score(t_b, b_xyz, t_n, n_i, lmn_matrix=None)
    return np.asarray(times), np.asarray(score, float)


def _build_scores(evt) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute composite boundary scores per probe on native FGM cadence."""

    scores: Dict[str, np.ndarray] = {}
    times: Dict[str, np.ndarray] = {}
    for p in ov.PROBES:
        d = evt.get(p, {})
        if "B_gsm" not in d or "N_tot" not in d:
            continue
        t_b, b_xyz = d["B_gsm"]
        t_n, n_i = d["N_tot"]
        tb_arr, sc = _composite_score_single(t_b, b_xyz, t_n, n_i)
        scores[p] = sc
        # Convert to datetime64 for easier comparison and plotting
        if hasattr(tb_arr[0], "timestamp"):
            dt = np.array([np.datetime64(t) for t in tb_arr])
        else:
            dt = tb_arr.astype("datetime64[s]")
        times[p] = dt
    return times, scores


def _detect_threshold_crossings(times, scores, threshold: float = 0.4, min_sep_s: float = 10.0) -> Dict[str, List[np.datetime64]]:
    """Identify discrete crossings where composite score >= threshold."""

    from test_boundary_threshold_case import detect_crossings  # type: ignore

    out: Dict[str, List[np.datetime64]] = {}
    for p, t_arr in times.items():
        sc = scores.get(p)
        if sc is None or sc.size == 0:
            continue
        # detect_crossings expects seconds or datetime objects; convert to seconds
        if np.issubdtype(t_arr.dtype, np.datetime64):
            sec = t_arr.astype("datetime64[s]").astype("int64")
        else:
            sec = t_arr.astype("int64")
        cts = detect_crossings(sec, sc, threshold=threshold, min_separation_s=min_sep_s)
        out[p] = [np.datetime64(ct) for ct in cts]
    return out


def _build_region_labels(t_ref, evt, B_lmn) -> Tuple[Dict[str, List[Tuple[str, int, int]]], Dict[str, np.ndarray]]:
    """Run :func:`detect_crossings_multi` and map to per-sample region labels.

    There are two operating modes:

    1. **Primary (HPCA He⁺ available)**
       When HPCA He⁺ density is present (exposed as ``evt[p]['N_he']`` or
       ``evt[p]['He']``), the detector uses He⁺ as the cold-plasma tracer and
       optionally the total ion density ``N_tot`` for the He⁺ fraction and
       density-contrast terms. This corresponds to the design of
       :class:`mms_mp.boundary.DetectorCfg`, where ``he_in``/``he_out`` are
       thresholds in cm⁻³ for He⁺ inside vs. outside closed flux tubes.

    2. **Fallback (no He⁺; use N_tot as proxy)**
       When HPCA is unavailable but FPI ion number density ``N_tot`` exists, we
       construct a *pseudo-He* indicator from ``N_tot`` via a monotonic mapping
       that is high for low-density magnetosphere intervals and low for the
       high-density sheath. Physically, cold magnetospheric plasmas on this
       event have much lower total ion density than the magnetosheath, so
       ``N_tot`` can act as a coarse proxy for the presence of cold plasma.
       The resulting pseudo-He is fed into :func:`detect_crossings_multi`
       without altering its internal physics logic; BN rotation and density
       contrast still contribute as before.

    In both modes the function returns, per probe ``p``:

    - ``layers[p]``: a list of ``(state, i1, i2)`` intervals on the 1 s BN grid
      where ``state`` is ``'magnetosphere'``, ``'mp_layer'`` or ``'sheath'``;
    - ``labels[p]``: an array of per-sample region labels defined on ``t_ref``.
    """

    layers: Dict[str, List[Tuple[str, int, int]]] = {}
    labels: Dict[str, np.ndarray] = {}

    for p in ov.PROBES:
        if p not in B_lmn:
            continue

        d = evt.get(p, {})

        # Prefer explicit He+ density (HPCA moments) when available
        he_key = None
        if "N_he" in d:
            he_key = "N_he"
        elif "He" in d:
            he_key = "He"

        have_ni = "N_tot" in d
        if he_key is None and not have_ni:
            # Neither He+ nor total ion density available -> cannot classify
            continue

        t_bn, _, _, BN = B_lmn[p]

        if he_key is not None:
            # --- Primary mode: use HPCA He+ as cold-plasma tracer ---------
            t_he, he = d[he_key]
            he_df = data_loader.to_dataframe(t_he, he, cols=["He"])
            he_df = he_df.reindex(t_bn, method="nearest")
            he_vals = he_df["He"].values

            ni_vals = None
            if have_ni:
                t_ni, ni = d["N_tot"]
                ni_df = data_loader.to_dataframe(t_ni, ni, cols=["Ni"])
                ni_df = ni_df.reindex(t_bn, method="nearest")
                ni_vals = ni_df["Ni"].values

            cfg = DetectorCfg()
        else:
            # --- Fallback mode: derive a pseudo-He indicator from N_tot ----
            # On this event the magnetosheath has significantly higher total
            # ion density than the magnetosphere.  We therefore map N_tot to
            # a unitless pseudo-He quantity that is near 1 in low-density
            # (magnetospheric) intervals and near 0 in high-density (sheath)
            # intervals, so it can be consumed by the existing He+-based
            # scoring logic in DetectorCfg.
            t_ni, ni = d["N_tot"]
            ni_df = data_loader.to_dataframe(t_ni, ni, cols=["Ni"])
            ni_df = ni_df.reindex(t_bn, method="nearest")
            ni_vals = ni_df["Ni"].values

            # Characteristic density scale (cm^-3) separating typical
            # magnetospheric and sheath values for this event.  The exact
            # choice is not critical because the logistic mapping in
            # detect_crossings_multi is robust to modest rescaling.
            N0 = 1.0
            ni_clipped = np.clip(ni_vals, 0.0, None)
            he_vals = 1.0 / (1.0 + ni_clipped / N0)

            # Use the default DetectorCfg: its he_in / he_out thresholds now
            # apply to the pseudo-He indicator rather than a true He+ density.
            cfg = DetectorCfg()

        layer_list = detect_crossings_multi(t_bn, he_vals, BN, ni=ni_vals, cfg=cfg)
        layers[p] = layer_list

        # Build per-sample label on BN grid
        lab_bn = np.full(t_bn.shape, "invalid", dtype=object)
        for state, i1, i2 in layer_list:
            lab_bn[i1 : i2 + 1] = state

        # Interpolate labels onto reference grid by nearest-neighbour in time
        t_bn_ns = t_bn.astype("datetime64[ns]").astype("int64")
        t_ref_ns = t_ref.astype("datetime64[ns]").astype("int64")
        idx = np.searchsorted(t_bn_ns, t_ref_ns, side="left")
        idx = np.clip(idx, 0, len(t_bn_ns) - 1)
        labels[p] = lab_bn[idx]

    return layers, labels


def _make_shue_surface(sw_pressure_npa: float = 2.0, dipole_tilt_deg: float = 0.0, n_theta: int = 80, n_phi: int = 80):
    """Generate a nominal Shue (1997) magnetopause surface in GSM (R_E)."""

    r0 = 11.4 * sw_pressure_npa ** (-1.0 / 6.6)
    alpha = 0.58 - 0.007 * dipole_tilt_deg

    theta = np.linspace(-np.pi / 2, np.pi / 2, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    th, ph = np.meshgrid(theta, phi, indexing="ij")

    r = r0 * (2.0 / (1.0 + np.cos(th))) ** alpha
    x = r * np.cos(th)
    rho = r * np.sin(th)
    y = rho * np.cos(ph)
    z = rho * np.sin(ph)
    return x, y, z


def main() -> None:  # pragma: no cover - orchestration only
    # 1. Load event and build LMN + B_LMN, positions
    evt = _load_event()
    lmn_map = _build_algorithmic_lmn_map(evt)
    B_lmn = _rotate_B(evt, lmn_map)
    pos_re = _extract_pos_re(evt)

    # Use MMS1 B_LMN grid as the master animation time base
    t_master, *_ = B_lmn["1"]

    # 2. Composite boundary scores and discrete crossings
    score_times, score_vals = _build_scores(evt)
    crossings = _detect_threshold_crossings(score_times, score_vals, threshold=0.4, min_sep_s=10.0)

    # 3. Continuous region classification on t_master grid
    layers, region_labels = _build_region_labels(t_master, evt, B_lmn)

    # 4. Prepare animation time indices (Δt = 4 s)
    t0 = t_master[0].astype("datetime64[s]")
    t1 = t_master[-1].astype("datetime64[s]")
    n_frames = int((t1 - t0).astype(int) / 4) + 1
    print(f"[animation] t0={t0} t1={t1} n_frames={n_frames}")
    t_frames = t0 + np.arange(n_frames).astype("timedelta64[s]") * 4

    # Precompute mapping from frame times to indices in t_master
    t_master_s = t_master.astype("datetime64[s]")
    master_ns = t_master_s.astype("int64")
    frame_ns = t_frames.astype("int64")
    frame_idx = np.searchsorted(master_ns, frame_ns, side="left")
    frame_idx = np.clip(frame_idx, 0, len(master_ns) - 1)

    # 5. Build figure layout
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.0, 1.6], height_ratios=[1, 1, 1], hspace=0.05, wspace=0.1)

    ax_BL = fig.add_subplot(gs[0, 0])
    ax_BM = fig.add_subplot(gs[1, 0], sharex=ax_BL)
    ax_BN = fig.add_subplot(gs[2, 0], sharex=ax_BL)
    ax_3d = fig.add_subplot(gs[:, 1], projection="3d")

    colors = {"1": "C0", "2": "C1", "3": "C2", "4": "C3"}
    region_colors = {"magnetosphere": "#b0d5ff", "mp_layer": "#d3b0ff", "sheath": "#d0d0d0"}

    # Static BL/BM/BN curves
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

    # Region shading on BN panel (use MMS1 labels as reference if available)
    if "1" in layers:
        t_bn, *_ = B_lmn["1"]
        for state, i1, i2 in layers["1"]:
            ax_BN.axvspan(t_bn[i1], t_bn[i2], color=region_colors.get(state, "0.9"), alpha=0.25, linewidth=0)

    # Discrete crossing markers on all panels
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

    # Per-probe region label summary text
    region_text = ax_BN.text(0.01, 1.02, "", transform=ax_BN.transAxes, va="bottom", ha="left", fontsize=9)

    # Global timestamp text
    time_text = fig.text(0.02, 0.93, "", fontsize=10, weight="bold")

    # Time cursor lines
    cursor_BL = ax_BL.axvline(t_master[0], color="k", linestyle="-", linewidth=1.2)
    cursor_BM = ax_BM.axvline(t_master[0], color="k", linestyle="-", linewidth=1.2)
    cursor_BN = ax_BN.axvline(t_master[0], color="k", linestyle="-", linewidth=1.2)

    # 3D static content: Shue surface, trajectories, Earth sphere
    Xs, Ys, Zs = _make_shue_surface()
    ax_3d.plot_surface(Xs, Ys, Zs, rstride=2, cstride=2, color="#d0d8f0", alpha=0.25, linewidth=0.0)

    # Earth sphere (approximate)
    phi = np.linspace(0, 2 * np.pi, 60)
    theta_s = np.linspace(0, np.pi, 30)
    th_s, ph_s = np.meshgrid(theta_s, phi)
    r_e = 1.0
    xe = r_e * np.cos(th_s)
    ye = r_e * np.sin(th_s) * np.cos(ph_s)
    ze = r_e * np.sin(th_s) * np.sin(ph_s)
    ax_3d.plot_surface(xe, ye, ze, color="0.8", alpha=0.9, linewidth=0.0)

    # Pre-plot full trajectories and set axis limits
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

    ax_3d.set_xlabel("X_GSM (R_E)")
    ax_3d.set_ylabel("Y_GSM (R_E)")
    ax_3d.set_zlabel("Z_GSM (R_E)")
    ax_3d.view_init(elev=20.0, azim=-60.0)
    ax_3d.legend(loc="upper right", fontsize=8)

    # Animated 3D markers per probe
    scatters = {}
    for p, (t, x, y, z) in pos_re.items():
        c = colors.get(p, None)
        s = ax_3d.scatter([x[0]], [y[0]], [z[0]], s=35, color=c, edgecolor="k", zorder=5)
        scatters[p] = (s, np.asarray(x), np.asarray(y), np.asarray(z))

    def _format_region_summary(idx: int) -> str:
        parts = []
        for p in ov.PROBES:
            lab_arr = region_labels.get(p)
            if lab_arr is None or idx >= lab_arr.size:
                region = "?"
            else:
                region = str(lab_arr[idx])
            col = region_colors.get(region, "black")
            parts.append(f"{{{col}}}MMS{p}: {region}{{reset}}")
        # We cannot embed colours in text easily; keep plain but ordered
        return "   ".join([f"MMS{p}: {region_labels.get(p)[idx] if region_labels.get(p) is not None and idx < region_labels.get(p).size else '?'}" for p in ov.PROBES])

    def init():
        return (
            cursor_BL,
            cursor_BM,
            cursor_BN,
            *[s for s, *_ in scatters.values()],
            region_text,
            time_text,
        )

    def update(frame: int):
        k = int(frame_idx[frame])
        t = t_master[k]

        for cursor in (cursor_BL, cursor_BM, cursor_BN):
            cursor.set_xdata([t, t])

        # Update region text summary; hide invalid classifications to avoid
        # clutter and only show probes with a meaningful state.
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
            region_text.set_text("No valid data")

        # Update timestamp
        time_text.set_text(str(t))

        # Update 3D markers
        for p, (scat, x, y, z) in scatters.items():
            if k < x.size:
                scat._offsets3d = (np.array([x[k]]), np.array([y[k]]), np.array([z[k]]))
        return (
            cursor_BL,
            cursor_BM,
            cursor_BN,
            *[s for s, *_ in scatters.values()],
            region_text,
            time_text,
        )

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=False)

    out_path = EVENT_DIR / "mms_mp_3d_animation_20190127.mp4"
    # Use an explicit bitrate so the ~40 s, 1600x900 animation lands in the
    # 8–12 MB range rather than being over-compressed by ffmpeg defaults. The
    # extra_args nudge ffmpeg towards a roughly constant ~2 Mbit/s stream.
    writer = FFMpegWriter(
        fps=15,
        bitrate=2000,  # target ~2 Mbit/s so the 40 s animation is ~8–12 MB on disk
        extra_args=['-minrate', '2000k', '-maxrate', '2000k', '-bufsize', '4000k'],
    )
    anim.save(out_path, writer=writer, dpi=100)
    plt.close(fig)

    print(f"[animation] Wrote {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()

