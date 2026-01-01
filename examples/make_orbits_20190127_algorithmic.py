"""String-of-pearls orbit visualisation for 2019-01-27 using algorithmic LMN.

This script produces a 2D GSM X–Y "string-of-pearls" plot for MMS1–4 over
12:15–12:55 UT, marking the boundary crossings inferred from the optimised
``algorithmic_lmn`` configuration. For context it overlays a nominal Shue
(1997) magnetopause cross-section for a typical solar-wind pressure.

Outputs
-------

    results/events_pub/2019-01-27_1215-1255/orbit_string_of_pearls_algorithmic.png

The script uses only **local** ephemeris data via ``mms_mp.data_loader`` and
does not re-download MMS files.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Ensure repo root (which contains the ``examples`` and ``mms_mp`` packages)
# is importable when this script is run via ``python examples/...``.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mms_mp import data_loader
import examples.make_overview_20190127_algorithmic as ov  # type: ignore
from examples.make_overview_20190127_algorithmic import (  # type: ignore
    DEFAULT_TRANGE,
    DEFAULT_PROBES,
    DEFAULT_EVENT_DIR,
    configure_event,
    _load_algorithmic_crossings,
)


RE_KM = 6371.0


def _load_event():
    """Load ephemeris for the event window.

    HPCA and EDP are disabled to respect the strict local caching policy and to
    avoid failures when those instruments are unavailable.
    """

    return data_loader.load_event(
        list(ov.TRANGE),
        probes=list(ov.PROBES),
        data_rate_fgm="srvy",
        data_rate_fpi="fast",
        include_hpca=False,
        include_edp=False,
        include_ephem=True,
    )


def _extract_positions(evt) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return per-probe (t, x_RE, y_RE) arrays near the requested TRANGE.

    We first try to restrict strictly to the configured TRANGE. If that leaves
    fewer than two samples for a probe (which should not normally happen for
    MEC ephemeris), we progressively relax to a ±5 min margin and, as a final
    fallback, keep the full available interval.
    """

    t0, t1 = data_loader._parse_trange(list(ov.TRANGE))
    t0 = t0.astype("datetime64[ns]")
    t1 = t1.astype("datetime64[ns]")
    margin = np.timedelta64(5, "m")

    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for p in ov.PROBES:
        probe_data = evt.get(p, {})
        if "POS_gsm" not in probe_data:
            continue

        t_pos, pos = probe_data["POS_gsm"]
        t_arr = data_loader._to_datetime64_any(np.asarray(t_pos)).astype(
            "datetime64[ns]"
        )
        pos = np.asarray(pos, float)
        if pos.ndim != 2 or pos.shape[1] < 2:
            continue

        mask = (t_arr >= t0) & (t_arr <= t1)
        if np.count_nonzero(mask) < 2:
            # Relax to a small margin around TRANGE
            mask = (t_arr >= t0 - margin) & (t_arr <= t1 + margin)
        if np.count_nonzero(mask) < 2:
            # As a last resort, fall back to the full ephemeris track
            mask = slice(None)

        t_sel = t_arr[mask]
        pos_sel = pos[mask]
        if t_sel.size < 2:
            continue

        x_re = pos_sel[:, 0] / RE_KM
        y_re = pos_sel[:, 1] / RE_KM
        out[p] = (t_sel, x_re, y_re)

    return out


def _nearest_index(times: np.ndarray, target: np.datetime64) -> int:
    """Index of *times* closest to *target* (assumes non-empty)."""

    diffs = np.abs(times.astype("datetime64[ns]") - target.astype("datetime64[ns]"))
    return int(np.argmin(diffs))


def _shue_xy_curve(
    sw_pressure_npa: float = 2.0,
    dipole_tilt_deg: float = 0.0,
    n_pts: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Nominal Shue (1997) magnetopause X–Y cross-section in R_E.

    This mirrors the parametrisation used in :func:`mms_mp.coords._shue_normal`:

        r(θ) = r0 * (2 / (1 + cos θ))**α

    and returns the dayside curve for θ ∈ [-π/2, π/2] revolved into the X–Y
    plane. The result is approximate but sufficient for contextual orbit plots.
    """

    r0 = 11.4 * sw_pressure_npa ** (-1 / 6.6)
    alpha = 0.58 - 0.007 * dipole_tilt_deg

    theta = np.linspace(-np.pi / 2, np.pi / 2, n_pts)
    r = r0 * (2.0 / (1.0 + np.cos(theta))) ** alpha
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "String-of-pearls orbit visualisation with algorithmic crossings. "
            "Defaults reproduce the 2019-01-27 12:1512:55 UT event."
        ),
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_TRANGE[0],
        help="Start time UTC, e.g. 2019-01-27/12:15:00",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_TRANGE[1],
        help="End time UTC, e.g. 2019-01-27/12:55:00",
    )
    parser.add_argument(
        "--probes",
        default="".join(DEFAULT_PROBES),
        help=(
            "Probe selection, e.g. '1234', '12', or '1,3'. "
            "Characters are interpreted as individual probe IDs."
        ),
    )
    parser.add_argument(
        "--event-dir",
        default=str(DEFAULT_EVENT_DIR),
        help=(
            "Output directory for figures and diagnostics "
            "(defaults to results/events_pub/2019-01-27_1215-1255)."
        ),
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - event-specific plotting script
    args = _parse_args()
    configure_event(trange=(args.start, args.end), probes=args.probes, event_dir=args.event_dir)

    evt = _load_event()
    pos = _extract_positions(evt)
    crossings = _load_algorithmic_crossings(ov.EVENT_DIR)

    if not pos:
        raise RuntimeError("No ephemeris positions available for requested TRANGE")
    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    # Earth disc in GSM X–Y (1 R_E circle at origin)
    phi = np.linspace(0, 2 * np.pi, 200)
    ax.fill(
        np.cos(phi),
        np.sin(phi),
        color="0.85",
        edgecolor="k",
        linewidth=0.8,
        label="Earth (1 R$_E$)",
    )

    # Shue model dayside magnetopause
    x_shue, y_shue = _shue_xy_curve()
    ax.plot(x_shue, y_shue, "k--", linewidth=1.0, label="Shue '97 MP (~2 nPa)")

    colours = {"1": "C0", "2": "C1", "3": "C2", "4": "C3"}

    all_x: list[float] = []
    all_y: list[float] = []

    # Track whether we've already added a legend entry for the algorithmic
    # crossing markers so that the legend remains compact.
    crossing_label_added = False

    for p in ov.PROBES:
        if p not in pos:
            continue
        t_arr, x, y = pos[p]
        c = colours.get(p, None)
        ax.plot(x, y, label=f"MMS{p}", color=c, linewidth=1.4)

        # Crossing markers at nearest ephemeris sample to each algorithmic crossing
        t_cross = crossings.get(p, np.array([], dtype="datetime64[ns]"))
        for i, tx in enumerate(t_cross):
            idx = _nearest_index(t_arr, tx)
            ax.scatter(
                x[idx],
                y[idx],
                color=c,
                edgecolor="k",
                s=35,
                zorder=5,
                label=(
                    "Algorithmic crossings"
                    if not crossing_label_added and i == 0
                    else None
                ),
            )
        if t_cross.size:
            crossing_label_added = True

        all_x.append(x.min())
        all_x.append(x.max())
        all_y.append(y.min())
        all_y.append(y.max())

    if all_x and all_y:
        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)
        margin = 1.5
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X$_{GSM}$ (R$_E$)")
    ax.set_ylabel("Y$_{GSM}$ (R$_E$)")
    ax.set_title("MMS orbits and algorithmic crossings")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)

    out_path = ov.EVENT_DIR / "orbit_string_of_pearls_algorithmic.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"[orbits] Wrote {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
