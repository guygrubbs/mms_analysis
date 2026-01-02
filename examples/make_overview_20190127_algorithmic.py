"""Per-probe overview figures for 2019-01-27 using optimised algorithmic LMN.

Each overview panel includes, for a single MMS probe:

- B_L, B_M, B_N (algorithmic LMN) at 1 s cadence
- Ion (DIS) spectrogram with algorithmic crossing markers
- Electron (DES) spectrogram with algorithmic crossing markers

All spectrograms are built from **local** MMS data only using
`pyspedas.mms.fpi` distribution products (``dis-dist`` / ``des-dist``)
with a fallback to FPI omni energy spectra
(``*_energyspectr_omni_*``) when full distributions are not available,
and the ``mms_mp.spectra`` helpers. No re-downloads are triggered.

Outputs are written under::

    results/events_pub/2019-01-27_1215-1255/

with filenames::

    overview_mms{p}.png

where ``p`` is 1–4.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

import mms_mp as mp
from mms_mp import data_loader


# Default configuration for the canonical 2019-01-27 event. The script can be
# reused for other crossings via CLI arguments, but these values remain the
# publication defaults.
DEFAULT_TRANGE = ("2019-01-27/12:15:00", "2019-01-27/12:55:00")
DEFAULT_PROBES = ["1", "2", "3", "4"]
DEFAULT_EVENT_DIR = pathlib.Path("results/events_pub/2019-01-27_1215-1255")

TRANGE = DEFAULT_TRANGE
PROBES = list(DEFAULT_PROBES)
EVENT_DIR = DEFAULT_EVENT_DIR
EVENT_DIR.mkdir(parents=True, exist_ok=True)


def configure_event(
    trange: Tuple[str, str] | None = None,
    probes: tuple[str, ...] | list[str] | str | None = None,
    event_dir: str | pathlib.Path | None = None,
):
    """Update global event configuration for this and companion scripts.

    Parameters
    ----------
    trange:
        Two-element iterable of UTC strings (``start``, ``end``) in the
        ``YYYY-mm-dd/HH:MM:SS`` format.
    probes:
        Iterable or compact string of probe indices (e.g. ``["1","2"]`` or
        ``"12"`` / ``"1,2"``).
    event_dir:
        Output directory for figures and diagnostics.
    """

    global TRANGE, PROBES, EVENT_DIR

    if trange is not None:
        TRANGE = (str(trange[0]), str(trange[1]))

    if probes is not None:
        if isinstance(probes, str):
            tokens = probes.replace(",", " ").split()
            parsed: list[str] = []
            for tok in tokens:
                parsed.extend(list(tok))
            PROBES = sorted({p for p in parsed if p})
        else:
            PROBES = [str(p) for p in probes]

    if event_dir is not None:
        EVENT_DIR = pathlib.Path(event_dir)
        EVENT_DIR.mkdir(parents=True, exist_ok=True)

    return TRANGE, PROBES, EVENT_DIR


def _load_event():
    """Minimal event load: FGM, FPI moments, MEC only.

    HPCA and EDP are disabled to respect the strict local caching policy and
    to avoid failing when those instruments are unavailable for this
    interval.
    """

    return data_loader.load_event(
        list(TRANGE),
        probes=list(PROBES),
        data_rate_fgm="srvy",
        data_rate_fpi="fast",
        include_hpca=False,
        include_edp=False,
        include_ephem=True,
    )


def _build_algorithmic_lmn_map(evt, window_half_width_s: float = 30.0) -> Dict[str, Dict[str, np.ndarray]]:
    """Wrapper around ``coords.algorithmic_lmn`` for this specific event.

    We reproduce the logic from ``analyze_20190127_dn_shear`` rather than
    importing it directly to keep this script self-contained and unit-test
    friendly.
    """

    from mms_mp.coords import algorithmic_lmn

    b_times: Dict[str, np.ndarray] = {}
    b_vals: Dict[str, np.ndarray] = {}
    pos_times: Dict[str, np.ndarray] = {}
    pos_vals: Dict[str, np.ndarray] = {}

    for p in PROBES:
        tB, B = evt[p]["B_gsm"]
        tpos, pos = evt[p]["POS_gsm"]
        b_times[p] = np.asarray(tB)
        b_vals[p] = np.asarray(B)
        pos_times[p] = np.asarray(tpos)
        pos_vals[p] = np.asarray(pos)

    # Crossing times approximated from the curated event analysis.
    #
    # ``coords.algorithmic_lmn`` expects ``t_cross`` in **epoch seconds**
    # (float).  The original implementation here passed ``numpy.datetime64``
    # objects directly, which are interpreted as nanoseconds since 1970 when
    # cast to ``float``.  That mismatch (seconds vs. nanoseconds) caused the
    # MVA windows to expand to the full interval and subtly distorted the
    # blended normal, yielding an N direction nearly **tangential** to the
    # Shue model normal for this event.
    #
    # To avoid this 1e9 scaling error we explicitly convert the curated UTC
    # times to epoch seconds before calling ``algorithmic_lmn``.
    t_cross_utc = {
        "1": "2019-01-27T12:43:25",
        "2": "2019-01-27T12:43:26",
        "3": "2019-01-27T12:43:18",
        "4": "2019-01-27T12:43:26",
    }

    epoch = np.datetime64("1970-01-01T00:00:00", "ns")
    t_cross: Dict[str, float] = {}
    for p, ts in t_cross_utc.items():
        dt64 = np.datetime64(ts.replace("/", "T"), "ns")
        dt = dt64 - epoch
        # Convert ``timedelta64[ns]`` → float seconds
        t_cross[p] = dt / np.timedelta64(1, "s")

    lmn_per_probe = algorithmic_lmn(
        b_times=b_times,
        b_gsm=b_vals,
        pos_times=pos_times,
        pos_gsm_km=pos_vals,
        t_cross=t_cross,
        window_half_width_s=window_half_width_s,
        tangential_strategy="Bmean",
        normal_weights=(0.8, 0.15, 0.05),
    )

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for p, triad in lmn_per_probe.items():
        out[p] = {"L": triad.L, "M": triad.M, "N": triad.N}
    return out


def _rotate_B(evt, lmn_map) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Rotate B_gsm to (BL, BM, BN) on a 1 s grid for each probe."""

    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for p in PROBES:
        tB, B = evt[p]["B_gsm"]
        df_B = data_loader.to_dataframe(tB, B, cols=["Bx", "By", "Bz"])
        df_B = data_loader.resample(df_B, "1s")
        t = df_B.index.values.astype("datetime64[ns]")
        B_arr = df_B[["Bx", "By", "Bz"]].values

        triad = lmn_map.get(p)
        if triad is None:
            continue
        L = np.asarray(triad["L"], float)
        M = np.asarray(triad["M"], float)
        N = np.asarray(triad["N"], float)
        R = np.vstack([L, M, N])  # (3,3)
        BL, BM, BN = (B_arr @ R.T).T
        out[p] = (t, BL, BM, BN)
    return out


def _load_spectrogram_data(evt) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Load DIS/DES spectrograms for each probe using local data only.

    This helper deliberately avoids higher-level metadata wrappers and works
    directly with the standard MMS FPI products via pySPEDAS. For physically
    meaningful **energy flux** scaling it prefers the official
    ``*_energyspectr_omni_*`` variables (which are derived from the full
    distributions by the FPI team and carry documented energy-flux units).
    When those omni spectra are not available, it falls back to collapsing the
    4-D ``dis-dist`` / ``des-dist`` distributions with
    :func:`mms_mp.spectra._collapse_fpi` to obtain an omni-like (time, energy)
    array.

    Returns
    -------
    spec : dict
        ``spec[species][probe] = (t, e, dat2d)`` with
        ``species in {"dis", "des"}`` and ``probe`` in PROBES.
    """

    from pyspedas import mms, get_data
    from pytplot import data_quants

    tr_start, tr_end = data_loader._parse_trange(list(TRANGE))
    t0 = tr_start.astype("datetime64[ns]")
    t1 = tr_end.astype("datetime64[ns]")

    spec: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        "dis": {},
        "des": {},
    }

    # For DES we optionally build an empirical calibration that maps collapsed
    # 4-D distributions into the same energy-flux units as the official omni
    # spectra. We derive this using MMS1–3 (where both dist and omni typically
    # exist) and later apply the aggregate factor to MMS4 when only a dist
    # product is available.
    des_scale_samples: list[float] = []
    des_from_dist: dict[str, bool] = {}

    # Work species-by-species and probe-by-probe to keep the logic simple and
    # robust. For each (species, probe) pair we
    #   1) ensure the corresponding ``*-dist`` CDF is loaded (local cache only),
    #   2) locate a 4-D flux variable under the canonical ``mms{p}_{species}*``
    #      namespace, and
    #   3) locate an energy table (typically ``*_energy_fast``), reducing it to
    #      1-D before collapsing the flux with ``_collapse_fpi``.
    for species in ("dis", "des"):
        for p in PROBES:
            base = f"mms{p}_{species}"

            t_arr = None
            e_arr = None
            dat2d = None

            print(f"[spectrogram/debug] ==== species={species} probe={p} ====")

            # Prefer fast cadence, fall back to srvy if needed.
            for rate in ("fast", "srvy"):
                print(f"[spectrogram/debug]  trying data_rate={rate} for {base}-dist")
                try:
                    mms.fpi(
                        trange=list(TRANGE),
                        probe=p,
                        data_rate=rate,
                        datatype=f"{species}-dist",
                        level="l2",
                        time_clip=True,
                    )
                except Exception as exc:
                    print(f"[spectrogram/debug]   mms.fpi failed for {base} {rate}: {exc}")
                    continue

                # Snapshot of relevant tplot variables for this base name.
                keys_for_base = [k for k in data_quants.keys() if k.startswith(base)]
                print(
                    f"[spectrogram/debug]   data_quants keys for {base}: "
                    f"{keys_for_base}"
                )

                # Identify a flux variable.
                #
                # We *first* look for 2-D omni-directional energy spectra
                # (``*_energyspectr_omni_*``), which are the recommended FPI
                # energy-flux products with physically meaningful units. If no
                # suitable omni spectrum exists we fall back to a 4-D
                # distribution ("true" FPI dist) and collapse its angular
                # dimensions.
                flux_name: str | None = None
                flux_kind: str | None = None  # "4d" or "2d"

                # --- 2-D omni spectrum path (preferred for flux scaling) ---
                omni_candidates = [
                    f"{base}_energyspectr_omni_{rate}",
                    f"{base}_energyspectr_omni_fast",
                    f"{base}_energyspectr_omni_srvy",
                ]
                for nm in omni_candidates:
                    qa2 = data_quants.get(nm)
                    if qa2 is not None and getattr(qa2, "ndim", None) == 2:
                        flux_name = nm
                        flux_kind = "2d"
                        break
                if flux_name is not None:
                    print(
                        f"[spectrogram/debug]   using 2-D omni flux_name={flux_name}, "
                        f"ndim={getattr(data_quants.get(flux_name), 'ndim', None)}, "
                        f"shape={getattr(data_quants.get(flux_name), 'shape', None)}"
                    )
                else:
                    # --- 4-D distribution fallback ---
                    cand4 = f"{base}_dist_{rate}"
                    qa4 = data_quants.get(cand4)
                    if qa4 is None or getattr(qa4, "ndim", None) != 4:
                        cand4 = None
                        for k, v in data_quants.items():
                            if not k.startswith(base):
                                continue
                            if getattr(v, "ndim", None) == 4:
                                cand4 = k
                                break
                    if cand4 is not None:
                        flux_name = cand4
                        flux_kind = "4d"
                        print(
                            f"[spectrogram/debug]   no 2-D omni; using 4-D dist "
                            f"flux4_name={flux_name}, ndim="
                            f"{getattr(data_quants.get(flux_name), 'ndim', None)}, "
                            f"shape={getattr(data_quants.get(flux_name), 'shape', None)}"
                        )
                    else:
                        print(
                            f"[spectrogram/debug]   no 2-D omni or 4-D flux "
                            f"variable found for {base} at rate={rate}"
                        )
                        continue

                # Identify an energy table; favour explicit *_energy_{rate}.
                energy_var = None
                preferred_energy = [
                    f"{base}_energy_{rate}",
                    f"{base}_energy_fast",
                    f"{base}_energy_srvy",
                    f"{base}_energy",
                ]
                for nm in preferred_energy:
                    if nm in data_quants:
                        energy_var = nm
                        break
                if energy_var is None:
                    for k in data_quants.keys():
                        if k.startswith(f"{base}_energy"):
                            energy_var = k
                            break
                if energy_var is None:
                    print(
                        f"[spectrogram/debug]   no energy table found for {base} "
                        f"at rate={rate}"
                    )
                    continue

                print(
                    f"[spectrogram/debug]   using energy_var={energy_var}, "
                    f"shape={getattr(data_quants.get(energy_var), 'shape', None)}"
                )

                # Retrieve raw arrays. get_data may return (t, data) or
                # (t, data, metadata); we only care about the first two
                # entries.
                try:
                    flux_res = get_data(flux_name)
                    e_res = get_data(energy_var)
                except Exception as exc:
                    print(
                        f"[spectrogram/debug]   get_data failed for {flux_name} / "
                        f"{energy_var}: {exc}"
                    )
                    continue

                if not (isinstance(flux_res, (tuple, list)) and len(flux_res) >= 2):
                    print(
                        f"[spectrogram/debug]   flux_res malformed for {flux_name}: "
                        f"type={type(flux_res)}"
                    )
                    continue
                if not (isinstance(e_res, (tuple, list)) and len(e_res) >= 2):
                    print(
                        f"[spectrogram/debug]   e_res malformed for {energy_var}: "
                        f"type={type(e_res)}"
                    )
                    continue

                t_raw, flux_data = flux_res[0], flux_res[1]
                e_raw = e_res[1]
                if flux_data is None or e_raw is None:
                    print(
                        f"[spectrogram/debug]   flux_data or e_raw is None for {base} "
                        f"rate={rate}"
                    )
                    continue

                t_arr = data_loader._to_datetime64_any(np.asarray(t_raw))
                e_arr = np.asarray(e_raw)
                print(
                    f"[spectrogram/debug]   raw shapes: t={t_arr.shape}, "
                    f"flux={np.shape(flux_data)}, e_raw={e_arr.shape}"
                )

                # Reduce energy table to 1-D: typical shape for L2 dist is
                # (n_time, n_energy). We keep a representative energy grid by
                # averaging over time.
                if e_arr.ndim == 2:
                    # Choose axis=0 when (n_time, n_energy).
                    if e_arr.shape[0] >= e_arr.shape[1]:
                        e_arr = np.nanmean(e_arr, axis=0)
                    else:
                        e_arr = np.nanmean(e_arr, axis=1)
                elif e_arr.ndim > 2:
                    # Fallback: flatten then trim later to match flux.
                    e_arr = e_arr.reshape(-1)

                print(
                    f"[spectrogram/debug]   reduced energy shape={e_arr.shape} "
                    f"(e_len={e_arr.size})"
                )

                try:
                    if flux_kind == "4d":
                        dat2d = mp.spectra._collapse_fpi(
                            np.asarray(flux_data), e_len=e_arr.size, method="sum"
                        )
                    else:
                        # 2-D omni spectrum; ensure (time, energy) shape.
                        dat2d = np.asarray(flux_data)
                        if dat2d.ndim > 2:
                            # Conservatively collapse any extra axes.
                            dat2d = np.nansum(dat2d, axis=tuple(range(2, dat2d.ndim)))

                        # DES-only: where we have an official omni spectrum,
                        # estimate a dist->omni scale factor using the matching
                        # 4-D distribution so we can later rescale MMS4 when it
                        # lacks an omni product.
                        if species == "des":
                            cand4 = f"{base}_dist_{rate}"
                            qa4 = data_quants.get(cand4)
                            if qa4 is None or getattr(qa4, "ndim", None) != 4:
                                cand4 = None
                                for k, v in data_quants.items():
                                    if not k.startswith(base):
                                        continue
                                    if getattr(v, "ndim", None) == 4:
                                        cand4 = k
                                        break
                            if cand4 is not None:
                                try:
                                    dist_res = get_data(cand4)
                                    if (
                                        isinstance(dist_res, (tuple, list))
                                        and len(dist_res) >= 2
                                    ):
                                        dist_data = dist_res[1]
                                        if dist_data is not None:
                                            scale = mp.spectra.estimate_dist_to_omni_scale(
                                                flux4d=np.asarray(dist_data),
                                                omni2d=dat2d,
                                                e_len=e_arr.size,
                                            )
                                            if np.isfinite(scale) and scale > 0:
                                                des_scale_samples.append(scale)
                                                print(
                                                    "[spectrogram/debug]   DES "
                                                    "dist->omni scale sample from "
                                                    f"MMS{p} rate={rate}: {scale:.3e}"
                                                )
                                except Exception as exc2:
                                    print(
                                        "[spectrogram/debug]   DES dist->omni scale "
                                        f"estimation failed for {base} rate={rate}: {exc2}"
                                    )
                except Exception as exc:
                    print(
                        f"[spectrogram/debug]   spectrogram collapse failed for {base} "
                        f"rate={rate}: {exc}"
                    )
                    t_arr = e_arr = dat2d = None
                    continue

                dat2d = np.asarray(dat2d)
                nan_frac = float(np.isnan(dat2d).sum()) / float(dat2d.size)
                print(
                    f"[spectrogram/debug]   collapsed dat2d shape={dat2d.shape}, "
                    f"nan_frac={nan_frac:.3f}, "
                    f"min={np.nanmin(dat2d) if dat2d.size else 'NA'}, "
                    f"max={np.nanmax(dat2d) if dat2d.size else 'NA'}"
                )

                # Successfully built a spectrogram for this (species, probe).
                break

            if t_arr is None or e_arr is None or dat2d is None:
                print(
                    f"[spectrogram/debug]   no usable spectrogram built for {species} "
                    f"MMS{p} in any rate"
                )
                continue

            # Restrict to event window and ensure shapes match.
            t_arr = t_arr.astype("datetime64[ns]")
            mask = (t_arr >= t0) & (t_arr <= t1)
            n_in = int(np.count_nonzero(mask))
            if n_in < 4:
                t_min = t_arr[0] if t_arr.size else "NA"
                t_max = t_arr[-1] if t_arr.size else "NA"
                print(
                    f"[spectrogram/debug]   insufficient samples in TRANGE for "
                    f"{species} MMS{p}: count={n_in}, t_range_total=[{t_min}, {t_max}], "
                    f"TRANGE=[{t0}, {t1}]"
                )
                continue

            t_sel = t_arr[mask]
            dat_sel = np.asarray(dat2d)[mask]

            if dat_sel.ndim != 2 or dat_sel.shape[0] != t_sel.size:
                print(
                    f"[spectrogram/debug]   dat_sel shape mismatch for {species} MMS{p}: "
                    f"t_sel={t_sel.shape}, dat_sel={dat_sel.shape}"
                )
                continue

            if dat_sel.shape[1] != e_arr.size:
                # Conservative: trim to min dimension to keep axes consistent.
                nE = min(dat_sel.shape[1], e_arr.size)
                e_sel = e_arr[:nE]
                dat_sel = dat_sel[:, :nE]
            else:
                e_sel = e_arr

            nan_frac_sel = float(np.isnan(dat_sel).sum()) / float(dat_sel.size)
            print(
                f"[spectrogram/debug]   FINAL {species} MMS{p}: "
                f"t={t_sel.shape}, e={e_sel.shape}, dat={dat_sel.shape}, "
                f"nan_frac={nan_frac_sel:.3f}, t_range=[{t_sel[0]}, {t_sel[-1]}]"
            )

            spec[species][p] = (t_sel, e_sel, dat_sel)

            if species == "des":
                des_from_dist[p] = (flux_kind == "4d")

	    # Post-processing: if we successfully derived one or more DES
	    # dist->omni scale factors from MMS1-3 and MMS4 DES came from a
	    # distribution-only product, rescale MMS4 so its colour scale is
	    # comparable to the omni-based MMS1-3 spectrograms.
    des_spec = spec.get("des", {})
    if des_scale_samples and des_from_dist.get("4") and "4" in des_spec:
        scale_arr = np.asarray(des_scale_samples, float)
        good = scale_arr[np.isfinite(scale_arr) & (scale_arr > 0)]
        if good.size:
            scale = float(np.nanmedian(good))
            t4, e4, dat4 = des_spec["4"]
            des_spec["4"] = (t4, e4, dat4 * scale)
            spec["des"] = des_spec
            print(
                "[spectrogram/debug]   applied DES MMS4 dist->omni calibration: "
                f"scale={scale:.3e} based on {len(des_scale_samples)} sample(s)"
            )
    elif "4" in des_spec and des_from_dist.get("4"):
        print(
            "[spectrogram/debug]   no DES dist->omni calibration samples; "
            "leaving MMS4 DES distribution in native units"
        )

    return spec


def _load_algorithmic_crossings(event_dir: pathlib.Path) -> Dict[str, np.ndarray]:
    """Load algorithmic crossing times per probe from CSV.

    The CSV is generated by ``examples.analyze_20190127_dn_shear.py`` and
    must be present in ``event_dir``. If missing, this function returns an
    empty mapping and the plots will simply omit vertical markers.
    """

    import csv

    path = event_dir / "diagnostics" / "crossings_algorithmic.csv"
    if not path.exists():
        return {}

    out: Dict[str, list[np.datetime64]] = {p: [] for p in PROBES}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row.get("probe")
            ts = row.get("t_cross") or row.get("time")
            if p not in out or not ts:
                continue
            try:
                dt = np.datetime64(ts.replace("/", "T"), "ns")
            except Exception:
                continue
            out[p].append(dt)

    return {p: np.array(v, dtype="datetime64[ns]") for p, v in out.items() if v}


def _make_overview_for_probe(p: str,
                             B_lmn: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                             spec: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
                             crossings: Dict[str, np.ndarray]):
    """Create the three-panel overview figure for one probe."""

    if p not in B_lmn:
        return

    t, BL, BM, BN = B_lmn[p]

    # Diagnostics for B_LMN content before plotting
    if t.size == 0:
        print(f"[overview/debug] MMS{p} B_LMN: EMPTY time series")
        return
    try:
        bl_min, bl_max = float(np.nanmin(BL)), float(np.nanmax(BL))
        bm_min, bm_max = float(np.nanmin(BM)), float(np.nanmax(BM))
        bn_min, bn_max = float(np.nanmin(BN)), float(np.nanmax(BN))
        print(
            f"[overview/debug] MMS{p} B_LMN: t.shape={t.shape}, "
            f"BL=[{bl_min:.2f}, {bl_max:.2f}], "
            f"BM=[{bm_min:.2f}, {bm_max:.2f}], "
            f"BN=[{bn_min:.2f}, {bn_max:.2f}], "
            f"t_range=[{t[0]}, {t[-1]}]"
        )
    except ValueError:
        # In case all-NaN arrays slip through, still log and skip plotting
        print(f"[overview/debug] MMS{p} B_LMN: time series present but all-NaN")
        return
    dis = spec.get("dis", {}).get(p)
    des = spec.get("des", {}).get(p)

    # Use a 3x2 GridSpec so that the left-column axes (data panels) all share
    # identical widths, while the right column is reserved for colour bars.
    # This keeps the time axes perfectly aligned across panels even when
    # spectrograms have colour bars. Give the rows a bit more vertical
    # breathing room (hspace) so titles never overlap adjacent panels.
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[40, 1], hspace=0.25, wspace=0.05)

    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax)

    cax2 = fig.add_subplot(gs[1, 1])
    cax3 = fig.add_subplot(gs[2, 1])

    axes = [ax, ax2, ax3]

    # Panel 1 – B_L, B_M, B_N
    ax.plot(t, BL, label="B_L")
    ax.plot(t, BM, label="B_M")
    ax.plot(t, BN, label="B_N")
    ax.set_ylabel("B (nT)")
    ax.set_title(f"MMS{p} – B in algorithmic LMN frame")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 2 – Ion spectrogram
    if dis is not None:
        t_i, e_i, dat_i = dis
        ax_i, _ = mp.spectra.generic_spectrogram(
            t_i,
            e_i,
            dat_i,
            log10=True,
            ylabel="E$_i$ (eV)",
            title="Ion energy flux (DIS)",
            ax=ax2,
            cax=cax2,
            show=False,
            return_axes=True,
        )
        if ax_i is not None:
            ax2 = ax_i

    # Panel 3 – Electron spectrogram
    if des is not None:
        t_e, e_e, dat_e = des
        ax_e, _ = mp.spectra.generic_spectrogram(
            t_e,
            e_e,
            dat_e,
            log10=True,
            ylabel="E$_e$ (eV)",
            title="Electron energy flux (DES)",
            ax=ax3,
            cax=cax3,
            show=False,
            return_axes=True,
        )
        if ax_e is not None:
            ax3 = ax_e

    # Overlay algorithmic crossing markers where available
    t_cross = crossings.get(p, np.array([], dtype="datetime64[ns]"))
    for t_x in t_cross:
        for ax in axes:
            ax.axvline(t_x, color="k", linestyle="--", linewidth=0.8, alpha=0.7)

    # Enforce the configured TRANGE on all subplots to avoid subtle axis drift
    t0, t1 = data_loader._parse_trange(list(TRANGE))
    t0 = t0.astype("datetime64[ns]")
    t1 = t1.astype("datetime64[ns]")

    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlim(t0, t1)

    # Let the GridSpec hspace handle vertical spacing; tight_layout tends to
    # fight with the explicit colour-bar axes and can cause title overlap.
    fig.autofmt_xdate()

    out_path = EVENT_DIR / f"overview_mms{p}.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[overview] Wrote {out_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Per-probe overview figures for MMS using optimised algorithmic LMN. "
            "Defaults reproduce the 2019-01-27 12:1512:55 event."
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


def main() -> None:  # pragma: no cover - thin orchestration
    args = _parse_args()
    configure_event(trange=(args.start, args.end), probes=args.probes, event_dir=args.event_dir)

    evt = _load_event()
    lmn_map = _build_algorithmic_lmn_map(evt, window_half_width_s=30.0)
    B_lmn = _rotate_B(evt, lmn_map)

    # Build spectrograms. Due to a subtle pySPEDAS/pytplot initialisation
    # quirk, the very first call to ``_load_spectrogram_data`` sometimes
    # misses DES MMS4 even though a second call in the same process sees it
    # (and the standalone spectrogram script does plot DES MMS4). To make the
    # overview robust and keep behaviour consistent with the standalone
    # spectrograms, we call the loader twice and merge the results.
    spec_primary = _load_spectrogram_data(evt)
    spec_retry = _load_spectrogram_data(evt)
    for species in ("dis", "des"):
        spec_primary.get(species, {}).update(spec_retry.get(species, {}))
    spec = spec_primary
    crossings = _load_algorithmic_crossings(EVENT_DIR)

    for p in PROBES:
        _make_overview_for_probe(p, B_lmn, spec, crossings)


if __name__ == "__main__":  # pragma: no cover
    main()
