"""Standalone DIS/DES spectrograms with algorithmic crossing markers.

This script generates per-probe ion and electron spectrograms for the
2019-01-27 12:15–12:55 event using the same pySPEDAS-backed loaders and
algorithmic crossing times as the overview panels. It writes eight PNGs
under::

    results/events_pub/2019-01-27_1215-1255/

with filenames::

    spectrogram_ion_mms{p}.png
    spectrogram_electron_mms{p}.png

for MMS probes ``p = 1..4``. Where DES data are genuinely unavailable after
exhausting all local products, a placeholder PNG is written that documents the
gap instead of silently omitting the figure.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

# Ensure repo root (containing ``examples`` and ``mms_mp``) is on sys.path when
# this script is invoked as ``python examples/...`` from the repository root.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mms_mp as mp
from mms_mp import data_loader
import examples.make_overview_20190127_algorithmic as ov  # type: ignore
from examples.make_overview_20190127_algorithmic import (  # type: ignore
    DEFAULT_TRANGE,
    DEFAULT_PROBES,
    DEFAULT_EVENT_DIR,
    _load_event,
    _load_spectrogram_data,
    _load_algorithmic_crossings,
)


def _make_spectrograms_for_species(
    species: str,
    label: str,
    ylabel: str,
    filename_template: str,
) -> None:
    """Build spectrogram PNGs for the requested species ("dis" or "des")."""

    evt = _load_event()
    spec = _load_spectrogram_data(evt)
    crossings = _load_algorithmic_crossings(ov.EVENT_DIR)

    per_probe: Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = spec.get(
        species, {}
    )

    for p in ov.PROBES:
        out_path = ov.EVENT_DIR / filename_template.format(p=p)
        data = per_probe.get(p)

        fig, ax = plt.subplots(figsize=(9.0, 4.0))

        if data is None:
            ax.text(
                0.5,
                0.5,
                f"No {label} spectrogram available for MMS{p}\n"
                "(no suitable omni or distribution products in local cache)",
                ha="center",
                va="center",
                fontsize=9,
                transform=ax.transAxes,
            )
            ax.set_axis_off()
        else:
            t, e, dat = data
            ax_s, _ = mp.spectra.generic_spectrogram(
                t,
                e,
                dat,
                log10=True,
                ylabel=ylabel,
                title=f"MMS{p} {label}",
                ax=ax,
                show=False,
                return_axes=True,
            )
            if ax_s is not None:
                ax = ax_s

            # Overlay algorithmic crossing times as vertical lines
            t_cross = crossings.get(p, np.array([], dtype="datetime64[ns]"))
            for tx in t_cross:
                ax.axvline(
                    tx,
                    color="k",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.7,
                )

        # Enforce the configured TRANGE on the time axis for consistency
        t0, t1 = data_loader._parse_trange(list(ov.TRANGE))
        t0 = t0.astype("datetime64[ns]")
        t1 = t1.astype("datetime64[ns]")
        ax.set_xlim(t0, t1)

        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_path, dpi=220)
        plt.close(fig)

        print(f"[spectrograms] Wrote {out_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone DIS/DES spectrograms with algorithmic crossing markers. "
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


def main() -> None:  # pragma: no cover - event-specific plotting script
    args = _parse_args()
    ov.configure_event(trange=(args.start, args.end), probes=args.probes, event_dir=args.event_dir)

    # Ensure event directory exists (should already from other scripts)
    pathlib.Path(ov.EVENT_DIR).mkdir(parents=True, exist_ok=True)

    _make_spectrograms_for_species(
        species="dis",
        label="ion energy flux (DIS)",
        ylabel="E$_i$ (eV)",
        filename_template="spectrogram_ion_mms{p}.png",
    )

    _make_spectrograms_for_species(
        species="des",
        label="electron energy flux (DES)",
        ylabel="E$_e$ (eV)",
        filename_template="spectrogram_electron_mms{p}.png",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
