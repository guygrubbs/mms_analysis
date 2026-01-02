"""
Magnetic-only boundary crossing figures for 2019-01-27 12:15–12:55.

- Per-spacecraft plots of B in LMN (BL, BM, BN) with boundary layers and crossings
- Combined BN overview for MMS1–4
- CSV of crossing events with UTC times and GSM positions

Outputs: results/events_pub/2019-01-27_1215-1255
No data-source comparisons; clear publication-ready styling.
"""
from __future__ import annotations
import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure repo root on path
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mms_mp as mp

EVENT_DIR = pathlib.Path('results/events_pub/2019-01-27_1215-1255')
EVENT_DIR.mkdir(parents=True, exist_ok=True)
TRANGE = ('2019-01-27/12:15:00', '2019-01-27/12:55:00')
PROBES = ('1','2','3','4')

# We will use LMN from .sav when available (no comparison shown on plots)
try:
    from tools.idl_sav_import import load_idl_sav
    _SAV = load_idl_sav('references/IDL_Code/mp_lmn_systems_20190127_1215_1255_mp_ver2.sav')
    _LMN_PER_PROBE = _SAV.get('lmn', {})
except Exception:
    _LMN_PER_PROBE = {}


def _load_event():
	    # HPCA is not required for these magnetic-only figures; disable it to
	    # avoid failing when He+ moments are unavailable or cannot be
	    # reconstructed for this interval.
	    return mp.load_event(
	        list(TRANGE),
	        probes=list(PROBES),
	        include_ephem=True,
	        include_hpca=False,
	        data_rate_fgm='srvy',
	        data_rate_fpi='fast',
	    )


def _rotate_B_to_LMN(B_df: pd.DataFrame,
                     probe: str,
                     pos_mid: np.ndarray | None = None,
                     lmn_map: dict | None = None):
    """Rotate GSM B into LMN for one probe.

    Priority of LMN sources:
    1. Physics-driven *algorithmic* LMN (if provided via ``lmn_map``).
    2. Authoritative .sav LMN for this event.
    3. Legacy ``hybrid_lmn`` using a local B-window + midpoint position.
    """

    if lmn_map is not None and probe in lmn_map:
        entry = lmn_map[probe]
        L = np.asarray(entry['L'], float)
        M = np.asarray(entry['M'], float)
        N = np.asarray(entry['N'], float)
    elif probe in _LMN_PER_PROBE:
        entry = _LMN_PER_PROBE[probe]
        L = np.asarray(entry['L'], float)
        M = np.asarray(entry['M'], float)
        N = np.asarray(entry['N'], float)
    else:
        # Fallback: legacy hybrid_lmn (diagnostic only)
        mid = len(B_df) // 2
        i0 = max(0, mid - 200)
        i1 = min(len(B_df), mid + 200)
        B_win = B_df.iloc[i0:i1].values
        lmn = mp.coords.hybrid_lmn(B_win, pos_gsm_km=pos_mid, eig_ratio_thresh=2.0)
        L, M, N = lmn.L, lmn.M, lmn.N
    R = np.vstack([L,M,N]).T
    BL_MN = B_df.values @ R
    BL = pd.Series(BL_MN[:,0], index=B_df.index, name='BL')
    BM = pd.Series(BL_MN[:,1], index=B_df.index, name='BM')
    BN = pd.Series(BL_MN[:,2], index=B_df.index, name='BN')
    return BL, BM, BN


def _compute_crossings(t_idx: pd.DatetimeIndex, BN: pd.Series, evt_probe: dict, cadence='1s'):
    from mms_mp.boundary import detect_crossings_multi
    he = pd.Series(np.nan, index=t_idx)
    ni = pd.Series(np.nan, index=t_idx)
    if 'N_he' in evt_probe and evt_probe['N_he'][0] is not None:
        t_he, he_vals = evt_probe['N_he']
        he_df = mp.data_loader.to_dataframe(t_he, he_vals, cols=['He'])
        he_df = mp.data_loader.resample(he_df, cadence)
        he = he_df['He'].reindex(t_idx, method='nearest')
    if 'N_tot' in evt_probe and evt_probe['N_tot'][0] is not None:
        t_ni, ni_vals = evt_probe['N_tot']
        ni_df = mp.data_loader.to_dataframe(t_ni, ni_vals, cols=['Ni'])
        ni_df = mp.data_loader.resample(ni_df, cadence)
        ni = ni_df['Ni'].reindex(t_idx, method='nearest')
    layers = detect_crossings_multi(t_idx.values, he.values, BN.values, ni=ni.values)
    return layers


def _nearest_idx(times: np.ndarray, target: np.datetime64) -> int:
    # times: datetime64[ns] array
    # Return index of nearest time
    diffs = np.abs(times - target)
    return int(np.argmin(diffs))


def _position_at(time_utc: np.datetime64, pos_t: np.ndarray, pos_xyz: np.ndarray) -> np.ndarray:
    # Return GSM position [X,Y,Z] in km nearest to time_utc
    i = _nearest_idx(pos_t, time_utc)
    return np.asarray(pos_xyz[i], float)


def _per_probe_plot(probe: str, B_df: pd.DataFrame, BL: pd.Series, BM: pd.Series, BN: pd.Series,
                    layers, pos_t: np.ndarray, pos_xyz: np.ndarray, vt: dict):
    fig, ax = plt.subplots(figsize=(12, 5))
    # Plot BL, BM lightly; BN bold
    ax.plot(BL.index, BL.values, lw=0.8, alpha=0.7, label='B_L')
    ax.plot(BM.index, BM.values, lw=0.8, alpha=0.7, label='B_M')
    ax.plot(BN.index, BN.values, lw=1.3, color='k', label='B_N')
    ax.set_ylabel('B (nT) in LMN')
    ax.grid(True, alpha=0.25)
    # Shade vt intervals if provided.  Use *naive* UTC timestamps to stay
    # consistent with the DatetimeIndex produced by ``data_loader.to_dataframe``.
    # Mixing tz-aware (UTC) and naive timestamps on the same axis can cause the
    # B_LMN curves to fall outside the visible x-limits even though the data are
    # present, which is what manifested in the "overlay-only" boundary plots.
    vt_labeled = False
    if vt and probe in vt:
        for (t0s, t1s) in vt[probe]:
            t0 = pd.to_datetime(t0s)  # naive UTC
            t1 = pd.to_datetime(t1s)
            if t1 < t0:
                t0, t1 = t1, t0
            label = 'VT intervals' if not vt_labeled else None
            ax.axvspan(t0, t1, color="k", alpha=0.05, label=label)
            if label is not None:
                vt_labeled = True

    # Mark crossing start/end and annotate positions.  Use distinct legend
    # entries for magnetosphere-side vs magnetosheath/other crossings, without
    # duplicating entries when multiple crossings exist.
    annotations = []
    mag_labeled = False
    sheath_labeled = False
    for typ, i1, i2 in layers or []:
        for idx in [i1, i2]:
            if idx is None:
                continue
            idx = int(idx)
            if 0 <= idx < len(BN):
                t_cross = BN.index[idx]
                if typ == 'magnetosphere':
                    label = 'Magnetosphere-side crossing' if not mag_labeled else None
                    ax.axvline(t_cross, color='r', ls='--', lw=1.0, alpha=0.7, label=label)
                    if label is not None:
                        mag_labeled = True
                else:
                    label = 'Magnetosheath/other crossing' if not sheath_labeled else None
                    ax.axvline(t_cross, color='g', ls='--', lw=1.0, alpha=0.7, label=label)
                    if label is not None:
                        sheath_labeled = True
                pos = _position_at(t_cross.to_datetime64(), pos_t, pos_xyz)
                annotations.append({'probe': probe, 'type': typ, 'time': t_cross.isoformat(),
                                    'X_km': float(pos[0]), 'Y_km': float(pos[1]), 'Z_km': float(pos[2])})
    # Enforce the canonical TRANGE on the time axis using the same helper as
    # the overview script, which returns ``datetime64[ns]`` (naive UTC).
    t0, t1 = mp.data_loader._parse_trange(list(TRANGE))
    t0 = t0.astype("datetime64[ns]")
    t1 = t1.astype("datetime64[ns]")
    ax.set_xlim(t0, t1)
    ax.legend(loc='upper right', frameon=False)
    ax.set_title(f'MMS{probe} Magnetic Field (LMN) — Boundary Crossings')
    fig.autofmt_xdate()
    fig.tight_layout()
    out = EVENT_DIR / f'boundary_BLMN_mms{probe}.png'
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return annotations


def _combined_bn_plot(BN_map: dict, vt: dict, crossings_per_probe: dict):
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    # Track whether we've already added legend entries for each overlay type so
    # the combined legend remains compact and non-redundant.
    vt_labeled = False
    mag_labeled = False
    sheath_labeled = False
    for i, p in enumerate(PROBES):
        key = str(p)
        ax = axes[i]
        bn = BN_map.get(key)
        if bn is None:
            continue

        # B_N is indexed by a naive ``DatetimeIndex``; keep all other time
        # annotations naive as well so Matplotlib does not silently shift the
        # axis and hide the field curves outside the visible window.
        ax.plot(bn.index, bn.values, lw=1.1, color="k", label="B_N")

        is_first = (i == 0)

        if vt and key in vt:
            for (t0s, t1s) in vt[key]:
                t0 = pd.to_datetime(t0s)  # naive UTC
                t1 = pd.to_datetime(t1s)
                if t1 < t0:
                    t0, t1 = t1, t0
                label = "VT intervals" if (is_first and not vt_labeled) else None
                ax.axvspan(t0, t1, color="k", alpha=0.05, label=label)
                if label is not None:
                    vt_labeled = True
        # Crossing markers
        for typ, i1, i2 in (crossings_per_probe.get(key) or []):
            for idx in [i1, i2]:
                try:
                    t_cross = bn.index[int(idx)]
                except Exception:
                    continue
                if typ == 'magnetosphere':
                    label = "Magnetosphere-side crossing" if (is_first and not mag_labeled) else None
                    ax.axvline(t_cross, color='r', ls='--', lw=1.0, alpha=0.6, label=label)
                    if label is not None:
                        mag_labeled = True
                else:
                    label = "Magnetosheath/other crossing" if (is_first and not sheath_labeled) else None
                    ax.axvline(t_cross, color='g', ls='--', lw=1.0, alpha=0.6, label=label)
                    if label is not None:
                        sheath_labeled = True
        ax.set_ylabel(f'MMS{p}\nB_N (nT)')
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(loc='upper right', frameon=False)
    axes[-1].set_xlabel("Time (UTC)")

    # Apply a shared x-limit corresponding to the event TRANGE, again using
    # the same helper as the overview script for consistent naive UTC times.
    t0, t1 = mp.data_loader._parse_trange(list(TRANGE))
    t0 = t0.astype("datetime64[ns]")
    t1 = t1.astype("datetime64[ns]")
    axes[-1].set_xlim(t0, t1)
    fig.suptitle('Boundary Crossing Overview — B_N (LMN), MMS1–4')
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0,0,1,0.96])
    out = EVENT_DIR / 'combined_BN_overview.png'
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _vt_intervals():
    import re
    path = pathlib.Path('references/IDL_Code/requested_mp_motion_givenlmn_vion.pro')
    if not path.exists():
        return {}
    txt = path.read_text(encoding='utf-8', errors='ignore')
    block = re.findall(r"If time_string\(trange_full\[0\]\) eq '2019-01-27/04:00:00' then begin(.*?)endif", txt, flags=re.S)
    out = { '1': [], '2': [], '3': [], '4': [] }
    if block:
        b = block[0]
        for sc, tag in [('1','vt_mms1'), ('2','vt_mms2'), ('3','vt_mms3'), ('4','vt_mms4')]:
            m = re.search(tag + r"= time_double\(\[(.*?)\]", b, flags=re.S)
            if not m:
                continue
            arr = m.group(1)
            times = re.findall(r"'([0-9\-/:\.]+)'", arr)
            for i in range(0, len(times), 2):
                if i+1 < len(times):
                    out[sc].append((times[i], times[i+1]))
    return out


def main():
    evt = _load_event()
    vt = _vt_intervals()

    all_annotations = []
    BN_map = {}
    crossings_per_probe = {}

    # Prefer the optimised physics-driven algorithmic LMN for this event.
    try:
        from examples.analyze_20190127_dn_shear import _build_algorithmic_lmn_map

        lmn_alg_map = _build_algorithmic_lmn_map(evt, window_half_width_s=30.0)
        print("[boundary_BLMN] Using algorithmic LMN for B_LMN rotation.")
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(
            "[boundary_BLMN] Warning: algorithmic LMN unavailable; "
            "falling back to .sav / hybrid LMN:",
            exc,
        )
        lmn_alg_map = None

    for p in PROBES:
        key = str(p)
        # Build B_df and POS
        if 'B_gsm' not in evt[key] or 'POS_gsm' not in evt[key]:
            continue
        tB, B = evt[key]['B_gsm']
        B_df = mp.data_loader.to_dataframe(tB, B, cols=['Bx','By','Bz'])
        B_df = mp.data_loader.resample(B_df, '1s')
        tpos, pos = evt[key]['POS_gsm']
        pos_df = mp.data_loader.to_dataframe(tpos, pos, cols=['X','Y','Z'])

        # Rotate B → LMN
        pos_mid = pos_df.iloc[len(pos_df) // 2].values if len(pos_df) else None
        BL, BM, BN = _rotate_B_to_LMN(B_df, key, pos_mid, lmn_alg_map)
        BN_map[key] = BN

        # Crossings
        layers = _compute_crossings(B_df.index, BN, evt[key], cadence='1s')
        crossings_per_probe[key] = layers

        # Per-probe figure and annotations
        annotations = _per_probe_plot(key, B_df, BL, BM, BN, layers,
                                      pos_df.index.values.astype('datetime64[ns]'),
                                      pos_df.values, vt)
        all_annotations.extend(annotations)

    # Combined BN overview
    _combined_bn_plot(BN_map, vt, crossings_per_probe)

    # Save crossing events CSV
    if all_annotations:
        df = pd.DataFrame(all_annotations)
        df.to_csv(EVENT_DIR / 'crossing_events.csv', index=False)

    print(f'Wrote figures and crossing CSV to: {EVENT_DIR.resolve()}')


if __name__ == '__main__':
    main()

