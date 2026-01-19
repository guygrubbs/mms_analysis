"""LMN alignment diagnostics for MMS 2019-01-27 (12:15–12:55).

Compares expert .sav LMN triads against:
- single-spacecraft MVA normals
- mean magnetic field direction
- mean ion velocity direction
- multi-spacecraft timing normal
- current algorithmic_lmn normals

Outputs CSV:
results/events_pub/2019-01-27_1215-1255/diagnostics/lmn_alignment_20190127.csv
"""
from __future__ import annotations
import pathlib, sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mms_mp as mp
from mms_mp import coords
from mms_mp.multispacecraft import timing_normal
from tools.idl_sav_import import load_idl_sav
from examples import analyze_20190127_dn_shear as evmod

EVENT_DIR = evmod.EVENT_DIR
OUT = EVENT_DIR / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
TRANGE = evmod.TRANGE
PROBES = evmod.PROBES
SAV_PATH = ROOT / evmod.SAVS["mixed_1230_1243"]


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.shape != (3,) or b.shape != (3,):
        return np.nan
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    c = float(np.clip(abs(np.dot(a / na, b / nb)), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _load_event():
    return evmod._minimal_event(TRANGE, PROBES)


def main():
    sav = load_idl_sav(str(SAV_PATH))
    lmn_sav = sav.get("lmn", {})
    trange_lmn = sav.get("trange_lmn_per_probe", {})

    evt = _load_event()

    # Build per-probe B/Vi arrays and positions at crossing for timing
    b_times: dict[str, np.ndarray] = {}
    b_vals: dict[str, np.ndarray] = {}
    vi_times: dict[str, np.ndarray] = {}
    vi_vals: dict[str, np.ndarray] = {}
    pos_at_cross: dict[str, np.ndarray] = {}

    # Crossing times near main MP boundary (same as analysis script)
    t_cross_utc = {
        "1": "2019-01-27T12:43:25",
        "2": "2019-01-27T12:43:26",
        "3": "2019-01-27T12:43:18",
        "4": "2019-01-27T12:43:26",
    }
    t_cross: dict[str, float] = {}

    for p in PROBES:
        data = evt.get(p, {})
        if "B_gsm" not in data or "POS_gsm" not in data:
            continue
        tB, B = data["B_gsm"]
        tP, P = data["POS_gsm"]
        if tB is None or B is None or tP is None or P is None:
            continue
        tB = np.asarray(tB, float); B = np.asarray(B, float)
        tP = np.asarray(tP, float); P = np.asarray(P, float)
        b_times[p] = tB; b_vals[p] = B
        # Position at crossing for timing normal
        if p in t_cross_utc:
            tc = pd.Timestamp(t_cross_utc[p], tz="UTC").timestamp()
            t_cross[p] = tc
            # nearest neighbour sample to crossing time
            j = int(np.argmin(np.abs(tP - tc)))
            pos_at_cross[p] = P[j, :3]
        # Ion bulk velocity (GSE) for alignment diagnostics
        if "V_i_gse" in data:
            tV, V = data["V_i_gse"]
            if tV is not None and V is not None:
                vi_times[p] = np.asarray(tV, float)
                vi_vals[p] = np.asarray(V, float)

    # Algorithmic LMN with baseline parameters
    lmn_alg = {}
    if len(t_cross) >= 2:
        pos_times = {p: np.asarray(evt[p]["POS_gsm"][0], float) for p in b_times}
        pos_vals = {p: np.asarray(evt[p]["POS_gsm"][1], float) for p in b_times}
        lmn_alg = coords.algorithmic_lmn(
            b_times=b_times,
            b_gsm=b_vals,
            pos_times=pos_times,
            pos_gsm_km=pos_vals,
            t_cross=t_cross,
        )

    # Timing normal using positions at crossing
    n_timing = None
    if len(pos_at_cross) >= 2 and len(t_cross) >= 2:
        try:
            n_timing, _, _, _ = timing_normal(pos_at_cross, t_cross, return_diagnostics=True)
        except Exception:
            n_timing = None

    rows = []

    for p in PROBES:
        key = str(p)
        lm = lmn_sav.get(key)
        tr = trange_lmn.get(key)
        if not lm or tr is None or key not in b_times:
            continue
        N_sav = np.asarray(lm["N"], float)
        L_sav = np.asarray(lm["L"], float)
        M_sav = np.asarray(lm["M"], float)
        # Use the same to_df helper as the main analysis script for robust timing
        t0 = pd.to_datetime(float(tr[0]), unit="s", utc=True)
        t1 = pd.to_datetime(float(tr[1]), unit="s", utc=True)
        tB_raw, B_raw = evt[key]["B_gsm"]
        Bdf = evmod.to_df(tB_raw, B_raw, ["Bx", "By", "Bz"])
        Bdf_win = Bdf[(Bdf.index >= t0) & (Bdf.index <= t1)]
        if Bdf_win.empty:
            continue
        B_win = Bdf_win[["Bx", "By", "Bz"]].values
        # MVA on B in .sav window
        lm_mva = coords.mva(B_win)
        N_mva = lm_mva.N
        B_mean = np.nanmean(B_win, axis=0)
        # Mean Vi in same window (if available)
        Vi_mean = np.full(3, np.nan)
        if "V_i_gse" in evt[key]:
            tV_raw, V_raw = evt[key]["V_i_gse"]
            Vdf = evmod.to_df(tV_raw, V_raw, ["Vx", "Vy", "Vz"])
            Vdf_win = Vdf[(Vdf.index >= t0) & (Vdf.index <= t1)]
            if not Vdf_win.empty:
                Vi_mean = np.nanmean(Vdf_win[["Vx", "Vy", "Vz"]].values, axis=0)
        # Tangential projections in plane ⟂ N_sav
        def proj_tan(v):
            v = np.asarray(v, float)
            return v - np.dot(v, N_sav) * N_sav
        B_t = proj_tan(B_mean)
        Vi_t = proj_tan(Vi_mean)
        # Algorithmic N for this probe (if available)
        N_alg = None
        if key in lmn_alg:
            N_alg = np.asarray(lmn_alg[key].N, float)
        row = {
            "probe": key,
            "ang_N_sav_vs_N_mva_deg": _angle_deg(N_sav, N_mva),
            "ang_N_sav_vs_N_alg_deg": _angle_deg(N_sav, N_alg) if N_alg is not None else np.nan,
            "ang_N_sav_vs_N_timing_deg": _angle_deg(N_sav, n_timing) if n_timing is not None else np.nan,
            "ang_L_sav_vs_Bmean_tan_deg": _angle_deg(L_sav, B_t),
            "ang_M_sav_vs_Bmean_tan_deg": _angle_deg(M_sav, B_t),
            "ang_L_sav_vs_Vi_tan_deg": _angle_deg(L_sav, Vi_t),
            "ang_M_sav_vs_Vi_tan_deg": _angle_deg(M_sav, Vi_t),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = OUT / "lmn_alignment_20190127.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    main()

