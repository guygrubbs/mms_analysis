"""Parameter sweep for physics-driven algorithmic LMN (2019-01-27 event).

Sweeps over:
- window_half_width_s
- normal_weights (timing, MVA, Shue)
- tangential_strategy

For each configuration, computes per-probe N-angle vs expert .sav LMN normal
and BN correlation vs .sav BN, then writes a CSV for offline inspection.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Iterable, Tuple, Dict

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mms_mp as mp
from mms_mp import coords
from tools.idl_sav_import import load_idl_sav
from examples import analyze_20190127_dn_shear as evmod

EVENT_DIR = evmod.EVENT_DIR
OUT = EVENT_DIR / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
TRANGE = evmod.TRANGE
PROBES = evmod.PROBES
SAV_PATH = ROOT / evmod.SAVS["all_1243"]

# Sweep grid
WINDOWS = [15.0, 20.0, 30.0, 40.0, 60.0]
NORMAL_WEIGHTS: list[Tuple[float, float, float]] = [
    (0.7, 0.2, 0.1),
    (0.5, 0.4, 0.1),
    (0.8, 0.15, 0.05),
]
TANGENTIAL_STRATEGIES = ["Bmean", "MVA", "Vi"]


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != (3,) or b.shape != (3,):
        return np.nan
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    c = float(np.clip(abs(np.dot(a / na, b / nb)), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _corr(a: pd.Series, b: pd.Series) -> float:
    join = pd.concat([a, b], axis=1).dropna()
    if len(join) == 0:
        return np.nan
    try:
        return float(np.corrcoef(join.iloc[:, 0].values, join.iloc[:, 1].values)[0, 1])
    except Exception:
        return np.nan


def _prepare_sav_and_event():
    sav = load_idl_sav(str(SAV_PATH))
    lmn_sav = sav.get("lmn", {})
    evt = evmod._minimal_event(TRANGE, PROBES)

    Bdf: Dict[str, pd.DataFrame] = {}
    BN_sav: Dict[str, pd.Series] = {}
    N_sav: Dict[str, np.ndarray] = {}

    for p in PROBES:
        key = str(p)
        if key not in evt or "B_gsm" not in evt[key] or key not in lmn_sav:
            continue
        tB, B = evt[key]["B_gsm"]
        Bdf_p = evmod.to_df(tB, B, ["Bx", "By", "Bz"])
        Bdf[key] = Bdf_p
        L = np.asarray(lmn_sav[key]["L"], float)
        M = np.asarray(lmn_sav[key]["M"], float)
        N = np.asarray(lmn_sav[key]["N"], float)
        N_sav[key] = N / (np.linalg.norm(N) + 1e-12)
        R_sav = np.vstack([L, M, N]).T
        B_lmn_sav = Bdf_p.values @ R_sav
        BN_sav[key] = pd.Series(B_lmn_sav[:, 2], index=Bdf_p.index, name=f"BN_sav_{key}")

    return sav, evt, Bdf, BN_sav, N_sav


def _build_inputs_for_algorithmic_lmn(evt) -> tuple:
    b_times: Dict[str, np.ndarray] = {}
    b_vals: Dict[str, np.ndarray] = {}
    pos_times: Dict[str, np.ndarray] = {}
    pos_vals: Dict[str, np.ndarray] = {}
    vi_times: Dict[str, np.ndarray] = {}
    vi_vals: Dict[str, np.ndarray] = {}

    # Crossing times near main MP boundary (~12:43 UT), same as analysis script
    import pandas as pd  # local to avoid polluting module namespace

    t_cross_utc = {
        "1": "2019-01-27T12:43:25",
        "2": "2019-01-27T12:43:26",
        "3": "2019-01-27T12:43:18",
        "4": "2019-01-27T12:43:26",
    }
    t_cross: Dict[str, float] = {}

    for p in PROBES:
        key = str(p)
        data = evt.get(key, {})
        if "B_gsm" in data and "POS_gsm" in data:
            tB, B = data["B_gsm"]
            tP, P = data["POS_gsm"]
            if tB is not None and B is not None and tP is not None and P is not None:
                b_times[key] = np.asarray(tB, float)
                b_vals[key] = np.asarray(B, float)
                pos_times[key] = np.asarray(tP, float)
                pos_vals[key] = np.asarray(P, float)
        if key in t_cross_utc and key in b_times:
            t_cross[key] = pd.Timestamp(t_cross_utc[key], tz="UTC").timestamp()
        if "V_i_gse" in data:
            tV, V = data["V_i_gse"]
            if tV is not None and V is not None:
                vi_times[key] = np.asarray(tV, float)
                vi_vals[key] = np.asarray(V, float)

    return b_times, b_vals, pos_times, pos_vals, vi_times, vi_vals, t_cross


def main():
    sav, evt, Bdf, BN_sav, N_sav = _prepare_sav_and_event()
    b_times, b_vals, pos_times, pos_vals, vi_times, vi_vals, t_cross = _build_inputs_for_algorithmic_lmn(evt)

    rows: list[dict[str, object]] = []

    for w in WINDOWS:
        for (w_t, w_m, w_s) in NORMAL_WEIGHTS:
            for strat in TANGENTIAL_STRATEGIES:
                row: dict[str, object] = {
                    "window_half_width_s": float(w),
                    "w_timing": float(w_t),
                    "w_mva": float(w_m),
                    "w_shue": float(w_s),
                    "tangential_strategy": strat,
                }
                for p in PROBES:
                    ip = int(p)
                    row[f"N_angle_MMS{ip}_deg"] = np.nan
                    row[f"BN_corr_MMS{ip}"] = np.nan

                try:
                    use_vi = strat.lower() in {"vi", "v_i", "vin"}
                    lmn_alg = coords.algorithmic_lmn(
                        b_times=b_times,
                        b_gsm=b_vals,
                        pos_times=pos_times,
                        pos_gsm_km=pos_vals,
                        t_cross=t_cross,
                        window_half_width_s=w,
                        tangential_strategy=strat,
                        normal_weights=(w_t, w_m, w_s),
                        vi_times=vi_times if use_vi else None,
                        vi_gsm=vi_vals if use_vi else None,
                    )
                except Exception as e:  # record failure but continue grid
                    row["status"] = f"error: {e}"  # type: ignore[assignment]
                    rows.append(row)
                    continue

                N_angles: list[float] = []
                BN_corrs: list[float] = []

                for p in PROBES:
                    key = str(p)
                    if key not in lmn_alg or key not in N_sav or key not in Bdf:
                        continue
                    N_alg = np.asarray(lmn_alg[key].N, float)
                    N_alg /= np.linalg.norm(N_alg) + 1e-12
                    N_ref = N_sav[key]
                    ang = _angle_deg(N_ref, N_alg)
                    row[f"N_angle_MMS{int(p)}_deg"] = ang
                    if not np.isnan(ang):
                        N_angles.append(ang)

                    # BN correlation in .sav LMN frame vs algorithmic LMN frame
                    L_alg = np.asarray(lmn_alg[key].L, float)
                    M_alg = np.asarray(lmn_alg[key].M, float)
                    N_alg_full = np.asarray(lmn_alg[key].N, float)
                    R_alg = np.vstack([L_alg, M_alg, N_alg_full]).T
                    Bdf_p = Bdf[key]
                    BN_ref = BN_sav[key]
                    BN_alg = pd.Series(
                        (Bdf_p.values @ R_alg)[:, 2],
                        index=Bdf_p.index,
                        name=f"BN_alg_{key}",
                    )
                    corr = _corr(BN_ref, BN_alg)
                    row[f"BN_corr_MMS{int(p)}"] = corr
                    if not np.isnan(corr):
                        BN_corrs.append(corr)

                row["N_angle_mean_deg"] = float(np.nanmean(N_angles)) if N_angles else np.nan
                row["N_angle_max_deg"] = float(np.nanmax(N_angles)) if N_angles else np.nan
                row["BN_corr_mean"] = float(np.nanmean(BN_corrs)) if BN_corrs else np.nan
                row["BN_corr_min"] = float(np.nanmin(BN_corrs)) if BN_corrs else np.nan

                rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = OUT / "algorithmic_lmn_param_sweep.csv"
    df.to_csv(out_csv, index=False)

    # Simple best-configuration summary printed to stdout
    if not df.empty and {"BN_corr_min", "N_angle_mean_deg"}.issubset(df.columns):
        crit = (df["BN_corr_min"] >= 0.995) & (df["N_angle_mean_deg"] < 18.0)
        candidates = df[crit].copy()
        if not candidates.empty:
            best = candidates.sort_values(
                ["N_angle_mean_deg", "N_angle_max_deg"], ascending=[True, True]
            ).iloc[0]
            print(
                "Best configuration meeting BN_corr_min>=0.995 "
                "& mean N-angle<18deg:"
            )
            print(best.to_dict())
        else:
            best = df.sort_values(
                ["N_angle_mean_deg", "BN_corr_min"], ascending=[True, False]
            ).iloc[0]
            print(
                "No configuration met strict criteria; best overall by "
                "N-angle and BN_corr_min:"
            )
            print(best.to_dict())

    print(f"Wrote {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    main()

