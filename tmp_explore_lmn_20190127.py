from __future__ import annotations
import sys
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mms_mp as mp
from tools.idl_sav_import import load_idl_sav
from examples import analyze_20190127_dn_shear as an
from mms_mp import coords

TRANGE = ["2019-01-27/12:15:00", "2019-01-27/12:55:00"]
PROBES = ["1", "2", "3", "4"]


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Return acute angle between vectors a and b in degrees."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != (3,) or b.shape != (3,):
        return np.nan
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if not np.isfinite(na) or not np.isfinite(nb) or na == 0 or nb == 0:
        return np.nan
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    ang = np.degrees(np.arccos(c))
    return float(min(ang, 180.0 - ang))


def stats(y_ref: np.ndarray, y_cand: np.ndarray) -> dict:
    y_ref = np.asarray(y_ref, float)
    y_cand = np.asarray(y_cand, float)
    if y_ref.shape != y_cand.shape or y_ref.size == 0:
        return {"mae": np.nan, "rmse": np.nan, "corr": np.nan, "n": 0}
    mask = np.isfinite(y_ref) & np.isfinite(y_cand)
    if mask.sum() < 3:
        return {"mae": np.nan, "rmse": np.nan, "corr": np.nan, "n": int(mask.sum())}
    d = y_cand[mask] - y_ref[mask]
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d ** 2)))
    corr = float(np.corrcoef(y_ref[mask], y_cand[mask])[0, 1])
    return {"mae": mae, "rmse": rmse, "corr": corr, "n": int(mask.sum())}


if __name__ == "__main__":
    print("Loading .sav LMN and event data...")
    sav = load_idl_sav("mp_lmn_systems_20190127_1215-1255_mp-ver2b.sav")
    LMN = sav.get("lmn", {})
    TRANGE_LMN = sav.get("trange_lmn_per_probe", {})

    # Use the same minimal CDF-based loader as the diagnostics script
    evt = an._minimal_event(list(TRANGE), list(PROBES))

    for key in PROBES:
        if key not in evt or "B_gsm" not in evt[key] or key not in LMN:
            continue
        print(f"\n=== Probe MMS{key} ===")
        tB, B = evt[key]["B_gsm"]
        Bdf = an.to_df(tB, B, ["Bx", "By", "Bz"])
        print(f"  Bdf length: {len(Bdf)}; index span: {Bdf.index.min()} → {Bdf.index.max()}")

        # Reference LMN triad from .sav
        L_sav = np.asarray(LMN[key]["L"], float)
        M_sav = np.asarray(LMN[key]["M"], float)
        N_sav = np.asarray(LMN[key]["N"], float)
        print(f"  L_sav[{key}] = {L_sav}")
        print(f"  M_sav[{key}] = {M_sav}")
        print(f"  N_sav[{key}] = {N_sav}")
        R_sav = np.vstack([L_sav, M_sav, N_sav]).T  # XYZ->LMN

        t_range = TRANGE_LMN.get(key)
        if t_range is not None and len(t_range) >= 2:
            t0 = pd.to_datetime(float(t_range[0]), unit="s", utc=True)
            t1 = pd.to_datetime(float(t_range[1]), unit="s", utc=True)
            Bwin = Bdf.loc[(Bdf.index >= t0) & (Bdf.index <= t1)]
            print(f"  Using LMN window {t0} → {t1}; Bwin length from range: {len(Bwin)}")
        else:
            mid = len(Bdf) // 2
            i0 = max(0, mid - 200); i1 = min(len(Bdf), mid + 200)
            Bwin = Bdf.iloc[i0:i1]
            print(f"  No LMN trange; fallback window indices {i0}:{i1}, len={len(Bwin)}")
        if len(Bwin) < 10:
            # Fallback: use +/-200 points around center, matching diagnostics script
            mid = len(Bdf) // 2
            i0 = max(0, mid - 200)
            i1 = min(len(Bdf), mid + 200)
            Bwin = Bdf.iloc[i0:i1]
            print(f"  Fallback +/-200 around center → indices {i0}:{i1}, len={len(Bwin)}")
        if len(Bwin) < 10:
            print("  Too few points in Bwin even after fallback, skipping.")
            continue

        Bseg = Bwin.values
        # Reference B in .sav LMN frame (from CDF B_gsm + .sav LMN)
        B_lmn_sav = Bseg @ R_sav

        # Candidate 1: pure MVA over Bseg (1 s resampled)
        lmn_mva = coords.mva(Bseg)
        R_mva = np.vstack([lmn_mva.L, lmn_mva.M, lmn_mva.N]).T
        B_lmn_mva = Bseg @ R_mva
        angN_mva = angle_deg(N_sav, lmn_mva.N)
        # Quick orientation diagnostics for MVA vs .sav
        lam_max, lam_mid, lam_min = lmn_mva.eigvals
        print(f"  MVA eigenvalues: lam_max={lam_max:.3g}, lam_mid={lam_mid:.3g}, lam_min={lam_min:.3g}")
        print(f"  MVA eigenvalue ratios: r_max_mid={lmn_mva.r_max_mid:.2f}, r_mid_min={lmn_mva.r_mid_min:.2f}")
        print(f"    Angles N_sav vs (L_MVA,M_MVA,N_MVA) = "
              f"({angle_deg(N_sav, lmn_mva.L):.1f}, {angle_deg(N_sav, lmn_mva.M):.1f}, {angN_mva:.1f}) deg")
        print(f"    Angles L_sav vs (L_MVA,M_MVA,N_MVA) = "
              f"({angle_deg(L_sav, lmn_mva.L):.1f}, {angle_deg(L_sav, lmn_mva.M):.1f}, {angle_deg(L_sav, lmn_mva.N):.1f}) deg")
        print(f"    Angles M_sav vs (L_MVA,M_MVA,N_MVA) = "
              f"({angle_deg(M_sav, lmn_mva.L):.1f}, {angle_deg(M_sav, lmn_mva.M):.1f}, {angle_deg(M_sav, lmn_mva.N):.1f}) deg")

        # Rotation matrix that maps MVA basis to .sav basis (event-specific)
        E_mva = np.column_stack((lmn_mva.L, lmn_mva.M, lmn_mva.N))
        E_sav = np.column_stack((L_sav, M_sav, N_sav))
        Q = E_mva.T @ E_sav
        print("  Q = E_mva^T @ E_sav (MVA→sav basis):")
        with np.printoptions(precision=3, suppress=True):
            print(Q)

        # Candidate 1b: MVA on full-cadence B within same window
        try:
            t_full = np.asarray(tB, dtype=float)
            t0_sec = float(t_range[0])
            t1_sec = float(t_range[1])
            mask_full = (t_full >= t0_sec) & (t_full <= t1_sec)
            B_full_win = B[mask_full, :3]
            if B_full_win.shape[0] >= 10:
                lmn_mva_full = coords.mva(B_full_win)
                angN_mva_full = angle_deg(N_sav, lmn_mva_full.N)
                print(f"  MVA(full) eigenvalue ratios: r_max_mid={lmn_mva_full.r_max_mid:.2f}, r_mid_min={lmn_mva_full.r_mid_min:.2f}")
                print(f"    Angles N_sav vs (L_MVAfull,M_MVAfull,N_MVAfull) = "
                      f"({angle_deg(N_sav, lmn_mva_full.L):.1f}, {angle_deg(N_sav, lmn_mva_full.M):.1f}, {angN_mva_full:.1f}) deg")
            else:
                print("  [warn] Too few full-cadence samples in window for MVA(full)")
        except Exception as e:
            print(f"  [warn] MVA(full) failed: {e}")

        # Candidate 2: hybrid_lmn with Shue allowed (position from MEC if available)
        pos_mid = None
        if "POS_gsm" in evt[key]:
            tP, POS = evt[key]["POS_gsm"]
            POSdf = an.to_df(tP, POS, ["X", "Y", "Z"])
            t_mid = Bwin.index[len(Bwin) // 2]
            try:
                pos_mid = POSdf.reindex([t_mid], method="nearest").iloc[0].values
            except Exception:
                pos_mid = POSdf.mean().values
        if pos_mid is not None:
            lmn_hybrid = coords.hybrid_lmn(Bseg, pos_gsm_km=pos_mid, eig_ratio_thresh=2.0)
            R_hyb = np.vstack([lmn_hybrid.L, lmn_hybrid.M, lmn_hybrid.N]).T
            B_lmn_hyb = Bseg @ R_hyb
            angN_hyb = angle_deg(N_sav, lmn_hybrid.N)
        else:
            lmn_hybrid = None
            angN_hyb = np.nan
            B_lmn_hyb = np.full_like(B_lmn_sav, np.nan)

        # Candidate 3: forced Shue-based LMN (ignore MVA eigen-ratios)
        if pos_mid is not None:
            lmn_shue = coords._shue_based_lmn(pos_mid)
            R_shue = np.vstack([lmn_shue.L, lmn_shue.M, lmn_shue.N]).T
            B_lmn_shue = Bseg @ R_shue
            angN_shue = angle_deg(N_sav, lmn_shue.N)
        else:
            lmn_shue = None
            angN_shue = np.nan
            B_lmn_shue = np.full_like(B_lmn_sav, np.nan)

        for label, Bcand, angN in [
            ("MVA", B_lmn_mva, angN_mva),
            ("Hybrid", B_lmn_hyb, angN_hyb),
            ("Shue", B_lmn_shue, angN_shue),
        ]:
            sL = stats(B_lmn_sav[:, 0], Bcand[:, 0])
            sM = stats(B_lmn_sav[:, 1], Bcand[:, 1])
            sN = stats(B_lmn_sav[:, 2], Bcand[:, 2])
            print(f"  {label}: angle(N_sav,N_{label})={angN:5.1f} deg")
            print(f"    BL: MAE={sL['mae']:.3f} RMSE={sL['rmse']:.3f} corr={sL['corr']:.3f} (n={sL['n']})")
            print(f"    BM: MAE={sM['mae']:.3f} RMSE={sM['rmse']:.3f} corr={sM['corr']:.3f} (n={sM['n']})")
            print(f"    BN: MAE={sN['mae']:.3f} RMSE={sN['rmse']:.3f} corr={sN['corr']:.3f} (n={sN['n']})")

