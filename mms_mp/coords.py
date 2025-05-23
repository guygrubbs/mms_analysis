"""
coords.py
=========

Local-LMN coordinate helpers used throughout the MMS-MP toolkit.

Key changes (May 2025)
----------------------
▶ Robust NaN filtering before MVA  
▶ Automatic fall-back to Shue model normal when MVA fails  
▶ Optional use of `pyspedas.lmn_matrix_make` when eigen-ratios are poor  
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import warnings

# ────────────────────────────────────────────────────────────────────────────
# Optional PySPEDAS helper (works on ≥ 1.7.20 only)
try:
    from pyspedas import lmn_matrix_make          # type: ignore
    _HAVE_PS_LMN = True
except Exception:                                 # pragma: no cover
    _HAVE_PS_LMN = False
# ────────────────────────────────────────────────────────────────────────────


# =========================================================================== #
#                                Dataclass                                   #
# =========================================================================== #
@dataclass
class LMN:
    """Container for an LMN triad (rows of *R*)."""
    L: np.ndarray                        # (3,)
    M: np.ndarray                        # (3,)
    N: np.ndarray                        # (3,)
    R: np.ndarray                        # 3 × 3 rotation matrix (rows = L, M, N)

    eigvals: Tuple[float, float, float]
    r_max_mid: float                     # λ_max / λ_mid
    r_mid_min: float                     # λ_mid / λ_min

    meta: Optional[dict] = None          # optional pyspedas output, etc.

    # ------------------------------------------------------------------ #
    def to_lmn(self, vec_xyz: np.ndarray) -> np.ndarray:
        """
        Rotate a vector (or array of vectors) from GSM → LMN.
        Accepts shapes (3,) or (N, 3) or (N, ≥ 3); extra columns are ignored.
        """
        v = vec_xyz[..., :3]
        return (self.R @ v.T).T


# =========================================================================== #
#                   Step 1 — classical minimum-variance                        #
# =========================================================================== #
def _do_mva(b_xyz: np.ndarray) -> LMN:
    """
    Minimum-variance analysis (Sonnerup & Cahill, 1967).
    Returns an *LMN* instance.  Raises ValueError if B is unusable.
    """
    # ── discard rows with NaNs / infs ─────────────────────────────────
    good = np.isfinite(b_xyz).all(axis=1)
    b_use = b_xyz[good]
    if b_use.shape[0] < 3:
        raise ValueError("too few finite B samples")

    # ── covariance & eigen-decomposition ─────────────────────────────
    b_mean = b_use.mean(axis=0, keepdims=True)
    cov = (b_use - b_mean).T @ (b_use - b_mean) / (b_use.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eig(cov)

    # sort so that L has largest eigen-value
    order = np.argsort(eigvals)[::-1]          # λ_max, λ_mid, λ_min
    eigvals = eigvals[order]
    R = eigvecs[:, order].T                   # rows = eigen-vectors

    # normalise rows (paranoia – should already be unit length)
    R = R / np.linalg.norm(R, axis=1, keepdims=True)

    return LMN(L=R[0], M=R[1], N=R[2], R=R,
               eigvals=tuple(eigvals),
               r_max_mid=eigvals[0] / eigvals[1] if eigvals[1] else np.inf,
               r_mid_min=eigvals[1] / eigvals[2] if eigvals[2] else np.inf)


# =========================================================================== #
#             Step 2 — Shue (1997) magnetopause normal (fallback)             #
# =========================================================================== #
def _shue_normal(pos_gsm_km: np.ndarray,
                 sw_pressure_npa: float = 2.0,
                 dipole_tilt_deg: float = 0.0) -> np.ndarray:
    """
    Outward magnetopause normal from Shue et al. (1997) model (GSM coords).
    Only the *direction* is needed for LMN-building; r₀, α constants use
    nominal solar-wind pressure and zero dipole tilt.
    """
    r0 = 11.4 * sw_pressure_npa ** (-1 / 6.6)
    alpha = 0.58 - 0.007 * dipole_tilt_deg

    RE = 6371.0
    r_vec = pos_gsm_km / RE
    r_mag = np.linalg.norm(r_vec)
    if r_mag == 0.0:
        return np.array([1., 0., 0.])         # degenerate – pick ±X̂

    theta = np.arccos(r_vec[2] / r_mag)
    _ = r0 * (2 / (1 + np.cos(theta))) ** alpha   # model distance (not used)
    return r_vec / r_mag


# =========================================================================== #
#                     Public API — hybrid LMN builder                         #
# =========================================================================== #
def hybrid_lmn(b_xyz: np.ndarray,
               pos_gsm_km: Optional[np.ndarray] = None,
               eig_ratio_thresh: float = 5.0,
               cache_key: Optional[str] = None) -> LMN:
    """
    Compute an LMN triad using:

    1. Classical MVA when λ_max/λ_mid and λ_mid/λ_min ≥ `eig_ratio_thresh`.
    2. Otherwise, try `pyspedas.lmn_matrix_make` (if available & position given).
    3. Finally, fall back to Shue outward normal + right-handed completion.
    """
    if cache_key:          # small LRU cache for repeated calls on same data
        return _cached_hybrid(cache_key,
                              b_xyz.tobytes(), b_xyz.shape,
                              None if pos_gsm_km is None else pos_gsm_km.tobytes(),
                              eig_ratio_thresh)

    return _compute_hybrid(b_xyz, pos_gsm_km, eig_ratio_thresh)


@lru_cache(maxsize=64)
def _cached_hybrid(_key: str,
                   b_bytes: bytes,
                   shape: Tuple[int, int],
                   pos_bytes: Optional[bytes],
                   thresh: float) -> LMN:
    b = np.frombuffer(b_bytes).reshape(shape)
    pos = None if pos_bytes is None else np.frombuffer(pos_bytes)
    return _compute_hybrid(b, pos, thresh)


# --------------------------------------------------------------------------- #
#                              core helper                                    #
# --------------------------------------------------------------------------- #
def _compute_hybrid(b_xyz: np.ndarray,
                    pos_gsm_km: Optional[np.ndarray],
                    eig_ratio_thresh: float) -> LMN:
    """
    Internal: do MVA → maybe PySPEDAS → maybe Shue.
    Always returns a valid *LMN* instance.
    """
    try:
        lm = _do_mva(b_xyz)
    except ValueError as err:
        warnings.warn(f"[coords] MVA failed ('{err}') – using Shue model normal")
        return _shue_based_lmn(pos_gsm_km)

    # good MVA?
    if lm.r_max_mid >= eig_ratio_thresh and lm.r_mid_min >= eig_ratio_thresh:
        return lm

    # weak eigen-ratios → try PySPEDAS (needs position)
    if _HAVE_PS_LMN and pos_gsm_km is not None:
        try:
            ps = lmn_matrix_make(b_xyz, pos_gsm_km, coord_system='GSM',
                                 verbose=False)
            R = np.vstack((ps['lmn'][0], ps['lmn'][1], ps['lmn'][2]))
            return LMN(R[0], R[1], R[2], R,
                       lm.eigvals, lm.r_max_mid, lm.r_mid_min,
                       meta=ps)
        except Exception as e:                      # pragma: no cover
            warnings.warn(f'pyspedas.lmn_matrix_make failed: {e}')

    # fall back to Shue
    warnings.warn('[coords] Weak eigen-ratios – using Shue model normal')
    return _shue_based_lmn(pos_gsm_km, base_lmn=lm)


# --------------------------------------------------------------------------- #
def _shue_based_lmn(pos_gsm_km: Optional[np.ndarray],
                    base_lmn: Optional[LMN] = None) -> LMN:
    """
    Build an LMN triad from Shue normal plus a perpendicular vector.
    If *base_lmn* supplied, re-use its eigen-values for bookkeeping.
    """
    if pos_gsm_km is None:
        # cannot do Shue without position – fabricate a dummy triad
        dummy = np.eye(3)
        return LMN(dummy[0], dummy[1], dummy[2], dummy,
                   (np.nan, np.nan, np.nan), np.nan, np.nan)

    N = _shue_normal(pos_gsm_km)
    # choose an L perpendicular to N
    trial = np.cross([0, 0, 1], N)
    if np.linalg.norm(trial) < 1e-3:                # N almost ẑ
        trial = np.cross([0, 1, 0], N)
    L = trial / np.linalg.norm(trial)
    M = np.cross(N, L);  M /= np.linalg.norm(M)

    R = np.vstack((L, M, N))
    ev = (np.nan, np.nan, np.nan) if base_lmn is None else base_lmn.eigvals
    return LMN(L, M, N, R,
               ev,
               getattr(base_lmn, 'r_max_mid', np.nan),
               getattr(base_lmn, 'r_mid_min', np.nan))
