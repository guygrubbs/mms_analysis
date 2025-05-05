"""
coords.py
=========

Local-LMN coordinate helpers used throughout the MMS-MP toolkit.

Key changes (May 2025)
----------------------
▶ Uses `pyspedas.lmn_matrix_make` (v1.7.20+) when eigenvalue ratios are poor  
▶ Retains Shue-model fallback if PySPEDAS < 1.7.20 or import fails  
▶ Tidier dataclass with `meta` field that stores any LMN metadata
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import warnings

# --------------------------------------------------------------------
# Attempt to import the new LMN helper from PySPEDAS ≥ 1.7.20
try:
    from pyspedas import lmn_matrix_make   # type: ignore
    _HAVE_PS_LMN = True
except Exception:                          # pragma: no cover
    _HAVE_PS_LMN = False

# --------------------------------------------------------------------
# Dataclass for LMN output
# --------------------------------------------------------------------
@dataclass
class LMN:
    L: np.ndarray
    M: np.ndarray
    N: np.ndarray
    R: np.ndarray                 # 3×3 rotation (rows = L, M, N)
    eigvals: Tuple[float, float, float]
    r_max_mid: float
    r_mid_min: float
    meta: Optional[dict] = None   # Optional metadata (pyspedas output)

    # rotate vectors → LMN
    def to_lmn(self, vec_xyz: np.ndarray) -> np.ndarray:
        return (self.R @ vec_xyz.T).T


# --------------------------------------------------------------------
# Classical minimum-variance analysis
# --------------------------------------------------------------------
def _mva(b_xyz: np.ndarray) -> LMN:
    db = b_xyz - b_xyz.mean(axis=0)
    cov = db.T @ db / db.shape[0]
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    L, M, _ = eigvecs.T
    L /= np.linalg.norm(L)
    M = M - np.dot(M, L) * L
    M /= np.linalg.norm(M)
    N = np.cross(L, M); N /= np.linalg.norm(N)
    R = np.vstack((L, M, N))

    return LMN(L, M, N, R,
               tuple(eigvals),
               eigvals[0] / eigvals[1],
               eigvals[1] / eigvals[2])

# --------------------------------------------------------------------
# Shue-model normal (fallback)
# --------------------------------------------------------------------
def _shue_normal(pos_gsm_km: np.ndarray,
                 sw_pressure_npa: float = 2.0,
                 dipole_tilt_deg: float = 0.0) -> np.ndarray:
    r0 = 11.4 * sw_pressure_npa ** (-1 / 6.6)
    alpha = 0.58 - 0.007 * dipole_tilt_deg
    RE = 6371.0
    r_vec = pos_gsm_km / RE
    r_mag = np.linalg.norm(r_vec)
    if r_mag == 0:
        return np.array([1., 0., 0.])
    theta = np.arccos(r_vec[2] / r_mag)
    _ = r0 * (2 / (1 + np.cos(theta))) ** alpha  # model distance (unused)
    nhat = r_vec / r_mag
    return nhat

# --------------------------------------------------------------------
# Public: hybrid LMN
# --------------------------------------------------------------------
def hybrid_lmn(b_xyz: np.ndarray,
               pos_gsm_km: Optional[np.ndarray] = None,
               eig_ratio_thresh: float = 5.0,
               cache_key: Optional[str] = None) -> LMN:
    """
    Create an LMN triad.

    1. Do MVA → if both λ_max/λ_mid and λ_mid/λ_min ≥ `eig_ratio_thresh`,
       keep it.
    2. Else, try PySPEDAS `lmn_matrix_make` (if available).
    3. Else, build triad using Shue (1997) outward normal.
    """
    if cache_key:
        return _cached_hybrid(cache_key, b_xyz.tobytes(), b_xyz.shape,
                              None if pos_gsm_km is None else pos_gsm_km.tobytes(),
                              eig_ratio_thresh)

    return _compute_hybrid(b_xyz, pos_gsm_km, eig_ratio_thresh)


@lru_cache(maxsize=64)
def _cached_hybrid(key: str,
                   b_bytes: bytes,
                   shape: Tuple[int, int],
                   pos_bytes: Optional[bytes],
                   thresh: float) -> LMN:
    b = np.frombuffer(b_bytes).reshape(shape)
    pos = None if pos_bytes is None else np.frombuffer(pos_bytes)
    return _compute_hybrid(b, pos, thresh)


def _compute_hybrid(b_xyz: np.ndarray,
                    pos_gsm_km: Optional[np.ndarray],
                    thresh: float) -> LMN:
    lm = _mva(b_xyz)
    if lm.r_max_mid >= thresh and lm.r_mid_min >= thresh:
        return lm  # good MVA

    # ----------------------------------------------------------------
    # Weak eigenvalues → attempt PySPEDAS LMN (if present)
    # ----------------------------------------------------------------
    if _HAVE_PS_LMN and pos_gsm_km is not None:
        try:
            lmn_res = lmn_matrix_make(b_xyz, pos_gsm_km,
                                      coord_system='GSM',
                                      verbose=False)
            R = np.vstack((lmn_res['lmn'][0],
                           lmn_res['lmn'][1],
                           lmn_res['lmn'][2]))
            return LMN(R[0], R[1], R[2], R,
                       lm.eigvals, lm.r_max_mid, lm.r_mid_min,
                       meta=lmn_res)
        except Exception as e:         # pragma: no cover
            warnings.warn(f'pyspedas.lmn_matrix_make failed: {e}')

    # ----------------------------------------------------------------
    # Last resort → Shue outward normal + mean-B cross-product
    # ----------------------------------------------------------------
    if pos_gsm_km is None:
        warnings.warn('Poor eigenvalue ratios and no spacecraft position; '
                      'returning low-quality MVA triad.')
        return lm

    nhat = _shue_normal(pos_gsm_km)
    Bmean = b_xyz.mean(axis=0)
    L = np.cross(Bmean, nhat)
    if np.linalg.norm(L) < 1e-3:
        L = np.cross([0, 1, 0], nhat)
    L /= np.linalg.norm(L)
    M = np.cross(nhat, L); M /= np.linalg.norm(M)
    N = nhat
    R = np.vstack((L, M, N))
    warnings.warn('Using Shue-model normal due to poor MVA quality.')
    return LMN(L, M, N, R,
               lm.eigvals, lm.r_max_mid, lm.r_mid_min)
