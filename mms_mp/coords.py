# mms_mp/coords.py
# ------------------------------------------------------------
# Local LMN coordinate utilities (expanded)
# ------------------------------------------------------------
# New features:
#   • Eigenvalue–ratio uncertainty metrics
#   • Shue et al. (1997) model‐normal fallback
#   • “Hybrid” LMN (prefers MVA, falls back to model if λ ratios poor)
#   • Optional LRU cache to avoid recomputing LMN for the same window
# ------------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import warnings

# ------------------------------------------------------------------
# Dataclass
# ------------------------------------------------------------------
@dataclass
class LMN:
    """Container for LMN basis + diagnostics."""
    L: np.ndarray            # (3,)
    M: np.ndarray            # (3,)
    N: np.ndarray            # (3,)
    R: np.ndarray            # (3, 3)  rows = L, M, N
    eigvals: Tuple[float, float, float]   # λ_max, λ_mid, λ_min
    r_max_mid: float         # λ_max / λ_mid
    r_mid_min: float         # λ_mid / λ_min

    # Convenience rotation --------------------------------------------------
    def to_lmn(self, vec_xyz: np.ndarray) -> np.ndarray:
        """Rotate N×3 GSM/GSE vectors → LMN."""
        return (self.R @ vec_xyz.T).T

# ------------------------------------------------------------------
# Classical MVA (minimum variance on B)
# ------------------------------------------------------------------
def mva(b_xyz: np.ndarray,
        eigen_sort: bool = True) -> LMN:
    """
    Perform MVA on a B-field slice (N×3, GSM) and return LMN dataclass.
    """
    if b_xyz.ndim != 2 or b_xyz.shape[1] != 3:
        raise ValueError("b_xyz must be (N, 3) array")
    db = b_xyz - b_xyz.mean(axis=0)
    cov = db.T @ db / db.shape[0]  # covariance
    eigvals, eigvecs = np.linalg.eig(cov)

    if eigen_sort:
        order = np.argsort(eigvals)[::-1]          # λ_max → λ_min
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

    L_vec, M_vec, N_vec = eigvecs.T
    # Orthonormalise for safety
    L_vec /= np.linalg.norm(L_vec)
    M_vec -= np.dot(M_vec, L_vec) * L_vec
    M_vec /= np.linalg.norm(M_vec)
    N_vec = np.cross(L_vec, M_vec)
    N_vec /= np.linalg.norm(N_vec)

    R = np.vstack((L_vec, M_vec, N_vec))
    r_max_mid = eigvals[0] / eigvals[1]
    r_mid_min = eigvals[1] / eigvals[2]

    return LMN(L_vec, M_vec, N_vec, R,
               tuple(eigvals), r_max_mid, r_mid_min)

# ------------------------------------------------------------------
# Shue et al. (1997) empirical magnetopause normal
# ------------------------------------------------------------------
def _shue_normal(pos_gsm: np.ndarray,
                 sw_pressure_nPa: float = 2.0,
                 dipole_tilt_deg: float = 0.0) -> np.ndarray:
    """
    Compute outward magnetopause normal using Shue et al. 1997 model.
    Only requires spacecraft position in GSM (km).
    """
    # Shue model r(θ) = r0 (2 / (1+cosθ))^α
    # Normal is gradient; here approximate as vector from origin to model surface.
    # r0 & α depend on solar-wind dynamic pressure (nPa) and dipole tilt.

    # === parameters ===
    r0 = 11.4 * sw_pressure_nPa**(-1/6.6)          # Earth radii
    alpha = 0.58 - 0.007 * dipole_tilt_deg

    # Convert pos to RE and spherical
    RE = 6371.0
    r_vec = pos_gsm / RE
    x, y, z = r_vec
    r = np.linalg.norm(r_vec)
    if r == 0:  # origin—undefined
        return np.array([1.0, 0.0, 0.0])

    theta = np.arccos(z / r)        # polar angle from +Z
    r_model = r0 * (2 / (1 + np.cos(theta)))**alpha
    # Normal is radial at surface; take outward unit vector from Earth centre
    n = r_vec / np.linalg.norm(r_vec)
    # Ensure n points outward (positive dot with pos)
    if np.dot(n, r_vec) < 0:
        n = -n
    return n

# ------------------------------------------------------------------
# Hybrid LMN factory
# ------------------------------------------------------------------
def hybrid_lmn(b_xyz: np.ndarray,
               pos_gsm: Optional[np.ndarray] = None,
               eig_ratio_thresh: float = 5.0,
               cache_key: Optional[str] = None) -> LMN:
    """
    Prefer MVA but fall back to Shue-model normal when eigenvalue ratios are poor.
    pos_gsm – spacecraft position (km GSM), required for model fallback.
    eig_ratio_thresh – require both λ_max/λ_mid and λ_mid/λ_min ≥ this.
    cache_key – optional immutable key → caches LMN for repeated calls.
    """
    if cache_key is None:
        return _compute_hybrid(b_xyz, pos_gsm, eig_ratio_thresh)

    # tiny wrapper around lru_cache (cannot hash ndarray directly)
    return _cached_hybrid(cache_key, b_xyz.tobytes(), tuple(b_xyz.shape),
                          None if pos_gsm is None else tuple(pos_gsm),
                          eig_ratio_thresh)

@lru_cache(maxsize=64)
def _cached_hybrid(key: str,
                   b_bytes: bytes,
                   b_shape: Tuple[int, int],
                   pos_tuple: Optional[Tuple[float, float, float]],
                   eig_ratio_thresh: float) -> LMN:
    b_xyz = np.frombuffer(b_bytes).reshape(b_shape)
    pos = None if pos_tuple is None else np.array(pos_tuple)
    return _compute_hybrid(b_xyz, pos, eig_ratio_thresh)

# ------------------------------------------------------------------
# Internal compute
# ------------------------------------------------------------------
def _compute_hybrid(b_xyz: np.ndarray,
                    pos_gsm: Optional[np.ndarray],
                    ratio_thresh: float) -> LMN:
    lm = mva(b_xyz)
    if lm.r_max_mid >= ratio_thresh and lm.r_mid_min >= ratio_thresh:
        return lm  # good MVA

    # Weak eigen ratios → fall back or blend
    if pos_gsm is None:
        warnings.warn("Poor MVA eigenvalue ratios but no spacecraft position provided; "
                      "returning MVA result anyway.")
        return lm

    n_model = _shue_normal(pos_gsm)
    # Create orthonormal triad: project L along B × N_model
    # Use mean B as guide for L direction (max variance often along B)
    Bmean = b_xyz.mean(axis=0)
    L_vec = np.cross(Bmean, n_model)
    if np.linalg.norm(L_vec) < 1e-3:
        # fallback to any orthogonal
        L_vec = np.cross([0, 1, 0], n_model)
    L_vec /= np.linalg.norm(L_vec)
    M_vec = np.cross(n_model, L_vec); M_vec /= np.linalg.norm(M_vec)
    N_vec = n_model / np.linalg.norm(n_model)
    R = np.vstack((L_vec, M_vec, N_vec))
    lm_model = LMN(L_vec, M_vec, N_vec, R, lm.eigvals, lm.r_max_mid, lm.r_mid_min)
    warnings.warn("Using model-normal LMN due to poor eigenvalue ratios "
                  f"(λ_max/λ_mid={lm.r_max_mid:.1f}, λ_mid/λ_min={lm.r_mid_min:.1f}).")
    return lm_model
