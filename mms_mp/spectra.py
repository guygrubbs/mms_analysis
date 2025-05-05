# mms_mp/spectra.py
# ============================================================
# Energy-time spectrogram helpers for MMS particle instruments
# ------------------------------------------------------------
# 2025-05 update
#   • Added HPCA omnidirectional spectrogram (`hpca_ion_spectrogram`)
#   • Automatic rebin/resample option so FPI and HPCA can share the
#     same time grid (leverages mms_mp.resample)
#   • Optional quality-mask argument (bad samples → transparent pixels)
#   • Robust handling of zeros / negatives before log10
#   • `return_axes=False` lets callers skip matplotlib import when
#     generating figures inside external notebooks
# ============================================================
from __future__ import annotations
from typing import Optional, Tuple, Literal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .resample import resample as _resample_single   # optional utility


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------
def _prep_log(data: np.ndarray,
              log10: bool,
              floor: float = 1e-30) -> np.ndarray:
    """
    Apply log10 safely: everything ≤0 becomes NaN; small values floored.
    """
    if not log10:
        return data
    dat = data.copy()
    dat[dat <= 0] = np.nan
    return np.log10(np.maximum(dat, floor))


def _generic_pcolormesh(t: np.ndarray,
                        e: np.ndarray,
                        z: np.ndarray,
                        *,
                        ylabel: str,
                        clabel: str,
                        vmin: Optional[float],
                        vmax: Optional[float],
                        cmap: str,
                        title: str,
                        mask: Optional[np.ndarray],
                        ax: Optional[plt.Axes],
                        return_axes: bool,
                        show: bool):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    if mask is not None:
        z = z.copy()
        z[~mask] = np.nan

    T, E = np.meshgrid(t, e, indexing='ij')
    pcm = ax.pcolormesh(T, E, z,
                        cmap=cmap,
                        shading='auto',
                        vmin=vmin, vmax=vmax)
    ax.set_yscale('log')
    ax.set_ylabel(ylabel)
    if np.issubdtype(t.dtype, np.datetime64):
        ax.set_xlabel('UTC')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    else:
        ax.set_xlabel('Time (s)')
    if title:
        ax.set_title(title)
    cb = plt.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(clabel)
    if show:
        plt.tight_layout(); plt.show()
    return (ax, cb) if return_axes else None


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def generic_spectrogram(t: np.ndarray,
                        e: np.ndarray,
                        data: np.ndarray,
                        *,
                        log10: bool = True,
                        vmin: Optional[float] = None,
                        vmax: Optional[float] = None,
                        cmap: str = 'viridis',
                        ylabel: str = 'Energy (eV)',
                        title: str = '',
                        mask: Optional[np.ndarray] = None,
                        ax: Optional[plt.Axes] = None,
                        show: bool = True,
                        return_axes: bool = False):
    """
    Quick-draw pcolormesh for any 2-D spectrogram.

    Parameters
    ----------
    t, e, data : (N,) (M,) (N,M) arrays
    log10      : take log10(data)  (default True)
    mask       : boolean same shape as data – False→transparent
    vmin/vmax  : colour limits **after** log10 if log10=True
    """
    z = _prep_log(data, log10)
    clabel = 'log$_{10}$ Flux' if log10 else 'Flux'
    return _generic_pcolormesh(t, e, z,
                               ylabel=ylabel,
                               clabel=clabel,
                               vmin=vmin, vmax=vmax,
                               cmap=cmap, title=title,
                               mask=mask, ax=ax,
                               return_axes=return_axes,
                               show=show)


# ----------------------------------------------------------------------
# FPI helpers
# ----------------------------------------------------------------------
def _collapse_fpi(flux4d: np.ndarray,
                  method: Literal['sum', 'mean'] = 'sum') -> np.ndarray:
    """
    Collapse phi×theta into omni (method='sum') or pitch-avg (mean).
    """
    if flux4d.ndim != 4:
        raise ValueError("FPI burst flux must be 4-D (t,e,phi,theta)")
    if method == 'sum':
        return flux4d.sum(axis=(2, 3))
    return flux4d.mean(axis=(2, 3))


def fpi_ion_spectrogram(t: np.ndarray,
                        e: np.ndarray,
                        flux4d: np.ndarray,
                        *,
                        collapse: Literal['sum', 'mean'] = 'sum',
                        **kw):
    """
    Ion energy flux (FPI-DIS burst) spectrogram.
    Extra kwargs pass straight to generic_spectrogram().
    """
    dat2d = _collapse_fpi(flux4d, collapse)
    return generic_spectrogram(t, e, dat2d,
                               ylabel='E$_i$ (eV)',
                               title='Ion energy flux',
                               **kw)


def fpi_electron_spectrogram(t: np.ndarray,
                             e: np.ndarray,
                             flux4d: np.ndarray,
                             *,
                             collapse: Literal['sum', 'mean'] = 'sum',
                             **kw):
    dat2d = _collapse_fpi(flux4d, collapse)
    return generic_spectrogram(t, e, dat2d,
                               ylabel='E$_e$ (eV)',
                               title='Electron energy flux',
                               **kw)


# ----------------------------------------------------------------------
# HPCA helper
# ----------------------------------------------------------------------
def hpca_ion_spectrogram(t: np.ndarray,
                         e: np.ndarray,
                         omni: np.ndarray,
                         *,
                         log10: bool = True,
                         species: str = 'H+',
                         mask: Optional[np.ndarray] = None,
                         **kw):
    """
    HPCA provides omni energy spectra per species.
      omni : (Ntime, Nenergy) array
    """
    return generic_spectrogram(t, e, omni,
                               log10=log10,
                               ylabel=f'E({species}) (eV)',
                               title=f'HPCA {species} energy flux',
                               mask=mask,
                               **kw)


# ----------------------------------------------------------------------
# Optional: automatic time-grid uniformisation
# ----------------------------------------------------------------------
def spectrogram_with_resample(t: np.ndarray,
                              e: np.ndarray,
                              data: np.ndarray,
                              *,
                              cadence: str = '1s',
                              method: str = 'nearest',
                              **kw):
    """
    Convenience wrapper — resamples the *time* axis to uniform cadence
    before plotting (good for overlaying with other instrument plots).

    Uses mms_mp.resample.resample() internally.
    """
    from .resample import resample as _res
    t_uni, d_uni, good = _res(t, data, cadence=cadence, method=method)
    return generic_spectrogram(t_uni, e, d_uni,
                               mask=good, **kw)
