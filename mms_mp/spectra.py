# mms_mp/spectra.py
# ------------------------------------------------------------
# Quick-look FPI / HPCA spectrogram helpers  – NEW MODULE
# ------------------------------------------------------------
# Goals
# -----
# • Provide pcolormesh wrappers to visualise energy–time (or
#   velocity–time) spectrograms from MMS instruments.
# • Keep dependency-light: rely only on numpy + matplotlib.
#
# Supported functions
# -------------------
# • fpi_ion_spectrogram()    – for FPI-DIS “ion energy flux”
# • fpi_electron_spectrogram()   – for FPI-DES
# • generic_spectrogram()    – underlying engine; accepts any
#                               time axis, energy axis, 2-D data.
# ------------------------------------------------------------
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, Tuple, Literal, Iterable, Dict


# ------------------------------------------------------------------
# Core pcolormesh engine
# ------------------------------------------------------------------
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
                        ax: Optional[plt.Axes] = None,
                        show: bool = True) -> plt.Axes:
    """
    Parameters
    ----------
    t     : 1-D time array (datetime64 or float seconds)
    e     : 1-D energy/velocity axis (length M) — must match data shape
    data  : 2-D array (len(t) × len(e))
    log10 : plot log10(data)
    vmin/vmax : colour limits (post-log)
    cmap  : matplotlib colormap
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    if log10:
        with np.errstate(invalid='ignore'):
            data_plot = np.log10(np.abs(data))
        clabel = 'log$_{10}$ Flux'
    else:
        data_plot = data
        clabel = 'Flux'

    T, E = np.meshgrid(t, e, indexing='ij')
    pcm = ax.pcolormesh(T, E, data_plot,
                        cmap=cmap,
                        shading='auto',
                        vmin=vmin, vmax=vmax)
    ax.set_yscale('log')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('UTC' if np.issubdtype(t.dtype, np.datetime64)
                  else 'Time (s)')
    if np.issubdtype(t.dtype, np.datetime64):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    if title:
        ax.set_title(title)
    cbar = plt.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(clabel)
    if show:
        plt.tight_layout(); plt.show()
    return ax


# ------------------------------------------------------------------
# Convenience wrappers for FPI data
# ------------------------------------------------------------------
def _extract_fpi_spectra(t_epoch: np.ndarray,
                         flux: np.ndarray,
                         energies: np.ndarray,
                         omnicone: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reshape FPI burst omni-flux for spectrogram:
        • Inputs follow pySPEDAS: flux shape (Ntime, Nenergy, Nphi, Ntheta)
        • If omnicone=True  →  sum over solid angle (phi, theta)
        • Returns t, e, data2D (Ntime × Nenergy)
    """
    if flux.ndim != 4:
        raise ValueError("Expect 4-D FPI flux array (t, e, phi, theta)")
    if omnicone:
        # integrate over angular bins (∑ sinθ dθ dφ)
        solid = flux.sum(axis=(2, 3))
        data2d = solid
    else:
        # keep pitch angle collapsed only over phi
        data2d = flux.mean(axis=2)  # still θ dimension
    return t_epoch, energies, data2d


def fpi_ion_spectrogram(t: np.ndarray,
                        flux_4d: np.ndarray,
                        energies: np.ndarray,
                        *,
                        title: str = 'Ion energy flux',
                        ax: Optional[plt.Axes] = None,
                        show: bool = True,
                        **kwargs):
    """
    Quick ion spectrogram from FPI-DIS burst omni energy flux.

    kwargs passed to generic_spectrogram().
    """
    t_sec, e, dat = _extract_fpi_spectra(t, flux_4d, energies)
    ax = generic_spectrogram(t_sec, e, dat,
                             ylabel='E$_i$ (eV)',
                             title=title,
                             ax=ax, show=show, **kwargs)
    return ax


def fpi_electron_spectrogram(t: np.ndarray,
                             flux_4d: np.ndarray,
                             energies: np.ndarray,
                             *,
                             title: str = 'Electron energy flux',
                             ax: Optional[plt.Axes] = None,
                             show: bool = True,
                             **kwargs):
    t_sec, e, dat = _extract_fpi_spectra(t, flux_4d, energies)
    ax = generic_spectrogram(t_sec, e, dat,
                             ylabel='E$_e$ (eV)',
                             title=title,
                             ax=ax, show=show, **kwargs)
    return ax
