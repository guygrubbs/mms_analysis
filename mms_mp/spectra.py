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

# Additional spectral analysis helpers expected by tests
try:
    from scipy import signal as scipy_signal
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    import numpy as _np
    _HAVE_SCIPY = False
    class _CompatSignal:
        # Minimal replacements using numpy
        def welch(self, x, fs, nperseg=256):
            n = len(x)
            freqs = _np.fft.rfftfreq(n, d=1.0/fs)
            fft = _np.fft.rfft(x * _np.hanning(n))
            psd = (fft.conj()*fft).real / (fs*n)
            return freqs, psd
        def csd(self, x, y, fs, nperseg=256):
            n = len(x)
            freqs = _np.fft.rfftfreq(n, d=1.0/fs)
            X = _np.fft.rfft(x * _np.hanning(n))
            Y = _np.fft.rfft(y * _np.hanning(n))
            Pxy = X * _np.conj(Y) / (fs*n)
            return freqs, Pxy
        def coherence(self, x, y, fs, nperseg=256):
            f, Pxy = self.csd(x, y, fs=fs, nperseg=nperseg)
            f, Px = self.welch(x, fs=fs, nperseg=nperseg)
            f, Py = self.welch(y, fs=fs, nperseg=nperseg)
            Cxy = (abs(Pxy)**2) / (Px*Py + 1e-30)
            return f, Cxy
    scipy_signal = _CompatSignal()


def calculate_psd(signal_1d: np.ndarray, *, fs: float, nperseg: int = 256):
    f, Pxx = scipy_signal.welch(signal_1d, fs=fs, nperseg=nperseg)
    return f, Pxx


def cross_spectral_analysis(x: np.ndarray, y: np.ndarray, *, fs: float, nperseg: int = 256):
    f, Pxy = scipy_signal.csd(x, y, fs=fs, nperseg=nperseg)
    f2, Cxy = scipy_signal.coherence(x, y, fs=fs, nperseg=nperseg)
    # phase of cross-spectrum
    phase = np.angle(Pxy)
    return f, np.abs(Pxy), Cxy, phase


def wavelet_analysis(x: np.ndarray, *, fs: float, f_min: float, f_max: float, n_freqs: int = 50):
    """Simple CWT-like time-frequency analysis; uses SciPy if available, otherwise numpy STFT fallback."""
    x = np.asarray(x)
    if _HAVE_SCIPY and hasattr(scipy_signal, 'cwt'):
        widths = np.linspace(1, n_freqs, n_freqs)
        try:
            scalogram = scipy_signal.cwt(x, scipy_signal.morlet2, widths, w=6)
            freqs = fs * 6 / (2*np.pi*widths)
        except Exception:
            scalogram = scipy_signal.cwt(x, scipy_signal.ricker, widths)
            freqs = fs / widths
        mask = (freqs >= f_min) & (freqs <= f_max)
        freqs = freqs[mask]
        scalogram = np.abs(scalogram[mask])
        times = np.arange(len(x)) / fs
        return freqs, times, scalogram
    # Fallback: STFT using numpy FFT over sliding windows
    win = int(max(16, min(len(x)//8, fs)))
    hop = max(1, win//4)
    window = np.hanning(win)
    specs = []
    times = []
    for start in range(0, len(x)-win+1, hop):
        seg = x[start:start+win] * window
        X = np.fft.rfft(seg)
        freqs = np.fft.rfftfreq(win, d=1.0/fs)
        specs.append(np.abs(X))
        times.append((start + win/2)/fs)
    S = np.array(specs).T  # shape (F, T)
    # limit frequency range
    mask = (freqs >= f_min) & (freqs <= f_max)
    return freqs[mask], np.array(times), S[mask]


def analyze_magnetic_fluctuations(B_xyz: np.ndarray, *, fs: float) -> dict:
    # Sum PSD across components
    comp = []
    f_ref = None
    for i in range(3):
        f, P = calculate_psd(B_xyz[:, i], fs=fs, nperseg=256)
        if f_ref is None:
            f_ref = f
        comp.append(P)
    psd_total = np.sum(np.vstack(comp), axis=0)
    return {'frequencies': f_ref, 'psd_total': psd_total}



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
                        return_axes: bool = False,
                        clabel: Optional[str] = None):
    """
    Quick-draw pcolormesh for any 2-D spectrogram.

    Parameters
    ----------
    t, e, data : (N,) (M,) (N,M) arrays
    log10      : take log10(data)  (default True)
    mask       : boolean same shape as data – False→transparent
    vmin/vmax  : colour limits **after** log10 if log10=True
    clabel     : optional colorbar label override (e.g., 'log10 dJ (keV/(cm^2 s sr keV))')
    """
    z = _prep_log(data, log10)
    cbl = clabel if clabel is not None else ('log$_{10}$ Flux' if log10 else 'Flux')
    return _generic_pcolormesh(t, e, z,
                               ylabel=ylabel,
                               clabel=cbl,
                               vmin=vmin, vmax=vmax,
                               cmap=cmap, title=title,
                               mask=mask, ax=ax,
                               return_axes=return_axes,
                               show=show)


# ----------------------------------------------------------------------
# FPI helpers
# ----------------------------------------------------------------------
def _collapse_fpi(flux4d: np.ndarray,
                  e_len: int,
                  method: Literal['sum', 'mean'] = 'sum') -> np.ndarray:
    """
    Collapse angular dimensions into omni.
    Detect which axis corresponds to energy by matching e_len, then sum/mean over the other two.
    Accepts either (t, e, phi, theta) or (t, phi, theta, e).
    """
    if flux4d.ndim != 4:
        raise ValueError("FPI burst flux must be 4-D (t,*,*,*)")
    # Identify energy axis by length
    axes = list(range(4))
    # Assume axis 0 is time
    cand = [ax for ax in axes[1:] if flux4d.shape[ax] == e_len]
    if not cand:
        # fallback: assume axis=1 is energy
        e_ax = 1
    else:
        e_ax = cand[0]
    # Angular axes = the two non-time, non-energy axes
    ang_axes = tuple(ax for ax in axes[1:] if ax != e_ax)
    if method == 'mean':
        dat2d = flux4d.mean(axis=ang_axes)
    else:
        dat2d = flux4d.sum(axis=ang_axes)
    # Ensure (time, energy) ordering
    if e_ax == 3:
        # result shape is (t, e) already after summing ang axes
        return dat2d
    if e_ax == 1:
        return dat2d
    # If energy axis was 2 before summing (shouldn't happen), transpose accordingly
    return dat2d


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
    dat2d = _collapse_fpi(flux4d, e_len=e.size, method=collapse)
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
    dat2d = _collapse_fpi(flux4d, e_len=e.size, method=collapse)
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
