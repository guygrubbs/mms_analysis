# mms_mp/visualize.py
# ---------------------------------------------------------------------
# Publication-ready plotting helpers
# ---------------------------------------------------------------------
# • _set_style()              – one-shot Matplotlib style
# • shade_layers()            – translucent layer shading
# • summary_single()          – 5-panel quick-look per probe
# • overlay_multi()           – stack / overlay variable across probes
# • plot_displacement()       – displacement-versus-time with σ band
# ---------------------------------------------------------------------
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Iterable, Literal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =====================================================================
# Global style
# =====================================================================
def _set_style() -> None:
    plt.style.use('seaborn-v0_8-white')
    plt.rcParams.update({
        'axes.prop_cycle': plt.cycler(
            'color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 10,
        'figure.dpi': 120,
        'savefig.dpi': 300,
        'axes.titleweight': 'bold'
    })

_set_style()

# =====================================================================
# Internal helper – plot only if finite data exist
# =====================================================================
def _safe_plot(ax: plt.Axes,
               t: np.ndarray,
               y: np.ndarray,
               *args, **kwargs) -> None:
    """Silently skip all-NaN series so autoscaling behaves."""
    if np.isfinite(y).any():
        ax.plot(t, y, *args, **kwargs)

# =====================================================================
# Layer shading
# =====================================================================
def shade_layers(ax: plt.Axes,
                 layers : List[Tuple[str, int, int]],
                 t      : np.ndarray,
                 *,
                 colors : Dict[str, str] | None = None) -> None:
    """
    layers = list of (layer_type, i_start, i_end) indices into `t`
    """
    if colors is None:
        colors = dict(sheath='lightgrey',
                      mp_layer='violet',
                      magnetosphere='skyblue')

    for typ, i1, i2 in layers:
        ax.axvspan(t[i1], t[i2],
                   color=colors.get(typ, 'lightgrey'),
                   alpha=0.15, lw=0)

# =====================================================================
# 5-panel quick-look for a single spacecraft
# =====================================================================
def summary_single(t       : np.ndarray,
                   B_lmn   : np.ndarray,
                   N_i     : np.ndarray,
                   N_e     : np.ndarray,
                   N_he    : np.ndarray,
                   vN_i    : np.ndarray,
                   vN_e    : np.ndarray,
                   vN_he   : np.ndarray,
                   *,
                   layers  : Optional[List[Tuple[str, int, int]]] = None,
                   title   : str = '',
                   figsize : Tuple[int, int] = (9, 9),
                   show    : bool = True,
                   ax      : Optional[Iterable[plt.Axes]] = None
                   ) -> List[plt.Axes]:
    """
    Panels
    ------
    0  – B_L, B_M, B_N  (LMN coordinates)
    1  – N_i, N_e, N_He
    2  – V_N (ions, electrons, He⁺)
    3  – blank (caller can add displacement)
    4  – blank spacer
    """
    if ax is None:
        fig, ax = plt.subplots(5, 1, sharex=True, figsize=figsize)
    ax = list(ax)

    # -------- Panel 0 : Magnetic field -------------------------------
    _safe_plot(ax[0], t, B_lmn[:, 0], label='B$_L$')
    _safe_plot(ax[0], t, B_lmn[:, 1], label='B$_M$')
    _safe_plot(ax[0], t, B_lmn[:, 2], label='B$_N$')
    ax[0].set_ylabel('B (nT)')
    ax[0].legend(loc='upper right')

    # -------- Panel 1 : Densities ------------------------------------
    _safe_plot(ax[1], t, N_i,  label='N$_i$')
    _safe_plot(ax[1], t, N_e,  label='N$_e$', lw=1.2)
    _safe_plot(ax[1], t, N_he, label='N(He$^+$)', lw=1.2)
    ax[1].set_ylabel('Density (cm$^{-3}$)')
    ax[1].legend(loc='upper right')

    # -------- Panel 2 : Normal velocities ----------------------------
    _safe_plot(ax[2], t, vN_i,  label='V$_N$ ions')
    _safe_plot(ax[2], t, vN_e,  label='V$_N$ e$^-$',  lw=1.2)
    _safe_plot(ax[2], t, vN_he, label='V$_N$ He$^+$', lw=1.2)
    ax[2].set_ylabel('V$_N$ (km/s)')
    ax[2].legend(loc='upper right')

    # -------- Shade layers (top 3 panels) ----------------------------
    if layers:
        for a in ax[:4]:
            shade_layers(a, layers, t)

    # -------- Panel 3 : |B| and dynamic pressure --------------------
    B_mag = np.linalg.norm(B_lmn[:, :3], axis=1)
    _safe_plot(ax[3], t, B_mag, label='|B|')
    ax[3].set_ylabel('|B| (nT)')

    dyn_ax = ax[3].twinx()
    mp_kg = 1.67262192369e-27
    cm3_to_m3 = 1e6
    km_to_m = 1e3

    def _dyn_pressure(density_cm3: np.ndarray, velocity_kms: np.ndarray, mass_factor: float = 1.0) -> np.ndarray:
        density_si = np.asarray(density_cm3) * cm3_to_m3 * mp_kg * mass_factor
        velocity_si = np.asarray(velocity_kms) * km_to_m
        return density_si * (velocity_si ** 2) * 1e9

    ion_pdyn = _dyn_pressure(N_i, vN_i)
    dyn_ax.plot(t, ion_pdyn, color='#d62728', label='P$_{dyn}$ ions')
    if np.isfinite(N_he).any() and np.isfinite(vN_he).any():
        he_pdyn = _dyn_pressure(N_he, vN_he, mass_factor=4.0)
        dyn_ax.plot(t, he_pdyn, color='#ff7f0e', linestyle='--', label='P$_{dyn}$ He$^+$')
    dyn_ax.set_ylabel('Dynamic Pressure (nPa)')
    ax[3]._dynamic_axis = dyn_ax

    handles, labels = ax[3].get_legend_handles_labels()
    handles2, labels2 = dyn_ax.get_legend_handles_labels()
    if handles or handles2:
        dyn_ax.legend(handles + handles2, labels + labels2, loc='upper right')

    # -------- Panel 4 : charge balance & He fraction -----------------
    delta_n = np.asarray(N_e) - np.asarray(N_i)
    _safe_plot(ax[4], t, delta_n, label='ΔN (e⁻ − ions)')
    if np.isfinite(delta_n).any():
        ax[4].axhline(0.0, color='k', linestyle=':', linewidth=0.8)
    ax[4].set_ylabel('ΔN (cm$^{-3}$)')

    he_frac = np.full_like(np.asarray(N_he, dtype=float), np.nan)
    denom = np.asarray(N_i, dtype=float)
    mask = np.isfinite(denom) & (np.abs(denom) > 0)
    he_frac[mask] = np.asarray(N_he, dtype=float)[mask] / denom[mask]
    frac_ax = ax[4].twinx()
    frac_ax.plot(t, he_frac, color='#9467bd', label='He$^+$/N$_i$')
    frac_ax.set_ylabel('He$^+$ Fraction')
    ax[4]._he_fraction_axis = frac_ax

    handles4, labels4 = ax[4].get_legend_handles_labels()
    handles4b, labels4b = frac_ax.get_legend_handles_labels()
    if handles4 or handles4b:
        frac_ax.legend(handles4 + handles4b, labels4 + labels4b, loc='upper right')

    # -------- Shared x-axis formatting -------------------------------
    ax[-1].set_xlabel('UTC')
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    if title:
        ax[0].set_title(title)

    if show:
        plt.tight_layout()
        plt.show()

    return ax

# =====================================================================
# Overlay a variable across multiple probes (timing plots)
# =====================================================================
def overlay_multi(overlay_dict: Dict[str, Dict[str, np.ndarray]],
                  *,
                  var      : str,
                  ref_probe: str,
                  probes   : Iterable[str] = ('1', '2', '3', '4'),
                  ylabel   : str = '',
                  title    : str = '',
                  ax       : Optional[plt.Axes] = None,
                  show     : bool = True) -> plt.Axes:
    """
    overlay_dict[var][probe] == 2-col array [[Δt, value], ...]
    produced by e.g. stack_aligned().
    """
    if var not in overlay_dict:
        raise KeyError(f'missing overlay variable: {var}')

    data_block = overlay_dict[var]
    if ref_probe not in data_block:
        raise KeyError(f'reference probe {ref_probe} missing from overlays for {var}')

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 3))

    plotted: bool = False
    for probe in probes:
        series = data_block.get(probe)
        if series is None:
            continue
        arr = np.asarray(series)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f'overlay array for probe {probe} must be (N, 2) [Δt, value]')
        if not np.isfinite(arr[:, 1]).any():
            continue

        label = f'MMS{probe}' if probe != ref_probe else f'MMS{probe} (ref)'
        lw = 2.5 if probe == ref_probe else 1.3
        ax.plot(arr[:, 0], arr[:, 1], label=label, lw=lw)
        plotted = True

    if not plotted:
        raise ValueError(f'no finite data available for {var}')

    ax.axvline(0.0, color='k', lw=0.8, linestyle=':')
    ax.set_xlabel(f'Δt relative to MMS{ref_probe} (s)')
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc='upper right')

    if show:
        plt.tight_layout()
        plt.show()

    return ax

# ---------------------------------------------------------------------
# Additional plotting helpers expected by tests
# ---------------------------------------------------------------------

def plot_spectrogram(ax: plt.Axes,
                     t: np.ndarray,
                     e: np.ndarray,
                     data: np.ndarray,
                     *,
                     title: str = '',
                     ylabel: str = 'Energy (eV)',
                     clabel: str = 'Flux'):
    from .spectra import generic_spectrogram
    # Return the QuadMesh/artist so callers can assert creation
    return generic_spectrogram(t, e, data, ylabel=ylabel, title=title, show=False, ax=ax, return_axes=False) or ax.collections[-1]


def plot_magnetic_field(axes: Iterable[plt.Axes],
                        t: np.ndarray,
                        B_xyz: np.ndarray,
                        *, labels=None, colors=None):
    axes = list(axes)
    labels = labels or ['Bx', 'By', 'Bz']
    colors = colors or [None, None, None]
    for i in range(3):
        _safe_plot(axes[i], t, B_xyz[:, i], label=labels[i], color=colors[i])
        axes[i].legend(loc='upper right')
    if np.issubdtype(np.asarray(t).dtype, np.datetime64):
        axes[-1].set_xlabel('Time (UTC)')
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    else:
        axes[-1].set_xlabel('Time (s)')


def plot_boundary_structure(axes: Iterable[plt.Axes],
                            x: np.ndarray,
                            series_list: Iterable[np.ndarray],
                            *, labels=None, title: str = ''):
    axes = list(axes)
    labels = labels or [f'Var {i+1}' for i in range(len(series_list))]
    for ax, y, lab in zip(axes, series_list, labels):
        _safe_plot(ax, x, y, label=lab)
        ax.legend(loc='upper right')
        ax.axvline(0.0, color='k', lw=0.8, alpha=0.5)
    if title:
        axes[0].set_title(title)

    # Note: overlay plotting handled in overlay_multi();
    # this helper only draws the provided series with boundary marker.
    return axes

# =====================================================================
# Displacement versus time
# =====================================================================
def plot_displacement(t        : np.ndarray,
                      disp_km  : np.ndarray,
                      sigma    : Optional[np.ndarray] = None,
                      *,
                      ax       : Optional[plt.Axes] = None,
                      title    : str = 'Magnetopause displacement',
                      show     : bool = True) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 3))

    _safe_plot(ax, t, disp_km, label='Displacement')
    if sigma is not None and np.isfinite(sigma).any():
        ax.fill_between(t, disp_km - sigma, disp_km + sigma,
                        color='grey', alpha=0.3, label='±1σ')

    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel('Δs (km)')
    ax.set_xlabel('Time (UTC)' if np.issubdtype(t.dtype, np.datetime64)
                  else 'Seconds from start')
    ax.set_title(title)
    ax.legend()

    if show:
        plt.tight_layout()
        plt.show()
    return ax
