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
        for a in ax[:3]:
            shade_layers(a, layers, t)

    # -------- Panels 3 & 4 : placeholders ----------------------------
    ax[3].axis('off')
    ax[4].axis('off')

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
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    col_cycle = ax._get_lines.prop_cycler

    for p in probes:
        if p not in overlay_dict[var]:
            continue
        col = next(col_cycle)['color']
        data = overlay_dict[var][p]
        ax.plot(data[:, 0], data[:, 1], label=f'MMS{p}', color=col)

    ax.axvline(0, color='k', ls='--', lw=1, alpha=.6)
    ax.set_xlabel(f'Δt from crossing of MMS{ref_probe} (s)')
    ax.set_ylabel(ylabel or var)
    ax.set_title(title or f'{var} overlay')
    ax.legend()
    if show:
        plt.tight_layout()
        plt.show()
    return ax

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
