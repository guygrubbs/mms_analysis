# mms_mp/visualize.py
# ------------------------------------------------------------
# Publication-ready plotting helpers  (upgraded)
# ------------------------------------------------------------
# What’s new
# ----------
# • “House” matplotlib style (tight, Science-ready).
# • Layer-shade & crossing-line helpers.
# • `summary_single()`        – 4-panel quick-look for one probe.
# • `overlay_multi()`         – stack/overlay any variable across probes.
# • `plot_displacement()`     – displacement vs time with σ-band.
# • All APIs accept **ax=…** for subplot reuse.
# ------------------------------------------------------------
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Iterable, Literal
import matplotlib.dates as mdates

# ------------------------------------------------------------------
# Global style (call once)
# ------------------------------------------------------------------
def _set_style():
    plt.style.use('seaborn-v0_8-white')
    plt.rcParams.update({
        'axes.prop_cycle': plt.cycler('color',
             ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 10,
        'figure.dpi': 120,
        'savefig.dpi': 300
    })

_set_style()

# ------------------------------------------------------------------
# Helper: shade layers
# ------------------------------------------------------------------
def shade_layers(ax: plt.Axes,
                 layers: List[Tuple[str, int, int]],
                 t: np.ndarray,
                 colors: Dict[str, str] = None):
    """
    Draw translucent shading for each layer onto axis.
    layers : list of (layer_type, i_start, i_end)
    """
    if colors is None:
        colors = dict(sheath='lightgrey', mp_layer='violet', magnetosphere='skyblue')
    for typ, i1, i2 in layers:
        ax.axvspan(t[i1], t[i2], color=colors.get(typ, 'lightgrey'), alpha=0.15, lw=0)

# ------------------------------------------------------------------
# Single-probe quick-look
# ------------------------------------------------------------------
def summary_single(t: np.ndarray,
                   B_lmn: np.ndarray,
                   N_tot: np.ndarray,
                   N_he: np.ndarray,
                   vN_tot: np.ndarray,
                   vN_he: np.ndarray,
                   *,
                   layers: Optional[List[Tuple[str, int, int]]] = None,
                   title: str = '',
                   figsize=(9, 8),
                   show: bool = True,
                   ax: Optional[Iterable[plt.Axes]] = None
                   ) -> List[plt.Axes]:
    """
    4-panel overview (B, densities, velocities, displacement).
    """
    if ax is None:
        fig, ax = plt.subplots(4, 1, sharex=True, figsize=figsize)
    ax = list(ax)

    ax[0].plot(t, B_lmn[:, 0], label='B_L')
    ax[0].plot(t, B_lmn[:, 1], label='B_M')
    ax[0].plot(t, B_lmn[:, 2], label='B_N')
    ax[0].set_ylabel('B (nT)'); ax[0].legend(loc='upper right')

    ax[1].plot(t, N_tot, label='N_i total')
    ax[1].plot(t, N_he, label='N(He+)', lw=1.3)
    ax[1].set_ylabel('Density (cm$^{-3}$)'); ax[1].legend(loc='upper right')

    ax[2].plot(t, vN_tot, label='V_N ion')
    ax[2].plot(t, vN_he, label='V_N He+', lw=1.3)
    ax[2].set_ylabel('V$_N$ (km/s)'); ax[2].legend(loc='upper right')

    if layers:
        for a in ax[:3]:
            shade_layers(a, layers, t)

    ax[3].axis('off')  # free slot (for displacement externally)

    ax[-1].set_xlabel('UTC')
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    if title:
        ax[0].set_title(title)
    if show:
        plt.tight_layout()
        plt.show()
    return ax


# ------------------------------------------------------------------
# Overlay multi-probe variable
# ------------------------------------------------------------------
def overlay_multi(overlay_dict: Dict[str, Dict[str, np.ndarray]],
                  *,
                  var: str,
                  ref_probe: str,
                  probes: Iterable[str] = ('1', '2', '3', '4'),
                  ylabel: str = '',
                  title: str = '',
                  ax: Optional[plt.Axes] = None,
                  show: bool = True):
    """
    overlay_dict[var][probe] = 2-col array [[Δt, value], ...]  (from stack_aligned)
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
        plt.tight_layout(); plt.show()
    return ax


# ------------------------------------------------------------------
# Displacement plot
# ------------------------------------------------------------------
def plot_displacement(t: np.ndarray,
                      disp_km: np.ndarray,
                      sigma: Optional[np.ndarray] = None,
                      *,
                      ax: Optional[plt.Axes] = None,
                      title: str = 'Magnetopause displacement',
                      show: bool = True):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    ax.plot(t, disp_km, label='Displacement')
    if sigma is not None:
        ax.fill_between(t, disp_km - sigma, disp_km + sigma,
                        color='grey', alpha=0.3, label='±1σ')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel('Δs (km)')
    ax.set_xlabel('Time (UTC)' if np.issubdtype(t.dtype, np.datetime64)
                  else 'Seconds from start')
    ax.set_title(title)
    ax.legend()
    if show:
        plt.tight_layout(); plt.show()
    return ax
