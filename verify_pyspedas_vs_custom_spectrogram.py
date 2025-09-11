import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from datetime import datetime

# Add vendored pySPEDAS to path and import
import sys
sys.path.insert(0, os.path.abspath('references/pyspedas-master'))
from pyspedas.projects.mms.particles.mms_part_getspec import mms_part_getspec
from pyspedas.projects.mms.mms_config import CONFIG
from pyspedas.tplot_tools import get_data, data_quants
from pyspedas.tplot_tools.MPLPlotter.specplot import specplot_make_1d_ybins
from pyspedas.tplot_tools.MPLPlotter.tplot import tplot


def ensure_dt_list(times):
    if isinstance(times, np.ndarray):
        times = times.tolist()
    out = []
    for t in times:
        if isinstance(t, (np.datetime64,)):
            # Convert numpy datetime64 to python datetime
            ts = t.astype('datetime64[ns]').astype('int64') / 1e9
            out.append(datetime.utcfromtimestamp(ts))
        elif isinstance(t, (int, float)):
            out.append(datetime.utcfromtimestamp(float(t)))
        else:
            out.append(t)
    return out


def plot_custom(times, energy_centers, z, out_png, title):
    # Compute Y bin boundaries similar to specplot
    ybins, _dir = specplot_make_1d_ybins(z, energy_centers, ylog=True, no_regrid=True)
    # specplot_make_1d_ybins returns just the bin boundaries when no_regrid=True
    if ybins is None or len(np.atleast_1d(ybins)) < 2:
        # Fallback: simple geometric midpoints
        e = np.asarray(energy_centers)
        Ee = np.empty(e.size + 1)
        Ee[1:-1] = np.sqrt(e[:-1] * e[1:])
        Ee[0] = e[0]**2 / Ee[1]
        Ee[-1] = e[-1]**2 / Ee[-2]
        ybins = Ee
    times = ensure_dt_list(times)
    Xc = mdates.date2num(times)
    Xe = np.empty(len(Xc) + 1)
    if len(Xc) > 1:
        Xe[1:-1] = 0.5 * (Xc[:-1] + Xc[1:])
        Xe[0] = Xc[0] - (Xc[1] - Xc[0]) / 2.0
        Xe[-1] = Xc[-1] + (Xc[-1] - Xc[-2]) / 2.0
    else:
        Xe[:] = [Xc[0] - 1.0/1440, Xc[0] + 1.0/1440]

    z = np.asarray(z)
    # z is (Nt, Ne); pcolormesh expects (Ny, Nx) for Z relative to edges
    # We have edges: (len(ybins), len(Xe)). Need Z.T shape (len(ybins)-1, len(Xe)-1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    fig.suptitle(title, fontsize=12, fontweight='bold')
    vmin = np.nanpercentile(z, 5)
    vmax = np.nanpercentile(z, 99)
    pcm = ax.pcolormesh(Xe, ybins, np.where(z <= 0, np.nan, z).T, shading='auto',
                        norm=mcolors.LogNorm(vmin=max(vmin, 1e-2), vmax=max(vmax, 1.0)),
                        cmap='viridis')
    cbar = fig.colorbar(pcm, ax=ax, pad=0.01)
    cbar.set_label('eflux (arb.)')
    ax.set_ylabel('Energy (eV)')
    ax.set_xlabel('Time (UT)')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.grid(True, axis='y', which='both', alpha=0.2)
    # Format time axis
    converter = mdates.ConciseDateConverter()
    ax.xaxis.set_major_locator(converter.get_locator(None, None))
    ax.xaxis.set_major_formatter(converter.get_formatter(None, None))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    # Ensure local cache
    data_root = os.path.abspath('mms_data')
    os.makedirs(data_root, exist_ok=True)
    CONFIG['local_data_dir'] = data_root
    print('[info] CONFIG.local_data_dir =', CONFIG['local_data_dir'])

    trange = ['2019-01-27 12:15:00', '2019-01-27 12:50:00']
    probe = '3'

    for species, sr in [('Ion','i'), ('Electron','e')]:
        print(f'\n=== MMS{probe} {species} BRST ===')
        # Build energy spectrogram using pySPEDAS (allow default regridding)
        out_vars = mms_part_getspec(instrument='fpi', probe=probe, species=sr, data_rate='brst',
                                    trange=trange, output=['energy'], units='eflux',
                                    center_measurement=False, spdf=False,
                                    no_regrid=False)  # let pySPEDAS do its thing
        if not out_vars:
            print('   [warn] mms_part_getspec returned no variables')
            continue
        # Find the energy spectrogram var (y=Z, v=energy)
        en_var = None
        for v in out_vars:
            if v.endswith('_energy'):
                en_var = v
                break
        if en_var is None:
            print('   [warn] No *_energy variable in outputs:', out_vars)
            continue
        print('   [info] Using pySPEDAS var:', en_var)

        # Save the "official" pySPEDAS plot
        tplot(en_var, save_png=f'pyspedas_{en_var}.png', trange=trange)

        # Pull out arrays to plot with custom method
        g = get_data(en_var)
        if isinstance(g, dict):
            times = g.get('x')
            z = g.get('y')  # pySPEDAS stores Z in 'y'
            energy = g.get('v')  # energy bins in 'v'
        else:
            times, z, energy = g[0], g[1], (g[2] if len(g) > 2 else None)
        if energy is None:
            print('   [warn] No energy vector available from tplot var; skipping custom plot')
            continue
        print('   [shape] z:', np.shape(z), 'energy:', np.shape(energy), 'times:', len(times))
        # If energy is 2D (Nt, Ne), use first row
        energy_centers = energy[0] if isinstance(energy, np.ndarray) and energy.ndim == 2 else energy
        # Our custom plot
        plot_custom(times, energy_centers, z, out_png=f'custom_{en_var}.png', title=f'Custom {en_var}')


if __name__ == '__main__':
    import matplotlib as mpl
    main()

