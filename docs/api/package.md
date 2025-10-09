# Package API Reference

## Package-level imports

The `mms_mp` package provides convenient access to key functions and classes:

```python
import numpy as np
import mms_mp

# Key functions available at package level
data = mms_mp.load_event(trange, probes=['1', '2'])
lmn = mms_mp.hybrid_lmn(B_data, pos_gsm_km=position)
layers = mms_mp.detect_crossings_multi(time, he_density, B_normal, ni=ion_density)
disp = mms_mp.integrate_disp(time, velocity, max_step_s=0.5)
print(disp.segment_count, disp.n_gaps_filled)
n_hat, V_ph, sigma, diag = mms_mp.timing_normal(positions, times, return_diagnostics=True)
print(diag['condition_number'])
t_grid, vars, good = mms_mp.merge_vars(var_dict, cadence='150ms')

# Configuration classes
cfg = mms_mp.DetectorCfg(he_in=0.25, he_out=0.15, he_frac_in=0.08)
```

## Package Information

```python
import mms_mp

print(mms_mp.__version__)    # Package version
print(mms_mp.__author__)     # Author information
print(mms_mp.__license__)    # License type
```

## Module Access

All modules are available as attributes:

```python
import mms_mp

# Access modules directly
mms_mp.data_loader.load_event(...)
mms_mp.coords.hybrid_lmn(...)
mms_mp.boundary.detect_crossings_multi(...)
mms_mp.motion.integrate_disp(...)
mms_mp.multispacecraft.timing_normal(...)
mms_mp.visualize.summary_single(...)
mms_mp.resample.merge_vars(...)
mms_mp.electric.exb_velocity_sync(...)
mms_mp.quality.build_quality_masks(...)
mms_mp.spectra.plot_fpi_spectrogram(...)
mms_mp.thickness.layer_thicknesses(...)
mms_mp.cli.main(...)
```

`mms_mp.visualize.summary_single` renders five physics-focused panels, including
magnetic field components, species densities/velocities, normal dynamic pressure,
and charge balance with He⁺ fraction.  Dynamic pressure is computed as
$n m_p v_N^2$ (converted to nPa) for ions and scaled by four for He⁺, so the
panels report physically meaningful momentum flux without manual tweaking.  Use
`mms_mp.visualize.overlay_multi` to compare aligned probe series – the helper now
validates that each input array is shaped `(N, 2)` `[Δt, value]`, highlights the
reference probe, and labels the x-axis as Δt relative to the selected spacecraft.

## Available Modules

| Module | Description |
|--------|-------------|
| `data_loader` | CDF download and variable extraction |
| `coords` | LMN transforms (MVA / model / hybrid) |
| `resample` | Multi-variable resampling/merging helpers |
| `electric` | E×B drift, v_N selection (quality-aware) |
| `quality` | Instrument quality-flag masks |
| `boundary` | Multi-parameter boundary detector |
| `motion` | Integrate v_N → displacement ±σ (adaptive steps + metadata) |
| `multispacecraft` | Timing method (n̂, V_ph, diagnostics) + alignment |
| `visualize` | Publication-ready plotting helpers |
| `spectra` | FPI ion/electron spectrograms |
| `thickness` | Layer thickness calculation utilities |
| `cli` | Command-line pipeline |

## Quick Start Example

```python
import mms_mp

# 1. Load data
trange = ['2019-01-27/12:20:00', '2019-01-27/12:40:00']
evt = mms_mp.load_event(trange, probes=['1', '2'])

# Access provenance metadata
meta = evt['__meta__']
print(meta['download_summary']['fgm'])

# 2. Process MMS1 data
d = evt['1']
t_grid, vars_grid, good = mms_mp.merge_vars({
    'He': (d['N_he'][0], d['N_he'][1]),
    'Ni': (d['N_tot'][0], d['N_tot'][1]),
    'B': (d['B_gsm'][0], d['B_gsm'][1]),
}, cadence='150ms')

# 3. Coordinate transformation
mid_idx = len(d['B_gsm'][0]) // 2
B_slice = d['B_gsm'][1][mid_idx-64:mid_idx+64, :3]
lmn = mms_mp.hybrid_lmn(B_slice, formation_type='auto')
B_lmn = lmn.to_lmn(vars_grid['B'][:, :3])
print(lmn.method, lmn.meta['eig_ratio_thresholds'])

# 4. Boundary detection
cfg = mms_mp.DetectorCfg(he_in=0.25, he_out=0.15, he_frac_in=0.08)
layers = mms_mp.detect_crossings_multi(
    t_grid,
    vars_grid['He'],
    B_lmn[:, 2],
    ni=vars_grid['Ni'],
    cfg=cfg,
)

print(f"Found {len(layers)} boundary layers")

# 5. E×B drift with magnetic screening
v_exb, quality = mms_mp.exb_velocity(
    evt['1']['E_gsm'][1],
    evt['1']['B_gsm'][1],
    unit_E='mV/m',
    unit_B='nT',
    min_b=1.0,
    return_quality=True,
)
vn_choice = mms_mp.normal_velocity(
    evt['1']['V_bulk_lmn'][1],
    v_exb,
    b_mag_nT=np.linalg.norm(evt['1']['B_gsm'][1], axis=1),
    return_metadata=True,
)
print(vn_choice.source[:5])  # → array(['exb', 'exb', 'bulk', ...], dtype=object)
```

`formation_type` tunes the eigenvalue thresholds used to accept an MVA solution.
If the ratios fall below the table in [Physics Units and Conventions](../physics-units-and-conventions.md),
`hybrid_lmn` transparently falls back to pySPEDAS (when installed) or the
Shue (1997) model normal and records the outcome in `LMN.method` and
`LMN.meta['eig_ratio_thresholds']`.

## Error Handling

The package uses standard Python exceptions:

- `ImportError`: Missing dependencies
- `ValueError`: Invalid parameter values
- `FileNotFoundError`: Missing data files
- `RuntimeError`: Processing errors

Example:

```python
try:
    evt = mms_mp.load_event(trange, probes=['1'])
except Exception as e:
    print(f"Data loading failed: {e}")
```

## Configuration

### PySPEDAS Cache

Data is cached in `~/.pyspedas/` by default. To change:

```python
import os
os.environ['SPEDAS_DATA_DIR'] = '/path/to/cache'
```

### Logging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
