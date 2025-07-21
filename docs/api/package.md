# Package API Reference

## Package-level imports

The `mms_mp` package provides convenient access to key functions and classes:

```python
import mms_mp

# Key functions available at package level
data = mms_mp.load_event(trange, probes=['1', '2'])
lmn = mms_mp.hybrid_lmn(B_data, pos_gsm_km=position)
layers = mms_mp.detect_crossings_multi(time, density, B_normal)
disp = mms_mp.integrate_disp(time, velocity)
n_hat, V_ph, sigma = mms_mp.timing_normal(positions, times)
t_grid, vars, good = mms_mp.merge_vars(var_dict, cadence='150ms')

# Configuration classes
cfg = mms_mp.DetectorCfg(he_in=0.25, he_out=0.15, BN_tol=1.0)
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

## Available Modules

| Module | Description |
|--------|-------------|
| `data_loader` | CDF download and variable extraction |
| `coords` | LMN transforms (MVA / model / hybrid) |
| `resample` | Multi-variable resampling/merging helpers |
| `electric` | E×B drift, v_N selection |
| `quality` | Instrument quality-flag masks |
| `boundary` | Multi-parameter boundary detector |
| `motion` | Integrate v_N → displacement ±σ |
| `multispacecraft` | Timing method (n̂, V_ph) + alignment |
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

# 2. Process MMS1 data
d = evt['1']
t_grid, vars_grid, good = mms_mp.merge_vars({
    'He': (d['N_he'][0], d['N_he'][1]),
    'B': (d['B_gsm'][0], d['B_gsm'][1]),
}, cadence='150ms')

# 3. Coordinate transformation
mid_idx = len(d['B_gsm'][0]) // 2
B_slice = d['B_gsm'][1][mid_idx-64:mid_idx+64, :3]
lmn = mms_mp.hybrid_lmn(B_slice)
B_lmn = lmn.to_lmn(vars_grid['B'][:, :3])

# 4. Boundary detection
cfg = mms_mp.DetectorCfg(he_in=0.25, he_out=0.15)
layers = mms_mp.detect_crossings_multi(
    t_grid, vars_grid['He'], B_lmn[:, 2], cfg=cfg
)

print(f"Found {len(layers)} boundary layers")
```

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
