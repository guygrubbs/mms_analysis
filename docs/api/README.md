# API Reference

Complete documentation of all modules, classes, and functions in the MMS Magnetopause Analysis Toolkit.

## Module Overview

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| [`data_loader`](data_loader.md) | CDF download and variable extraction | `load_event()`, `load_ephemeris()` |
| [`coords`](coords.md) | Coordinate transformations | `hybrid_lmn()`, `LMNTransform` |
| [`resample`](resample.md) | Multi-variable resampling | `merge_vars()`, `resample()` |
| [`boundary`](boundary.md) | Boundary detection | `detect_crossings_multi()`, `DetectorCfg` |
| [`electric`](electric.md) | E×B drift calculations | `exb_velocity()`, `normal_velocity()` |
| [`motion`](motion.md) | Displacement integration | `integrate_disp()`, `DispResult` |
| [`multispacecraft`](multispacecraft.md) | Timing analysis | `timing_normal()`, `stack_aligned()` |
| [`visualize`](visualize.md) | Plotting utilities | `summary_single()`, `overlay_multi()` |
| [`quality`](quality.md) | Data quality assessment | Quality flag handling |
| [`spectra`](spectra.md) | Spectrograms | FPI ion/electron spectrograms |
| [`cli`](cli.md) | Command-line interface | Batch processing |

## Quick Reference

### Common Workflows

```python
# Load data
data = mp.data_loader.load_event(trange, probes=['1', '2'])

# Resample
t, vars, good = mp.resample.merge_vars(var_dict, cadence='150ms')

# Transform coordinates
lmn = mp.coords.hybrid_lmn(B_data)

# Detect boundaries
layers = mp.boundary.detect_crossings_multi(t, density, B_normal)

# Multi-spacecraft timing
n_hat, V_ph, sigma = mp.multispacecraft.timing_normal(positions, times)
```

### Data Structures

- **Time arrays:** `numpy.datetime64[ns]` or seconds since epoch
- **Spatial data:** GSM coordinates in km
- **Magnetic field:** nT in GSM or LMN coordinates
- **Velocities:** km/s in various coordinate systems
- **Densities:** particles/cm³

### Error Handling

All functions raise appropriate exceptions:
- `ValueError` for invalid parameters
- `RuntimeError` for processing failures
- `FileNotFoundError` for missing data files