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
| [`motion`](motion.md) | Displacement integration (adaptive) | `integrate_disp()`, `DispResult` |
| [`multispacecraft`](multispacecraft.md) | Timing analysis + diagnostics | `timing_normal()`, `stack_aligned()` |
| [`visualize`](visualize.md) | Plotting utilities | `summary_single()`, `overlay_multi()` |
| [`quality`](quality.md) | Data quality assessment | Quality flag handling |
| [`spectra`](spectra.md) | Energy spectrograms | `fpi_ion_spectrogram()`, `fpi_electron_spectrogram()` |
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

# Detect boundaries (He⁺ density, total density, Bₙ)
layers = mp.boundary.detect_crossings_multi(t, he_density, B_normal, ni=ion_density)

# Multi-spacecraft timing with diagnostics
n_hat, V_ph, sigma, diag = mp.multispacecraft.timing_normal(positions, times, return_diagnostics=True)
print(diag['condition_number'])
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