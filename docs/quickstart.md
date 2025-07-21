# Quick Start Guide

Get up and running with the MMS Magnetopause Analysis Toolkit in under 5 minutes.

## 1. First Analysis (Command Line)

```bash
# Analyze a 1-hour magnetopause event
python -m mms_mp.cli \
    --start 2019-11-12T04:00 \
    --end 2019-11-12T05:00 \
    --probes 1 2 3 4 \
    --plot
```

**Output:**
- `results/mp_analysis.json` - All numerical results
- `results/layer_thickness.csv` - Boundary layer measurements
- `results/figures/` - Quick-look plots for each spacecraft

## 2. Interactive Demo

```bash
# Run the notebook-style demo
python main_analysis.py
```

This opens interactive plots and prints results to console.

## 3. Python API Example

```python
import mms_mp as mp

# Load data for one spacecraft
trange = ['2019-11-12T04:00', '2019-11-12T05:00']
data = mp.data_loader.load_event(trange, probes=['2'])

# Get MMS2 data
d = data['2']

# Resample to common time grid
t, vars_grid, good = mp.resample.merge_vars({
    'Ni': (d['N_tot'][0], d['N_tot'][1]),
    'B': (d['B_gsm'][0], d['B_gsm'][1])
}, cadence='150ms')

# Transform to boundary coordinates
lmn = mp.coords.hybrid_lmn(d['B_gsm'][1])
B_lmn = lmn.to_lmn(vars_grid['B'])

# Detect boundary crossings
layers = mp.boundary.detect_crossings_multi(
    t, vars_grid['Ni'], B_lmn[:, 2], good_mask=good['Ni']
)

print(f"Found {len(layers)} boundary layers")
```

## 4. Understanding the Output

### JSON Results Structure
```json
{
  "crossings_sec": {"1": 1573531200.0, "2": 1573531205.0},
  "positions_km": {"1": [1000, 2000, 3000], "2": [1100, 2100, 3100]},
  "boundary_normal": [0.8, 0.5, 0.3],
  "phase_speed_km_s": 45.2,
  "phase_speed_sigma": 5.1,
  "layer_thickness_km": {"1": {"magnetopause": 120.5}}
}
```

### CSV Output
| spacecraft | layer_type | thickness_km | crossing_time |
|------------|------------|--------------|---------------|
| 1 | magnetopause | 120.5 | 2019-11-12T04:20:00 |
| 2 | magnetopause | 115.2 | 2019-11-12T04:20:05 |

## 5. Key Concepts

| Term | Definition |
|------|------------|
| **Event** | Time interval analyzed across all spacecraft |
| **Probe** | MMS spacecraft number (1-4) |
| **Layer** | Detected boundary region (e.g., magnetopause, current sheet) |
| **LMN Coordinates** | Boundary-normal coordinate system |
| **Phase Speed** | Boundary motion velocity from timing analysis |

## 6. Next Steps

- **Detailed workflows:** See [User Guide](user-guide/)
- **API documentation:** Browse [API Reference](api/)
- **Real examples:** Check [Examples](examples/)
- **Customization:** Read [Developer Guide](developer-guide/)

## 7. Getting Help

- Check [Troubleshooting](troubleshooting.md) for common issues
- Open an issue on GitHub for bugs
- Join discussions for usage questions