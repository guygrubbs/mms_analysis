# MMS Ion and Electron Spectrograms

## Overview

The MMS-MP package includes comprehensive support for generating ion and electron energy spectrograms from real MMS FPI (Fast Plasma Investigation) data. This capability is essential for analyzing plasma regime transitions across magnetopause boundaries.

## Quick Start

### Generate Spectrograms for 2019-01-27 Event

```bash
python create_mms_spectrograms_2019_01_27.py
```

This script generates three spectrogram files:
- `mms_ion_spectrograms_2019_01_27.png` - Ion energy flux for all 4 spacecraft
- `mms_electron_spectrograms_2019_01_27.png` - Electron energy flux for all 4 spacecraft  
- `mms_combined_spectrograms_2019_01_27.png` - Detailed ion/electron comparison

## API Usage

### Basic Spectrogram Creation

```python
import mms_mp.spectra as spectra
import numpy as np

# Example data
times = np.array([...])  # datetime array
energies = np.logspace(1, 4, 32)  # 10 eV to 10 keV
flux_data = np.random.rand(len(times), len(energies))

# Create ion spectrogram
spectra.fpi_ion_spectrogram(
    times, energies, flux_data,
    title='Ion Energy Flux',
    cmap='plasma'
)

# Create electron spectrogram  
spectra.fpi_electron_spectrogram(
    times, energies, flux_data,
    title='Electron Energy Flux',
    cmap='viridis'
)
```

### Loading FPI Energy Distribution Data

```python
from pyspedas.projects import mms
from pyspedas import get_data

# Load ion distributions (DIS)
mms.mms_load_fpi(
    trange=['2019-01-27/12:25:00', '2019-01-27/12:35:00'],
    probe='1',
    data_rate='brst',  # or 'fast'
    level='l2',
    datatype='dis-dist'
)

# Load electron distributions (DES)
mms.mms_load_fpi(
    trange=['2019-01-27/12:25:00', '2019-01-27/12:35:00'],
    probe='1', 
    data_rate='brst',
    level='l2',
    datatype='des-dist'
)

# Extract data
times, ion_flux = get_data('mms1_dis_energyspectr_omni_brst')
times, electron_flux = get_data('mms1_des_energyspectr_omni_brst')
```

## Data Products

### FPI Energy Distributions

The MMS FPI instrument provides 3D particle distribution functions that can be processed into energy spectrograms:

- **DIS (Dual Ion Spectrometers)**: Ion energy distributions
- **DES (Dual Electron Spectrometers)**: Electron energy distributions
- **Energy Range**: 10 eV to 30 keV (32 energy channels)
- **Time Resolution**: 
  - Burst mode: ~150 ms
  - Fast mode: ~4.5 s

### Spectrogram Format

All spectrograms use the standard format:
- **X-axis**: Time (UT)
- **Y-axis**: Energy (eV, logarithmic scale)
- **Colorbar**: log₁₀ Flux [cm⁻²s⁻¹sr⁻¹eV⁻¹]

## Physics Interpretation

### Ion Spectrograms

Ion energy spectrograms reveal:
- **Thermal populations**: Low energy (10-100 eV) bulk plasma
- **Beam populations**: Mid energy (100-1000 eV) accelerated ions
- **Energetic populations**: High energy (1-30 keV) ring current ions
- **Temperature variations**: Across magnetopause boundary

### Electron Spectrograms

Electron energy spectrograms show:
- **Thermal electrons**: Low energy (10-100 eV) core population
- **Suprathermal electrons**: Mid energy (100-1000 eV) halo population
- **Energetic electrons**: High energy (1-30 keV) radiation belt electrons
- **Acceleration signatures**: Energy-dependent flux enhancements

### Magnetopause Signatures

Key features to look for:
- **Density transitions**: Flux level changes across boundary
- **Temperature changes**: Spectral shape evolution
- **Acceleration regions**: Energy-dependent enhancements
- **Multi-spacecraft timing**: Formation-scale dynamics

## File Naming Convention

Generated spectrogram files follow the pattern:
```
mms_{species}_spectrograms_{YYYY_MM_DD}.png
```

Where:
- `{species}`: `ion`, `electron`, or `combined`
- `{YYYY_MM_DD}`: Event date (e.g., `2019_01_27`)

## Examples

### 2019-01-27 Magnetopause Crossing

The included example script demonstrates spectrogram generation for a well-documented magnetopause crossing event:

- **Event Time**: 2019-01-27 12:30:50 UT
- **Analysis Period**: 12:25:00 - 12:35:00 UT (10 minutes)
- **Data Mode**: Burst mode (high time resolution)
- **Spacecraft**: All 4 MMS spacecraft in tetrahedral formation

This event shows clear plasma regime transitions with:
- Ion heating across the magnetopause
- Electron acceleration in the boundary layer
- Multi-spacecraft timing of the crossing

## Troubleshooting

### Common Issues

1. **Missing FPI data**: Script automatically falls back to synthetic data
2. **Download timeouts**: Large burst mode files may take time to download
3. **Memory usage**: Burst mode data can be memory intensive

### Data Availability

FPI energy distribution data is available for:
- **Time Period**: 2015-present (MMS mission duration)
- **Data Modes**: Fast (always available) and Burst (event-triggered)
- **Coverage**: Near-Earth magnetosphere and solar wind

## References

- MMS FPI Instrument: [Pollock et al., 2016](https://doi.org/10.1007/s11214-016-0245-4)
- Energy Spectrogram Analysis: [Burch et al., 2016](https://doi.org/10.1126/science.aaf2939)
- Magnetopause Physics: [Fuselier et al., 2016](https://doi.org/10.1002/2016JA022774)
