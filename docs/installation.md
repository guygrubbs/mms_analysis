# Installation Guide

## System Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.9 - 3.12 | Python 3.13 not yet supported by PySPEDAS |
| Operating System | Windows, macOS, Linux | Tested on all platforms |
| Memory | 8+ GB RAM | For processing burst-mode data |
| Storage | 10+ GB free | CDF cache can grow large |

## Installation Methods

### Method 1: pip install (Recommended)

```bash
# Create virtual environment
python -m venv mms_mp_env
source mms_mp_env/bin/activate  # Windows: mms_mp_env\Scripts\activate

# Install from PyPI (when available)
pip install mms-magnetopause

# Or install from source
pip install git+https://github.com/your-org/mms-magnetopause.git
```

### Method 2: Development Install

```bash
# Clone repository
git clone https://github.com/your-org/mms-magnetopause.git
cd mms-magnetopause

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

### Method 3: Conda Environment

```bash
# Create conda environment
conda create -n mms_mp python=3.11
conda activate mms_mp

# Install dependencies
conda install numpy scipy pandas matplotlib
pip install pyspedas>=1.7.20

# Install toolkit
pip install -e .
```

## Dependency Details

### Core Dependencies
- **PySPEDAS ≥ 1.7.20** - MMS data loading and CDF handling
- **NumPy** - Numerical computations
- **SciPy** - Scientific computing (integration, interpolation)
- **pandas** - Data manipulation and CSV output
- **Matplotlib** - Plotting and visualization

### Optional Dependencies
- **tqdm** - Progress bars (recommended for long downloads)
- **jupyter** - For notebook examples
- **pytest** - For running tests

## Verification

Test your installation:

```python
import mms_mp
print(f"MMS-MP version: {mms_mp.__version__}")

# Quick test
from mms_mp import data_loader
print("Installation successful!")
```

### Command Line Interface

Test the CLI:

```bash
# Show help
python -m mms_mp --help

# Alternative CLI access
python -m mms_mp.cli --help
```

### Package Installation

For development or to install from source:

```bash
# Clone repository
git clone https://github.com/your-org/mms-magnetopause.git
cd mms-magnetopause

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Configuration

### PySPEDAS Cache Directory
By default, CDFs are cached in `~/.pyspedas/`. To change:

```python
import os
os.environ['SPEDAS_DATA_DIR'] = '/path/to/your/cache'
```

### Memory Settings
For large datasets, increase memory limits:

```python
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
```

## Troubleshooting Installation

### Common Issues

1. **PySPEDAS version conflicts**
   ```bash
   pip install --upgrade pyspedas>=1.7.20
   ```

2. **SSL certificate errors**
   ```bash
   pip install --trusted-host pypi.org --trusted-host pypi.python.org mms-magnetopause
   ```

3. **Permission errors on Windows**
   - Run terminal as Administrator
   - Or use `--user` flag: `pip install --user mms-magnetopause`

4. **Memory errors during installation**
   ```bash
   pip install --no-cache-dir mms-magnetopause
   ```

See [Troubleshooting](troubleshooting.md) for more solutions.