# Examples

This directory contains example notebooks and scripts demonstrating how to use the MMS Magnetopause Analysis Toolkit.

## Notebooks

### 1. Basic Usage (`01_basic_usage.ipynb`)
- Loading MMS data
- Basic boundary detection
- Simple visualization

### 2. Advanced Analysis (`02_advanced_analysis.ipynb`)
- Multi-spacecraft timing analysis
- Layer thickness calculation
- Publication-quality plots

### 3. Custom Workflows (`03_custom_workflows.ipynb`)
- Customizing detection parameters
- Working with different time ranges
- Batch processing multiple events

## Scripts

### `example_script.py`
A standalone Python script showing the complete workflow without Jupyter.

## Running the Examples

### Prerequisites
```bash
# Install the package with notebook dependencies
pip install -e ".[notebooks]"

# Or install jupyter separately
pip install jupyter ipykernel
```

### Launch Jupyter
```bash
jupyter notebook examples/
```

### Run Scripts
```bash
python examples/example_script.py
```

## Data Requirements

The examples use publicly available MMS data that will be automatically downloaded from CDAWeb on first run. Make sure you have:

1. Internet connection for data download
2. Sufficient disk space (~100MB for cache)
3. Python packages: numpy, scipy, pandas, matplotlib, pyspedas

## Example Events

The notebooks demonstrate analysis of well-known magnetopause crossing events:

- **2019-01-27 12:20-12:40 UT**: Clear boundary crossing with all 4 spacecraft
- **2019-11-12 04:00-05:00 UT**: Multiple crossings with good timing geometry
- **2017-07-11 22:30-23:30 UT**: Classic reconnection event

## Troubleshooting

If you encounter issues:

1. **Data download fails**: Check internet connection and try again
2. **Import errors**: Ensure all dependencies are installed
3. **Plotting issues**: Make sure matplotlib backend is properly configured
4. **Memory issues**: Try reducing the time range or using lower cadence data

For more help, see the main documentation in `docs/`.
