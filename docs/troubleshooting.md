# Troubleshooting Guide

Common issues and solutions for the MMS Magnetopause Analysis Toolkit.

## Installation Issues

### PySPEDAS Version Conflicts
**Problem:** `TypeError: mms_load_state() got an unexpected keyword 'notplot'`

**Solution:**
```bash
pip install --upgrade pyspedas>=1.7.20
```

### SSL Certificate Errors
**Problem:** Certificate verification failed during pip install

**Solution:**
```bash
pip install --trusted-host pypi.org --trusted-host pypi.python.org mms-magnetopause
```

### Memory Errors During Installation
**Problem:** Installation fails with memory errors

**Solution:**
```bash
pip install --no-cache-dir mms-magnetopause
```

## Data Loading Issues

### Missing Variables
**Problem:** `Missing var mms4_des_numberdensity_fast`

**Cause:** MMS4 DES data not available at requested cadence/time

**Solution:** Toolkit automatically falls back to survey mode, then fills with NaN

### CDAWeb Connection Timeout
**Problem:** Downloads fail with timeout errors

**Solution:**
```python
import pyspedas
pyspedas.set_downloads_timeout(300)  # 5 minutes
```

### Large Cache Directory
**Problem:** `~/.pyspedas/` directory consuming too much disk space

**Solution:**
```bash
# Clean old files (older than 30 days)
find ~/.pyspedas -name "*.cdf" -mtime +30 -delete

# Or remove entire cache
rm -rf ~/.pyspedas
```

## Analysis Issues

### No Boundary Crossings Detected
**Problem:** `detect_crossings_multi()` returns empty list

**Possible Causes:**
1. Thresholds too strict for event
2. Poor data quality
3. Wrong time interval

**Solutions:**
```python
# Adjust detection thresholds
cfg = mp.boundary.DetectorCfg(
    he_in=0.1,     # Increase from default 0.05
    he_out=0.02,   # Increase from default 0.01
    BN_tol=1.0     # Decrease from default 2.0
)

# Check data quality
good_fraction = np.mean(good_mask)
print(f"Good data fraction: {good_fraction:.2f}")

# Visualize raw data
mp.visualize.summary_single(t, B_lmn, Ni, Ne, He, vN_i, vN_e, vN_he)
```

### Timing Analysis Fails
**Problem:** `timing_normal()` returns NaN values

**Cause:** Insufficient spacecraft separation or poor crossing time estimates

**Solution:**
```python
# Check spacecraft separation
separations = []
for i, sc1 in enumerate(positions.keys()):
    for sc2 in list(positions.keys())[i+1:]:
        sep = np.linalg.norm(positions[sc1] - positions[sc2])
        separations.append(sep)
        
min_sep = min(separations)
print(f"Minimum spacecraft separation: {min_sep:.1f} km")

# Require minimum separation
if min_sep < 100:  # km
    print("Warning: Spacecraft too close for reliable timing")
```

### Memory Errors During Processing
**Problem:** Out of memory errors with large datasets

**Solutions:**
```python
# Reduce time resolution
t, vars, good = mp.resample.merge_vars(data, cadence='1s')  # Instead of 150ms

# Process shorter time intervals
for start, end in time_chunks:
    result = analyze_interval([start, end])

# Use survey mode instead of burst
data = mp.data_loader.load_event(trange, cadence='srvy')
```

## Visualization Issues

### Plots Not Displaying
**Problem:** `plt.show()` doesn't display plots

**Solutions:**
```python
# For Jupyter notebooks
%matplotlib inline

# For scripts, use non-interactive backend
import matplotlib
matplotlib.use('Agg')  # For saving files only

# Or force interactive backend
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

### Poor Plot Quality
**Problem:** Plots look pixelated or have poor resolution

**Solution:**
```python
# Increase DPI for better quality
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# Use vector format for publications
plt.savefig('figure.pdf', format='pdf')
```

## Performance Issues

### Slow Data Loading
**Problem:** `load_event()` takes very long time

**Solutions:**
```python
# Load only needed probes
data = mp.data_loader.load_event(trange, probes=['1', '2'])

# Skip optional instruments
data = mp.data_loader.load_event(trange, include_hpca=False, include_edp=False)

# Use survey mode for quick analysis
data = mp.data_loader.load_event(trange, cadence='srvy')
```

### Slow Resampling
**Problem:** `merge_vars()` is very slow

**Solution:**
```python
# Use faster interpolation method
t, vars, good = mp.resample.merge_vars(data, method='nearest')

# Reduce output resolution
t, vars, good = mp.resample.merge_vars(data, cadence='1s')
```

## Getting Help

### Check Logs
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Report Issues
When reporting bugs, include:
1. Python version and operating system
2. Complete error traceback
3. Minimal code example
4. Data time range and spacecraft

### Community Support
- GitHub Issues: Bug reports and feature requests
- Discussions: Usage questions and tips
- Email: Direct contact for sensitive issues

## Diagnostic Commands

```python
# Check installation
import mms_mp
print(f"Version: {mms_mp.__version__}")

# Check dependencies
import pyspedas, numpy, scipy, pandas, matplotlib
print("All dependencies imported successfully")

# Test data loading
try:
    data = mp.data_loader.load_event(['2019-11-12T04:00', '2019-11-12T04:10'], 
                                     probes=['1'])
    print("Data loading test: PASSED")
except Exception as e:
    print(f"Data loading test: FAILED - {e}")

# Check cache directory
import os
cache_dir = os.path.expanduser('~/.pyspedas')
if os.path.exists(cache_dir):
    size_gb = sum(os.path.getsize(os.path.join(dirpath, filename))
                  for dirpath, dirnames, filenames in os.walk(cache_dir)
                  for filename in filenames) / 1e9
    print(f"Cache directory size: {size_gb:.1f} GB")
```