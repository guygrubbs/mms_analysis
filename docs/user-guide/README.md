# User Guide

Comprehensive guide to using the MMS Magnetopause Analysis Toolkit for scientific analysis.

## Contents

1. **[Basic Workflows](basic-workflows.md)** - Common analysis patterns
2. **[Advanced Analysis](advanced-analysis.md)** - Multi-spacecraft timing, uncertainty analysis
3. **[Data Quality](data-quality.md)** - Understanding and handling data quality issues
4. **[Coordinate Systems](coordinate-systems.md)** - LMN transforms and boundary normals
5. **[Boundary Detection](boundary-detection.md)** - Configuring the multi-parameter detector
6. **[Visualization](visualization.md)** - Creating publication-quality plots
7. **[Batch Processing](batch-processing.md)** - Analyzing multiple events
8. **[Performance Tips](performance-tips.md)** - Optimizing for large datasets

## Scientific Background

### Magnetopause Physics

The magnetopause is the boundary between Earth's magnetosphere and the solar wind. Key physical processes include:

- **Magnetic reconnection** - Conversion of magnetic energy to kinetic energy
- **Boundary motion** - Dynamic response to solar wind pressure changes
- **Layer structure** - Multiple boundary regions with distinct properties

### MMS Mission

The Magnetospheric Multiscale (MMS) mission consists of four identical spacecraft in tetrahedral formation, providing:

- **High-resolution measurements** - Up to 8192 samples/second in burst mode
- **Multi-point observations** - Spatial gradients and timing analysis
- **Comprehensive instrumentation** - Particles, fields, and waves

### Analysis Approach

This toolkit implements a multi-step analysis:

1. **Data loading** - Automatic CDF download and quality filtering
2. **Coordinate transformation** - Hybrid LMN using MVA + model fallback
3. **Boundary detection** - Multi-parameter logic with hysteresis
4. **Motion analysis** - Velocity integration and displacement calculation
5. **Timing analysis** - Multi-spacecraft boundary normal and phase speed

## Key Concepts

### Time Ranges
Always specify times in ISO format:
```python
trange = ['2019-11-12T04:00:00', '2019-11-12T05:00:00']
```

### Coordinate Systems
- **GSM** - Geocentric Solar Magnetospheric
- **GSE** - Geocentric Solar Ecliptic  
- **LMN** - Boundary-normal coordinates (L=along boundary, M=perpendicular, N=normal)

### Data Cadence
- **Survey (srvy)** - 4-8 second resolution, always available
- **Fast (fast)** - 30-150 ms resolution, routine operations
- **Burst (brst)** - 8-150 ms resolution, triggered events

### Quality Flags
All MMS data includes quality indicators:
- **0** - Good data
- **1** - Caution (use with care)
- **2** - Bad data (exclude from analysis)
- **3** - Fill value (no measurement)