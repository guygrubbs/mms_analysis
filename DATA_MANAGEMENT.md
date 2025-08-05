# MMS Data Management

## Local Data Caching Strategy

This project uses efficient local data caching to minimize downloads and maximize analysis speed.

### Current Status
- **Data cache**: `pydata/` directory (45+ GB)
- **Files cached**: 1,000+ data files
- **Coverage**: 2019-01-27 event with burst mode data
- **Git status**: Excluded from version control

### How It Works
1. **First download**: PySpedas downloads data from NASA MMS SDC
2. **Local storage**: Files saved in `pydata/` directory structure
3. **Subsequent use**: PySpedas checks local cache before downloading
4. **No re-downloads**: Same data files never downloaded twice

### Data Structure
```
pydata/
├── mms1/
│   ├── fgm/brst/l2/2019/01/27/    # Burst mode magnetic field
│   ├── fpi/brst/l2/dis-moms/      # Ion moments (burst)
│   ├── fpi/brst/l2/des-moms/      # Electron moments (burst)
│   ├── fpi/brst/l2/dis-dist/      # Ion distributions (burst)
│   ├── fpi/brst/l2/des-dist/      # Electron distributions (burst)
│   └── mec/srvy/l2/epht89q/       # Ephemeris data
├── mms2/ (same structure)
├── mms3/ (same structure)
└── mms4/ (same structure)
```

### Benefits
- ✅ **Fast analysis**: No waiting for downloads
- ✅ **Offline capable**: Work without internet connection
- ✅ **Bandwidth efficient**: Download once, use many times
- ✅ **Reproducible**: Same data files for consistent results

### Best Practices
1. **Keep pydata/ local**: Never commit to Git (too large)
2. **Share code only**: Commit analysis scripts and results
3. **Document data sources**: Include event times in code comments
4. **Clean periodically**: Remove data for old events you're not using

### Cache Management
```bash
# Check cache size
du -sh pydata/

# Remove specific event data
rm -rf pydata/*/*/2019/01/27/

# Clear all cache (will re-download on next use)
rm -rf pydata/
```

### Data Sources
All data from NASA MMS Science Data Center:
- **FGM**: Fluxgate Magnetometer (magnetic field)
- **FPI**: Fast Plasma Investigation (plasma moments and distributions)
- **MEC**: Magnetic Ephemeris and Coordinates (spacecraft positions)
- **EDP**: Electric Double Probes (electric field)

### Current Event: 2019-01-27 12:30:50 UT
- **Type**: Magnetopause crossing
- **Data modes**: Burst mode (highest resolution)
- **Coverage**: All 4 MMS spacecraft
- **Quality**: Publication-ready
