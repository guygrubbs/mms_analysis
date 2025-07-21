# Changelog

All notable changes to the MMS Magnetopause Analysis Toolkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-07-21

### Added
- **Package Structure**: Proper Python package with `__init__.py`
- **Version Management**: Package version information (`__version__`)
- **Package Configuration**: `pyproject.toml` for modern Python packaging
- **License**: MIT License file
- **CLI Support**: Command-line interface accessible via `python -m mms_mp`
- **Installation Support**: Pip-installable package with proper dependencies
- **Development Tools**: 
  - Pre-commit configuration
  - Makefile for common tasks
  - .gitignore for Python projects
- **Documentation**:
  - Comprehensive API documentation
  - Installation guide updates
  - Package-level documentation
- **Examples**:
  - Jupyter notebook examples
  - Standalone Python script examples
  - Example data and workflows
- **Testing**:
  - Basic package tests
  - Test fixtures and configuration
  - Test directory structure

### Changed
- **Requirements**: Updated to consistent version specifications
- **README**: Updated to reflect actual codebase structure
- **Module Documentation**: Added missing `thickness.py` module to documentation
- **Import Structure**: Fixed relative imports in quality module
- **CLI Access**: Added `__main__.py` for proper module execution

### Fixed
- **Package Imports**: Resolved import issues for proper package functionality
- **CLI Execution**: Fixed `python -m mms_mp.cli` execution
- **Documentation Consistency**: Aligned README with actual codebase
- **Version Requirements**: Fixed PySPEDAS version inconsistencies

### Technical Details

#### Package Structure
```
mms_mp/
├── __init__.py         # Package initialization (NEW)
├── __main__.py         # CLI entry point (NEW)
├── data_loader.py      # CDF download + variable extraction
├── coords.py           # LMN transforms (MVA / model / hybrid)
├── resample.py         # Multi-var resampling/merging helpers
├── electric.py         # E×B drift, v_N selection
├── quality.py          # Instrument quality-flag masks (FIXED imports)
├── boundary.py         # Multi-parameter boundary detector
├── motion.py           # Integrate v_N → displacement ±σ
├── multispacecraft.py  # Timing method (n̂, V_ph) + alignment
├── visualize.py        # Publication-ready plotting helpers
├── spectra.py          # FPI ion/electron spectrograms
├── thickness.py        # Layer thickness calculation utilities
└── cli.py              # Command-line pipeline
```

#### New Files
- `LICENSE` - MIT License
- `pyproject.toml` - Modern Python package configuration
- `CHANGELOG.md` - This changelog
- `.pre-commit-config.yaml` - Code quality tools
- `Makefile` - Development automation
- `examples/` - Example notebooks and scripts
- `tests/` - Test suite structure
- `docs/api/package.md` - Package API documentation

#### Dependencies
- **Core**: pyspedas>=1.7.20, numpy>=1.20.0, scipy>=1.7.0, pandas>=1.3.0, matplotlib>=3.3.0
- **Optional**: tqdm (progress bars), jupyter (notebooks), pytest (testing)

#### Installation
```bash
# Install from source
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with notebook support
pip install -e ".[notebooks]"
```

#### CLI Usage
```bash
# New ways to access CLI
python -m mms_mp --help
python -m mms_mp.cli --help

# Package installation also provides mms-mp command (if on PATH)
mms-mp --help
```

### Migration Notes

For users upgrading from previous versions:

1. **Import Changes**: The package can now be imported as `import mms_mp`
2. **CLI Access**: Use `python -m mms_mp` instead of direct script execution
3. **Installation**: Use `pip install -e .` for development installation
4. **Dependencies**: Check that all dependencies meet minimum version requirements

### Known Issues

- The `mms-mp` command may not be on PATH depending on Python installation
- Use `python -m mms_mp` as the recommended CLI access method

### Contributors

- MMS-MP Development Team

---

## [Unreleased]

### Planned
- Type hints for better IDE support
- Comprehensive docstrings
- Additional example notebooks
- Performance optimizations
- Extended test coverage
