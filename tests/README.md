# Tests

This directory contains the test suite for the MMS Magnetopause Analysis Toolkit.

## Running Tests

To run all tests:
```bash
pytest tests/
```

To run tests with coverage:
```bash
pytest tests/ --cov=mms_mp --cov-report=html
```

To run a specific test file:
```bash
pytest tests/test_package.py
```

## Test Structure

- `test_package.py` - Basic package and import tests
- `conftest.py` - Shared fixtures and test configuration
- Individual module tests (to be added):
  - `test_data_loader.py`
  - `test_coords.py`
  - `test_boundary.py`
  - `test_motion.py`
  - `test_multispacecraft.py`
  - etc.

## Test Data

Test fixtures provide synthetic data that mimics MMS observations:
- Magnetic field vectors
- Plasma densities with boundary signatures
- Spacecraft positions
- Time arrays

## Contributing Tests

When adding new functionality:
1. Add corresponding tests in the appropriate test file
2. Use the existing fixtures when possible
3. Follow the naming convention `test_*`
4. Include both positive and negative test cases
5. Test edge cases and error conditions
