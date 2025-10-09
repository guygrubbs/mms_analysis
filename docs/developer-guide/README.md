# Developer Guide

Guide for contributing to and extending the MMS Magnetopause Analysis Toolkit.

## Contents

1. **[Contributing](contributing.md)** - How to contribute code
2. **[Architecture](architecture.md)** - Code organization and design patterns
3. **[Testing](testing.md)** - Running and writing tests
4. **[Documentation](documentation.md)** - Documentation standards
5. **[Release Process](release-process.md)** - How releases are made
6. **[Extending](extending.md)** - Adding new functionality

## Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/mms-magnetopause.git
cd mms-magnetopause

# Create development environment
python -m venv dev_env
source dev_env/bin/activate

# Install in development mode with test dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

## Code Standards

### Style Guide
- **PEP 8** compliance with 88-character line length
- **Black** auto-formatter
- **isort** for import sorting
- **Type hints** for all public functions

### Documentation
- **Docstrings** in NumPy format
- **Type annotations** for parameters and returns
- **Examples** in docstrings for complex functions

### Testing
- **pytest** framework
- **Minimum 80% coverage** for new code
- **Integration tests** for end-to-end workflows
- **Mock external dependencies** (CDAWeb, file I/O)

## Architecture Overview

```
mms_mp/
├── data_loader.py      # External data interface
├── coords.py           # Mathematical transformations  
├── resample.py         # Signal processing
├── boundary.py         # Physics algorithms
├── motion.py           # Numerical integration
├── multispacecraft.py  # Multi-point analysis
├── visualize.py        # Presentation layer
└── cli.py             # User interface
```

### Design Principles

1. **Separation of concerns** - Each module has a single responsibility
2. **Functional programming** - Pure functions where possible
3. **Error handling** - Graceful degradation and informative messages
4. **Performance** - Vectorized operations, minimal copying
5. **Extensibility** - Plugin architecture for custom algorithms

## Adding New Features

### 1. New Boundary Detector
```python
# mms_mp/boundary_custom.py
def detect_custom_boundary(t, data, **kwargs):
    """Custom boundary detection algorithm."""
    # Implementation here
    return layers
```

### 2. New Coordinate System
```python
# mms_mp/coords.py
class CustomTransform(CoordinateTransform):
    """Custom coordinate transformation."""
    
    def to_custom(self, vectors):
        """Transform to custom coordinates."""
        return self.rotation_matrix @ vectors.T
```

### 3. New Visualization
```python
# mms_mp/visualize.py
def plot_custom(data, **kwargs):
    """Custom plotting function."""
    fig, ax = plt.subplots()
    # Plotting code here
    return fig, ax
```

## Testing Guidelines

### Unit Tests
```python
# tests/test_boundary.py
def test_detect_crossings_basic():
    """Test basic boundary detection."""
    t = np.arange(100, dtype=float)
    he = np.ones(100) * 0.3
    he[:60] = 0.05  # Magnetosheath
    BN = np.concatenate([np.full(60, -8.0), np.linspace(-2.0, 2.0, 20), np.full(20, 4.0)])
    ni = np.concatenate([np.full(60, 10.0), np.linspace(8.0, 5.0, 20), np.full(20, 4.0)])

    layers = detect_crossings_multi(t, he, BN, ni=ni)
    assert len(layers) == 3
    assert layers[0][0] == 'sheath'
```

### Integration Tests
```python
# tests/test_integration.py
def test_full_workflow():
    """Test complete analysis workflow."""
    # Mock data loading
    with patch('mms_mp.data_loader.load_event') as mock_load:
        mock_load.return_value = create_test_data()
        
        # Run analysis
        result = run_analysis(['2019-11-12T04:00', '2019-11-12T05:00'])
        
        # Verify results
        assert 'boundary_normal' in result
        assert len(result['crossings_sec']) > 0
```

## Performance Considerations

### Memory Management
- Use **views** instead of copies where possible
- **Chunked processing** for large datasets
- **Lazy loading** of optional data

### Computational Efficiency
- **Vectorized operations** with NumPy
- **Caching** of expensive computations
- **Parallel processing** for independent spacecraft

### I/O Optimization
- **Batch downloads** from CDAWeb
- **Compressed storage** for intermediate results
- **Streaming** for real-time applications