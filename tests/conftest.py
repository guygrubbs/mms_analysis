"""
Pytest configuration and fixtures for MMS-MP tests
"""
import os
# Force non-interactive matplotlib backend early to avoid GUI hangs
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Ensure any accidental plt.show() during tests is a no-op
plt.show = lambda *args, **kwargs: None

import pytest
import numpy as np
import sys
import faulthandler
from datetime import datetime

# Auto-close figures after each test to prevent resource buildup
@pytest.fixture(autouse=True)
def _close_figures():
    yield
    try:
        import matplotlib.pyplot as _plt
        _plt.close('all')
    except Exception:
        pass

# Dump stack traces if a test runs too long; helps diagnose rare stalls on Windows
@pytest.fixture(autouse=True)
def _faulthandler_timeout():
    faulthandler.enable()
    # Schedule a dump if the interpreter appears stuck >120s
    faulthandler.dump_traceback_later(120, repeat=False, file=sys.stderr)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def sample_time_array():
    """Sample time array for testing"""
    return np.linspace(0, 3600, 1000)  # 1 hour, 1000 points


@pytest.fixture
def sample_magnetic_field():
    """Sample magnetic field data for testing"""
    n_points = 1000
    # Simple synthetic B-field with some variation
    Bx = 10 + 2 * np.sin(np.linspace(0, 4*np.pi, n_points))
    By = 5 + np.cos(np.linspace(0, 6*np.pi, n_points))
    Bz = -2 + 0.5 * np.sin(np.linspace(0, 8*np.pi, n_points))
    return np.column_stack([Bx, By, Bz])


@pytest.fixture
def sample_position():
    """Sample spacecraft position for testing"""
    # Simple position in GSM coordinates (km)
    return np.array([10000.0, 5000.0, -2000.0])


@pytest.fixture
def sample_density():
    """Sample density data for testing"""
    n_points = 1000
    # Synthetic density with boundary crossing signature
    base_density = 0.1 + 0.05 * np.random.random(n_points)
    # Add a step function to simulate boundary crossing
    step_start = n_points // 3
    step_end = 2 * n_points // 3
    base_density[step_start:step_end] = 0.3 + 0.1 * np.random.random(step_end - step_start)
    return base_density
