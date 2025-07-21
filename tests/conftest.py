"""
Pytest configuration and fixtures for MMS-MP tests
"""
import pytest
import numpy as np
from datetime import datetime


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
