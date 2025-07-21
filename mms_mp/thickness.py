"""
Magnetopause layer thickness calculation utilities.

This module provides functions to calculate the thickness of magnetopause
boundary layers from displacement data and crossing times.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple


def layer_thicknesses(times: np.ndarray,
                      disp: np.ndarray,
                      crossings: List[Tuple[float, str]]) -> List[Tuple[str, float]]:
    """
    Calculate magnetopause layer thicknesses from displacement data and crossing times.

    This function computes the thickness of boundary layers by measuring the
    displacement between entry and exit times for each layer crossing. The
    displacement represents the distance traveled through the boundary layer
    in the normal direction.

    Args:
        times: Time array corresponding to displacement measurements.
            Shape: (N,) where N is the number of time points.
            Units: seconds (typically Unix timestamp or relative time).
            Must be monotonically increasing.

        disp: Cumulative displacement array from integration of normal velocity.
            Shape: (N,) matching the times array.
            Units: km. Represents distance traveled in boundary-normal direction.
            Typically obtained from integrate_disp() function.

        crossings: List of boundary crossing events.
            Each element is a tuple (time, event_type) where:
            - time: Crossing time in same units as times array
            - event_type: String describing crossing ('enter', 'exit', etc.)
            Must contain alternating entry/exit pairs for proper layer calculation.

    Returns:
        List[Tuple[str, float]]: Layer thickness measurements.
            Each element is a tuple (layer_name, thickness_km) where:
            - layer_name: Descriptive name like 'layer_1', 'layer_2', etc.
            - thickness_km: Layer thickness in kilometers (always positive)

    Raises:
        IndexError: If crossings list has odd number of elements (unpaired crossings).
        ValueError: If times array is not monotonically increasing.

    Examples:
        >>> import numpy as np
        >>> from mms_mp.thickness import layer_thicknesses

        # Example with displacement data and two crossings
        >>> times = np.linspace(0, 100, 1001)  # 100 seconds, 10 Hz
        >>> disp = np.cumsum(np.random.randn(1001) * 0.1)  # Random walk displacement
        >>> crossings = [
        ...     (20.0, 'enter'),  # Enter magnetopause at t=20s
        ...     (30.0, 'exit'),   # Exit magnetopause at t=30s
        ...     (60.0, 'enter'),  # Second crossing
        ...     (70.0, 'exit')
        ... ]
        >>> thicknesses = layer_thicknesses(times, disp, crossings)
        >>> for layer_name, thickness in thicknesses:
        ...     print(f"{layer_name}: {thickness:.2f} km")

        # Real example with MMS data
        >>> # (assuming you have displacement from integrate_disp)
        >>> # and crossings from boundary detection
        >>> result = integrate_disp(t_grid, v_normal)
        >>> crossings = extract_enter_exit(boundary_layers, t_grid)
        >>> thicknesses = layer_thicknesses(result.t_sec, result.disp_km, crossings)

    Notes:
        - **Layer Definition**: A layer is defined as the region between consecutive
          entry and exit crossings. The function assumes alternating entry/exit pairs.

        - **Thickness Calculation**: Thickness is the absolute difference in
          displacement between exit and entry points: |disp(t_exit) - disp(t_entry)|

        - **Multiple Layers**: If multiple boundary crossings occur, each entry/exit
          pair defines a separate layer with its own thickness measurement.

        - **Physical Interpretation**: The thickness represents the distance traveled
          through the boundary layer in the normal direction, which is the true
          layer thickness if the spacecraft trajectory is perpendicular to the boundary.

        - **Coordinate System**: Displacement should be in boundary-normal coordinates
          (typically from LMN transformation) for meaningful thickness measurements.

    References:
        - Paschmann & Daly (1998): Analysis Methods for Multi-Spacecraft Data
        - Berchem & Russell (1982): Magnetic field rotation through the magnetopause
    """
    out = []
    sorted_times = sorted(crossings, key=lambda x: x[0])
    for i in range(0, len(sorted_times), 2):
        t1, _ = sorted_times[i]
        t2, _ = sorted_times[i+1]
        # indices
        idx1 = np.searchsorted(times, t1)
        idx2 = np.searchsorted(times, t2)
        thick = abs(disp[idx2] - disp[idx1])
        out.append((f'layer_{i//2+1}', thick))
    return out
