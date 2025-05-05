import numpy as np
from typing import List, Tuple

def layer_thicknesses(times: np.ndarray,
                      disp: np.ndarray,
                      crossings: List[Tuple[float, str]]) -> List[Tuple[str, float]]:
    """
    crossings must contain alternating ('enter','exit',...) times.
    Returns list of (layer_name, thickness_km).
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
