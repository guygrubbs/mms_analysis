# mms_mp/__init__.py
"""
MMS Magnetopause Analysis Toolkit

A Python-only, research-grade toolkit that reproduces—and extends—the IDL workflow 
used to study magnetopause reconnection events with NASA's Magnetospheric Multiscale 
(MMS) mission data.

Features:
- Automatic CDF download (FGM / FPI / HPCA / EDP / ephemeris)
- Hybrid LMN (MVA + Shue model fallback)
- Cold-ion + B_N boundary detector (hysteresis)
- E×B drift + He⁺ bulk blending
- Displacement integration ±σ
- 2-to-4 SC timing → n̂, V_phase ±σ
- One-command CLI + CSV / JSON / PNG output
"""

__version__ = "1.0.0"
__author__ = "MMS-MP Development Team"
__email__ = "contact@example.com"
__license__ = "MIT"

# Core modules
from . import data_loader
from . import coords
from . import resample
from . import electric
from . import quality
from . import boundary
from . import motion
from . import multispacecraft
from . import visualize
from . import spectra
from . import thickness
from . import formation_detection
from . import ephemeris
from . import cli

# Make key functions easily accessible
from .data_loader import load_event
from .coords import hybrid_lmn
from .boundary import detect_crossings_multi, DetectorCfg
from .motion import integrate_disp
from .multispacecraft import timing_normal
from .resample import merge_vars
from .electric import exb_velocity, normal_velocity
from .formation_detection import detect_formation_type, FormationType, analyze_formation_from_event_data
from .ephemeris import get_mec_ephemeris_manager, validate_mec_data_usage

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    
    # Modules
    "data_loader",
    "coords",
    "resample",
    "electric",
    "quality",
    "boundary",
    "motion",
    "multispacecraft",
    "visualize",
    "spectra",
    "thickness",
    "formation_detection",
    "ephemeris",
    "cli",
    
    # Key functions
    "load_event",
    "hybrid_lmn",
    "detect_crossings_multi",
    "DetectorCfg",
    "integrate_disp",
    "timing_normal",
    "merge_vars",
    "exb_velocity",
    "normal_velocity",
    "detect_formation_type",
    "FormationType",
    "analyze_formation_from_event_data",
    "get_mec_ephemeris_manager",
    "validate_mec_data_usage",
]
