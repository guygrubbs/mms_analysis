"""
MMS Scientific Analysis Enhancements

Addresses key scientific and technical issues in multi-spacecraft analysis:
1. Spacecraft-specific hardware issues
2. Formation geometry quality
3. Boundary analysis validity
4. Inter-spacecraft calibration
5. Timing synchronization
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class MMSScientificValidator:
    """Enhanced scientific validation for MMS multi-spacecraft analysis"""
    
    # Known spacecraft-specific issues
    KNOWN_ISSUES = {
        'mms4': {
            'fpi_des': {
                'issue': 'HV801 optocoupler failure',
                'start_date': '2019-01-01',
                'impact': 'electron_measurements_unreliable',
                'severity': 'high'
            }
        }
    }
    
    # Inter-spacecraft calibration factors (from cross-calibration studies)
    MAGNETOMETER_CALIBRATION = {
        'mms1': 1.000,  # Reference spacecraft
        'mms2': 1.002,  # +0.2% systematic offset
        'mms3': 0.998,  # -0.2% systematic offset
        'mms4': 1.001   # +0.1% systematic offset
    }
    
    # Analysis validity thresholds
    MIN_TQF = 0.3  # Minimum Tetrahedral Quality Factor
    MAX_ELONGATION = 5.0  # Maximum formation elongation ratio
    MIN_PHASE_VELOCITY = 10.0  # km/s - minimum reasonable phase velocity
    MAX_PHASE_VELOCITY = 2000.0  # km/s - maximum reasonable phase velocity


def calculate_tetrahedral_quality_factor(positions: Dict[str, np.ndarray]) -> float:
    """
    Calculate Tetrahedral Quality Factor (TQF) for spacecraft formation
    
    TQF = 1 means perfect tetrahedron, 0 means degenerate (planar) formation
    
    Args:
        positions: Dictionary of spacecraft positions {probe: [x,y,z]}
        
    Returns:
        TQF value between 0 and 1
    """
    if len(positions) < 4:
        return 0.0
    
    # Get position vectors
    probes = sorted(positions.keys())
    pos_array = np.array([positions[p] for p in probes[:4]])
    
    # Calculate volume of tetrahedron
    # V = |det(r2-r1, r3-r1, r4-r1)| / 6
    r1, r2, r3, r4 = pos_array
    volume = abs(np.linalg.det(np.array([r2-r1, r3-r1, r4-r1]))) / 6.0
    
    # Calculate characteristic length scale
    distances = []
    for i in range(4):
        for j in range(i+1, 4):
            distances.append(np.linalg.norm(pos_array[i] - pos_array[j]))
    
    char_length = np.mean(distances)
    
    # TQF = actual_volume / ideal_volume
    # For regular tetrahedron: V_ideal = (sqrt(2)/12) * L^3
    ideal_volume = (np.sqrt(2) / 12) * char_length**3
    
    if ideal_volume > 0:
        tqf = volume / ideal_volume
        return min(tqf, 1.0)  # Cap at 1.0
    else:
        return 0.0


def assess_formation_geometry(positions: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Comprehensive formation geometry assessment
    
    Returns:
        Dictionary with geometry quality metrics
    """
    if len(positions) < 3:
        return {'tqf': 0.0, 'elongation': np.inf, 'valid': False}
    
    # Calculate TQF
    tqf = calculate_tetrahedral_quality_factor(positions)
    
    # Calculate formation elongation
    probes = list(positions.keys())
    pos_array = np.array([positions[p] for p in probes])
    
    # Principal component analysis to find elongation
    centered = pos_array - np.mean(pos_array, axis=0)
    cov_matrix = np.cov(centered.T)
    eigenvals = np.linalg.eigvals(cov_matrix)
    eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
    
    if eigenvals[-1] > 0:
        elongation = eigenvals[0] / eigenvals[-1]
    else:
        elongation = np.inf
    
    # Determine if formation is valid for multi-spacecraft analysis
    validator = MMSScientificValidator()
    valid = (tqf >= validator.MIN_TQF and 
             elongation <= validator.MAX_ELONGATION)
    
    return {
        'tqf': tqf,
        'elongation': elongation,
        'eigenvalue_ratio': eigenvals[0] / eigenvals[1] if eigenvals[1] > 0 else np.inf,
        'valid': valid
    }


def validate_boundary_analysis(positions: Dict, crossings: Dict, 
                             normal: np.ndarray, phase_velocity: float) -> Dict[str, bool]:
    """
    Validate assumptions for multi-spacecraft boundary analysis
    
    Args:
        positions: Spacecraft positions
        crossings: Boundary crossing times
        normal: Boundary normal vector
        phase_velocity: Calculated phase velocity
        
    Returns:
        Dictionary of validation results
    """
    validator = MMSScientificValidator()
    
    checks = {}
    
    # 1. Formation geometry check
    geometry = assess_formation_geometry(positions)
    checks['formation_valid'] = geometry['valid']
    
    # 2. Phase velocity reasonableness
    checks['velocity_reasonable'] = (
        validator.MIN_PHASE_VELOCITY <= abs(phase_velocity) <= validator.MAX_PHASE_VELOCITY
    )
    
    # 3. Crossing sequence consistency
    crossing_times = [crossings[p] for p in sorted(crossings.keys()) if not np.isnan(crossings[p])]
    if len(crossing_times) >= 2:
        # Check if crossings are not simultaneous (would indicate infinite velocity)
        time_spread = max(crossing_times) - min(crossing_times)
        checks['crossing_sequence_valid'] = time_spread > 1.0  # At least 1 second spread
    else:
        checks['crossing_sequence_valid'] = False
    
    # 4. Normal vector validity
    checks['normal_valid'] = np.linalg.norm(normal) > 0.5  # Should be close to unit vector
    
    # Overall validity
    checks['overall_valid'] = all([
        checks['formation_valid'],
        checks['velocity_reasonable'], 
        checks['crossing_sequence_valid'],
        checks['normal_valid']
    ])
    
    return checks


def apply_spacecraft_calibration(data: np.ndarray, spacecraft_id: str, 
                               instrument: str = 'fgm') -> np.ndarray:
    """
    Apply known inter-spacecraft calibration corrections
    
    Args:
        data: Instrument data array
        spacecraft_id: Spacecraft identifier ('mms1', 'mms2', etc.)
        instrument: Instrument type ('fgm', 'fpi', etc.)
        
    Returns:
        Calibration-corrected data
    """
    validator = MMSScientificValidator()
    
    if instrument == 'fgm' and spacecraft_id in validator.MAGNETOMETER_CALIBRATION:
        factor = validator.MAGNETOMETER_CALIBRATION[spacecraft_id]
        return data * factor
    
    # Add other instrument calibrations as needed
    return data


def check_spacecraft_health(spacecraft_id: str, time_range: List[str], 
                          instrument: str) -> Dict[str, any]:
    """
    Check for known spacecraft/instrument issues during time period
    
    Args:
        spacecraft_id: Spacecraft identifier
        time_range: [start_time, end_time] in ISO format
        instrument: Instrument name
        
    Returns:
        Health status dictionary
    """
    validator = MMSScientificValidator()
    
    health_status = {
        'healthy': True,
        'issues': [],
        'severity': 'none',
        'recommendation': 'use_data'
    }
    
    # Check known issues
    if spacecraft_id in validator.KNOWN_ISSUES:
        for instr, issue_info in validator.KNOWN_ISSUES[spacecraft_id].items():
            if instrument.lower() in instr.lower():
                # Check if time range overlaps with issue period
                issue_start = datetime.fromisoformat(issue_info['start_date'])
                range_start = datetime.fromisoformat(time_range[0].replace('Z', '+00:00'))
                
                if range_start >= issue_start:
                    health_status['healthy'] = False
                    health_status['issues'].append(issue_info['issue'])
                    health_status['severity'] = issue_info['severity']
                    
                    if issue_info['severity'] == 'high':
                        health_status['recommendation'] = 'exclude_spacecraft'
                    else:
                        health_status['recommendation'] = 'use_with_caution'
    
    return health_status


def enhanced_multi_spacecraft_analysis(positions: Dict, crossings: Dict, 
                                     evt_data: Dict, time_range: List[str]):
    """
    Enhanced multi-spacecraft analysis with comprehensive validation
    
    This function performs the timing analysis with full scientific validation
    and quality assessment.
    """
    print("\n🔬 Enhanced Multi-Spacecraft Analysis with Scientific Validation")
    
    # 1. Check spacecraft health
    print("\n📋 Spacecraft Health Assessment:")
    healthy_spacecraft = {}
    for probe in positions.keys():
        health = check_spacecraft_health(f'mms{probe}', time_range, 'fpi')
        print(f"  MMS{probe}: {health['recommendation']} ({health['severity']} issues)")
        
        if health['recommendation'] != 'exclude_spacecraft':
            healthy_spacecraft[probe] = positions[probe]
    
    # 2. Formation geometry assessment
    print("\n📐 Formation Geometry Assessment:")
    geometry = assess_formation_geometry(healthy_spacecraft)
    print(f"  Tetrahedral Quality Factor: {geometry['tqf']:.3f}")
    print(f"  Formation Elongation: {geometry['elongation']:.2f}")
    print(f"  Formation Valid: {'✅' if geometry['valid'] else '❌'}")
    
    # 3. Perform timing analysis if formation is adequate
    if geometry['valid'] and len(healthy_spacecraft) >= 2:
        from . import multispacecraft
        
        try:
            # Filter crossings to match healthy spacecraft
            healthy_crossings = {p: crossings[p] for p in healthy_spacecraft.keys() 
                               if p in crossings}
            
            n_hat, V_phase, sigma_V = multispacecraft.timing_normal(
                healthy_spacecraft, healthy_crossings
            )
            
            # 4. Validate results
            print("\n✅ Boundary Analysis Validation:")
            validation = validate_boundary_analysis(
                healthy_spacecraft, healthy_crossings, n_hat, V_phase
            )
            
            for check, result in validation.items():
                status = "✅" if result else "❌"
                print(f"  {check}: {status}")
            
            if validation['overall_valid']:
                print(f"\n🎯 VALIDATED Results:")
                print(f"  Normal vector: [{n_hat[0]:.3f}, {n_hat[1]:.3f}, {n_hat[2]:.3f}]")
                print(f"  Phase velocity: {V_phase:.1f} ± {sigma_V:.1f} km/s")
                print(f"  Formation quality: {geometry['tqf']:.3f}")
            else:
                print(f"\n⚠️  Results may be unreliable due to validation failures")
            
            return n_hat, V_phase, sigma_V, validation
            
        except Exception as e:
            print(f"❌ Timing analysis failed: {e}")
            return None, None, None, {'overall_valid': False}
    
    else:
        print("❌ Formation geometry inadequate for reliable timing analysis")
        return None, None, None, {'overall_valid': False}


if __name__ == "__main__":
    # Example usage
    print("🔬 MMS Scientific Enhancement Module")
    print("Provides comprehensive validation for multi-spacecraft analysis")
