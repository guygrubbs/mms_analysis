"""
MMS Operational Mode Enhancements

This module provides enhanced data loading and analysis capabilities
that account for MMS mission operational realities.
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class MMSOperationalConfig:
    """Configuration for MMS operational mode awareness"""
    
    # Known problematic periods for specific spacecraft/instruments
    KNOWN_ISSUES = {
        'mms4': {
            'hpca': [
                ('2015-09-01', '2015-10-15'),  # Example: HPCA issues
            ],
            'fpi': [
                ('2019-11-01', '2019-11-30'),  # Example: FPI issues
            ]
        }
    }
    
    # Data quality thresholds
    MIN_DATA_COVERAGE = 0.7  # Minimum 70% data coverage
    MIN_CROSSING_CONFIDENCE = 0.5  # Minimum crossing detection confidence
    
    # Multi-spacecraft analysis requirements
    MIN_SPACECRAFT_FOR_TIMING = 2
    PREFERRED_SPACECRAFT_FOR_TIMING = 3


def assess_data_quality(spacecraft_data: Dict, probe: str) -> Dict[str, float]:
    """
    Assess data quality for a specific spacecraft
    
    Args:
        spacecraft_data: Data dictionary for one spacecraft
        probe: Spacecraft identifier ('1', '2', '3', '4')
        
    Returns:
        Quality metrics dictionary
    """
    quality = {}
    
    # Check data coverage for key instruments
    for instrument in ['B_gsm', 'N_he', 'N_e', 'V_he_gsm']:
        if instrument in spacecraft_data:
            data = spacecraft_data[instrument][1]
            if data.ndim == 1:
                valid_fraction = np.sum(~np.isnan(data)) / len(data)
            else:
                valid_fraction = np.sum(~np.isnan(data).any(axis=1)) / len(data)
            quality[f'{instrument}_coverage'] = valid_fraction
        else:
            quality[f'{instrument}_coverage'] = 0.0
    
    # Overall quality score
    key_instruments = ['B_gsm_coverage', 'N_he_coverage']
    if all(k in quality for k in key_instruments):
        quality['overall'] = np.mean([quality[k] for k in key_instruments])
    else:
        quality['overall'] = 0.0
    
    return quality


def filter_spacecraft_for_timing(positions: Dict, crossings: Dict, 
                                evt_data: Dict) -> Tuple[Dict, Dict]:
    """
    Filter spacecraft for multi-spacecraft timing analysis based on data quality
    
    Args:
        positions: Spacecraft positions dictionary
        crossings: Crossing times dictionary  
        evt_data: Full event data dictionary
        
    Returns:
        Filtered positions and crossings dictionaries
    """
    config = MMSOperationalConfig()
    
    # Assess quality for each spacecraft
    quality_scores = {}
    for probe in positions.keys():
        if probe in evt_data:
            quality = assess_data_quality(evt_data[probe], probe)
            quality_scores[probe] = quality['overall']
            
            print(f"  MMS{probe} data quality: {quality['overall']:.2f}")
            
            # Check for known issues
            if f'mms{probe}' in config.KNOWN_ISSUES:
                print(f"  ⚠️  MMS{probe} has known instrument issues")
        else:
            quality_scores[probe] = 0.0
    
    # Filter based on quality and valid crossings
    good_spacecraft = {}
    good_crossings = {}
    
    for probe in positions.keys():
        # Check data quality
        if quality_scores.get(probe, 0) < config.MIN_DATA_COVERAGE:
            print(f"  ❌ MMS{probe} excluded: poor data quality ({quality_scores.get(probe, 0):.2f})")
            continue
            
        # Check crossing validity
        if probe not in crossings or np.isnan(crossings[probe]):
            print(f"  ❌ MMS{probe} excluded: no valid crossing detected")
            continue
            
        # Include this spacecraft
        good_spacecraft[probe] = positions[probe]
        good_crossings[probe] = crossings[probe]
        print(f"  ✅ MMS{probe} included in timing analysis")
    
    # Check if we have enough spacecraft
    n_good = len(good_spacecraft)
    if n_good < config.MIN_SPACECRAFT_FOR_TIMING:
        print(f"  ⚠️  Only {n_good} spacecraft available (minimum {config.MIN_SPACECRAFT_FOR_TIMING})")
        
    return good_spacecraft, good_crossings


def get_optimal_data_rates(trange: List[str]) -> Dict[str, str]:
    """
    Determine optimal data rates based on time period and MMS operational mode
    
    Args:
        trange: Time range [start, end] in ISO format
        
    Returns:
        Dictionary of optimal data rates for each instrument
    """
    # Parse time range
    start_time = datetime.fromisoformat(trange[0].replace('Z', '+00:00'))
    
    # Default rates
    rates = {
        'fgm': 'fast',
        'fpi': 'fast', 
        'hpca': 'fast'
    }
    
    # Check if this might be a burst mode period
    # (This is simplified - real implementation would check MMS operations database)
    hour = start_time.hour
    if 6 <= hour <= 18:  # Daytime - more likely to have burst mode
        print("  📡 Daytime period - checking for burst mode data")
        rates['fgm'] = 'brst'  # Try burst first, fallback to fast
    
    return rates


def enhanced_load_event(trange: List[str], probes: List[str], **kwargs):
    """
    Enhanced event loading with MMS operational awareness
    
    This function wraps the standard load_event with additional intelligence
    about MMS operational modes and data quality.
    """
    from . import data_loader
    
    print("🛰️  Enhanced MMS loading with operational awareness")
    
    # Get optimal data rates for this time period
    optimal_rates = get_optimal_data_rates(trange)
    print(f"  📊 Optimal data rates: {optimal_rates}")
    
    # Update kwargs with optimal rates if not specified
    kwargs.setdefault('data_rate_fgm', optimal_rates['fgm'])
    kwargs.setdefault('data_rate_fpi', optimal_rates['fpi'])
    kwargs.setdefault('data_rate_hpca', optimal_rates['hpca'])
    
    # Enable fallback modes
    kwargs.setdefault('include_srvy', True)
    kwargs.setdefault('include_slow', True)
    
    # Load data with enhanced parameters
    evt = data_loader.load_event(trange, probes, **kwargs)
    
    # Assess and report data quality
    print("\n📋 Data Quality Assessment:")
    for probe in probes:
        if probe in evt:
            quality = assess_data_quality(evt[probe], probe)
            print(f"  MMS{probe}: {quality['overall']:.1%} overall quality")
            
            # Report specific issues
            for instrument, coverage in quality.items():
                if 'coverage' in instrument and coverage < 0.5:
                    print(f"    ⚠️  {instrument}: {coverage:.1%} coverage")
        else:
            print(f"  MMS{probe}: No data available")
    
    return evt


# Example usage functions
def robust_timing_analysis(positions: Dict, crossings: Dict, evt_data: Dict):
    """
    Perform timing analysis with automatic spacecraft filtering
    """
    from . import multispacecraft
    
    print("\n🔍 Robust Multi-Spacecraft Timing Analysis")
    
    # Filter spacecraft based on data quality
    good_pos, good_cross = filter_spacecraft_for_timing(positions, crossings, evt_data)
    
    if len(good_pos) < 2:
        print("❌ Insufficient spacecraft for timing analysis")
        return None, None, None
    
    try:
        # Perform timing analysis with filtered spacecraft
        n_hat, V_phase, sigma_V = multispacecraft.timing_normal(good_pos, good_cross)
        
        print(f"✅ Timing analysis successful with {len(good_pos)} spacecraft")
        print(f"   Normal vector: [{n_hat[0]:.3f}, {n_hat[1]:.3f}, {n_hat[2]:.3f}]")
        print(f"   Phase velocity: {V_phase:.1f} ± {sigma_V:.1f} km/s")
        
        return n_hat, V_phase, sigma_V
        
    except Exception as e:
        print(f"❌ Timing analysis failed: {e}")
        return None, None, None


if __name__ == "__main__":
    # Example of enhanced loading
    trange = ['2019-11-12T04:00:00', '2019-11-12T05:00:00']
    probes = ['1', '2', '3', '4']
    
    evt = enhanced_load_event(trange, probes)
    print("\n🎯 Enhanced loading complete with operational awareness!")
