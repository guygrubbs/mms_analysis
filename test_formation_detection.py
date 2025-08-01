#!/usr/bin/env python3
"""
Test Formation Detection Module
==============================

Simple test to verify the formation detection module works correctly
and can distinguish between different formation types.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path to import mms_mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mms_mp.formation_detection import detect_formation_type, print_formation_analysis, FormationType


def create_test_formations():
    """Create test formations of different types"""
    
    formations = {}
    
    # 1. String-of-pearls formation (high linearity)
    formations['string_of_pearls'] = {
        '1': np.array([0.0, 0.0, 0.0]),      # km
        '2': np.array([100.0, 0.0, 0.0]),    # 100 km along X
        '3': np.array([200.0, 0.0, 0.0]),    # 200 km along X
        '4': np.array([300.0, 0.0, 0.0])     # 300 km along X
    }
    
    # 2. Tetrahedral formation (balanced 3D)
    formations['tetrahedral'] = {
        '1': np.array([0.0, 0.0, 0.0]),
        '2': np.array([100.0, 0.0, 0.0]),
        '3': np.array([50.0, 86.6, 0.0]),    # 60° in XY plane
        '4': np.array([50.0, 28.9, 81.6])    # Above plane
    }
    
    # 3. Planar formation (high planarity)
    formations['planar'] = {
        '1': np.array([0.0, 0.0, 0.0]),
        '2': np.array([100.0, 0.0, 0.0]),
        '3': np.array([50.0, 86.6, 0.0]),
        '4': np.array([-50.0, 86.6, 0.0])    # All in XY plane
    }
    
    # 4. Collapsed formation (very small separations)
    formations['collapsed'] = {
        '1': np.array([0.0, 0.0, 0.0]),
        '2': np.array([2.0, 0.0, 0.0]),      # 2 km separation
        '3': np.array([0.0, 2.0, 0.0]),
        '4': np.array([0.0, 0.0, 2.0])
    }
    
    # 5. Realistic string-of-pearls with some scatter
    formations['realistic_string'] = {
        '1': np.array([0.0, 5.0, 2.0]),      # Small Y,Z offsets
        '2': np.array([120.0, -3.0, 1.0]),   # Mainly along X
        '3': np.array([240.0, 2.0, -1.0]),
        '4': np.array([360.0, -1.0, 3.0])
    }
    
    return formations


def test_formation_detection():
    """Test the formation detection on known formation types"""
    
    print("FORMATION DETECTION TEST")
    print("=" * 50)
    
    formations = create_test_formations()
    
    for formation_name, positions in formations.items():
        print(f"\n🔍 Testing: {formation_name.upper()}")
        print("-" * 30)
        
        # Detect formation type
        analysis = detect_formation_type(positions)
        
        # Print results
        print_formation_analysis(analysis, verbose=False)
        
        # Check if detection matches expectation
        detected_type = analysis.formation_type.value
        expected_matches = {
            'string_of_pearls': ['string_of_pearls', 'linear'],
            'tetrahedral': ['tetrahedral'],
            'planar': ['planar'],
            'collapsed': ['collapsed'],
            'realistic_string': ['string_of_pearls', 'linear']
        }
        
        if detected_type in expected_matches.get(formation_name, []):
            print(f"✅ CORRECT: Expected {formation_name}, detected {detected_type}")
        else:
            print(f"❌ INCORRECT: Expected {formation_name}, detected {detected_type}")
        
        print(f"Confidence: {analysis.confidence:.3f}")


def test_real_mms_positions():
    """Test with realistic MMS positions"""
    
    print(f"\n" + "=" * 50)
    print("REALISTIC MMS POSITIONS TEST")
    print("=" * 50)
    
    # Typical MMS positions during string-of-pearls phase
    # Based on approximate positions from 2019 timeframe
    RE_km = 6371.0
    
    # String-of-pearls configuration (approximate)
    string_positions = {
        '1': np.array([10.5, 3.2, 1.8]) * RE_km,     # ~11.5 RE
        '2': np.array([10.6, 3.1, 1.7]) * RE_km,     # Slightly ahead
        '3': np.array([10.7, 3.0, 1.6]) * RE_km,     # Further ahead
        '4': np.array([10.8, 2.9, 1.5]) * RE_km      # Furthest ahead
    }
    
    print(f"\n🔍 Testing: REALISTIC STRING-OF-PEARLS")
    print("-" * 40)
    
    analysis = detect_formation_type(string_positions)
    print_formation_analysis(analysis, verbose=True)
    
    # Check if string-of-pearls is detected
    if analysis.formation_type in [FormationType.STRING_OF_PEARLS, FormationType.LINEAR]:
        print(f"✅ CORRECT: String-of-pearls formation detected")
    else:
        print(f"❌ INCORRECT: Expected string-of-pearls, got {analysis.formation_type.value}")
        print("   This suggests the formation detection needs tuning!")


def main():
    """Main test function"""
    
    # Test with synthetic formations
    test_formation_detection()
    
    # Test with realistic MMS positions
    test_real_mms_positions()
    
    print(f"\n" + "=" * 50)
    print("FORMATION DETECTION TEST COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
