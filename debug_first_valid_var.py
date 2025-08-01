#!/usr/bin/env python3
"""
Debug _first_valid_var Function
===============================

This script debugs exactly what's happening in the _first_valid_var function
to understand why it's not finding the MEC variables.
"""

import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path to import mms_mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mms_mp
from pytplot import data_quants
from pyspedas import get_data
from mms_mp.data_loader import _first_valid_var, _is_valid


def debug_first_valid_var_step_by_step():
    """Debug _first_valid_var step by step"""
    
    print("🔍 Debugging _first_valid_var Function")
    print("=" * 50)
    
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    try:
        # Load event data to trigger MEC loading
        print(f"Loading event data...")
        evt = mms_mp.load_event(
            trange=trange,
            probes=['1'],
            include_ephem=True,
            include_edp=False
        )
        
        print(f"Event data loaded. Checking data_quants...")
        
        # Check what MEC variables are in data_quants
        mec_vars = [var for var in data_quants.keys() if 'mms1_mec' in var]
        print(f"MEC variables in data_quants: {len(mec_vars)}")
        for var in sorted(mec_vars):
            print(f"   {var}")
        
        # Test the exact patterns used by data loader
        print(f"\n🔍 Testing _first_valid_var patterns for MMS1:")
        
        pos_patterns = [
            'mms1_mec_r_gsm',
            'mms1_mec_r_gse', 
            'mms1_defeph_pos',
            'mms1_state_pos_gsm',
            'mms1_orbatt_r_gsm'
        ]
        
        print(f"\nPosition patterns:")
        for pattern in pos_patterns:
            in_data_quants = pattern in data_quants
            print(f"   {pattern}: in_data_quants = {in_data_quants}")
            
            if in_data_quants:
                # Test _is_valid function
                is_valid_result = _is_valid(pattern, expect_cols=3)
                print(f"      _is_valid result: {is_valid_result}")
                
                if not is_valid_result:
                    # Debug why it's not valid
                    try:
                        times, data = get_data(pattern)
                        print(f"      Data shape: {data.shape}")
                        print(f"      Times length: {len(times)}")
                        print(f"      Data has finite values: {np.isfinite(data).any()}")
                        print(f"      Data min: {np.nanmin(data)}")
                        print(f"      Data max: {np.nanmax(data)}")
                        
                        # Check specific validation criteria
                        print(f"      Validation details:")
                        print(f"         times is not None: {times is not None}")
                        print(f"         data is not None: {data is not None}")
                        print(f"         len(times) > 0: {len(times) > 0}")
                        print(f"         data.ndim == 2: {data.ndim == 2}")
                        if data.ndim == 2:
                            print(f"         data.shape[1] == 3: {data.shape[1] == 3}")
                        print(f"         np.isfinite(data).any(): {np.isfinite(data).any()}")
                        
                        # Show sample data
                        if len(data) > 0:
                            sample_idx = len(data) // 2
                            sample = data[sample_idx]
                            print(f"      Sample data: {sample}")
                            print(f"      Sample magnitude: {np.linalg.norm(sample):.1f}")
                            
                    except Exception as e:
                        print(f"      Error getting data: {e}")
        
        # Test _first_valid_var with the exact pattern list
        print(f"\n🔍 Testing _first_valid_var with position patterns:")
        pos_result = _first_valid_var(pos_patterns, expect_cols=3)
        print(f"   _first_valid_var result: {pos_result}")
        
        if pos_result:
            print(f"   ✅ _first_valid_var found valid variable: {pos_result}")
            
            # Test extracting data from the result
            try:
                times, data = get_data(pos_result)
                sample_idx = len(data) // 2
                sample = data[sample_idx]
                print(f"   Sample position: {sample}")
                print(f"   Sample magnitude: {np.linalg.norm(sample):.1f} km")
                
                if not np.any(np.isnan(sample)) and np.linalg.norm(sample) > 10000:
                    print(f"   ✅ Sample data is valid and reasonable")
                else:
                    print(f"   ❌ Sample data is invalid (NaN: {np.any(np.isnan(sample))}, mag: {np.linalg.norm(sample):.1f})")
                    
            except Exception as e:
                print(f"   ❌ Error extracting data from result: {e}")
        else:
            print(f"   ❌ _first_valid_var returned None")
        
        # Test velocity patterns
        print(f"\n🔍 Testing velocity patterns:")
        vel_patterns = [
            'mms1_mec_v_gsm',
            'mms1_mec_v_gse',
            'mms1_defeph_vel',
            'mms1_state_vel_gsm'
        ]
        
        vel_result = _first_valid_var(vel_patterns, expect_cols=3)
        print(f"   _first_valid_var velocity result: {vel_result}")
        
        return pos_result is not None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_manual_mec_extraction():
    """Test manual MEC data extraction to compare with data loader"""
    
    print(f"\n🔍 Testing Manual MEC Data Extraction")
    print("=" * 50)
    
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    # Clear data_quants and load MEC data directly
    data_quants.clear()
    
    from pyspedas.projects import mms
    
    try:
        # Load MEC data directly
        print(f"Loading MEC data directly...")
        mms.mms_load_mec(
            trange=trange,
            probe='1',
            data_rate='srvy',
            level='l2',
            datatype='epht89q',
            time_clip=True,
            notplot=False
        )
        
        # Extract position manually
        pos_var = 'mms1_mec_r_gsm'
        if pos_var in data_quants:
            times, pos_data = get_data(pos_var)
            
            # Find closest time
            if hasattr(times[0], 'timestamp'):
                time_diffs = [abs((t - center_time).total_seconds()) for t in times]
            else:
                target_timestamp = center_time.timestamp()
                time_diffs = [abs(t - target_timestamp) for t in times]
            
            closest_index = np.argmin(time_diffs)
            position = pos_data[closest_index]
            
            print(f"Manual extraction:")
            print(f"   Position: {position}")
            print(f"   Magnitude: {np.linalg.norm(position):.1f} km")
            print(f"   NaN check: {np.any(np.isnan(position))}")
            
            if not np.any(np.isnan(position)) and np.linalg.norm(position) > 10000:
                print(f"   ✅ Manual extraction successful")
                return True
            else:
                print(f"   ❌ Manual extraction failed")
                return False
        else:
            print(f"   ❌ {pos_var} not found in data_quants")
            return False
            
    except Exception as e:
        print(f"❌ Error in manual extraction: {e}")
        return False


def main():
    """Main debug function"""
    
    print("DEBUG _first_valid_var FUNCTION")
    print("=" * 80)
    print("Investigating why _first_valid_var is not finding MEC variables")
    print("=" * 80)
    
    # Run debug tests
    test1_success = debug_first_valid_var_step_by_step()
    test2_success = test_manual_mec_extraction()
    
    print(f"\n" + "=" * 80)
    print("DEBUG SUMMARY")
    print("=" * 80)
    
    if test1_success:
        print("✅ _first_valid_var is working correctly")
    else:
        print("❌ _first_valid_var has issues")
    
    if test2_success:
        print("✅ Manual MEC extraction works")
    else:
        print("❌ Manual MEC extraction fails")
    
    if test1_success and test2_success:
        print("🎯 Both methods work - data loader should be fixed")
    elif test2_success and not test1_success:
        print("⚠️ Manual works but _first_valid_var doesn't - need to fix validation")
    else:
        print("❌ Fundamental MEC data issue")


if __name__ == "__main__":
    main()
