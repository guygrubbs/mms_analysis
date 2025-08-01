#!/usr/bin/env python3
"""
Debug Data Loader MEC Integration
=================================

This script debugs exactly what happens in the data loader when trying
to load MEC ephemeris data and why _first_valid_var might be failing.
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


def debug_mec_loading_step_by_step():
    """Debug MEC loading step by step"""
    
    print("🔍 Debugging MEC Loading Step by Step")
    print("=" * 50)
    
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    print(f"Time range: {trange[0]} to {trange[1]}")
    
    # Clear data_quants
    data_quants.clear()
    print(f"Cleared data_quants: {len(data_quants)} variables")
    
    # Step 1: Load MEC data directly (we know this works)
    print(f"\n📊 Step 1: Loading MEC data directly...")
    
    from pyspedas.projects import mms
    
    for probe in ['1', '2', '3', '4']:
        print(f"\nLoading MMS{probe} MEC data...")
        
        try:
            result = mms.mms_load_mec(
                trange=trange,
                probe=probe,
                data_rate='srvy',
                level='l2',
                datatype='epht89q',
                time_clip=True,
                notplot=False
            )
            
            # Check what MEC variables were loaded
            mec_vars = [var for var in data_quants.keys() if f'mms{probe}_mec' in var]
            print(f"   MEC variables loaded: {len(mec_vars)}")
            
            for var in sorted(mec_vars):
                if 'r_gsm' in var or 'v_gsm' in var:
                    print(f"      {var}")
                    
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    # Step 2: Test _first_valid_var function
    print(f"\n📊 Step 2: Testing _first_valid_var function...")
    
    for probe in ['1', '2', '3', '4']:
        print(f"\nTesting MMS{probe}:")
        
        # Test position patterns
        pos_patterns = [
            f'mms{probe}_mec_r_gsm',
            f'mms{probe}_mec_r_gse',
            f'mms{probe}_defeph_pos',
            f'mms{probe}_state_pos_gsm'
        ]
        
        print(f"   Position patterns:")
        for pattern in pos_patterns:
            in_data_quants = pattern in data_quants
            if in_data_quants:
                is_valid = _is_valid(pattern, expect_cols=3)
                print(f"      {pattern}: in_data_quants={in_data_quants}, is_valid={is_valid}")
                
                if not is_valid:
                    # Debug why it's not valid
                    try:
                        times, data = get_data(pattern)
                        print(f"         Data shape: {data.shape}, finite values: {np.isfinite(data).any()}")
                        print(f"         Min abs value: {np.nanmin(np.abs(data))}")
                        if len(data) > 0:
                            sample = data[len(data)//2]
                            print(f"         Sample: {sample}")
                    except Exception as e:
                        print(f"         get_data error: {e}")
            else:
                print(f"      {pattern}: in_data_quants={in_data_quants}")
        
        # Test _first_valid_var with position patterns
        pos_result = _first_valid_var(pos_patterns, expect_cols=3)
        print(f"   _first_valid_var result for positions: {pos_result}")
        
        # Test velocity patterns
        vel_patterns = [
            f'mms{probe}_mec_v_gsm',
            f'mms{probe}_mec_v_gse',
            f'mms{probe}_defeph_vel',
            f'mms{probe}_state_vel_gsm'
        ]
        
        vel_result = _first_valid_var(vel_patterns, expect_cols=3)
        print(f"   _first_valid_var result for velocities: {vel_result}")
    
    # Step 3: Test data loader integration
    print(f"\n📊 Step 3: Testing data loader integration...")
    
    try:
        # Load event data through data_loader
        evt = mms_mp.load_event(
            trange=trange,
            probes=['1', '2', '3', '4'],
            include_ephem=True,
            include_edp=False
        )
        
        print(f"Event data loaded successfully")
        
        for probe in ['1', '2', '3', '4']:
            if probe in evt:
                print(f"\nMMS{probe} event data:")
                
                if 'POS_gsm' in evt[probe]:
                    times, pos_data = evt[probe]['POS_gsm']
                    print(f"   POS_gsm: {len(times)} points, shape {pos_data.shape}")
                    
                    # Check for NaN
                    if len(pos_data) > 0:
                        sample_idx = len(pos_data) // 2
                        sample = pos_data[sample_idx]
                        is_nan = np.any(np.isnan(sample))
                        print(f"   Sample position: {sample} (NaN: {is_nan})")
                        
                        if not is_nan:
                            print(f"   ✅ Real position data loaded!")
                        else:
                            print(f"   ❌ Position data is NaN (synthetic fallback)")
                else:
                    print(f"   ❌ No POS_gsm data")
                
                if 'VEL_gsm' in evt[probe]:
                    times, vel_data = evt[probe]['VEL_gsm']
                    print(f"   VEL_gsm: {len(times)} points, shape {vel_data.shape}")
                    
                    if len(vel_data) > 0:
                        sample_idx = len(vel_data) // 2
                        sample = vel_data[sample_idx]
                        is_nan = np.any(np.isnan(sample))
                        print(f"   Sample velocity: {sample} (NaN: {is_nan})")
                        
                        if not is_nan:
                            print(f"   ✅ Real velocity data loaded!")
                        else:
                            print(f"   ❌ Velocity data is NaN (synthetic fallback)")
                else:
                    print(f"   ❌ No VEL_gsm data")
            else:
                print(f"\nMMS{probe}: Not in event data")
                
    except Exception as e:
        print(f"❌ Data loader error: {e}")
        import traceback
        traceback.print_exc()


def test_mec_data_validation():
    """Test MEC data validation specifically"""
    
    print(f"\n🔍 Testing MEC Data Validation")
    print("=" * 50)
    
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    # Clear and load MEC data
    data_quants.clear()
    
    from pyspedas.projects import mms
    
    # Load MEC data for MMS1
    mms.mms_load_mec(
        trange=trange,
        probe='1',
        data_rate='srvy',
        level='l2',
        datatype='epht89q',
        time_clip=True,
        notplot=False
    )
    
    # Test specific MEC variable validation
    test_var = 'mms1_mec_r_gsm'
    
    if test_var in data_quants:
        print(f"\n📊 Testing validation for {test_var}:")
        
        try:
            times, data = get_data(test_var)
            print(f"   Data loaded: shape {data.shape}, {len(times)} time points")
            
            # Check validation criteria
            print(f"   Times not None: {times is not None}")
            print(f"   Data not None: {data is not None}")
            print(f"   Length > 0: {len(times) > 0}")
            print(f"   Expected columns (3): {data.ndim == 2 and data.shape[1] == 3}")
            print(f"   Has finite values: {np.isfinite(data).any()}")
            print(f"   Min abs value: {np.nanmin(np.abs(data))}")
            print(f"   Max abs value: {np.nanmax(np.abs(data))}")
            
            # Test _is_valid function
            is_valid_result = _is_valid(test_var, expect_cols=3)
            print(f"   _is_valid result: {is_valid_result}")
            
            # Show sample data
            if len(data) > 0:
                sample_idx = len(data) // 2
                sample = data[sample_idx]
                print(f"   Sample data: {sample}")
                
                # Check if sample is reasonable
                magnitude = np.linalg.norm(sample)
                print(f"   Sample magnitude: {magnitude:.1f} km")
                
                if 30000 < magnitude < 100000:  # Reasonable MMS orbit range
                    print(f"   ✅ Sample data looks reasonable")
                else:
                    print(f"   ⚠️ Sample data magnitude unusual")
            
        except Exception as e:
            print(f"   ❌ Error testing {test_var}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ {test_var} not found in data_quants")


def main():
    """Main debug function"""
    
    print("MEC DATA LOADER DEBUG")
    print("=" * 80)
    print("Debugging why data loader is not using real MEC data")
    print("=" * 80)
    
    debug_mec_loading_step_by_step()
    test_mec_data_validation()


if __name__ == "__main__":
    main()
