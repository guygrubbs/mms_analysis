#!/usr/bin/env python3
"""
Debug Data Quants
=================

This script investigates what variables are actually in data_quants
after loading MEC data through the data_loader.
"""

import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path to import mms_mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mms_mp
from pytplot import data_quants


def debug_data_quants_after_loading():
    """Debug what's in data_quants after load_event"""
    
    print("🔍 Debugging Data Quants After Loading")
    print("=" * 50)
    
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    print(f"Time range: {trange[0]} to {trange[1]}")
    
    # Clear data_quants first
    data_quants.clear()
    print(f"Cleared data_quants: {len(data_quants)} variables")
    
    try:
        # Load event data
        print(f"\nLoading event data...")
        evt = mms_mp.load_event(
            trange=trange,
            probes=['1', '2', '3', '4'],
            include_ephem=True,
            include_edp=False
        )
        
        print(f"Event data loaded successfully")
        print(f"Variables in data_quants after loading: {len(data_quants)}")
        
        # Look for MEC variables
        mec_vars = []
        pos_vars = []
        vel_vars = []
        
        for var_name in data_quants.keys():
            if 'mec' in var_name.lower():
                mec_vars.append(var_name)
            if any(keyword in var_name.lower() for keyword in ['pos', 'r_']):
                pos_vars.append(var_name)
            if any(keyword in var_name.lower() for keyword in ['vel', 'v_']):
                vel_vars.append(var_name)
        
        print(f"\nMEC variables in data_quants: {len(mec_vars)}")
        for var in sorted(mec_vars):
            times = data_quants[var].times
            data = data_quants[var].values
            print(f"   {var}: {len(times)} points, shape {data.shape}")
            
            # Show sample data
            if len(data) > 0:
                sample_idx = len(data) // 2
                if len(data.shape) > 1 and data.shape[1] >= 3:
                    print(f"      Sample: [{data[sample_idx][0]:.1f}, {data[sample_idx][1]:.1f}, {data[sample_idx][2]:.1f}]")
        
        print(f"\nPosition-related variables in data_quants: {len(pos_vars)}")
        for var in sorted(pos_vars):
            print(f"   {var}")
        
        print(f"\nVelocity-related variables in data_quants: {len(vel_vars)}")
        for var in sorted(vel_vars):
            print(f"   {var}")
        
        # Test the _first_valid_var function directly
        print(f"\n🔍 Testing _first_valid_var function:")
        
        from mms_mp.data_loader import _first_valid_var
        
        # Test patterns for MMS1
        test_patterns = [
            ['mms1_mec_r_gsm', 'mms1_mec_r_gse', 'mms1_defeph_pos'],
            ['mms1_mec_v_gsm', 'mms1_mec_v_gse', 'mms1_defeph_vel']
        ]
        
        for i, patterns in enumerate(test_patterns):
            var_type = "position" if i == 0 else "velocity"
            print(f"\n   Testing {var_type} patterns for MMS1:")
            for pattern in patterns:
                if pattern in data_quants:
                    print(f"      ✅ {pattern}: Found in data_quants")
                    result = _first_valid_var([pattern], expect_cols=3)
                    if result:
                        print(f"         ✅ _first_valid_var returned: {result}")
                    else:
                        print(f"         ❌ _first_valid_var returned None (invalid data)")
                else:
                    print(f"      ❌ {pattern}: Not in data_quants")
            
            # Test the full pattern list
            result = _first_valid_var(patterns, expect_cols=3)
            if result:
                print(f"      ✅ _first_valid_var with full list returned: {result}")
            else:
                print(f"      ❌ _first_valid_var with full list returned None")
        
        # Check what's actually in the event data structure
        print(f"\n🔍 Event Data Structure:")
        for probe in ['1', '2', '3', '4']:
            if probe in evt:
                print(f"\nMMS{probe} variables in evt:")
                for var_name in evt[probe].keys():
                    times, data = evt[probe][var_name]
                    print(f"   {var_name}: {len(times)} points, shape {data.shape}")
                    
                    # Check for NaN data
                    if len(data) > 0:
                        nan_count = np.sum(np.isnan(data))
                        total_elements = data.size
                        if nan_count == total_elements:
                            print(f"      ❌ All data is NaN")
                        elif nan_count > 0:
                            print(f"      ⚠️ {nan_count}/{total_elements} elements are NaN")
                        else:
                            print(f"      ✅ No NaN data")
                            # Show sample
                            sample_idx = len(data) // 2
                            if len(data.shape) > 1 and data.shape[1] >= 3:
                                print(f"         Sample: [{data[sample_idx][0]:.1f}, {data[sample_idx][1]:.1f}, {data[sample_idx][2]:.1f}]")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main debug function"""
    
    print("DATA QUANTS DEBUG")
    print("=" * 80)
    print("Investigating data_quants contents after MEC loading")
    print("=" * 80)
    
    debug_data_quants_after_loading()


if __name__ == "__main__":
    main()
