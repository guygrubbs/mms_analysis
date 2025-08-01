#!/usr/bin/env python3
"""
Debug MEC Variables
===================

This script investigates what variables are actually available
in the loaded MEC data to fix the variable name matching.
"""

import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path to import mms_mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mms_mp
from pytplot import data_quants


def debug_loaded_variables():
    """Debug what variables are actually loaded"""
    
    print("🔍 Debugging Loaded MEC Variables")
    print("=" * 50)
    
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    print(f"Time range: {trange[0]} to {trange[1]}")
    
    try:
        # Load event data
        evt = mms_mp.load_event(
            trange=trange,
            probes=['1', '2', '3', '4'],
            include_ephem=True,
            include_edp=False
        )
        
        print(f"\n📊 Event Data Structure:")
        for probe in ['1', '2', '3', '4']:
            if probe in evt:
                print(f"\nMMS{probe} variables:")
                for var_name in evt[probe].keys():
                    times, data = evt[probe][var_name]
                    print(f"   {var_name}: {len(times)} points, shape {data.shape}")
        
        print(f"\n📊 PyTplot Data Quants (all loaded variables):")
        print(f"Total variables loaded: {len(data_quants)}")
        
        # Look for MEC-related variables
        mec_vars = []
        for var_name in data_quants.keys():
            if 'mec' in var_name.lower():
                mec_vars.append(var_name)
        
        print(f"\nMEC variables found: {len(mec_vars)}")
        for var in sorted(mec_vars):
            print(f"   {var}")
        
        # Look for position-related variables
        pos_vars = []
        for var_name in data_quants.keys():
            if any(keyword in var_name.lower() for keyword in ['pos', 'r_', 'position']):
                pos_vars.append(var_name)
        
        print(f"\nPosition-related variables found: {len(pos_vars)}")
        for var in sorted(pos_vars):
            print(f"   {var}")
        
        # Look for velocity-related variables
        vel_vars = []
        for var_name in data_quants.keys():
            if any(keyword in var_name.lower() for keyword in ['vel', 'v_', 'velocity']):
                vel_vars.append(var_name)
        
        print(f"\nVelocity-related variables found: {len(vel_vars)}")
        for var in sorted(vel_vars):
            print(f"   {var}")
        
        # Test direct access to MEC variables
        print(f"\n🔍 Testing Direct MEC Variable Access:")
        for probe in ['1', '2', '3', '4']:
            print(f"\nMMS{probe}:")
            
            # Try different MEC variable name patterns
            mec_patterns = [
                f'mms{probe}_mec_r_gsm',
                f'mms{probe}_mec_r_gse', 
                f'mms{probe}_mec_v_gsm',
                f'mms{probe}_mec_v_gse',
                f'mms{probe}_mec_pos_gsm',
                f'mms{probe}_mec_pos_gse',
                f'mms{probe}_mec_vel_gsm',
                f'mms{probe}_mec_vel_gse'
            ]
            
            for pattern in mec_patterns:
                if pattern in data_quants:
                    times, data = data_quants[pattern].values, data_quants[pattern].times
                    print(f"   ✅ {pattern}: {len(times)} points, shape {data.shape}")
                    
                    # Show sample data
                    if len(data) > 0:
                        sample_idx = len(data) // 2
                        print(f"      Sample: [{data[sample_idx][0]:.1f}, {data[sample_idx][1]:.1f}, {data[sample_idx][2]:.1f}]")
                else:
                    print(f"   ❌ {pattern}: Not found")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_direct_mec_loading():
    """Test loading MEC data directly with PySpedas"""
    
    print(f"\n🔍 Testing Direct MEC Loading")
    print("=" * 50)
    
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    try:
        from pyspedas.projects import mms
        
        # Load MEC data directly
        for probe in ['1', '2', '3', '4']:
            print(f"\nLoading MMS{probe} MEC data directly...")
            
            mms.mms_load_mec(
                trange=trange,
                probe=probe,
                data_rate='srvy',
                level='l2',
                datatype='epht89q',
                time_clip=True,
                notplot=False
            )
            
            # Check what variables were loaded
            mec_vars_for_probe = []
            for var_name in data_quants.keys():
                if f'mms{probe}_mec' in var_name:
                    mec_vars_for_probe.append(var_name)
            
            print(f"   Variables loaded: {len(mec_vars_for_probe)}")
            for var in sorted(mec_vars_for_probe):
                times = data_quants[var].times
                data = data_quants[var].values
                print(f"      {var}: {len(times)} points, shape {data.shape}")
                
                # Show sample data for position/velocity
                if any(keyword in var for keyword in ['r_', 'v_', 'pos', 'vel']):
                    if len(data) > 0:
                        sample_idx = len(data) // 2
                        if len(data.shape) > 1 and data.shape[1] >= 3:
                            print(f"         Sample: [{data[sample_idx][0]:.1f}, {data[sample_idx][1]:.1f}, {data[sample_idx][2]:.1f}]")
                        else:
                            print(f"         Sample: {data[sample_idx]}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main debug function"""
    
    print("MEC VARIABLE DEBUG")
    print("=" * 80)
    print("Investigating MEC variable names and data availability")
    print("=" * 80)
    
    # Run debug tests
    tests = [
        ("Loaded Variables Analysis", debug_loaded_variables),
        ("Direct MEC Loading", test_direct_mec_loading)
    ]
    
    for test_name, test_func in tests:
        print(f"\n" + "=" * 80)
        print(f"TEST: {test_name}")
        print("=" * 80)
        
        try:
            test_func()
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")


if __name__ == "__main__":
    main()
