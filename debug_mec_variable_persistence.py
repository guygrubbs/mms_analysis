#!/usr/bin/env python3
"""
Debug MEC Variable Persistence
==============================

This script investigates why MEC variables disappear from data_quants
during the data loading process.
"""

import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path to import mms_mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pytplot import data_quants
from pyspedas.projects import mms


def test_mec_variable_persistence():
    """Test MEC variable persistence through loading process"""
    
    print("🔍 Testing MEC Variable Persistence")
    print("=" * 50)
    
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    # Step 1: Clear data_quants and check initial state
    print(f"\n📊 Step 1: Initial state")
    data_quants.clear()
    print(f"   data_quants cleared: {len(data_quants)} variables")
    
    # Step 2: Load MEC data directly and check
    print(f"\n📊 Step 2: Load MEC data directly")
    
    try:
        result = mms.mms_load_mec(
            trange=trange,
            probe='1',
            data_rate='srvy',
            level='l2',
            datatype='epht89q',
            time_clip=True,
            notplot=False
        )
        
        print(f"   MEC loading result: {result}")
        print(f"   Variables in data_quants after MEC loading: {len(data_quants)}")
        
        # List all variables
        all_vars = list(data_quants.keys())
        mec_vars = [var for var in all_vars if 'mms1_mec' in var]
        
        print(f"   Total variables: {len(all_vars)}")
        print(f"   MEC variables: {len(mec_vars)}")
        
        if mec_vars:
            print(f"   MEC variables found:")
            for var in sorted(mec_vars):
                print(f"      {var}")
        else:
            print(f"   ❌ No MEC variables found")
            print(f"   All variables:")
            for var in sorted(all_vars):
                print(f"      {var}")
        
        # Step 3: Test if we can access MEC position data
        if 'mms1_mec_r_gsm' in data_quants:
            print(f"\n📊 Step 3: Test MEC position access")
            
            from pyspedas import get_data
            times, pos_data = get_data('mms1_mec_r_gsm')
            
            print(f"   Position data shape: {pos_data.shape}")
            print(f"   Times length: {len(times)}")
            
            if len(pos_data) > 0:
                sample_idx = len(pos_data) // 2
                sample = pos_data[sample_idx]
                print(f"   Sample position: {sample}")
                print(f"   Sample magnitude: {np.linalg.norm(sample):.1f} km")
                
                if not np.any(np.isnan(sample)) and np.linalg.norm(sample) > 10000:
                    print(f"   ✅ MEC position data is valid")
                    return True
                else:
                    print(f"   ❌ MEC position data is invalid")
                    return False
            else:
                print(f"   ❌ No position data")
                return False
        else:
            print(f"\n❌ mms1_mec_r_gsm not found in data_quants")
            return False
            
    except Exception as e:
        print(f"   ❌ Error loading MEC data: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loader_variable_clearing():
    """Test if data_loader clears variables during loading"""
    
    print(f"\n🔍 Testing Data Loader Variable Clearing")
    print("=" * 50)
    
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    # Step 1: Load MEC data directly first
    print(f"\n📊 Step 1: Load MEC data directly")
    data_quants.clear()
    
    try:
        mms.mms_load_mec(
            trange=trange,
            probe='1',
            data_rate='srvy',
            level='l2',
            datatype='epht89q',
            time_clip=True,
            notplot=False
        )
        
        mec_vars_before = [var for var in data_quants.keys() if 'mms1_mec' in var]
        print(f"   MEC variables before data_loader: {len(mec_vars_before)}")
        
        # Step 2: Now call data_loader and see what happens
        print(f"\n📊 Step 2: Call data_loader")
        
        import mms_mp
        
        # Monitor data_quants during loading
        print(f"   Variables before load_event: {len(data_quants)}")
        
        evt = mms_mp.load_event(
            trange=trange,
            probes=['1'],
            include_ephem=True,
            include_edp=False
        )
        
        mec_vars_after = [var for var in data_quants.keys() if 'mms1_mec' in var]
        print(f"   Variables after load_event: {len(data_quants)}")
        print(f"   MEC variables after data_loader: {len(mec_vars_after)}")
        
        if len(mec_vars_before) > 0 and len(mec_vars_after) == 0:
            print(f"   ❌ Data loader cleared MEC variables!")
            return False
        elif len(mec_vars_after) > 0:
            print(f"   ✅ MEC variables survived data loading")
            return True
        else:
            print(f"   ⚠️ No MEC variables found before or after")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alternative_mec_loading():
    """Test alternative MEC loading methods"""
    
    print(f"\n🔍 Testing Alternative MEC Loading Methods")
    print("=" * 50)
    
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    data_quants.clear()
    
    try:
        # Method 1: Load with specific variable names
        print(f"\n📊 Method 1: Load with specific variable names")
        
        result1 = mms.mms_load_mec(
            trange=trange,
            probe='1',
            data_rate='srvy',
            level='l2',
            datatype='epht89q',
            time_clip=True,
            notplot=False,
            varnames=['mms1_mec_r_gsm', 'mms1_mec_v_gsm']
        )
        
        vars_method1 = [var for var in data_quants.keys() if 'mms1_mec' in var]
        print(f"   Method 1 variables: {len(vars_method1)}")
        for var in vars_method1:
            print(f"      {var}")
        
        # Method 2: Load all variables
        data_quants.clear()
        print(f"\n📊 Method 2: Load all variables")
        
        result2 = mms.mms_load_mec(
            trange=trange,
            probe='1',
            data_rate='srvy',
            level='l2',
            datatype='epht89q',
            time_clip=True,
            notplot=False,
            varformat='*'
        )
        
        vars_method2 = [var for var in data_quants.keys() if 'mms1_mec' in var]
        print(f"   Method 2 variables: {len(vars_method2)}")
        
        # Method 3: Load with get_support_data
        data_quants.clear()
        print(f"\n📊 Method 3: Load with get_support_data")
        
        result3 = mms.mms_load_mec(
            trange=trange,
            probe='1',
            data_rate='srvy',
            level='l2',
            datatype='epht89q',
            time_clip=True,
            notplot=False,
            get_support_data=True
        )
        
        vars_method3 = [var for var in data_quants.keys() if 'mms1_mec' in var]
        print(f"   Method 3 variables: {len(vars_method3)}")
        
        # Find the best method
        best_method = max([
            (1, len(vars_method1)),
            (2, len(vars_method2)), 
            (3, len(vars_method3))
        ], key=lambda x: x[1])
        
        print(f"\n   Best method: Method {best_method[0]} with {best_method[1]} variables")
        
        return best_method[1] > 0
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main debug function"""
    
    print("DEBUG MEC VARIABLE PERSISTENCE")
    print("=" * 80)
    print("Investigating why MEC variables disappear from data_quants")
    print("=" * 80)
    
    # Run tests
    test1_success = test_mec_variable_persistence()
    test2_success = test_data_loader_variable_clearing()
    test3_success = test_alternative_mec_loading()
    
    print(f"\n" + "=" * 80)
    print("DEBUG SUMMARY")
    print("=" * 80)
    
    print(f"MEC variable persistence: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"Data loader compatibility: {'✅ PASS' if test2_success else '❌ FAIL'}")
    print(f"Alternative loading methods: {'✅ PASS' if test3_success else '❌ FAIL'}")
    
    if test1_success:
        print("🎯 MEC variables can be loaded and accessed")
    else:
        print("❌ Fundamental MEC loading issue")
    
    if test2_success:
        print("🎯 Data loader preserves MEC variables")
    else:
        print("❌ Data loader clears MEC variables")
    
    if test3_success:
        print("🎯 Alternative loading methods work")
    else:
        print("❌ All MEC loading methods fail")


if __name__ == "__main__":
    main()
