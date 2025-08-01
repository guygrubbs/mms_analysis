#!/usr/bin/env python3
"""
Inspect MEC CDF File
====================

This script directly examines the MEC CDF file to see what variables
are actually available and their exact names.
"""

import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from cdflib import CDF
    CDF_AVAILABLE = True
except ImportError:
    print("cdflib not available - trying alternative method")
    CDF_AVAILABLE = False

from pytplot import data_quants
from pyspedas.projects import mms


def inspect_mec_cdf_file():
    """Inspect MEC CDF file directly"""
    
    print("🔍 Inspecting MEC CDF File")
    print("=" * 50)
    
    # Path to a known MEC file
    mec_file = "pydata/mms1/mec/srvy/l2/epht89q/2019/01/mms1_mec_srvy_l2_epht89q_20190127_v2.2.2.cdf"
    
    if not os.path.exists(mec_file):
        print(f"❌ MEC file not found: {mec_file}")
        return False
    
    print(f"📁 File: {mec_file}")
    print(f"📊 File size: {os.path.getsize(mec_file) / 1024:.1f} KB")
    
    if CDF_AVAILABLE:
        try:
            # Open CDF file directly
            with CDF(mec_file) as cdf:
                print(f"\n📊 CDF File Info:")
                print(f"   Variables: {len(cdf.cdf_info()['zVariables'])}")
                
                print(f"\n📊 Available Variables:")
                for var_name in cdf.cdf_info()['zVariables']:
                    var_info = cdf.varinq(var_name)
                    print(f"   {var_name}: {var_info['Data_Type_Description']}, shape {var_info['Dim_Sizes']}")
                
                # Look for position and velocity variables
                pos_vars = []
                vel_vars = []
                
                for var_name in cdf.cdf_info()['zVariables']:
                    var_lower = var_name.lower()
                    if any(keyword in var_lower for keyword in ['pos', 'r_', 'position']):
                        pos_vars.append(var_name)
                    if any(keyword in var_lower for keyword in ['vel', 'v_', 'velocity']):
                        vel_vars.append(var_name)
                
                print(f"\n📊 Position Variables: {len(pos_vars)}")
                for var in pos_vars:
                    data = cdf.varget(var)
                    print(f"   {var}: shape {data.shape}")
                    if len(data) > 0:
                        sample_idx = len(data) // 2
                        if len(data.shape) > 1 and data.shape[1] >= 3:
                            print(f"      Sample: [{data[sample_idx][0]:.1f}, {data[sample_idx][1]:.1f}, {data[sample_idx][2]:.1f}]")
                
                print(f"\n📊 Velocity Variables: {len(vel_vars)}")
                for var in vel_vars:
                    data = cdf.varget(var)
                    print(f"   {var}: shape {data.shape}")
                    if len(data) > 0:
                        sample_idx = len(data) // 2
                        if len(data.shape) > 1 and data.shape[1] >= 3:
                            print(f"      Sample: [{data[sample_idx][0]:.2f}, {data[sample_idx][1]:.2f}, {data[sample_idx][2]:.2f}]")
                
                return True
                
        except Exception as e:
            print(f"❌ Error reading CDF file: {e}")
            return False
    else:
        print("⚠️ cdflib not available - cannot inspect CDF directly")
        return False


def test_pyspedas_mec_loading():
    """Test PySpedas MEC loading with verbose output"""
    
    print(f"\n🔍 Testing PySpedas MEC Loading")
    print("=" * 50)
    
    from datetime import datetime, timedelta
    
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    print(f"Time range: {trange[0]} to {trange[1]}")
    
    try:
        # Clear any existing data
        data_quants.clear()
        
        # Load MEC data for MMS1
        print(f"\nLoading MMS1 MEC data...")
        result = mms.mms_load_mec(
            trange=trange,
            probe='1',
            data_rate='srvy',
            level='l2',
            datatype='epht89q',
            time_clip=True,
            notplot=False,
            varnames=['mms1_mec_r_gsm', 'mms1_mec_v_gsm', 'mms1_mec_r_gse', 'mms1_mec_v_gse']
        )
        
        print(f"PySpedas result: {result}")
        
        print(f"\nVariables in data_quants after loading:")
        print(f"Total variables: {len(data_quants)}")
        
        for var_name in sorted(data_quants.keys()):
            if 'mms1' in var_name and 'mec' in var_name:
                times = data_quants[var_name].times
                data = data_quants[var_name].values
                print(f"   {var_name}: {len(times)} points, shape {data.shape}")
                
                # Show sample data
                if len(data) > 0:
                    sample_idx = len(data) // 2
                    if len(data.shape) > 1 and data.shape[1] >= 3:
                        print(f"      Sample: [{data[sample_idx][0]:.1f}, {data[sample_idx][1]:.1f}, {data[sample_idx][2]:.1f}]")
                    else:
                        print(f"      Sample: {data[sample_idx]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alternative_mec_loading():
    """Test alternative MEC loading methods"""
    
    print(f"\n🔍 Testing Alternative MEC Loading")
    print("=" * 50)
    
    from datetime import datetime, timedelta
    
    center_time = datetime(2019, 1, 27, 12, 30, 50)
    trange = [
        (center_time - timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S'),
        (center_time + timedelta(minutes=5)).strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    try:
        # Try loading with different parameters
        print(f"Trying different MEC loading parameters...")
        
        # Method 1: Load all variables
        print(f"\nMethod 1: Load all MEC variables")
        result1 = mms.mms_load_mec(
            trange=trange,
            probe='1',
            data_rate='srvy',
            level='l2',
            datatype='epht89q',
            time_clip=True,
            get_support_data=True,
            varformat='*'
        )
        print(f"Result 1: {result1}")
        
        # Check what was loaded
        mec_vars = [var for var in data_quants.keys() if 'mms1' in var and 'mec' in var]
        print(f"MEC variables loaded: {len(mec_vars)}")
        for var in sorted(mec_vars):
            print(f"   {var}")
        
        return len(mec_vars) > 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main inspection function"""
    
    print("MEC CDF FILE INSPECTION")
    print("=" * 80)
    print("Investigating MEC file structure and variable names")
    print("=" * 80)
    
    # Run inspections
    tests = [
        ("CDF File Inspection", inspect_mec_cdf_file),
        ("PySpedas MEC Loading", test_pyspedas_mec_loading),
        ("Alternative MEC Loading", test_alternative_mec_loading)
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
