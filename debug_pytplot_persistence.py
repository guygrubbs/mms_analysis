#!/usr/bin/env python3
"""
Debug PyTplot Data Persistence
==============================

This script investigates why MEC variables disappear from pytplot
after loading multiple spacecraft data.
"""

import numpy as np
from datetime import datetime
from mms_mp.data_loader import _load_state
from pytplot import data_quants, get_data
import warnings
warnings.filterwarnings('ignore')

def main():
    """Debug pytplot data persistence"""
    
    print("🔍 DEBUGGING PYTPLOT DATA PERSISTENCE")
    print("=" * 50)
    
    # Event parameters
    trange = ['2019-01-27/12:25:00', '2019-01-27/12:35:00']
    
    print(f"📡 Testing MEC data loading for: {trange[0]} to {trange[1]}")
    
    # Test loading MEC data one spacecraft at a time
    for probe in ['1', '2', '3', '4']:
        print(f"\n🛰️ Loading MEC data for MMS{probe}:")
        
        # Check what's in pytplot before loading
        before_vars = list(data_quants.keys())
        mec_before = [v for v in before_vars if 'mec' in v.lower()]
        print(f"   Before loading: {len(mec_before)} MEC variables in pytplot")
        if mec_before:
            print(f"      Existing MEC vars: {mec_before[:3]}...")  # Show first 3
        
        try:
            # Load MEC data for this spacecraft
            result = _load_state(trange, probe)
            
            # Check what's in pytplot after loading
            after_vars = list(data_quants.keys())
            mec_after = [v for v in after_vars if 'mec' in v.lower()]
            print(f"   After loading: {len(mec_after)} MEC variables in pytplot")
            
            # Check specifically for this spacecraft's variables
            this_probe_mec = [v for v in mec_after if f'mms{probe}_mec' in v]
            print(f"   MMS{probe} MEC vars: {len(this_probe_mec)}")
            
            # Check for other spacecraft's variables
            other_probe_mec = [v for v in mec_after if 'mec' in v and f'mms{probe}_mec' not in v]
            print(f"   Other spacecraft MEC vars: {len(other_probe_mec)}")
            
            if this_probe_mec:
                # Try to access the position data immediately
                pos_var = f'mms{probe}_mec_r_gsm'
                if pos_var in data_quants:
                    try:
                        times, pos_data = get_data(pos_var)
                        print(f"   ✅ Successfully accessed {pos_var}: {len(times)} points")
                        
                        # Check data quality
                        nan_count = np.isnan(pos_data).sum()
                        print(f"      NaN count: {nan_count}/{pos_data.size}")
                        
                        if nan_count == 0:
                            mid_pos = pos_data[len(pos_data)//2]
                            print(f"      Sample position: [{mid_pos[0]:.1f}, {mid_pos[1]:.1f}, {mid_pos[2]:.1f}] km")
                        
                    except Exception as e:
                        print(f"   ❌ Error accessing {pos_var}: {e}")
                else:
                    print(f"   ❌ {pos_var} not found in data_quants")
            
            # Show what happened to previous spacecraft data
            if probe != '1':
                prev_probe = str(int(probe) - 1)
                prev_vars = [v for v in mec_after if f'mms{prev_probe}_mec' in v]
                print(f"   Previous MMS{prev_probe} vars still present: {len(prev_vars)}")
                
        except Exception as e:
            print(f"   ❌ MEC loading failed: {e}")
    
    # Final check - what's left in pytplot?
    print(f"\n📊 FINAL PYTPLOT STATE:")
    print("=" * 30)
    
    final_vars = list(data_quants.keys())
    final_mec = [v for v in final_vars if 'mec' in v.lower()]
    
    print(f"Total variables: {len(final_vars)}")
    print(f"MEC variables: {len(final_mec)}")
    
    # Group by spacecraft
    for probe in ['1', '2', '3', '4']:
        probe_mec = [v for v in final_mec if f'mms{probe}_mec' in v]
        print(f"   MMS{probe}: {len(probe_mec)} MEC variables")
        if probe_mec:
            print(f"      Examples: {probe_mec[:2]}")
    
    # Test if we can access all spacecraft positions simultaneously
    print(f"\n🎯 SIMULTANEOUS ACCESS TEST:")
    print("=" * 35)
    
    positions = {}
    for probe in ['1', '2', '3', '4']:
        pos_var = f'mms{probe}_mec_r_gsm'
        if pos_var in data_quants:
            try:
                times, pos_data = get_data(pos_var)
                mid_pos = pos_data[len(pos_data)//2]
                positions[probe] = mid_pos
                print(f"   ✅ MMS{probe}: [{mid_pos[0]:.1f}, {mid_pos[1]:.1f}, {mid_pos[2]:.1f}] km")
            except Exception as e:
                print(f"   ❌ MMS{probe}: Error accessing data: {e}")
        else:
            print(f"   ❌ MMS{probe}: Variable {pos_var} not found")
    
    if len(positions) == 4:
        print(f"\n🎉 SUCCESS! All spacecraft positions accessible")
        
        # Calculate ordering
        x_positions = {probe: positions[probe][0] for probe in positions.keys()}
        x_ordered = sorted(positions.keys(), key=lambda p: x_positions[p])
        
        print(f"X-GSM ordering: {'-'.join(x_ordered)}")
        print(f"Expected:       2-1-4-3")
        print(f"Match: {'✅ YES' if x_ordered == ['2', '1', '4', '3'] else '❌ NO'}")
        
    else:
        print(f"❌ Only {len(positions)} spacecraft positions accessible")
        print("   This confirms the MEC data persistence issue")

if __name__ == "__main__":
    main()
