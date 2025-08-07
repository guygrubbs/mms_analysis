#!/usr/bin/env python3
"""
Simple MEC Debug Test
====================

Quick test to see the debugging output from the data loader.
"""

import numpy as np
from datetime import datetime
from mms_mp.data_loader import load_event
import warnings
warnings.filterwarnings('ignore')

def main():
    """Simple MEC debug test"""
    
    print("🔧 SIMPLE MEC DEBUG TEST")
    print("=" * 30)
    
    # Event parameters - very short window
    trange = ['2019-01-27/12:30:00', '2019-01-27/12:31:00']
    
    print(f"📡 Loading MMS data for short window: {trange[0]} to {trange[1]}")
    
    try:
        # Load data for just one spacecraft to see debugging
        data = load_event(
            trange=trange,
            probes=['1'],  # Just MMS1 for debugging
            data_rate_fgm='brst',
            data_rate_fpi='brst'
        )
        
        print("✅ Data loaded successfully")
        
        # Check what we got
        if '1' in data:
            probe_data = data['1']
            print(f"\nMMS1 variables loaded:")
            for key in sorted(probe_data.keys()):
                times, data_array = probe_data[key]
                print(f"   {key}: {len(times)} points, shape {data_array.shape}")
                
                if key == 'POS_gsm':
                    nan_count = np.isnan(data_array).sum()
                    print(f"      NaN count: {nan_count}/{data_array.size}")
                    if nan_count < data_array.size:
                        print(f"      Sample position: {data_array[0]}")
                
        else:
            print("❌ No data for MMS1")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
