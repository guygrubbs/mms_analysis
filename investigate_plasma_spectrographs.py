#!/usr/bin/env python3
"""
Investigate Plasma Spectrographs
================================

This script investigates what plasma spectrograph data is available
for the 2019-01-27 event and creates proper ion and electron spectrographs.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path to import mms_mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyspedas.projects import mms
from pytplot import get_data, data_quants


def investigate_fpi_data_availability():
    """Investigate what FPI data is available"""
    
    print("🔍 INVESTIGATING FPI PLASMA DATA AVAILABILITY")
    print("=" * 80)
    
    # Event time
    event_time = datetime(2019, 1, 27, 12, 30, 50)
    
    # Load full day to avoid time clipping issues
    full_day_start = datetime(2019, 1, 27, 0, 0, 0)
    full_day_end = datetime(2019, 1, 27, 23, 59, 59)
    
    trange_full = [
        full_day_start.strftime('%Y-%m-%d/%H:%M:%S'),
        full_day_end.strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    print(f"Event time: {event_time.strftime('%Y-%m-%d %H:%M:%S')} UT")
    print(f"Loading full day: {trange_full}")
    
    # Test different FPI data types and rates
    fpi_tests = [
        # Ion data
        {'datatype': 'dis-moms', 'data_rate': 'fast', 'description': 'Ion moments (fast)'},
        {'datatype': 'dis-moms', 'data_rate': 'brst', 'description': 'Ion moments (burst)'},
        {'datatype': 'dis-dist', 'data_rate': 'fast', 'description': 'Ion distributions (fast)'},
        {'datatype': 'dis-dist', 'data_rate': 'brst', 'description': 'Ion distributions (burst)'},
        
        # Electron data
        {'datatype': 'des-moms', 'data_rate': 'fast', 'description': 'Electron moments (fast)'},
        {'datatype': 'des-moms', 'data_rate': 'brst', 'description': 'Electron moments (burst)'},
        {'datatype': 'des-dist', 'data_rate': 'fast', 'description': 'Electron distributions (fast)'},
        {'datatype': 'des-dist', 'data_rate': 'brst', 'description': 'Electron distributions (burst)'},
    ]
    
    available_data = {}
    
    for test in fpi_tests:
        print(f"\n📊 Testing: {test['description']}")
        
        try:
            # Clear previous data
            data_quants.clear()
            
            # Load FPI data
            result = mms.mms_load_fpi(
                trange=trange_full,
                probe='1',  # Test with MMS1 first
                data_rate=test['data_rate'],
                level='l2',
                datatype=test['datatype'],
                time_clip=False,
                notplot=False
            )
            
            # Check what variables were loaded
            fpi_vars = [var for var in data_quants.keys() if 'mms1' in var and ('dis' in var or 'des' in var)]
            
            if fpi_vars:
                print(f"   ✅ Data loaded: {len(fpi_vars)} variables")
                
                # Check time coverage for key variables
                for var in fpi_vars[:5]:  # Check first 5 variables
                    try:
                        times, data = get_data(var)
                        
                        if len(times) > 0:
                            # Convert times
                            if hasattr(times[0], 'strftime'):
                                time_objects = times
                            else:
                                time_objects = [datetime.fromtimestamp(t) for t in times]
                            
                            first_time = time_objects[0]
                            last_time = time_objects[-1]
                            
                            print(f"      {var}: {len(times)} points")
                            print(f"         Range: {first_time.strftime('%H:%M:%S')} to {last_time.strftime('%H:%M:%S')} UT")
                            
                            # Check if event time is covered
                            if first_time <= event_time <= last_time:
                                print(f"         ✅ Event time covered")
                                
                                # Store this as available data
                                if test['datatype'] not in available_data:
                                    available_data[test['datatype']] = {}
                                available_data[test['datatype']][test['data_rate']] = {
                                    'variables': fpi_vars,
                                    'time_coverage': (first_time, last_time),
                                    'data_points': len(times)
                                }
                            else:
                                print(f"         ❌ Event time not covered")
                        else:
                            print(f"      {var}: No data points")
                            
                    except Exception as e:
                        print(f"      {var}: Error - {e}")
            else:
                print(f"   ❌ No FPI variables loaded")
                print(f"   Available variables: {list(data_quants.keys())[:5]}")
                
        except Exception as e:
            print(f"   ❌ Error loading {test['description']}: {e}")
    
    return available_data


def create_ion_spectrograph(available_data):
    """Create ion energy spectrograph"""
    
    print(f"\n🔍 CREATING ION SPECTROGRAPH")
    print("=" * 80)
    
    # Check what ion data is available
    ion_data_types = [key for key in available_data.keys() if 'dis' in key]
    
    if not ion_data_types:
        print("❌ No ion data available")
        return False
    
    print(f"Available ion data types: {ion_data_types}")
    
    # Event time and window
    event_time = datetime(2019, 1, 27, 12, 30, 50)
    start_time = event_time - timedelta(hours=1)
    end_time = event_time + timedelta(hours=1)
    
    # Try to load ion distribution data for spectrograph
    for data_type in ion_data_types:
        if 'dist' in data_type:  # Distribution data for spectrograph
            print(f"\n📊 Attempting ion spectrograph with {data_type}")
            
            try:
                # Clear and load distribution data
                data_quants.clear()
                
                # Load full day
                full_day_start = datetime(2019, 1, 27, 0, 0, 0)
                full_day_end = datetime(2019, 1, 27, 23, 59, 59)
                
                trange_full = [
                    full_day_start.strftime('%Y-%m-%d/%H:%M:%S'),
                    full_day_end.strftime('%Y-%m-%d/%H:%M:%S')
                ]
                
                # Try fast mode first
                result = mms.mms_load_fpi(
                    trange=trange_full,
                    probe=['1', '2', '3', '4'],
                    data_rate='fast',
                    level='l2',
                    datatype=data_type,
                    time_clip=False,
                    notplot=False
                )
                
                # Look for energy spectra variables
                energy_vars = [var for var in data_quants.keys() if 'energy' in var.lower() and 'dis' in var]
                eflux_vars = [var for var in data_quants.keys() if 'eflux' in var.lower() and 'dis' in var]
                
                print(f"   Energy variables: {len(energy_vars)}")
                print(f"   Energy flux variables: {len(eflux_vars)}")
                
                if energy_vars or eflux_vars:
                    # Create spectrograph plot
                    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
                    
                    for i, probe in enumerate(['1', '2', '3', '4']):
                        # Look for energy flux data for this probe
                        probe_eflux_vars = [var for var in eflux_vars if f'mms{probe}' in var]
                        
                        if probe_eflux_vars:
                            var_name = probe_eflux_vars[0]
                            times, spec_data = get_data(var_name)
                            
                            if len(times) > 0:
                                # Convert times
                                if hasattr(times[0], 'strftime'):
                                    time_objects = times
                                else:
                                    time_objects = [datetime.fromtimestamp(t) for t in times]
                                
                                # Filter to event window
                                window_mask = [(start_time <= t <= end_time) for t in time_objects]
                                
                                if any(window_mask):
                                    event_times = [t for t, mask in zip(time_objects, window_mask) if mask]
                                    event_spec = spec_data[window_mask]
                                    
                                    if len(event_spec.shape) == 2 and event_spec.shape[1] > 1:
                                        # Create spectrogram
                                        im = axes[i].imshow(event_spec.T, aspect='auto', origin='lower',
                                                          extent=[0, len(event_times), 0, event_spec.shape[1]],
                                                          cmap='jet', interpolation='nearest')
                                        
                                        axes[i].set_ylabel(f'MMS{probe}\nEnergy Channel')
                                        
                                        # Add colorbar
                                        plt.colorbar(im, ax=axes[i], label='Ion Energy Flux')
                                        
                                        print(f"   ✅ MMS{probe} ion spectrograph: {len(event_times)} time points, {event_spec.shape[1]} energy channels")
                                    else:
                                        axes[i].text(0.5, 0.5, f'MMS{probe}: Invalid spectrogram data shape', 
                                                   transform=axes[i].transAxes, ha='center', va='center')
                                else:
                                    axes[i].text(0.5, 0.5, f'MMS{probe}: No data in event window', 
                                               transform=axes[i].transAxes, ha='center', va='center')
                            else:
                                axes[i].text(0.5, 0.5, f'MMS{probe}: No data points', 
                                           transform=axes[i].transAxes, ha='center', va='center')
                        else:
                            axes[i].text(0.5, 0.5, f'MMS{probe}: No energy flux data available', 
                                       transform=axes[i].transAxes, ha='center', va='center')
                        
                        axes[i].grid(True, alpha=0.3)
                        axes[i].axvline(len([t for t in time_objects if start_time <= t <= end_time]) // 2, 
                                      color='red', linestyle='--', alpha=0.7, linewidth=2)
                    
                    axes[-1].set_xlabel('Time Index')
                    plt.suptitle(f'MMS Ion Energy Spectrographs: {event_time.strftime("%Y-%m-%d %H:%M:%S")} UT Event', fontsize=16)
                    plt.tight_layout()
                    plt.savefig('results_final/ion_spectrographs.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"✅ Ion spectrograph saved to results_final/ion_spectrographs.png")
                    return True
                else:
                    print(f"   ❌ No energy spectra variables found")
                    
            except Exception as e:
                print(f"   ❌ Error creating ion spectrograph: {e}")
                import traceback
                traceback.print_exc()
    
    return False


def create_electron_spectrograph(available_data):
    """Create electron energy spectrograph"""
    
    print(f"\n🔍 CREATING ELECTRON SPECTROGRAPH")
    print("=" * 80)
    
    # Check what electron data is available
    electron_data_types = [key for key in available_data.keys() if 'des' in key]
    
    if not electron_data_types:
        print("❌ No electron data available")
        return False
    
    print(f"Available electron data types: {electron_data_types}")
    
    # Event time and window
    event_time = datetime(2019, 1, 27, 12, 30, 50)
    start_time = event_time - timedelta(hours=1)
    end_time = event_time + timedelta(hours=1)
    
    # Try to load electron distribution data for spectrograph
    for data_type in electron_data_types:
        if 'dist' in data_type:  # Distribution data for spectrograph
            print(f"\n📊 Attempting electron spectrograph with {data_type}")
            
            try:
                # Clear and load distribution data
                data_quants.clear()
                
                # Load full day
                full_day_start = datetime(2019, 1, 27, 0, 0, 0)
                full_day_end = datetime(2019, 1, 27, 23, 59, 59)
                
                trange_full = [
                    full_day_start.strftime('%Y-%m-%d/%H:%M:%S'),
                    full_day_end.strftime('%Y-%m-%d/%H:%M:%S')
                ]
                
                # Try fast mode first
                result = mms.mms_load_fpi(
                    trange=trange_full,
                    probe=['1', '2', '3', '4'],
                    data_rate='fast',
                    level='l2',
                    datatype=data_type,
                    time_clip=False,
                    notplot=False
                )
                
                # Look for energy spectra variables
                energy_vars = [var for var in data_quants.keys() if 'energy' in var.lower() and 'des' in var]
                eflux_vars = [var for var in data_quants.keys() if 'eflux' in var.lower() and 'des' in var]
                
                print(f"   Energy variables: {len(energy_vars)}")
                print(f"   Energy flux variables: {len(eflux_vars)}")
                
                if energy_vars or eflux_vars:
                    # Create spectrograph plot
                    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
                    
                    for i, probe in enumerate(['1', '2', '3', '4']):
                        # Look for energy flux data for this probe
                        probe_eflux_vars = [var for var in eflux_vars if f'mms{probe}' in var]
                        
                        if probe_eflux_vars:
                            var_name = probe_eflux_vars[0]
                            times, spec_data = get_data(var_name)
                            
                            if len(times) > 0:
                                # Convert times
                                if hasattr(times[0], 'strftime'):
                                    time_objects = times
                                else:
                                    time_objects = [datetime.fromtimestamp(t) for t in times]
                                
                                # Filter to event window
                                window_mask = [(start_time <= t <= end_time) for t in time_objects]
                                
                                if any(window_mask):
                                    event_times = [t for t, mask in zip(time_objects, window_mask) if mask]
                                    event_spec = spec_data[window_mask]
                                    
                                    if len(event_spec.shape) == 2 and event_spec.shape[1] > 1:
                                        # Create spectrogram
                                        im = axes[i].imshow(event_spec.T, aspect='auto', origin='lower',
                                                          extent=[0, len(event_times), 0, event_spec.shape[1]],
                                                          cmap='jet', interpolation='nearest')
                                        
                                        axes[i].set_ylabel(f'MMS{probe}\nEnergy Channel')
                                        
                                        # Add colorbar
                                        plt.colorbar(im, ax=axes[i], label='Electron Energy Flux')
                                        
                                        print(f"   ✅ MMS{probe} electron spectrograph: {len(event_times)} time points, {event_spec.shape[1]} energy channels")
                                    else:
                                        axes[i].text(0.5, 0.5, f'MMS{probe}: Invalid spectrogram data shape', 
                                                   transform=axes[i].transAxes, ha='center', va='center')
                                else:
                                    axes[i].text(0.5, 0.5, f'MMS{probe}: No data in event window', 
                                               transform=axes[i].transAxes, ha='center', va='center')
                            else:
                                axes[i].text(0.5, 0.5, f'MMS{probe}: No data points', 
                                           transform=axes[i].transAxes, ha='center', va='center')
                        else:
                            axes[i].text(0.5, 0.5, f'MMS{probe}: No energy flux data available', 
                                       transform=axes[i].transAxes, ha='center', va='center')
                        
                        axes[i].grid(True, alpha=0.3)
                    
                    axes[-1].set_xlabel('Time Index')
                    plt.suptitle(f'MMS Electron Energy Spectrographs: {event_time.strftime("%Y-%m-%d %H:%M:%S")} UT Event', fontsize=16)
                    plt.tight_layout()
                    plt.savefig('results_final/electron_spectrographs.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"✅ Electron spectrograph saved to results_final/electron_spectrographs.png")
                    return True
                else:
                    print(f"   ❌ No energy spectra variables found")
                    
            except Exception as e:
                print(f"   ❌ Error creating electron spectrograph: {e}")
                import traceback
                traceback.print_exc()
    
    return False


def main():
    """Main investigation function"""

    print("INVESTIGATE PLASMA SPECTROGRAPHS")
    print("=" * 80)
    print("Investigating plasma data availability for 2019-01-27 event")
    print("=" * 80)

    # Create results directory if needed
    results_dir = "results_final"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Run investigations
    print("Phase 1: Investigating FPI data availability...")
    available_data = investigate_fpi_data_availability()

    print(f"\nPhase 2: Creating spectrographs...")

    # Try to create spectrographs
    ion_success = create_ion_spectrograph(available_data)
    electron_success = create_electron_spectrograph(available_data)

    # Summary
    print(f"\n" + "=" * 80)
    print("INVESTIGATION SUMMARY")
    print("=" * 80)

    print(f"Available data types: {list(available_data.keys())}")
    print(f"Ion spectrograph creation: {'✅ Success' if ion_success else '❌ Failed'}")
    print(f"Electron spectrograph creation: {'✅ Success' if electron_success else '❌ Failed'}")

    if not (ion_success or electron_success):
        print("\n🔍 RECOMMENDATIONS:")
        print("1. Check if FPI data files exist for this date")
        print("2. Try different data rates (brst vs fast)")
        print("3. Check data availability on MMS Science Data Center")
        print("4. Consider using moments data for plasma analysis")
        print("5. Check if this time period has known data gaps")

    return ion_success or electron_success


if __name__ == "__main__":
    success = main()

    if success:
        print(f"\n🎯 PLASMA SPECTROGRAPH INVESTIGATION: SUCCESS")
        print(f"✅ Spectrograph data found and plotted")
    else:
        print(f"\n⚠️ PLASMA SPECTROGRAPH INVESTIGATION: ISSUES FOUND")
        print(f"❌ Limited or no spectrograph data available")
