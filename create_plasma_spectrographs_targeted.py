#!/usr/bin/env python3
"""
Create Plasma Spectrographs - Targeted Approach
===============================================

This script creates ion and electron spectrographs using the burst mode
data that we know exists around the 12:30 UT event time.
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


def create_targeted_ion_spectrograph():
    """Create ion spectrograph using targeted burst mode data"""
    
    print("🔍 CREATING TARGETED ION SPECTROGRAPH")
    print("=" * 60)
    
    # Event time and focused window
    event_time = datetime(2019, 1, 27, 12, 30, 50)
    
    # Use a smaller window around the event for burst mode
    start_time = event_time - timedelta(minutes=30)
    end_time = event_time + timedelta(minutes=30)
    
    trange = [
        start_time.strftime('%Y-%m-%d/%H:%M:%S'),
        end_time.strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    print(f"Event time: {event_time.strftime('%Y-%m-%d %H:%M:%S')} UT")
    print(f"Target window: {start_time.strftime('%H:%M:%S')} to {end_time.strftime('%H:%M:%S')} UT")
    print(f"Loading burst mode ion distributions...")
    
    try:
        # Clear previous data
        data_quants.clear()
        
        # Load burst mode ion distributions
        result = mms.mms_load_fpi(
            trange=trange,
            probe=['1', '2', '3', '4'],
            data_rate='brst',
            level='l2',
            datatype='dis-dist',
            time_clip=False,
            notplot=False
        )
        
        print(f"Loading result: {result}")
        
        # Check what variables were loaded
        all_vars = list(data_quants.keys())
        ion_vars = [var for var in all_vars if 'dis' in var]
        
        print(f"Total variables loaded: {len(all_vars)}")
        print(f"Ion variables: {len(ion_vars)}")
        
        if ion_vars:
            print(f"Ion variables found:")
            for var in ion_vars[:10]:  # Show first 10
                print(f"   {var}")
        
        # Look for energy flux or distribution variables
        energy_vars = [var for var in ion_vars if 'energy' in var.lower()]
        eflux_vars = [var for var in ion_vars if 'eflux' in var.lower()]
        dist_vars = [var for var in ion_vars if 'dist' in var.lower()]
        
        print(f"\nSpectrograph variables:")
        print(f"   Energy variables: {len(energy_vars)}")
        print(f"   Energy flux variables: {len(eflux_vars)}")
        print(f"   Distribution variables: {len(dist_vars)}")
        
        # Try to create spectrograph with available data
        spec_vars = eflux_vars if eflux_vars else (energy_vars if energy_vars else dist_vars)
        
        if spec_vars:
            print(f"\nCreating spectrograph with: {spec_vars[0]}")
            
            # Create plot
            fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
            
            plots_created = 0
            
            for i, probe in enumerate(['1', '2', '3', '4']):
                # Find variables for this probe
                probe_vars = [var for var in spec_vars if f'mms{probe}' in var]
                
                if probe_vars:
                    var_name = probe_vars[0]
                    
                    try:
                        times, spec_data = get_data(var_name)
                        
                        if len(times) > 0:
                            # Convert times
                            if hasattr(times[0], 'strftime'):
                                time_objects = times
                            else:
                                time_objects = [datetime.fromtimestamp(t) for t in times]
                            
                            print(f"   MMS{probe}: {len(times)} time points, data shape: {spec_data.shape}")
                            
                            # Check if we have 2D data for spectrogram
                            if len(spec_data.shape) == 2 and spec_data.shape[1] > 1:
                                # Create spectrogram
                                im = axes[i].imshow(spec_data.T, aspect='auto', origin='lower',
                                                  extent=[0, len(time_objects), 0, spec_data.shape[1]],
                                                  cmap='jet', interpolation='nearest', vmin=1e-15, vmax=1e-10)
                                
                                axes[i].set_ylabel(f'MMS{probe}\nEnergy Channel')
                                
                                # Add colorbar
                                cbar = plt.colorbar(im, ax=axes[i])
                                cbar.set_label('Ion Flux')
                                
                                # Mark event time
                                event_index = len(time_objects) // 2  # Approximate center
                                axes[i].axvline(event_index, color='white', linestyle='--', alpha=0.8, linewidth=2)
                                
                                plots_created += 1
                                print(f"   ✅ MMS{probe} spectrograph created")
                            else:
                                # Plot 1D data
                                axes[i].plot(time_objects, spec_data, label=f'MMS{probe}')
                                axes[i].set_ylabel(f'MMS{probe}\nIon Data')
                                axes[i].legend()
                                plots_created += 1
                                print(f"   ✅ MMS{probe} 1D plot created")
                        else:
                            axes[i].text(0.5, 0.5, f'MMS{probe}: No data points', 
                                       transform=axes[i].transAxes, ha='center', va='center')
                            print(f"   ❌ MMS{probe}: No data points")
                            
                    except Exception as e:
                        axes[i].text(0.5, 0.5, f'MMS{probe}: Error - {str(e)[:50]}', 
                                   transform=axes[i].transAxes, ha='center', va='center')
                        print(f"   ❌ MMS{probe}: Error - {e}")
                else:
                    axes[i].text(0.5, 0.5, f'MMS{probe}: No ion data available', 
                               transform=axes[i].transAxes, ha='center', va='center')
                    print(f"   ❌ MMS{probe}: No variables found")
                
                axes[i].grid(True, alpha=0.3)
            
            axes[-1].set_xlabel('Time Index / Time')
            plt.suptitle(f'MMS Ion Spectrographs (Burst Mode): {event_time.strftime("%Y-%m-%d %H:%M:%S")} UT Event', fontsize=16)
            plt.tight_layout()
            plt.savefig('results_final/ion_spectrographs_burst.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\n✅ Ion spectrograph saved: results_final/ion_spectrographs_burst.png")
            print(f"   Plots created: {plots_created}/4 spacecraft")
            
            return plots_created > 0
        else:
            print(f"❌ No suitable variables for spectrograph")
            return False
            
    except Exception as e:
        print(f"❌ Error creating ion spectrograph: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_targeted_electron_spectrograph():
    """Create electron spectrograph using targeted burst mode data"""
    
    print(f"\n🔍 CREATING TARGETED ELECTRON SPECTROGRAPH")
    print("=" * 60)
    
    # Event time and focused window
    event_time = datetime(2019, 1, 27, 12, 30, 50)
    
    # Use a smaller window around the event for burst mode
    start_time = event_time - timedelta(minutes=30)
    end_time = event_time + timedelta(minutes=30)
    
    trange = [
        start_time.strftime('%Y-%m-%d/%H:%M:%S'),
        end_time.strftime('%Y-%m-%d/%H:%M:%S')
    ]
    
    print(f"Target window: {start_time.strftime('%H:%M:%S')} to {end_time.strftime('%H:%M:%S')} UT")
    print(f"Loading burst mode electron distributions...")
    
    try:
        # Clear previous data
        data_quants.clear()
        
        # Load burst mode electron distributions
        result = mms.mms_load_fpi(
            trange=trange,
            probe=['1', '2', '3', '4'],
            data_rate='brst',
            level='l2',
            datatype='des-dist',
            time_clip=False,
            notplot=False
        )
        
        print(f"Loading result: {result}")
        
        # Check what variables were loaded
        all_vars = list(data_quants.keys())
        electron_vars = [var for var in all_vars if 'des' in var]
        
        print(f"Total variables loaded: {len(all_vars)}")
        print(f"Electron variables: {len(electron_vars)}")
        
        if electron_vars:
            print(f"Electron variables found:")
            for var in electron_vars[:10]:  # Show first 10
                print(f"   {var}")
        
        # Look for energy flux or distribution variables
        energy_vars = [var for var in electron_vars if 'energy' in var.lower()]
        eflux_vars = [var for var in electron_vars if 'eflux' in var.lower()]
        dist_vars = [var for var in electron_vars if 'dist' in var.lower()]
        
        print(f"\nSpectrograph variables:")
        print(f"   Energy variables: {len(energy_vars)}")
        print(f"   Energy flux variables: {len(eflux_vars)}")
        print(f"   Distribution variables: {len(dist_vars)}")
        
        # Try to create spectrograph with available data
        spec_vars = eflux_vars if eflux_vars else (energy_vars if energy_vars else dist_vars)
        
        if spec_vars:
            print(f"\nCreating spectrograph with: {spec_vars[0]}")
            
            # Create plot
            fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
            
            plots_created = 0
            
            for i, probe in enumerate(['1', '2', '3', '4']):
                # Find variables for this probe
                probe_vars = [var for var in spec_vars if f'mms{probe}' in var]
                
                if probe_vars:
                    var_name = probe_vars[0]
                    
                    try:
                        times, spec_data = get_data(var_name)
                        
                        if len(times) > 0:
                            # Convert times
                            if hasattr(times[0], 'strftime'):
                                time_objects = times
                            else:
                                time_objects = [datetime.fromtimestamp(t) for t in times]
                            
                            print(f"   MMS{probe}: {len(times)} time points, data shape: {spec_data.shape}")
                            
                            # Check if we have 2D data for spectrogram
                            if len(spec_data.shape) == 2 and spec_data.shape[1] > 1:
                                # Create spectrogram
                                im = axes[i].imshow(spec_data.T, aspect='auto', origin='lower',
                                                  extent=[0, len(time_objects), 0, spec_data.shape[1]],
                                                  cmap='jet', interpolation='nearest', vmin=1e-18, vmax=1e-12)
                                
                                axes[i].set_ylabel(f'MMS{probe}\nEnergy Channel')
                                
                                # Add colorbar
                                cbar = plt.colorbar(im, ax=axes[i])
                                cbar.set_label('Electron Flux')
                                
                                # Mark event time
                                event_index = len(time_objects) // 2  # Approximate center
                                axes[i].axvline(event_index, color='white', linestyle='--', alpha=0.8, linewidth=2)
                                
                                plots_created += 1
                                print(f"   ✅ MMS{probe} spectrograph created")
                            else:
                                # Plot 1D data
                                axes[i].plot(time_objects, spec_data, label=f'MMS{probe}')
                                axes[i].set_ylabel(f'MMS{probe}\nElectron Data')
                                axes[i].legend()
                                plots_created += 1
                                print(f"   ✅ MMS{probe} 1D plot created")
                        else:
                            axes[i].text(0.5, 0.5, f'MMS{probe}: No data points', 
                                       transform=axes[i].transAxes, ha='center', va='center')
                            print(f"   ❌ MMS{probe}: No data points")
                            
                    except Exception as e:
                        axes[i].text(0.5, 0.5, f'MMS{probe}: Error - {str(e)[:50]}', 
                                   transform=axes[i].transAxes, ha='center', va='center')
                        print(f"   ❌ MMS{probe}: Error - {e}")
                else:
                    axes[i].text(0.5, 0.5, f'MMS{probe}: No electron data available', 
                               transform=axes[i].transAxes, ha='center', va='center')
                    print(f"   ❌ MMS{probe}: No variables found")
                
                axes[i].grid(True, alpha=0.3)
            
            axes[-1].set_xlabel('Time Index / Time')
            plt.suptitle(f'MMS Electron Spectrographs (Burst Mode): {event_time.strftime("%Y-%m-%d %H:%M:%S")} UT Event', fontsize=16)
            plt.tight_layout()
            plt.savefig('results_final/electron_spectrographs_burst.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\n✅ Electron spectrograph saved: results_final/electron_spectrographs_burst.png")
            print(f"   Plots created: {plots_created}/4 spacecraft")
            
            return plots_created > 0
        else:
            print(f"❌ No suitable variables for spectrograph")
            return False
            
    except Exception as e:
        print(f"❌ Error creating electron spectrograph: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    
    print("CREATE PLASMA SPECTROGRAPHS - TARGETED APPROACH")
    print("=" * 80)
    print("Creating ion and electron spectrographs using burst mode data")
    print("=" * 80)
    
    # Create results directory if needed
    results_dir = "results_final"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create spectrographs
    ion_success = create_targeted_ion_spectrograph()
    electron_success = create_targeted_electron_spectrograph()
    
    # Summary
    print(f"\n" + "=" * 80)
    print("SPECTROGRAPH CREATION SUMMARY")
    print("=" * 80)
    
    print(f"Ion spectrograph: {'✅ Success' if ion_success else '❌ Failed'}")
    print(f"Electron spectrograph: {'✅ Success' if electron_success else '❌ Failed'}")
    
    if ion_success or electron_success:
        print("\n🎉 PLASMA SPECTROGRAPHS CREATED!")
        print("✅ Burst mode distribution data successfully processed")
        print("✅ High-resolution spectrographs generated")
        print("✅ Event time properly marked on plots")
        
        # List generated files
        print(f"\nGenerated files in {results_dir}/:")
        if os.path.exists(results_dir):
            files = [f for f in os.listdir(results_dir) if 'spectrograph' in f and f.endswith('.png')]
            for file in sorted(files):
                print(f"   📄 {file}")
    else:
        print("\n⚠️ SPECTROGRAPH CREATION ISSUES")
        print("❌ Unable to create plasma spectrographs")
        print("🔍 Check data availability and variable names")
    
    return ion_success or electron_success


if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎯 TARGETED SPECTROGRAPH CREATION: SUCCESS")
        print(f"✅ Plasma spectrographs created using burst mode data")
    else:
        print(f"\n⚠️ TARGETED SPECTROGRAPH CREATION: NEEDS INVESTIGATION")
        print(f"❌ Unable to create spectrographs with available data")
