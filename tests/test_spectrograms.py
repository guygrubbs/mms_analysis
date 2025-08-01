#!/usr/bin/env python3
"""
Test suite for MMS ion and electron spectrogram generation
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys

# Add the parent directory to the path to import mms_mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mms_mp.spectra as spectra


class TestSpectrogramGeneration:
    """Test spectrogram generation functions"""
    
    def setup_method(self):
        """Set up test data"""
        # Create synthetic test data
        self.n_times = 50
        self.n_energies = 32
        
        # Time array
        start_time = datetime(2019, 1, 27, 12, 25, 0)
        self.times = [start_time + timedelta(seconds=i*10) for i in range(self.n_times)]
        
        # Energy array (10 eV to 30 keV)
        self.energies = np.logspace(1, 4.5, self.n_energies)
        
        # Synthetic flux data
        self.ion_flux = np.random.lognormal(10, 2, (self.n_times, self.n_energies))
        self.electron_flux = np.random.lognormal(12, 1.5, (self.n_times, self.n_energies))
    
    def test_generic_spectrogram(self):
        """Test generic spectrogram function"""
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Test basic spectrogram creation
        result = spectra.generic_spectrogram(
            np.array(self.times), 
            self.energies, 
            self.ion_flux,
            ax=ax,
            show=False,
            return_axes=True,
            title="Test Spectrogram"
        )
        
        # Check that result is returned
        assert result is not None
        assert len(result) == 2  # (ax, colorbar)
        
        # Check plot properties
        assert ax.get_title() == "Test Spectrogram"
        assert ax.get_ylabel() == "Energy (eV)"
        assert ax.get_yscale() == 'log'
        
        plt.close(fig)
    
    def test_fpi_ion_spectrogram(self):
        """Test FPI ion spectrogram function"""
        
        # Create 4D flux data (time, energy, phi, theta)
        flux_4d = np.random.lognormal(10, 2, (self.n_times, self.n_energies, 16, 8))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Test ion spectrogram
        result = spectra.fpi_ion_spectrogram(
            np.array(self.times),
            self.energies,
            flux_4d,
            ax=ax,
            show=False,
            return_axes=True
        )
        
        # Check that result is returned
        assert result is not None
        
        # Check plot properties
        assert ax.get_title() == "Ion energy flux"
        assert ax.get_ylabel() == "E$_i$ (eV)"
        
        plt.close(fig)
    
    def test_fpi_electron_spectrogram(self):
        """Test FPI electron spectrogram function"""
        
        # Create 4D flux data (time, energy, phi, theta)
        flux_4d = np.random.lognormal(12, 1.5, (self.n_times, self.n_energies, 16, 8))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Test electron spectrogram
        result = spectra.fpi_electron_spectrogram(
            np.array(self.times),
            self.energies,
            flux_4d,
            ax=ax,
            show=False,
            return_axes=True
        )
        
        # Check that result is returned
        assert result is not None
        
        # Check plot properties
        assert ax.get_title() == "Electron energy flux"
        assert ax.get_ylabel() == "E$_e$ (eV)"
        
        plt.close(fig)
    
    def test_hpca_ion_spectrogram(self):
        """Test HPCA ion spectrogram function"""
        
        # Create 2D omnidirectional data (time, energy)
        omni_data = np.random.lognormal(8, 2, (self.n_times, self.n_energies))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Test HPCA spectrogram
        result = spectra.hpca_ion_spectrogram(
            np.array(self.times),
            self.energies,
            omni_data,
            species='H+',
            ax=ax,
            show=False,
            return_axes=True
        )
        
        # Check that result is returned
        assert result is not None
        
        # Check plot properties
        assert ax.get_title() == "HPCA H+ energy flux"
        assert ax.get_ylabel() == "E(H+) (eV)"
        
        plt.close(fig)
    
    def test_spectrogram_with_mask(self):
        """Test spectrogram with quality mask"""
        
        # Create quality mask (some bad data points)
        mask = np.random.choice([True, False], size=(self.n_times, self.n_energies), p=[0.8, 0.2])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Test spectrogram with mask
        result = spectra.generic_spectrogram(
            np.array(self.times),
            self.energies,
            self.ion_flux,
            mask=mask,
            ax=ax,
            show=False,
            return_axes=True
        )
        
        # Check that result is returned
        assert result is not None
        
        plt.close(fig)
    
    def test_spectrogram_parameters(self):
        """Test spectrogram with various parameters"""
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Test with custom parameters
        result = spectra.generic_spectrogram(
            np.array(self.times),
            self.energies,
            self.ion_flux,
            log10=True,
            vmin=1e3,
            vmax=1e6,
            cmap='plasma',
            ylabel='Custom Energy (eV)',
            title='Custom Title',
            ax=ax,
            show=False,
            return_axes=True
        )
        
        # Check that result is returned
        assert result is not None
        
        # Check custom properties
        assert ax.get_title() == "Custom Title"
        assert ax.get_ylabel() == "Custom Energy (eV)"
        
        plt.close(fig)
    
    def test_energy_range_validation(self):
        """Test that energy ranges are handled correctly"""
        
        # Test with different energy ranges
        energies_low = np.linspace(10, 100, 20)  # Linear low energy
        energies_high = np.logspace(3, 5, 20)    # Log high energy
        
        flux_low = np.random.lognormal(8, 1, (self.n_times, 20))
        flux_high = np.random.lognormal(6, 1, (self.n_times, 20))
        
        # Test low energy range
        fig1, ax1 = plt.subplots()
        result1 = spectra.generic_spectrogram(
            np.array(self.times), energies_low, flux_low,
            ax=ax1, show=False, return_axes=True
        )
        assert result1 is not None
        plt.close(fig1)
        
        # Test high energy range
        fig2, ax2 = plt.subplots()
        result2 = spectra.generic_spectrogram(
            np.array(self.times), energies_high, flux_high,
            ax=ax2, show=False, return_axes=True
        )
        assert result2 is not None
        plt.close(fig2)


class TestSpectrogramScript:
    """Test the main spectrogram generation script"""
    
    def test_script_imports(self):
        """Test that the script can be imported without errors"""
        
        # Try to import the script
        try:
            import create_mms_spectrograms_2019_01_27
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import spectrogram script: {e}")
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation function"""
        
        try:
            from create_mms_spectrograms_2019_01_27 import create_synthetic_flux_data
            
            trange = ['2019-01-27/12:25:00', '2019-01-27/12:35:00']
            data = create_synthetic_flux_data(trange)
            
            # Check data structure
            assert 'mode' in data
            assert 'times' in data
            assert 'energies' in data
            assert 'ion_flux' in data
            assert 'electron_flux' in data
            
            # Check data shapes
            assert len(data['times']) > 0
            assert len(data['energies']) > 0
            assert data['ion_flux'].shape[0] == len(data['times'])
            assert data['ion_flux'].shape[1] == len(data['energies'])
            assert data['electron_flux'].shape[0] == len(data['times'])
            assert data['electron_flux'].shape[1] == len(data['energies'])
            
        except ImportError:
            pytest.skip("Spectrogram script not available for testing")


if __name__ == "__main__":
    pytest.main([__file__])
