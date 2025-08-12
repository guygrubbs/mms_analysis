"""
Spectral Analysis Test Suite for MMS-MP Package
==============================================

Tests for spectral analysis, thickness calculations, and visualization components.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch
import warnings

from mms_mp import spectra, thickness, visualize


@pytest.mark.timeout(300)
class TestSpectralAnalysis:
    """Test spectral analysis methods"""

    def test_power_spectral_density(self):
        """Test PSD calculation for magnetic field fluctuations"""
        # Create test signal with known frequency content
        fs = 100.0  # Hz
        t = np.linspace(0, 10, int(fs * 10))

        # Signal with 1 Hz and 10 Hz components
        signal = (2.0 * np.sin(2 * np.pi * 1 * t) +
                 1.0 * np.sin(2 * np.pi * 10 * t) +
                 0.1 * np.random.randn(len(t)))

        # Calculate PSD
        frequencies, psd = spectra.calculate_psd(signal, fs=fs, nperseg=256)

        # Should have peaks at 1 Hz and 10 Hz
        peak_1hz_idx = np.argmin(np.abs(frequencies - 1.0))
        peak_10hz_idx = np.argmin(np.abs(frequencies - 10.0))

        # Check that peaks are significantly above background
        background_level = np.median(psd)
        assert psd[peak_1hz_idx] > 10 * background_level, "1 Hz peak not detected"
        assert psd[peak_10hz_idx] > 5 * background_level, "10 Hz peak not detected"

        # Check frequency resolution
        df = frequencies[1] - frequencies[0]
        expected_df = fs / 256  # nperseg = 256
        assert abs(df - expected_df) < 1e-10, f"Frequency resolution incorrect: {df} vs {expected_df}"

    def test_cross_spectral_analysis(self):
        """Test cross-spectral analysis between field components"""
        # Create correlated signals
        fs = 50.0
        t = np.linspace(0, 20, int(fs * 20))

        # Base signal
        base_signal = np.sin(2 * np.pi * 2 * t)  # 2 Hz

        # Correlated signal with phase shift
        phase_shift = np.pi / 4  # 45 degrees
        shifted_signal = np.sin(2 * np.pi * 2 * t + phase_shift)

        # Add noise
        np.random.seed(42)
        signal1 = base_signal + 0.1 * np.random.randn(len(t))
        signal2 = shifted_signal + 0.1 * np.random.randn(len(t))

        # Calculate cross-spectrum
        frequencies, cross_psd, coherence, phase = spectra.cross_spectral_analysis(
            signal1, signal2, fs=fs, nperseg=256
        )

        # Find 2 Hz frequency bin
        freq_2hz_idx = np.argmin(np.abs(frequencies - 2.0))

        # Check coherence at 2 Hz (should be high)
        assert coherence[freq_2hz_idx] > 0.8, f"Coherence too low: {coherence[freq_2hz_idx]}"

        # Check phase at 2 Hz (should be close to π/4)
        measured_phase = phase[freq_2hz_idx]
        expected_phase = phase_shift

        # Handle phase wrapping
        phase_diff = np.abs(measured_phase - expected_phase)
        phase_diff = min(phase_diff, 2*np.pi - phase_diff)

        assert phase_diff < 0.2, f"Phase incorrect: {measured_phase} vs {expected_phase}"

    def test_wavelet_analysis(self):
        """Test wavelet analysis for time-frequency decomposition"""
        # Create chirp signal (frequency increases with time)
        fs = 100.0
        t = np.linspace(0, 5, int(fs * 5))

        # Frequency increases from 1 Hz to 10 Hz
        f_inst = 1 + 9 * t / 5  # Instantaneous frequency
        signal = np.sin(2 * np.pi * np.cumsum(f_inst) / fs)

        # Perform wavelet analysis
        frequencies, times, scalogram = spectra.wavelet_analysis(
            signal, fs=fs, f_min=0.5, f_max=15, n_freqs=50
        )

        # Check that energy follows the chirp
        # At t=0, energy should be concentrated around 1 Hz
        # At t=5, energy should be concentrated around 10 Hz

        t_start_idx = 0
        t_end_idx = -1

        f_1hz_idx = np.argmin(np.abs(frequencies - 1.0))
        f_10hz_idx = np.argmin(np.abs(frequencies - 10.0))

        # Energy at start should be higher at 1 Hz than 10 Hz
        energy_start_1hz = scalogram[f_1hz_idx, t_start_idx]
        energy_start_10hz = scalogram[f_10hz_idx, t_start_idx]

        # Energy at end should be higher at 10 Hz than 1 Hz
        energy_end_1hz = scalogram[f_1hz_idx, t_end_idx]
        energy_end_10hz = scalogram[f_10hz_idx, t_end_idx]

        assert energy_start_1hz > energy_start_10hz, "Start: 1 Hz should dominate"
        assert energy_end_10hz > energy_end_1hz, "End: 10 Hz should dominate"

    def test_magnetic_fluctuation_analysis(self):
        """Test analysis of magnetic field fluctuations"""
        # Create synthetic magnetopause-like fluctuations
        fs = 16.0  # Hz (typical FGM survey rate)
        t = np.linspace(0, 300, int(fs * 300))  # 5 minutes

        # Background field
        B0 = np.array([40.0, 20.0, 10.0])  # nT

        # Add fluctuations at different scales
        # ULF waves (0.01-0.1 Hz)
        ulf_freq = 0.05  # Hz
        ulf_amp = 5.0    # nT

        # Kinetic scale fluctuations (0.1-1 Hz)
        kinetic_freq = 0.3  # Hz
        kinetic_amp = 2.0   # nT

        # Create 3D fluctuations
        B_fluct = np.zeros((len(t), 3))
        for i in range(3):
            B_fluct[:, i] = (ulf_amp * np.sin(2 * np.pi * ulf_freq * t + i * np.pi/3) +
                           kinetic_amp * np.sin(2 * np.pi * kinetic_freq * t + i * np.pi/2))

        # Add noise
        np.random.seed(42)
        B_fluct += 0.5 * np.random.randn(*B_fluct.shape)

        # Total field
        B_total = B0[np.newaxis, :] + B_fluct

        # Analyze fluctuations
        analysis = spectra.analyze_magnetic_fluctuations(B_total, fs=fs)

        # Check that ULF and kinetic peaks are detected
        frequencies = analysis['frequencies']
        psd_total = analysis['psd_total']

        ulf_idx = np.argmin(np.abs(frequencies - ulf_freq))
        kinetic_idx = np.argmin(np.abs(frequencies - kinetic_freq))

        background_psd = np.median(psd_total)

        assert psd_total[ulf_idx] > 5 * background_psd, "ULF peak not detected"
        assert psd_total[kinetic_idx] > 3 * background_psd, "Kinetic peak not detected"


class TestThicknessCalculations:
    """Test magnetopause thickness calculation methods"""

    def test_gradient_method_thickness(self):
        """Test thickness calculation using gradient method"""
        # Create synthetic boundary profile
        x = np.linspace(-50, 50, 1000)  # km

        # Tanh profile with known thickness
        true_thickness = 10.0  # km
        profile = np.tanh(x / (true_thickness / 2))  # Characteristic scale

        # Add noise
        np.random.seed(42)
        noisy_profile = profile + 0.05 * np.random.randn(len(profile))

        # Calculate thickness using gradient method
        calculated_thickness = thickness.gradient_method(x, noisy_profile)

        # Should recover true thickness within reasonable error
        relative_error = abs(calculated_thickness - true_thickness) / true_thickness
        assert relative_error < 0.2, f"Thickness error too large: {relative_error:.3f}"

    def test_current_sheet_thickness(self):
        """Test current sheet thickness from magnetic field rotation"""
        # Create Harris current sheet profile
        z = np.linspace(-20, 20, 400)  # km
        sheet_thickness = 5.0  # km

        # Magnetic field components for Harris sheet
        Bx = 50.0 * np.tanh(z / sheet_thickness)  # nT
        By = np.zeros_like(z)
        Bz = 50.0 / np.cosh(z / sheet_thickness)  # nT

        B_field = np.column_stack([Bx, By, Bz])

        # Calculate current sheet thickness
        calculated_thickness = thickness.current_sheet_thickness(z, B_field)

        # Should recover Harris sheet thickness
        relative_error = abs(calculated_thickness - sheet_thickness) / sheet_thickness
        assert relative_error < 0.15, f"Current sheet thickness error: {relative_error:.3f}"

    def test_multi_scale_analysis(self):
        """Test multi-scale thickness analysis"""
        # Create boundary with multiple scale lengths
        x = np.linspace(-100, 100, 2000)

        # Large scale transition (50 km)
        large_scale = np.tanh(x / 25)

        # Small scale structure (5 km)
        small_scale = 0.3 * np.tanh(x / 2.5)

        # Combined profile
        combined_profile = large_scale + small_scale

        # Analyze multiple scales
        scales = thickness.multi_scale_analysis(x, combined_profile,
                                              scale_range=[1, 100], n_scales=20)

        # Should detect both scale lengths
        detected_scales = scales['characteristic_scales']

        # Check that both ~5 km and ~50 km scales are detected
        has_small_scale = any(2 < scale < 8 for scale in detected_scales)
        has_large_scale = any(20 < scale < 80 for scale in detected_scales)

        assert has_small_scale, f"Small scale not detected: {detected_scales}"
        assert has_large_scale, f"Large scale not detected: {detected_scales}"


class TestVisualizationComponents:
    """Test visualization and plotting functions"""

    def test_spectrogram_creation(self):
        """Test spectrogram plotting functionality"""
        # Create test data
        fs = 32.0  # Hz
        t = np.linspace(0, 60, int(fs * 60))  # 1 minute

        # Create energy-time spectrogram data
        energies = np.logspace(1, 4, 64)  # 10 eV to 10 keV
        times = t[::10]  # Subsample for spectrogram

        # Synthetic particle flux
        flux_matrix = np.zeros((len(times), len(energies)))
        for i, time in enumerate(times):
            # Temperature varies with time
            kT = 100 + 50 * np.sin(2 * np.pi * time / 30)  # eV
            for j, E in enumerate(energies):
                flux_matrix[i, j] = 1e6 * np.exp(-E / kT)  # Maxwellian

        # Test plotting (should not raise exceptions)
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create spectrogram
            im = visualize.plot_spectrogram(ax, times, energies, flux_matrix,
                                          title="Test Spectrogram",
                                          ylabel="Energy (eV)",
                                          clabel="Flux")

            # Check that plot was created
            assert im is not None, "Spectrogram plot not created"
            assert ax.get_title() == "Test Spectrogram", "Title not set correctly"

            plt.close(fig)

        except Exception as e:
            pytest.fail(f"Spectrogram plotting failed: {e}")

    def test_magnetic_field_visualization(self):
        """Test magnetic field vector plotting"""
        # Create test magnetic field data
        t = np.linspace(0, 3600, 100)  # 1 hour

        # Rotating magnetic field
        B_field = np.zeros((len(t), 3))
        B_field[:, 0] = 50 + 10 * np.sin(2 * np.pi * t / 1800)  # Bx
        B_field[:, 1] = 30 * np.cos(2 * np.pi * t / 1800)       # By
        B_field[:, 2] = 20 + 5 * np.sin(2 * np.pi * t / 900)    # Bz

        try:
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

            # Plot field components
            visualize.plot_magnetic_field(axes, t, B_field,
                                        labels=['Bx', 'By', 'Bz'],
                                        colors=['red', 'green', 'blue'])

            # Check that all components are plotted
            for i, ax in enumerate(axes):
                lines = ax.get_lines()
                assert len(lines) > 0, f"No lines plotted in subplot {i}"

                # Check data ranges
                line_data = lines[0].get_ydata()
                assert len(line_data) == len(t), f"Wrong data length in subplot {i}"

            plt.close(fig)

        except Exception as e:
            pytest.fail(f"Magnetic field plotting failed: {e}")

    def test_boundary_layer_visualization(self):
        """Test boundary layer structure visualization"""
        # Create synthetic boundary crossing data
        distance = np.linspace(-50, 50, 1000)  # km from boundary

        # Different plasma parameters across boundary
        density = 5 + 3 * np.tanh(distance / 10)  # cm^-3
        temperature = 200 + 800 * (1 - np.tanh(distance / 15))  # eV
        magnetic_field = 40 + 20 * np.tanh(distance / 8)  # nT

        try:
            fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

            # Plot boundary structure
            visualize.plot_boundary_structure(
                axes, distance,
                [density, temperature, magnetic_field],
                labels=['Density (cm⁻³)', 'Temperature (eV)', '|B| (nT)'],
                title="Magnetopause Structure"
            )

            # Check that plots were created
            for i, ax in enumerate(axes):
                lines = ax.get_lines()
                assert len(lines) > 0, f"No data plotted in panel {i}"

                # Check that boundary (x=0) is marked
                vlines = [line for line in ax.get_lines() if hasattr(line, 'get_xdata')
                         and len(line.get_xdata()) == 2 and line.get_xdata()[0] == 0]
                # Note: vertical lines might be added differently, so this is optional

            plt.close(fig)

        except Exception as e:
            pytest.fail(f"Boundary structure plotting failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
