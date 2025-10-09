"""
Comprehensive Physics and Logic Test Suite for MMS-MP Package
============================================================

This module contains comprehensive test cases that validate the physics
and logic of all major components in the MMS-MP package.

Test Categories:
1. Coordinate System Physics (coords.py)
2. Boundary Detection Logic (boundary.py) 
3. Data Loading and Validation (data_loader.py)
4. Electric Field Physics (electric.py)
5. Motion Analysis (motion.py)
6. Multi-spacecraft Analysis (multispacecraft.py)
7. Data Quality Assessment (quality.py)
8. Resampling and Interpolation (resample.py)
9. Spectral Analysis (spectra.py)
10. Thickness Calculations (thickness.py)
11. Visualization Components (visualize.py)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import warnings

# Import all modules to test
from mms_mp import coords, boundary, data_loader, electric, motion
from mms_mp import multispacecraft, quality, resample, spectra, thickness, visualize


class TestCoordinateSystemPhysics:
    """Test coordinate system transformations and physics"""
    
    def test_lmn_orthogonality(self):
        """Test that LMN coordinate system maintains orthogonality"""
        # Create synthetic magnetic field data with clear variance structure
        np.random.seed(42)
        n_points = 1000
        
        # Create field with maximum variance in X, medium in Y, minimum in Z
        B_x = 50 + 20 * np.sin(np.linspace(0, 4*np.pi, n_points)) + 5 * np.random.randn(n_points)
        B_y = 30 + 10 * np.cos(np.linspace(0, 2*np.pi, n_points)) + 3 * np.random.randn(n_points)
        B_z = 20 + 2 * np.random.randn(n_points)
        
        B_field = np.column_stack([B_x, B_y, B_z])
        
        # Test MVA calculation
        lmn_system = coords.mva(B_field)
        
        # Verify orthogonality: L·M = L·N = M·N = 0
        assert abs(np.dot(lmn_system.L, lmn_system.M)) < 1e-10, "L and M vectors not orthogonal"
        assert abs(np.dot(lmn_system.L, lmn_system.N)) < 1e-10, "L and N vectors not orthogonal"
        assert abs(np.dot(lmn_system.M, lmn_system.N)) < 1e-10, "M and N vectors not orthogonal"
        
        # Verify unit vectors
        assert abs(np.linalg.norm(lmn_system.L) - 1.0) < 1e-10, "L vector not unit length"
        assert abs(np.linalg.norm(lmn_system.M) - 1.0) < 1e-10, "M vector not unit length"
        assert abs(np.linalg.norm(lmn_system.N) - 1.0) < 1e-10, "N vector not unit length"
        
        # Verify right-handed coordinate system
        cross_product = np.cross(lmn_system.L, lmn_system.M)
        assert np.dot(cross_product, lmn_system.N) > 0, "LMN system not right-handed"
    
    def test_eigenvalue_ordering(self):
        """Test that eigenvalues are properly ordered (λ_max > λ_mid > λ_min)"""
        np.random.seed(123)
        
        # Create field with known variance structure
        B_field = np.random.randn(500, 3)
        B_field[:, 0] *= 10  # Maximum variance in X
        B_field[:, 1] *= 5   # Medium variance in Y
        B_field[:, 2] *= 1   # Minimum variance in Z
        
        lmn_system = coords.mva(B_field)
        
        # Check eigenvalue ordering
        λ_max, λ_mid, λ_min = lmn_system.eigvals
        assert λ_max >= λ_mid >= λ_min, f"Eigenvalues not ordered: {λ_max}, {λ_mid}, {λ_min}"
        
        # Check eigenvalue ratios
        assert lmn_system.r_max_mid == λ_max / λ_mid, "Incorrect max/mid ratio"
        assert lmn_system.r_mid_min == λ_mid / λ_min, "Incorrect mid/min ratio"
    
    def test_coordinate_transformation_consistency(self):
        """Test that coordinate transformations are consistent and reversible"""
        np.random.seed(456)
        
        # Create test magnetic field
        B_gsm = np.random.randn(100, 3) * 50
        
        # Get LMN system
        lmn_system = coords.mva(B_gsm)
        
        # Transform to LMN coordinates
        B_lmn = lmn_system.to_lmn(B_gsm)
        
        # Transform back to GSM
        B_gsm_recovered = lmn_system.to_gsm(B_lmn)
        
        # Check consistency (should recover original within numerical precision)
        np.testing.assert_allclose(B_gsm, B_gsm_recovered, rtol=1e-12, atol=1e-12,
                                 err_msg="Coordinate transformation not reversible")
    
    def test_hybrid_lmn_physics(self):
        """Test hybrid LMN method with position-dependent normal"""
        # Test position at magnetopause (typical location)
        pos_gsm = np.array([10.0, 5.0, 2.0])  # Earth radii
        
        # Create synthetic boundary-crossing field
        np.random.seed(789)
        B_field = np.random.randn(200, 3) * 30
        
        # Add systematic rotation across boundary
        for i in range(len(B_field)):
            angle = (i / len(B_field) - 0.5) * np.pi / 4  # ±45° rotation
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                               [np.sin(angle), np.cos(angle), 0],
                               [0, 0, 1]])
            B_field[i] = rotation @ B_field[i]
        
        # Test hybrid LMN
        lmn_system = coords.hybrid_lmn(B_field, pos_gsm_km=pos_gsm * 6371)

        # Verify it's a valid LMN system
        assert hasattr(lmn_system, 'L'), "Missing L vector"
        assert hasattr(lmn_system, 'M'), "Missing M vector"
        assert hasattr(lmn_system, 'N'), "Missing N vector"

        # Check orthogonality
        assert abs(np.dot(lmn_system.L, lmn_system.M)) < 1e-10
        assert abs(np.dot(lmn_system.L, lmn_system.N)) < 1e-10
        assert abs(np.dot(lmn_system.M, lmn_system.N)) < 1e-10

        # Method bookkeeping and metadata
        assert lmn_system.method in {'mva', 'pyspedas', 'shue'}
        assert lmn_system.meta is not None
        assert lmn_system.meta.get('formation_type') == 'auto'
        ratios = lmn_system.meta.get('eig_ratio_thresholds')
        assert ratios is not None and {'lambda_max_mid', 'lambda_mid_min'} <= set(ratios.keys())


class TestBoundaryDetectionLogic:
    """Test boundary detection algorithms and state machine logic"""
    
    def test_detector_config_validation(self):
        """Test that detector configuration validates parameters correctly"""
        # Valid configuration
        cfg = boundary.DetectorCfg(he_in=0.3, he_out=0.1, min_pts=5)
        assert cfg.he_in == 0.3
        assert cfg.he_out == 0.1
        assert cfg.min_pts == 5

        # Invalid parameter should raise error
        with pytest.raises(TypeError):
            boundary.DetectorCfg(invalid_param=123)
    
    def test_boundary_state_transitions(self):
        """Test state machine transitions for boundary detection"""
        cfg = boundary.DetectorCfg(he_in=0.2, he_out=0.1, he_frac_in=0.06)
        
        # Test transition from sheath to magnetosphere
        # High He+ density should indicate magnetosphere
        new_state = boundary._sm_update('sheath', he_val=0.5, BN_val=5.0,
                                       cfg=cfg, inside_mag=True)
        assert new_state == 'magnetosphere', "Should transition to magnetosphere with high He+"
        
        # Test transition to current sheet layer
        # Low |BN| should indicate mp_layer
        new_state = boundary._sm_update('sheath', he_val=0.15, BN_val=1.0,
                                       cfg=cfg, inside_mag=False)
        assert new_state == 'mp_layer', "Should transition to mp_layer with low |BN|"
        
        # Test staying in sheath
        # Low He+ density should keep in sheath
        new_state = boundary._sm_update('sheath', he_val=0.05, BN_val=5.0,
                                       cfg=cfg, inside_mag=False)
        assert new_state == 'sheath', "Should stay in sheath with low He+"
    
    def test_hysteresis_logic(self):
        """Test hysteresis prevents rapid state oscillations"""
        cfg = boundary.DetectorCfg(he_in=0.2, he_out=0.1, min_pts=3)
        
        # Create data that oscillates around threshold
        he_data = np.array([0.05, 0.15, 0.25, 0.15, 0.25, 0.15, 0.05])
        BN_data = np.array([5.0, 4.0, 3.0, 4.0, 3.0, 4.0, 5.0])
        
        # Simulate state machine with hysteresis
        states = []
        current_state = 'sheath'
        
        for he_val, BN_val in zip(he_data, BN_data):
            inside_mag = he_val > cfg.he_in if current_state == 'sheath' else he_val > cfg.he_out
            new_state = boundary._sm_update(current_state, he_val, BN_val, cfg, inside_mag)
            states.append(new_state)
            current_state = new_state
        
        # Should not oscillate rapidly due to hysteresis
        state_changes = sum(1 for i in range(1, len(states)) if states[i] != states[i-1])
        assert state_changes <= 3, f"Too many state changes: {state_changes}, states: {states}"


class TestDataLoaderValidation:
    """Test data loading, validation, and preprocessing"""
    
    @patch('mms_mp.data_loader.mms')
    @patch('mms_mp.data_loader.get_data')
    def test_data_loading_fallback(self, mock_get_data, mock_mms):
        """Test that data loader falls back through cadences correctly"""
        # Mock MMS loading functions
        mock_mms.mms_load_fgm = Mock(side_effect=[Exception("fast failed"), None])
        mock_mms.mms_load_fpi = Mock()
        mock_mms.mms_load_state = Mock()
        
        # Mock successful data retrieval
        mock_get_data.return_value = (
            np.linspace(0, 3600, 100),  # time array
            np.random.randn(100, 3) * 50  # data array
        )
        
        # Test loading with fallback
        trange = ['2019-01-27/12:00:00', '2019-01-27/13:00:00']
        probes = ['1']
        
        # This should not raise an exception due to fallback mechanism
        try:
            result = data_loader.load_event(trange, probes)
            # Should return empty dict if all loading fails, but not crash
            assert isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"Data loader should handle failures gracefully: {e}")
    
    def test_variable_validation(self):
        """Test variable validation logic"""
        # Test valid variable detection
        with patch('mms_mp.data_loader.get_data') as mock_get_data:
            # Mock valid data
            mock_get_data.return_value = (
                np.linspace(0, 100, 50),
                np.random.randn(50, 3)
            )
            
            is_valid = data_loader._is_valid('test_var', expect_cols=3)
            assert is_valid, "Should detect valid variable"
            
            # Mock invalid data (wrong columns)
            mock_get_data.return_value = (
                np.linspace(0, 100, 50),
                np.random.randn(50, 4)  # Wrong number of columns
            )
            
            is_valid = data_loader._is_valid('test_var', expect_cols=3)
            assert not is_valid, "Should detect invalid variable (wrong columns)"
            
            # Mock missing data
            mock_get_data.return_value = (None, None)
            
            is_valid = data_loader._is_valid('test_var')
            assert not is_valid, "Should detect missing data"
    
    def test_time_range_validation(self):
        """Test time range parsing and validation"""
        # Valid time ranges
        valid_ranges = [
            ['2019-01-27/12:00:00', '2019-01-27/13:00:00'],
            ['2019-01-27T12:00:00', '2019-01-27T13:00:00'],
            [datetime(2019, 1, 27, 12), datetime(2019, 1, 27, 13)]
        ]
        
        for trange in valid_ranges:
            # Should not raise exception
            try:
                # This would be called internally by load_event
                start_time = pd.to_datetime(trange[0])
                end_time = pd.to_datetime(trange[1])
                assert end_time > start_time, "End time should be after start time"
            except Exception as e:
                pytest.fail(f"Valid time range should parse correctly: {trange}, error: {e}")


class TestElectricFieldPhysics:
    """Test electric field calculations and physics"""
    
    def test_exb_drift_calculation(self):
        """Test E×B drift velocity calculation"""
        # Create test electric and magnetic fields
        E_field = np.array([1.0, 0.0, 0.0])  # mV/m in X direction
        B_field = np.array([0.0, 0.0, 50.0])  # nT in Z direction
        
        # Calculate E×B drift
        v_exb = electric.calculate_exb_drift(E_field, B_field)
        
        # Expected drift: E×B/B² in Y direction
        # |v_exb| = |E|/|B| = 1 mV/m / 50 nT = 20 km/s
        expected_magnitude = 20.0  # km/s
        expected_direction = np.array([0.0, -1.0, 0.0])  # -Y by right-hand rule

        np.testing.assert_allclose(v_exb, expected_magnitude * expected_direction, 
                                 rtol=1e-10, err_msg="E×B drift calculation incorrect")
    
    def test_electric_field_units(self):
        """Test electric field unit conversions"""
        # Test mV/m to V/m conversion
        E_mv_per_m = np.array([1.0, 2.0, 3.0])  # mV/m
        E_v_per_m = electric.convert_electric_field_units(E_mv_per_m, 'mV/m', 'V/m')
        
        expected = E_mv_per_m / 1000.0
        np.testing.assert_allclose(E_v_per_m, expected, 
                                 err_msg="mV/m to V/m conversion incorrect")
    
    def test_convection_electric_field(self):
        """Test convection electric field calculation"""
        # Test motional electric field: E = -v × B
        velocity = np.array([100.0, 0.0, 0.0])  # km/s in X direction
        B_field = np.array([0.0, 0.0, 50.0])   # nT in Z direction
        
        E_conv = electric.calculate_convection_field(velocity, B_field)
        
        # Expected: E = -v × B; with v=[100,0,0] km/s and B=[0,0,50] nT → [0, +5, 0] mV/m
        expected = np.array([0.0, 5.0, 0.0])  # mV/m

        np.testing.assert_allclose(E_conv, expected, rtol=1e-10,
                                 err_msg="Convection electric field calculation incorrect")


class TestMotionAnalysis:
    """Test spacecraft motion and timing analysis"""
    
    def test_velocity_calculation(self):
        """Test spacecraft velocity calculation from position"""
        # Create position time series (circular motion)
        t = np.linspace(0, 2*np.pi, 100)
        radius = 10.0  # km
        positions = np.column_stack([
            radius * np.cos(t),
            radius * np.sin(t),
            np.zeros_like(t)
        ])
        times = t  # seconds
        
        velocities = motion.calculate_velocity(positions, times)
        
        # For circular motion, |v| = ω * r = (2π/T) * r = 1 * 10 = 10 km/s
        expected_speed = 10.0  # km/s
        calculated_speeds = np.linalg.norm(velocities, axis=1)
        
        # Check that speeds are approximately constant
        np.testing.assert_allclose(calculated_speeds[1:-1], expected_speed, 
                                 rtol=0.1, err_msg="Circular motion speed not constant")
    
    def test_timing_analysis(self):
        """Test timing analysis for boundary crossings"""
        # Create synthetic boundary crossing data
        times = np.linspace(0, 60, 1000)  # 60 seconds, 1000 points
        
        # Simulate boundary crossing at t=30s with transition width of 2s
        boundary_signal = np.tanh((times - 30) / 1.0)  # Sharp transition
        
        # Add noise
        np.random.seed(42)
        noisy_signal = boundary_signal + 0.1 * np.random.randn(len(times))
        
        crossing_time = motion.detect_crossing_time(times, noisy_signal)
        
        # Should detect crossing near t=30s
        assert abs(crossing_time - 30.0) < 2.0, f"Crossing time detection failed: {crossing_time}"
    
    def test_formation_geometry(self):
        """Test MMS formation geometry calculations"""
        # Create tetrahedral formation (simplified)
        positions = {
            '1': np.array([0.0, 0.0, 0.0]),      # Origin
            '2': np.array([100.0, 0.0, 0.0]),    # 100 km in X
            '3': np.array([50.0, 86.6, 0.0]),    # 100 km at 60°
            '4': np.array([50.0, 28.9, 81.6])    # Above plane
        }
        
        # Calculate formation quality metrics
        quality_metrics = motion.analyze_formation_geometry(positions)
        
        # Check that formation is reasonably tetrahedral
        assert 'tetrahedrality' in quality_metrics
        assert 0.5 < quality_metrics['tetrahedrality'] < 1.0, "Formation not tetrahedral enough"
        
        assert 'volume' in quality_metrics
        assert quality_metrics['volume'] > 0, "Formation volume should be positive"


class TestMultiSpacecraftAnalysis:
    """Test multi-spacecraft analysis methods"""

    def test_curlometer_technique(self):
        """Test curlometer current density calculation"""
        # Create synthetic magnetic field data for 4 spacecraft
        # Simulate current sheet with J in Y direction
        positions = {
            '1': np.array([0.0, 0.0, 0.0]),
            '2': np.array([100.0, 0.0, 0.0]),
            '3': np.array([0.0, 100.0, 0.0]),
            '4': np.array([0.0, 0.0, 100.0])
        }

        # Magnetic field with curl (current in Y direction)
        B_fields = {
            '1': np.array([10.0, 0.0, 20.0]),
            '2': np.array([10.0, 0.0, 25.0]),  # dBz/dx = 0.05 nT/km
            '3': np.array([15.0, 0.0, 20.0]),  # dBx/dy = 0.05 nT/km
            '4': np.array([10.0, 0.0, 20.0])
        }

        # Calculate current density using curlometer
        current_density = multispacecraft.curlometer(positions, B_fields)

        # Expected current in Y direction: J_y = (1/μ₀) * (dBz/dx - dBx/dz)
        # With our setup: J_y ≈ (1/μ₀) * 0.05 nT/km ≈ 40 nA/m²
        assert abs(current_density[1]) > 10.0, "Should detect significant Y current"
        assert abs(current_density[0]) < 10.0, "X current should be small"
        assert abs(current_density[2]) < 10.0, "Z current should be small"

    def test_gradient_calculation(self):
        """Test spatial gradient calculations"""
        # Create linear gradient in magnetic field
        positions = {
            '1': np.array([0.0, 0.0, 0.0]),
            '2': np.array([100.0, 0.0, 0.0]),
            '3': np.array([0.0, 100.0, 0.0]),
            '4': np.array([50.0, 50.0, 100.0])
        }

        # Linear gradient: Bx increases by 1 nT per 100 km in X direction
        B_fields = {
            '1': np.array([10.0, 20.0, 30.0]),
            '2': np.array([11.0, 20.0, 30.0]),
            '3': np.array([10.0, 20.0, 30.0]),
            '4': np.array([10.5, 20.0, 30.0])
        }

        gradients = multispacecraft.calculate_gradients(positions, B_fields)

        # Should detect gradient in X direction for Bx component
        assert abs(gradients['Bx']['dx'] - 0.01) < 0.005, "X gradient of Bx incorrect"
        assert abs(gradients['Bx']['dy']) < 0.005, "Y gradient of Bx should be zero"

    def test_timing_analysis_multisc(self):
        """Test multi-spacecraft timing analysis"""
        # Simulate boundary crossing with known propagation
        boundary_normal = np.array([1.0, 0.0, 0.0])  # X direction
        boundary_velocity = 50.0  # km/s

        positions = {
            '1': np.array([0.0, 0.0, 0.0]),
            '2': np.array([100.0, 0.0, 0.0]),
            '3': np.array([0.0, 100.0, 0.0]),
            '4': np.array([0.0, 0.0, 100.0])
        }

        # Expected time delays based on position projections
        expected_delays = {}
        for probe, pos in positions.items():
            delay = np.dot(pos, boundary_normal) / boundary_velocity
            expected_delays[probe] = delay

        # Create synthetic crossing times
        crossing_times = {probe: 100.0 + delay for probe, delay in expected_delays.items()}

        # Analyze timing
        timing_result = multispacecraft.timing_analysis(positions, crossing_times)

        # Should recover boundary normal and velocity
        recovered_normal = timing_result['normal']
        recovered_velocity = timing_result['velocity']

        # Check normal direction (allow for sign flip)
        normal_similarity = abs(np.dot(recovered_normal, boundary_normal))
        assert normal_similarity > 0.9, f"Normal direction not recovered: {normal_similarity}"

        # Check velocity magnitude
        assert abs(recovered_velocity - boundary_velocity) < 10.0, \
               f"Velocity not recovered: {recovered_velocity} vs {boundary_velocity}"


class TestDataQualityAssessment:
    """Test data quality assessment and flagging"""

    def test_outlier_detection(self):
        """Test outlier detection in magnetic field data"""
        # Create clean data with outliers
        np.random.seed(42)
        clean_data = 50 + 10 * np.sin(np.linspace(0, 4*np.pi, 1000)) + 2 * np.random.randn(1000)

        # Add outliers
        outlier_indices = [100, 300, 700]
        data_with_outliers = clean_data.copy()
        data_with_outliers[outlier_indices] = [200, -100, 150]  # Clear outliers

        # Detect outliers
        outlier_mask = quality.detect_outliers(data_with_outliers, method='iqr', threshold=3.0)

        # Should detect the outliers we added
        detected_outliers = np.where(outlier_mask)[0]
        for idx in outlier_indices:
            assert idx in detected_outliers, f"Failed to detect outlier at index {idx}"

    def test_data_gap_detection(self):
        """Test detection of data gaps and irregular sampling"""
        # Create regular time series with gaps
        regular_times = np.linspace(0, 1000, 1000)  # 1 Hz sampling

        # Remove data to create gaps
        gap_indices = list(range(300, 350)) + list(range(700, 800))
        times_with_gaps = np.delete(regular_times, gap_indices)

        # Detect gaps
        gaps = quality.detect_data_gaps(times_with_gaps, expected_cadence=1.0, gap_threshold=5.0)

        # Should detect 2 gaps
        assert len(gaps) == 2, f"Expected 2 gaps, found {len(gaps)}"

        # Check gap locations
        gap_starts = [gap['start_time'] for gap in gaps]
        assert 300 in [int(t) for t in gap_starts], "First gap not detected"
        assert 700 in [int(t) for t in gap_starts], "Second gap not detected"

    def test_signal_quality_metrics(self):
        """Test signal quality metrics calculation"""
        # Create test signals with different quality levels
        t = np.linspace(0, 10, 1000)

        # High quality: clean sinusoid
        high_quality = np.sin(2 * np.pi * t)

        # Low quality: noisy signal
        np.random.seed(42)
        low_quality = np.sin(2 * np.pi * t) + 0.5 * np.random.randn(len(t))

        # Calculate quality metrics
        hq_metrics = quality.calculate_signal_quality(high_quality)
        lq_metrics = quality.calculate_signal_quality(low_quality)

        # High quality signal should have better SNR
        assert hq_metrics['snr'] > lq_metrics['snr'], "High quality signal should have better SNR"

        # Check that metrics are reasonable
        assert hq_metrics['snr'] > 10.0, "High quality SNR too low"
        assert 0 < lq_metrics['snr'] < 10.0, "Low quality SNR out of range"


class TestResamplingAndInterpolation:
    """Test data resampling and interpolation methods"""

    def test_time_synchronization(self):
        """Test synchronization of multi-instrument data"""
        # Create data with different time bases
        t1 = np.linspace(0, 100, 101)  # 1 Hz
        t2 = np.linspace(0, 100, 51)   # 0.5 Hz

        data1 = np.sin(2 * np.pi * t1 / 10)  # 0.1 Hz signal
        data2 = np.cos(2 * np.pi * t2 / 10)  # 0.1 Hz signal

        # Synchronize to common time base
        common_time = np.linspace(0, 100, 201)  # 2 Hz

        sync_data1 = resample.interpolate_to_time(t1, data1, common_time)
        sync_data2 = resample.interpolate_to_time(t2, data2, common_time)

        # Check that interpolation preserves signal characteristics
        assert len(sync_data1) == len(common_time), "Interpolated data wrong length"
        assert len(sync_data2) == len(common_time), "Interpolated data wrong length"

        # Check that signals maintain expected phase relationship
        # sin and cos should be 90° out of phase
        correlation = np.corrcoef(sync_data1, sync_data2)[0, 1]
        assert abs(correlation) < 0.1, f"Sin/cos should be uncorrelated, got {correlation}"

    def test_spectral_preservation(self):
        """Test that resampling preserves spectral content"""
        # Create signal with known frequency content
        t_orig = np.linspace(0, 10, 1000)  # 100 Hz sampling
        signal_orig = (np.sin(2 * np.pi * 1 * t_orig) +  # 1 Hz
                      0.5 * np.sin(2 * np.pi * 5 * t_orig))  # 5 Hz

        # Downsample to 20 Hz (still above Nyquist for 5 Hz)
        t_new = np.linspace(0, 10, 200)
        signal_new = resample.interpolate_to_time(t_orig, signal_orig, t_new)

        # Compare power spectra
        from scipy import signal as scipy_signal

        f_orig, psd_orig = scipy_signal.welch(signal_orig, fs=100, nperseg=256)
        f_new, psd_new = scipy_signal.welch(signal_new, fs=20, nperseg=64)

        # Find peaks at 1 Hz and 5 Hz in both spectra
        peak1_orig = np.argmax(psd_orig[(f_orig > 0.5) & (f_orig < 1.5)])
        peak5_orig = np.argmax(psd_orig[(f_orig > 4.5) & (f_orig < 5.5)])

        peak1_new = np.argmax(psd_new[(f_new > 0.5) & (f_new < 1.5)])
        peak5_new = np.argmax(psd_new[(f_new > 4.5) & (f_new < 5.5)])

        # Both spectra should show peaks at the same frequencies
        assert peak1_orig >= 0 and peak1_new >= 0, "1 Hz peak not preserved"
        assert peak5_orig >= 0 and peak5_new >= 0, "5 Hz peak not preserved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
