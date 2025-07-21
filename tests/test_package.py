"""
Test basic package functionality and imports
"""


def test_package_import():
    """Test that the package can be imported"""
    import mms_mp
    assert hasattr(mms_mp, '__version__')
    assert mms_mp.__version__ == "1.0.0"
    print("✓ Package import test passed")


def test_module_imports():
    """Test that all modules can be imported"""
    import mms_mp

    # Test that all expected modules are available
    expected_modules = [
        'data_loader', 'coords', 'resample', 'electric', 'quality',
        'boundary', 'motion', 'multispacecraft', 'visualize', 'spectra',
        'thickness', 'cli'
    ]

    for module_name in expected_modules:
        assert hasattr(mms_mp, module_name), f"Module {module_name} not found"
    print("✓ Module imports test passed")


def test_key_functions_available():
    """Test that key functions are available at package level"""
    import mms_mp

    # Test that key functions are accessible
    key_functions = [
        'load_event', 'hybrid_lmn', 'detect_crossings_multi',
        'integrate_disp', 'timing_normal', 'merge_vars'
    ]

    for func_name in key_functions:
        assert hasattr(mms_mp, func_name), f"Function {func_name} not found"
    print("✓ Key functions test passed")


def test_detector_cfg_available():
    """Test that DetectorCfg class is available"""
    import mms_mp
    assert hasattr(mms_mp, 'DetectorCfg')
    print("✓ DetectorCfg test passed")


def run_all_tests():
    """Run all tests"""
    print("Running MMS-MP package tests...")
    try:
        test_package_import()
        test_module_imports()
        test_key_functions_available()
        test_detector_cfg_available()
        print("\n🎉 All tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    run_all_tests()
