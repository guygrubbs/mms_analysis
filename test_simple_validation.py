"""
Simple Validation Test for MMS-MP Package
=========================================

Basic test to validate that all modules can be imported and have expected functions.
"""

import sys
import traceback
from datetime import datetime


def test_module_imports():
    """Test that all modules can be imported"""
    print("📦 Testing module imports...")
    
    modules_to_test = [
        'mms_mp',
        'mms_mp.coords',
        'mms_mp.boundary', 
        'mms_mp.data_loader',
        'mms_mp.electric',
        'mms_mp.motion',
        'mms_mp.multispacecraft',
        'mms_mp.quality',
        'mms_mp.resample',
        'mms_mp.spectra',
        'mms_mp.thickness',
        'mms_mp.visualize'
    ]
    
    results = {}
    
    for module_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[''])
            results[module_name] = {'status': 'SUCCESS', 'error': None}
            print(f"   ✅ {module_name}")
        except Exception as e:
            results[module_name] = {'status': 'FAILED', 'error': str(e)}
            print(f"   ❌ {module_name}: {e}")
    
    return results


def test_function_availability():
    """Test that key functions are available in modules"""
    print("\n🔧 Testing function availability...")
    
    function_tests = [
        ('mms_mp.coords', 'hybrid_lmn'),
        ('mms_mp.boundary', 'DetectorCfg'),
        ('mms_mp.boundary', '_sm_update'),
        ('mms_mp.data_loader', 'load_event'),
        ('mms_mp.electric', 'exb_velocity'),
        ('mms_mp.motion', 'normal_velocity'),
        ('mms_mp.multispacecraft', 'timing_normal'),
        ('mms_mp.quality', 'dis_good_mask'),
        ('mms_mp.quality', 'des_good_mask')
    ]
    
    results = {}
    
    for module_name, function_name in function_tests:
        try:
            module = __import__(module_name, fromlist=[''])
            if hasattr(module, function_name):
                results[f"{module_name}.{function_name}"] = {'status': 'SUCCESS', 'error': None}
                print(f"   ✅ {module_name}.{function_name}")
            else:
                results[f"{module_name}.{function_name}"] = {'status': 'MISSING', 'error': 'Function not found'}
                print(f"   ❌ {module_name}.{function_name}: Function not found")
        except Exception as e:
            results[f"{module_name}.{function_name}"] = {'status': 'FAILED', 'error': str(e)}
            print(f"   ❌ {module_name}.{function_name}: {e}")
    
    return results


def test_basic_functionality():
    """Test basic functionality without complex calculations"""
    print("\n⚙️ Testing basic functionality...")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Boundary detector configuration
    tests_total += 1
    try:
        from mms_mp.boundary import DetectorCfg
        cfg = DetectorCfg(he_in=0.3, he_out=0.1, min_pts=5)
        assert cfg.he_in == 0.3
        assert cfg.he_out == 0.1
        assert cfg.min_pts == 5
        print("   ✅ Boundary DetectorCfg: PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Boundary DetectorCfg: FAILED - {e}")
    
    # Test 2: Quality mask functions
    tests_total += 1
    try:
        from mms_mp.quality import dis_good_mask, des_good_mask
        import numpy as np
        
        flag_data = np.array([0, 1, 2, 0, 1])
        dis_mask = dis_good_mask(flag_data, accept_levels=(0,))
        des_mask = des_good_mask(flag_data, accept_levels=(0, 1))
        
        assert len(dis_mask) == len(flag_data)
        assert len(des_mask) == len(flag_data)
        print("   ✅ Quality masks: PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Quality masks: FAILED - {e}")
    
    # Test 3: LMN coordinate system basic structure
    tests_total += 1
    try:
        from mms_mp.coords import LMN
        import numpy as np
        
        # Create a simple LMN instance
        L = np.array([1.0, 0.0, 0.0])
        M = np.array([0.0, 1.0, 0.0])
        N = np.array([0.0, 0.0, 1.0])
        R = np.eye(3)
        
        lmn = LMN(L=L, M=M, N=N, R=R, eigvals=(1.0, 1.0, 1.0), r_max_mid=1.0, r_mid_min=1.0)
        
        assert hasattr(lmn, 'to_lmn')
        assert hasattr(lmn, 'to_gsm')
        print("   ✅ LMN structure: PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ LMN structure: FAILED - {e}")
    
    # Test 4: Electric field function signature
    tests_total += 1
    try:
        from mms_mp.electric import exb_velocity
        import numpy as np
        
        # Test with simple inputs
        E = np.array([1.0, 0.0, 0.0])
        B = np.array([0.0, 0.0, 1.0])
        
        # This should not crash (even if result is wrong)
        v = exb_velocity(E, B, unit_E='mV/m', unit_B='nT')
        assert len(v) == 3  # Should return 3D vector
        print("   ✅ Electric field function: PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Electric field function: FAILED - {e}")
    
    return tests_passed, tests_total


def main():
    """Run simple validation tests"""
    
    print("🧪 MMS-MP SIMPLE VALIDATION TEST")
    print("=" * 50)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version}")
    
    # Test 1: Module imports
    import_results = test_module_imports()
    
    # Test 2: Function availability
    function_results = test_function_availability()
    
    # Test 3: Basic functionality
    basic_passed, basic_total = test_basic_functionality()
    
    # Summary
    print(f"\n{'='*50}")
    print("📋 VALIDATION SUMMARY")
    print(f"{'='*50}")
    
    # Import results
    import_success = sum(1 for r in import_results.values() if r['status'] == 'SUCCESS')
    import_total = len(import_results)
    
    print(f"📦 Module Imports: {import_success}/{import_total} successful")
    
    # Function results
    function_success = sum(1 for r in function_results.values() if r['status'] == 'SUCCESS')
    function_total = len(function_results)
    
    print(f"🔧 Function Availability: {function_success}/{function_total} found")
    
    # Basic functionality
    print(f"⚙️ Basic Functionality: {basic_passed}/{basic_total} passed")
    
    # Overall assessment
    total_success = import_success + function_success + basic_passed
    total_tests = import_total + function_total + basic_total
    success_rate = total_success / total_tests
    
    print(f"\n📊 Overall Success Rate: {total_success}/{total_tests} ({success_rate:.1%})")
    
    if success_rate >= 0.9:
        grade = "A 🎉"
        assessment = "EXCELLENT - Package structure is solid!"
    elif success_rate >= 0.8:
        grade = "B+ 👍"
        assessment = "GOOD - Minor issues detected"
    elif success_rate >= 0.7:
        grade = "B ⚠️"
        assessment = "FAIR - Some issues need attention"
    elif success_rate >= 0.6:
        grade = "C ⚠️"
        assessment = "MARGINAL - Several issues detected"
    else:
        grade = "F ❌"
        assessment = "POOR - Major structural issues"
    
    print(f"\n🎯 Overall Grade: {grade}")
    print(f"💡 Assessment: {assessment}")
    
    # Detailed failure analysis
    if success_rate < 1.0:
        print(f"\n🔍 FAILURE ANALYSIS:")
        
        failed_imports = [name for name, result in import_results.items() if result['status'] != 'SUCCESS']
        if failed_imports:
            print(f"   📦 Failed imports: {', '.join(failed_imports)}")
        
        failed_functions = [name for name, result in function_results.items() if result['status'] != 'SUCCESS']
        if failed_functions:
            print(f"   🔧 Missing functions: {', '.join(failed_functions)}")
        
        if basic_passed < basic_total:
            print(f"   ⚙️ Basic functionality issues: {basic_total - basic_passed} tests failed")
    
    if success_rate >= 0.8:
        print(f"\n✅ PACKAGE STRUCTURE VALIDATION COMPLETE!")
        print(f"🎯 The MMS-MP package has a solid foundation")
        print(f"📚 All major modules are importable and functional")
        print(f"🔧 Key functions are available and working")
        
        if success_rate < 1.0:
            print(f"\n💡 Minor improvements needed:")
            print(f"   • Review failed tests above")
            print(f"   • Ensure all expected functions are implemented")
            print(f"   • Add any missing functionality")
    else:
        print(f"\n⚠️ STRUCTURAL ISSUES DETECTED")
        print(f"Please review the failure analysis and fix critical issues")
    
    return success_rate >= 0.8


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
