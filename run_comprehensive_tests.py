"""
Comprehensive Test Runner for MMS-MP Package
===========================================

This script runs all test suites and generates a comprehensive report
of the physics validation and code quality assessment.
"""

import subprocess
import sys
import os
import time
from datetime import datetime
import json


def run_test_suite(test_file, description):
    """Run a specific test suite and capture results"""
    print(f"\n{'='*80}")
    print(f"🧪 RUNNING: {description}")
    print(f"📁 File: {test_file}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Run pytest with verbose output and capture results
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            test_file, 
            '-v',           # Verbose output
            '--tb=short',   # Short traceback format
            '--durations=10'  # Show 10 slowest tests
        ], capture_output=True, text=True, timeout=300)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Parse results
        output = result.stdout
        error_output = result.stderr
        return_code = result.returncode
        
        # Count passed/failed tests
        lines = output.split('\n')
        passed_count = sum(1 for line in lines if ' PASSED' in line)
        failed_count = sum(1 for line in lines if ' FAILED' in line)
        error_count = sum(1 for line in lines if ' ERROR' in line)
        skipped_count = sum(1 for line in lines if ' SKIPPED' in line)
        
        # Determine status
        if return_code == 0:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        
        print(f"\n📊 RESULTS: {status}")
        print(f"   ⏱️  Duration: {duration:.2f} seconds")
        print(f"   ✅ Passed: {passed_count}")
        print(f"   ❌ Failed: {failed_count}")
        print(f"   ⚠️  Errors: {error_count}")
        print(f"   ⏭️  Skipped: {skipped_count}")
        
        if failed_count > 0 or error_count > 0:
            print(f"\n🔍 ERROR DETAILS:")
            print(output)
            if error_output:
                print(f"\n🚨 STDERR:")
                print(error_output)
        
        return {
            'test_file': test_file,
            'description': description,
            'status': 'PASSED' if return_code == 0 else 'FAILED',
            'duration': duration,
            'passed': passed_count,
            'failed': failed_count,
            'errors': error_count,
            'skipped': skipped_count,
            'output': output,
            'error_output': error_output
        }
        
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: Test suite exceeded 5 minutes")
        return {
            'test_file': test_file,
            'description': description,
            'status': 'TIMEOUT',
            'duration': 300,
            'passed': 0,
            'failed': 0,
            'errors': 1,
            'skipped': 0,
            'output': '',
            'error_output': 'Test suite timed out after 5 minutes'
        }
    
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")
        return {
            'test_file': test_file,
            'description': description,
            'status': 'ERROR',
            'duration': 0,
            'passed': 0,
            'failed': 0,
            'errors': 1,
            'skipped': 0,
            'output': '',
            'error_output': str(e)
        }


def generate_test_report(results):
    """Generate comprehensive test report"""
    
    print(f"\n{'='*80}")
    print(f"📋 COMPREHENSIVE TEST REPORT")
    print(f"{'='*80}")
    print(f"🕐 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🧪 Total Test Suites: {len(results)}")
    
    # Calculate totals
    total_passed = sum(r['passed'] for r in results)
    total_failed = sum(r['failed'] for r in results)
    total_errors = sum(r['errors'] for r in results)
    total_skipped = sum(r['skipped'] for r in results)
    total_duration = sum(r['duration'] for r in results)
    
    passed_suites = sum(1 for r in results if r['status'] == 'PASSED')
    failed_suites = sum(1 for r in results if r['status'] in ['FAILED', 'ERROR', 'TIMEOUT'])
    
    print(f"\n📊 OVERALL SUMMARY:")
    print(f"   ✅ Test Suites Passed: {passed_suites}/{len(results)}")
    print(f"   ❌ Test Suites Failed: {failed_suites}/{len(results)}")
    print(f"   ⏱️  Total Duration: {total_duration:.2f} seconds")
    print(f"\n📈 INDIVIDUAL TEST COUNTS:")
    print(f"   ✅ Total Passed: {total_passed}")
    print(f"   ❌ Total Failed: {total_failed}")
    print(f"   ⚠️  Total Errors: {total_errors}")
    print(f"   ⏭️  Total Skipped: {total_skipped}")
    
    # Detailed results by suite
    print(f"\n📋 DETAILED RESULTS BY SUITE:")
    print(f"{'Suite':<40} {'Status':<10} {'Pass':<6} {'Fail':<6} {'Error':<6} {'Time':<8}")
    print(f"{'-'*80}")
    
    for result in results:
        suite_name = os.path.basename(result['test_file']).replace('.py', '')
        status_icon = "✅" if result['status'] == 'PASSED' else "❌"
        
        print(f"{suite_name:<40} {status_icon:<10} {result['passed']:<6} "
              f"{result['failed']:<6} {result['errors']:<6} {result['duration']:<8.2f}")
    
    # Physics validation summary
    print(f"\n🔬 PHYSICS VALIDATION SUMMARY:")
    
    physics_tests = [
        ('test_comprehensive_physics.py', 'Core Physics Laws'),
        ('test_spectral_analysis.py', 'Spectral Analysis'),
        ('test_integration_workflow.py', 'End-to-End Workflow'),
        ('test_boundary_detection.py', 'Boundary Physics'),
        ('test_physics_calculations.py', 'Mathematical Physics')
    ]
    
    for test_file, description in physics_tests:
        result = next((r for r in results if test_file in r['test_file']), None)
        if result:
            status = "✅ VALIDATED" if result['status'] == 'PASSED' else "❌ FAILED"
            print(f"   {description:<30} {status}")
        else:
            print(f"   {description:<30} ⏭️ NOT RUN")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    
    if failed_suites == 0:
        print(f"   🎉 EXCELLENT: All test suites passed!")
        print(f"   ✅ Physics validation: COMPLETE")
        print(f"   ✅ Code quality: VERIFIED")
        print(f"   ✅ Integration: SUCCESSFUL")
        overall_grade = "A+"
    elif failed_suites <= 1:
        print(f"   👍 GOOD: Minor issues detected")
        print(f"   ⚠️  Physics validation: MOSTLY COMPLETE")
        print(f"   ✅ Code quality: GOOD")
        print(f"   ⚠️  Integration: MINOR ISSUES")
        overall_grade = "B+"
    elif failed_suites <= 2:
        print(f"   ⚠️  FAIR: Several issues need attention")
        print(f"   ❌ Physics validation: INCOMPLETE")
        print(f"   ⚠️  Code quality: NEEDS WORK")
        print(f"   ❌ Integration: ISSUES DETECTED")
        overall_grade = "C"
    else:
        print(f"   ❌ POOR: Major issues detected")
        print(f"   ❌ Physics validation: FAILED")
        print(f"   ❌ Code quality: POOR")
        print(f"   ❌ Integration: BROKEN")
        overall_grade = "F"
    
    print(f"\n🏆 OVERALL GRADE: {overall_grade}")
    
    # Save detailed report to file
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_suites': len(results),
            'passed_suites': passed_suites,
            'failed_suites': failed_suites,
            'total_tests': total_passed + total_failed + total_errors,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'total_errors': total_errors,
            'total_skipped': total_skipped,
            'total_duration': total_duration,
            'overall_grade': overall_grade
        },
        'detailed_results': results
    }
    
    with open('test_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: test_report.json")
    
    return overall_grade


def main():
    """Run comprehensive test suite"""
    
    print(f"🧪 MMS-MP COMPREHENSIVE TEST SUITE")
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version}")
    print(f"📁 Working Directory: {os.getcwd()}")
    
    # Define test suites to run
    test_suites = [
        ('tests/test_comprehensive_physics.py', 'Core Physics and Logic Validation'),
        ('tests/test_spectral_analysis.py', 'Spectral Analysis and Visualization'),
        ('tests/test_integration_workflow.py', 'End-to-End Integration Testing'),
        ('tests/test_boundary_detection.py', 'Boundary Detection Algorithms'),
        ('tests/test_physics_calculations.py', 'Mathematical Physics Calculations'),
        ('tests/test_package.py', 'Package Structure and Imports'),
        ('tests/test_comprehensive_validation.py', 'Comprehensive Validation Suite')
    ]
    
    # Check which test files exist
    existing_suites = []
    for test_file, description in test_suites:
        if os.path.exists(test_file):
            existing_suites.append((test_file, description))
        else:
            print(f"⚠️  Test file not found: {test_file}")
    
    print(f"\n🎯 Running {len(existing_suites)} test suites...")
    
    # Run all test suites
    results = []
    for test_file, description in existing_suites:
        result = run_test_suite(test_file, description)
        results.append(result)
    
    # Generate comprehensive report
    overall_grade = generate_test_report(results)
    
    # Exit with appropriate code
    failed_suites = sum(1 for r in results if r['status'] in ['FAILED', 'ERROR', 'TIMEOUT'])
    
    if failed_suites == 0:
        print(f"\n🎉 ALL TESTS PASSED! MMS-MP package is ready for production use.")
        sys.exit(0)
    else:
        print(f"\n⚠️  {failed_suites} test suite(s) failed. Please review and fix issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
