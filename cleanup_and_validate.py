"""
Cleanup and Validation Script for MMS-MP Package
===============================================

This script:
1. Removes outdated files and images
2. Validates the current codebase structure
3. Ensures all test files are properly configured
4. Generates a clean environment for testing
"""

import os
import glob
import shutil
from datetime import datetime
import json


def remove_outdated_files():
    """Remove outdated files that don't represent current functionality"""
    
    print("🧹 CLEANING UP OUTDATED FILES")
    print("=" * 50)
    
    # Patterns for outdated files to remove
    outdated_patterns = [
        # Old test scripts
        "*test_magnetopause*.py",
        "*mms_auto*.py", 
        "*debug_*.py",
        "*diagnose_*.py",
        "*investigate_*.py",
        "*extended_mms_*.py",
        "*simple_*.py",
        "*demo*.py",
        
        # Old visualization scripts
        "*create_*spectrograms*.py",
        "*create_*plasma*.py",
        "*visualization_demo*.py",
        
        # Old analysis scripts
        "*main_analysis*.py",
        "*full_script*.py",
        "*lmn_crossing*.py",
        "*operational_*.py",
        "*scientific_*.py",
        
        # Outdated images
        "*.png",
        "*.jpg", 
        "*.jpeg",
        "*.gif"
    ]
    
    removed_files = []
    
    for pattern in outdated_patterns:
        matching_files = glob.glob(pattern)
        for file_path in matching_files:
            # Skip files we want to keep
            if any(keep in file_path for keep in [
                'test_comprehensive_physics.py',
                'test_spectral_analysis.py', 
                'test_integration_workflow.py',
                'run_comprehensive_tests.py',
                'cleanup_and_validate.py'
            ]):
                continue
                
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    removed_files.append(file_path)
                    print(f"   ✅ Removed: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    removed_files.append(file_path)
                    print(f"   ✅ Removed directory: {file_path}")
            except Exception as e:
                print(f"   ❌ Failed to remove {file_path}: {e}")
    
    print(f"\n📊 Cleanup Summary:")
    print(f"   🗑️  Files removed: {len(removed_files)}")
    
    if removed_files:
        print(f"   📝 Removed files:")
        for file_path in removed_files[:10]:  # Show first 10
            print(f"      - {file_path}")
        if len(removed_files) > 10:
            print(f"      ... and {len(removed_files) - 10} more")
    
    return removed_files


def validate_package_structure():
    """Validate the MMS-MP package structure"""
    
    print(f"\n🔍 VALIDATING PACKAGE STRUCTURE")
    print("=" * 50)
    
    # Expected package structure
    expected_structure = {
        'mms_mp/': {
            '__init__.py': 'Package initialization',
            'coords.py': 'Coordinate system transformations',
            'boundary.py': 'Boundary detection algorithms',
            'data_loader.py': 'Data loading and preprocessing',
            'electric.py': 'Electric field calculations',
            'motion.py': 'Motion and timing analysis',
            'multispacecraft.py': 'Multi-spacecraft analysis',
            'quality.py': 'Data quality assessment',
            'resample.py': 'Data resampling and interpolation',
            'spectra.py': 'Spectral analysis methods',
            'thickness.py': 'Thickness calculations',
            'visualize.py': 'Visualization functions'
        },
        'tests/': {
            '__init__.py': 'Test package initialization',
            'conftest.py': 'Pytest configuration',
            'test_comprehensive_physics.py': 'Core physics validation',
            'test_spectral_analysis.py': 'Spectral analysis tests',
            'test_integration_workflow.py': 'Integration tests',
            'test_boundary_detection.py': 'Boundary detection tests',
            'test_physics_calculations.py': 'Physics calculation tests',
            'test_package.py': 'Package structure tests'
        }
    }
    
    validation_results = {}
    
    for directory, files in expected_structure.items():
        print(f"\n📁 Checking {directory}")
        
        if not os.path.exists(directory):
            print(f"   ❌ Directory missing: {directory}")
            validation_results[directory] = {'status': 'missing', 'files': {}}
            continue
        
        dir_results = {'status': 'exists', 'files': {}}
        
        for filename, description in files.items():
            filepath = os.path.join(directory, filename)
            
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"   ✅ {filename} ({file_size} bytes) - {description}")
                dir_results['files'][filename] = {
                    'status': 'exists',
                    'size': file_size,
                    'description': description
                }
            else:
                print(f"   ❌ {filename} - MISSING - {description}")
                dir_results['files'][filename] = {
                    'status': 'missing',
                    'description': description
                }
        
        validation_results[directory] = dir_results
    
    # Check for unexpected files
    print(f"\n🔍 Checking for unexpected files...")
    
    all_files = []
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if not file.startswith('.') and not file.endswith('.pyc'):
                rel_path = os.path.relpath(os.path.join(root, file))
                all_files.append(rel_path)
    
    expected_files = set()
    for directory, files in expected_structure.items():
        for filename in files.keys():
            expected_files.add(os.path.join(directory, filename).replace('\\', '/'))
    
    # Add other expected files
    expected_files.update([
        'run_comprehensive_tests.py',
        'cleanup_and_validate.py',
        'README.md',
        'setup.py',
        'requirements.txt'
    ])
    
    unexpected_files = []
    for file_path in all_files:
        normalized_path = file_path.replace('\\', '/')
        if normalized_path not in expected_files:
            unexpected_files.append(file_path)
    
    if unexpected_files:
        print(f"   ⚠️  Found {len(unexpected_files)} unexpected files:")
        for file_path in unexpected_files[:10]:
            print(f"      - {file_path}")
        if len(unexpected_files) > 10:
            print(f"      ... and {len(unexpected_files) - 10} more")
    else:
        print(f"   ✅ No unexpected files found")
    
    return validation_results, unexpected_files


def validate_test_configuration():
    """Validate test configuration and dependencies"""
    
    print(f"\n🧪 VALIDATING TEST CONFIGURATION")
    print("=" * 50)
    
    # Check if pytest is available
    try:
        import pytest
        print(f"   ✅ pytest available: {pytest.__version__}")
    except ImportError:
        print(f"   ❌ pytest not available - install with: pip install pytest")
        return False
    
    # Check if required test dependencies are available
    test_dependencies = [
        ('numpy', 'Numerical computations'),
        ('matplotlib', 'Plotting and visualization'),
        ('pandas', 'Data manipulation'),
        ('scipy', 'Scientific computing')
    ]
    
    missing_deps = []
    
    for dep_name, description in test_dependencies:
        try:
            module = __import__(dep_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"   ✅ {dep_name} available: {version} - {description}")
        except ImportError:
            print(f"   ❌ {dep_name} missing - {description}")
            missing_deps.append(dep_name)
    
    # Check test files syntax
    test_files = glob.glob('tests/test_*.py')
    syntax_errors = []
    
    print(f"\n🔍 Checking test file syntax...")
    
    for test_file in test_files:
        try:
            with open(test_file, 'r') as f:
                compile(f.read(), test_file, 'exec')
            print(f"   ✅ {test_file} - syntax OK")
        except SyntaxError as e:
            print(f"   ❌ {test_file} - syntax error: {e}")
            syntax_errors.append((test_file, str(e)))
        except Exception as e:
            print(f"   ⚠️  {test_file} - warning: {e}")
    
    return len(missing_deps) == 0 and len(syntax_errors) == 0


def generate_validation_report(removed_files, structure_validation, unexpected_files, test_config_ok):
    """Generate comprehensive validation report"""
    
    print(f"\n📋 VALIDATION REPORT")
    print("=" * 50)
    print(f"🕐 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Cleanup summary
    print(f"\n🧹 CLEANUP SUMMARY:")
    print(f"   🗑️  Files removed: {len(removed_files)}")
    
    # Structure validation summary
    missing_dirs = sum(1 for v in structure_validation.values() if v['status'] == 'missing')
    total_dirs = len(structure_validation)
    
    missing_files = 0
    total_files = 0
    for dir_info in structure_validation.values():
        for file_info in dir_info['files'].values():
            total_files += 1
            if file_info['status'] == 'missing':
                missing_files += 1
    
    print(f"\n📁 STRUCTURE VALIDATION:")
    print(f"   📂 Directories: {total_dirs - missing_dirs}/{total_dirs} present")
    print(f"   📄 Files: {total_files - missing_files}/{total_files} present")
    print(f"   ⚠️  Unexpected files: {len(unexpected_files)}")
    
    # Test configuration summary
    print(f"\n🧪 TEST CONFIGURATION:")
    if test_config_ok:
        print(f"   ✅ All dependencies available")
        print(f"   ✅ All test files have valid syntax")
    else:
        print(f"   ❌ Missing dependencies or syntax errors")
    
    # Overall assessment
    issues = missing_dirs + missing_files + len(unexpected_files) + (0 if test_config_ok else 1)
    
    if issues == 0:
        status = "✅ EXCELLENT"
        recommendation = "Ready for comprehensive testing"
    elif issues <= 2:
        status = "👍 GOOD"
        recommendation = "Minor issues, but ready for testing"
    elif issues <= 5:
        status = "⚠️ FAIR"
        recommendation = "Several issues need attention"
    else:
        status = "❌ POOR"
        recommendation = "Major issues must be resolved"
    
    print(f"\n🎯 OVERALL ASSESSMENT: {status}")
    print(f"   📊 Total issues: {issues}")
    print(f"   💡 Recommendation: {recommendation}")
    
    # Save detailed report
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'cleanup': {
            'files_removed': len(removed_files),
            'removed_files': removed_files
        },
        'structure': {
            'directories_missing': missing_dirs,
            'files_missing': missing_files,
            'unexpected_files': len(unexpected_files),
            'details': structure_validation
        },
        'test_config': {
            'status': 'OK' if test_config_ok else 'ISSUES'
        },
        'assessment': {
            'total_issues': issues,
            'status': status.split()[1],  # Remove emoji
            'recommendation': recommendation
        }
    }
    
    with open('validation_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: validation_report.json")
    
    return issues == 0


def main():
    """Run cleanup and validation"""
    
    print(f"🧹 MMS-MP CLEANUP AND VALIDATION")
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Working Directory: {os.getcwd()}")
    
    # Step 1: Remove outdated files
    removed_files = remove_outdated_files()
    
    # Step 2: Validate package structure
    structure_validation, unexpected_files = validate_package_structure()
    
    # Step 3: Validate test configuration
    test_config_ok = validate_test_configuration()
    
    # Step 4: Generate comprehensive report
    validation_passed = generate_validation_report(
        removed_files, structure_validation, unexpected_files, test_config_ok
    )
    
    if validation_passed:
        print(f"\n🎉 VALIDATION COMPLETE! Environment is ready for testing.")
        print(f"💡 Next step: Run 'python run_comprehensive_tests.py'")
        return True
    else:
        print(f"\n⚠️  VALIDATION ISSUES DETECTED! Please review and fix.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
