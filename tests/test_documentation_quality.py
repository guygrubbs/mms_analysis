"""
Test documentation quality and type hints

This module verifies that all public functions have proper docstrings
and type hints following Google/NumPy style conventions.
"""

import inspect
import sys
import os
from typing import get_type_hints

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mms_mp


class TestDocumentationQuality:
    """Test suite for documentation quality"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        
    def assert_condition(self, condition, description=""):
        """Assert that a condition is true"""
        if condition:
            print(f"✓ {description}")
            self.passed += 1
            return True
        else:
            print(f"❌ {description}")
            self.failed += 1
            return False
    
    def test_package_docstrings(self):
        """Test that key modules have proper docstrings"""
        print("\n=== Testing Module Docstrings ===")
        
        modules_to_test = [
            ('mms_mp', mms_mp),
            ('mms_mp.coords', mms_mp.coords),
            ('mms_mp.multispacecraft', mms_mp.multispacecraft),
            ('mms_mp.electric', mms_mp.electric),
            ('mms_mp.motion', mms_mp.motion),
            ('mms_mp.thickness', mms_mp.thickness),
            ('mms_mp.boundary', mms_mp.boundary),
            ('mms_mp.data_loader', mms_mp.data_loader),
            ('mms_mp.visualize', mms_mp.visualize),
        ]
        
        for module_name, module in modules_to_test:
            docstring = inspect.getdoc(module)
            has_docstring = docstring is not None and len(docstring.strip()) > 0
            self.assert_condition(
                has_docstring,
                f"{module_name} has module docstring"
            )
    
    def test_function_docstrings(self):
        """Test that key functions have comprehensive docstrings"""
        print("\n=== Testing Function Docstrings ===")
        
        functions_to_test = [
            ('hybrid_lmn', mms_mp.hybrid_lmn),
            ('timing_normal', mms_mp.timing_normal),
            ('exb_velocity', mms_mp.exb_velocity),
            ('integrate_disp', mms_mp.integrate_disp),
            ('layer_thicknesses', mms_mp.thickness.layer_thicknesses),
            ('detect_crossings_multi', mms_mp.detect_crossings_multi),
            ('load_event', mms_mp.load_event),
        ]
        
        for func_name, func in functions_to_test:
            docstring = inspect.getdoc(func)
            
            # Check if docstring exists
            has_docstring = docstring is not None and len(docstring.strip()) > 0
            self.assert_condition(
                has_docstring,
                f"{func_name} has docstring"
            )
            
            if has_docstring:
                # Check for key sections in Google-style docstrings
                has_args = 'Args:' in docstring or 'Parameters:' in docstring
                has_returns = 'Returns:' in docstring or 'Return:' in docstring
                has_examples = 'Examples:' in docstring or 'Example:' in docstring
                
                self.assert_condition(
                    has_args,
                    f"{func_name} documents arguments"
                )
                
                self.assert_condition(
                    has_returns,
                    f"{func_name} documents return values"
                )
                
                # Examples are nice to have but not required for all functions
                if has_examples:
                    print(f"✓ {func_name} includes examples")
                    self.passed += 1
    
    def test_type_hints(self):
        """Test that key functions have proper type hints"""
        print("\n=== Testing Type Hints ===")
        
        functions_to_test = [
            ('hybrid_lmn', mms_mp.coords.hybrid_lmn),
            ('timing_normal', mms_mp.multispacecraft.timing_normal),
            ('exb_velocity', mms_mp.electric.exb_velocity),
            ('integrate_disp', mms_mp.motion.integrate_disp),
            ('layer_thicknesses', mms_mp.thickness.layer_thicknesses),
        ]
        
        for func_name, func in functions_to_test:
            try:
                # Get type hints
                hints = get_type_hints(func)
                
                # Check if function has type hints
                has_hints = len(hints) > 0
                self.assert_condition(
                    has_hints,
                    f"{func_name} has type hints"
                )
                
                # Check for return type annotation
                has_return_type = 'return' in hints
                self.assert_condition(
                    has_return_type,
                    f"{func_name} has return type annotation"
                )
                
            except Exception as e:
                self.assert_condition(
                    False,
                    f"{func_name} type hints accessible (error: {e})"
                )
    
    def test_docstring_quality(self):
        """Test docstring quality for key functions"""
        print("\n=== Testing Docstring Quality ===")
        
        # Test the enhanced docstrings we added
        enhanced_functions = [
            ('hybrid_lmn', mms_mp.hybrid_lmn),
            ('timing_normal', mms_mp.timing_normal),
            ('exb_velocity', mms_mp.exb_velocity),
            ('integrate_disp', mms_mp.integrate_disp),
        ]
        
        for func_name, func in enhanced_functions:
            docstring = inspect.getdoc(func)
            
            if docstring:
                # Check for comprehensive content
                has_physics = any(word in docstring.lower() for word in 
                                ['physics', 'equation', 'formula', 'method'])
                has_references = 'References:' in docstring
                has_notes = 'Notes:' in docstring
                has_raises = 'Raises:' in docstring
                
                if has_physics:
                    print(f"✓ {func_name} includes physics background")
                    self.passed += 1
                    
                if has_references:
                    print(f"✓ {func_name} includes references")
                    self.passed += 1
                    
                if has_notes:
                    print(f"✓ {func_name} includes usage notes")
                    self.passed += 1
                    
                if has_raises:
                    print(f"✓ {func_name} documents exceptions")
                    self.passed += 1
    
    def test_package_metadata(self):
        """Test package metadata and version information"""
        print("\n=== Testing Package Metadata ===")
        
        # Check version information
        has_version = hasattr(mms_mp, '__version__')
        self.assert_condition(
            has_version,
            "Package has __version__ attribute"
        )
        
        if has_version:
            version_format = isinstance(mms_mp.__version__, str) and '.' in mms_mp.__version__
            self.assert_condition(
                version_format,
                f"Version format is valid: {mms_mp.__version__}"
            )
        
        # Check other metadata
        metadata_attrs = ['__author__', '__license__']
        for attr in metadata_attrs:
            has_attr = hasattr(mms_mp, attr)
            self.assert_condition(
                has_attr,
                f"Package has {attr} attribute"
            )
    
    def test_import_structure(self):
        """Test that key functions are importable at package level"""
        print("\n=== Testing Import Structure ===")
        
        key_functions = [
            'load_event', 'hybrid_lmn', 'detect_crossings_multi',
            'integrate_disp', 'timing_normal', 'merge_vars',
            'exb_velocity', 'normal_velocity'
        ]
        
        for func_name in key_functions:
            has_function = hasattr(mms_mp, func_name)
            self.assert_condition(
                has_function,
                f"{func_name} available at package level"
            )
            
            if has_function:
                func = getattr(mms_mp, func_name)
                is_callable = callable(func)
                self.assert_condition(
                    is_callable,
                    f"{func_name} is callable"
                )
    
    def run_all_tests(self):
        """Run all documentation quality tests"""
        print("📚 Running Documentation Quality Tests")
        print("=" * 45)
        
        self.test_package_docstrings()
        self.test_function_docstrings()
        self.test_type_hints()
        self.test_docstring_quality()
        self.test_package_metadata()
        self.test_import_structure()
        
        print(f"\n📊 Test Results:")
        print(f"   ✓ Passed: {self.passed}")
        print(f"   ❌ Failed: {self.failed}")
        print(f"   📈 Success Rate: {self.passed/(self.passed + self.failed)*100:.1f}%")
        
        if self.failed == 0:
            print("\n🎉 All documentation quality tests passed!")
            return True
        else:
            print(f"\n⚠️  {self.failed} test(s) failed - review documentation")
            return False


if __name__ == "__main__":
    tester = TestDocumentationQuality()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
