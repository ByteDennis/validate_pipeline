"""
Basic Object Examination:
    Built-in types (int, str, list, dict, etc.)
    Functions and methods
    Classes and instances
    Modules

Advanced Object Types:
    Generators and iterators
    Exception objects
    File objects and special objects
    Objects with custom __repr__ methods

Complex Scenarios:
    Nested and complex objects
    Large objects (performance testing)
    Circular references (should handle gracefully)
    Empty containers
    Metaclasses and advanced class features

Edge Cases:
    None values
    Empty containers
    Performance with large objects
    Objects with special properties
    
Before: 15+ separate test methods with repetitive code
After: Consolidated into parametrized tests with shared logic
"""

import pytest
import sys
import io
import time
import math
import json
from contextlib import redirect_stdout
import wat


# without this fixture, wat output is on screen and disappear shortly after
@pytest.fixture
def capture_wat():
    def _capture(obj):
        output = io.StringIO()           # Creates an in-memory text buffer
        with redirect_stdout(output):    # Redirects print statements to our buffer
            wat.wat(obj)                 # WAT prints to our buffer instead of console
        return output.getvalue()         # Returns the captured text as a string
    return _capture                      # Returns the function itself

@pytest.fixture
def sample_class():
    """Fixture providing a sample class for testing."""
    class TestClass:
        """A test class for WAT examination."""
        class_var = "I'm a class variable"
        def __init__(self, value):
            self.instance_var = value
        def instance_method(self):
            return self.instance_var
        @classmethod
        def class_method(cls):
            return cls.class_var
        @staticmethod
        def static_method():
            return "static"
        @property
        def prop(self):
            return self.instance_var * 2
    return TestClass


@pytest.fixture
def sample_instance(sample_class):
    """Fixture providing a sample class instance."""
    return sample_class("test_value")

@pytest.fixture
def complex_nested_object():
    """Fixture providing a complex nested object for testing."""
    return {
        "numbers": [1, 2, 3, 4, 5],
        "nested": {
            "inner": {"deep": "value"},
            "list_of_dicts": [{"a": 1}, {"b": 2}]
        },
        "function": lambda x: x * 2,
        "class_instance": type("DynamicClass", (), {"attr": "value"})()
    }


@pytest.fixture
def circular_reference_object():
    """Fixture providing an object with circular references."""
    circular_list = [1, 2, 3]
    circular_list.append(circular_list)
    return circular_list


@pytest.fixture
def custom_repr_class():
    """Fixture providing a class with custom __repr__."""
    class CustomRepr:
        def __init__(self, value):
            self.value = value
        def __repr__(self):
            return f"CustomRepr(value={self.value!r})"
        def __str__(self):
            return f"CustomRepr with value: {self.value}"
    return CustomRepr


@pytest.fixture
def metaclass_example():
    """Fixture providing a metaclass example."""
    class MetaClass(type):
        def __new__(cls, name, bases, attrs):
            attrs['meta_attr'] = 'added by metaclass'
            return super().__new__(cls, name, bases, attrs)
    
    class TestMetaClass(metaclass=MetaClass):
        pass
    
    return TestMetaClass


# Test Classes
class TestWATCore:
    """Test core functionality of WAT - Python object examination tool."""
    
    @pytest.mark.parametrize("obj,expected_type", [
        (42, "int"),
        (3.14, "float"),
        ("string", "str"),
        ([1, 2, 3], "list"),
        ((1, 2, 3), "tuple"),
        ({"a": 1, "b": 2}, "dict"),
        ({1, 2, 3}, "set"),
        (True, "bool"),
        (None, "NoneType"),
        (b"bytes", "bytes"),
        (bytearray(b"test"), "bytearray"),
        (frozenset([1, 2, 3]), "frozenset"),
    ])
    def test_builtin_types(self, capture_wat, obj, expected_type):
        """Test WAT with various built-in Python types."""
        output = capture_wat(obj)
        assert expected_type.lower() in output.lower()
        assert len(output.strip()) > 0
    
    @pytest.mark.parametrize("func,expected_keyword", [
        (lambda x: x, "function"),
        (len, "builtin"),
        (str.upper, "method"),
        (print, "builtin"),
    ])
    def test_functions_and_methods(self, capture_wat, func, expected_keyword):
        """Test WAT with functions and methods."""
        output = capture_wat(func)
        assert expected_keyword.lower() in output.lower()
    
    @pytest.mark.parametrize("module,module_name", [
        (math, "math"),
        (json, "json"),
        (sys, "sys"),
        (io, "io"),
    ])
    def test_modules(self, capture_wat, module, module_name):
        """Test WAT with various modules."""
        output = capture_wat(module)
        assert "module" in output.lower()
        assert module_name in output.lower()
    
    @pytest.mark.parametrize("empty_obj,expected_type", [
        ([], "list"),
        ({}, "dict"),
        (set(), "set"),
        (tuple(), "tuple"),
        ("", "str"),
        (frozenset(), "frozenset"),
    ])
    def test_empty_containers(self, capture_wat, empty_obj, expected_type):
        """Test WAT with empty containers."""
        output = capture_wat(empty_obj)
        assert expected_type.lower() in output.lower()
        assert len(output.strip()) > 0
    
    def test_class_examination(self, capture_wat, sample_class):
        """Test WAT with class objects."""
        output = capture_wat(sample_class)
        print(output)
    
    def test_instance_examination(self, capture_wat, sample_instance):
        """Test WAT with class instances."""
        output = capture_wat(sample_instance)
        print(output)
    
    def test_complex_objects(self, capture_wat, complex_nested_object):
        """Test WAT with complex nested objects."""
        output = capture_wat(complex_nested_object)
        print(output)
    
    def test_generators_and_iterators(self, capture_wat):
        """Test WAT with generators and iterators."""
        def test_generator():
            for i in range(3):
                yield i
        gen = test_generator()
        wat.wat(gen)
        print()
        iterator = iter([1, 2, 3])
        wat.wat(iterator)
    
    def test_exceptions(self, capture_wat):
        """Test WAT with exception objects."""
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            print(capture_wat(e))
    
    def test_circular_references(self, capture_wat, circular_reference_object):
        """Test WAT with objects containing circular references."""
        # WAT should handle this gracefully without infinite recursion
        try:
            output = capture_wat(circular_reference_object)
            assert "list" in output.lower()
        except RecursionError:
            pytest.fail("WAT should handle circular references gracefully")
    
    def test_custom_repr_objects(self, capture_wat, custom_repr_class):
        """Test WAT with objects that have custom __repr__ methods."""
        obj = custom_repr_class("test")
        output = capture_wat(obj)
        assert "CustomRepr" in output
        assert len(output.strip()) > 0
    
    def test_metaclass_objects(self, capture_wat, metaclass_example):
        """Test WAT with metaclasses."""
        output = capture_wat(metaclass_example)
        assert "class" in output.lower()
        assert "TestMetaClass" in output
        
        instance = metaclass_example()
        output = capture_wat(instance)
        assert "TestMetaClass" in output


class TestWATSpecialCases:
    """Test special cases and edge conditions for WAT."""
    
    @pytest.mark.parametrize("size", [100, 1000, 5000])
    def test_large_lists(self, capture_wat, size):
        """Test WAT with large lists of different sizes."""
        large_list = list(range(size))
        output = capture_wat(large_list)
        assert "list" in output.lower()
        assert str(size) in output
    
    @pytest.mark.parametrize("size", [50, 100, 500])
    def test_large_dicts(self, capture_wat, size):
        """Test WAT with large dictionaries of different sizes."""
        large_dict = {f"key_{i}": f"value_{i}" for i in range(size)}
        output = capture_wat(large_dict)
        assert "dict" in output.lower()
        assert str(size) in output
    
    @pytest.mark.parametrize("special_obj,expected_content", [
        (range(10), "range"),
        (slice(1, 10, 2), "slice"),
        (Ellipsis, "ellipsis"),
        (NotImplemented, "NotImplemented"),
    ])
    def test_special_objects(self, capture_wat, special_obj, expected_content):
        """Test WAT with special Python objects."""
        output = capture_wat(special_obj)
        assert expected_content.lower() in output.lower()
    
    def test_file_objects(self, capture_wat):
        """Test WAT with file objects."""
        with open(__file__, 'r') as f:
            output = capture_wat(f)
            assert "file" in output.lower() or "TextIOWrapper" in output
    
    def test_wat_method_chaining(self):
        """Test if WAT returns None and doesn't break chains."""
        test_list = [1, 2, 3]
        result = wat.wat(test_list)
        assert result is None
    
    def test_wat_import_accessibility(self):
        """Test that WAT can be imported and used correctly."""
        import wat
        assert hasattr(wat, 'wat')
        assert callable(wat.wat)
        
        try:
            wat.wat("test")
        except Exception as e:
            pytest.fail(f"Basic WAT usage should not raise exception: {e}")


class TestWATPerformance:
    """Test WAT performance characteristics."""
    
    @pytest.mark.parametrize("obj_factory", [
        lambda: list(range(1000)),
        lambda: {f"key_{i}": i for i in range(500)},
        lambda: "x" * 1000,
        lambda: tuple(range(1000)),
        lambda: set(range(500)),
    ])
    def test_performance_smoke_test(self, capture_wat, obj_factory):
        """Smoke test to ensure WAT doesn't hang on various object sizes."""
        obj = obj_factory()
        start_time = time.time()
        capture_wat(obj)
        end_time = time.time()
        
        # Should complete in reasonable time (less than 5 seconds)
        duration = end_time - start_time
        assert duration < 5.0, f"WAT took {duration:.2f}s for {type(obj).__name__}"


class TestWATOutputQuality:
    """Test the quality and format of WAT output."""
    
    @pytest.mark.parametrize("test_obj", [
        {"key": "value", "number": 42},
        [1, 2, 3, "mixed", {"nested": True}],
        "simple string",
        42,
        None,
    ])
    def test_output_format_quality(self, capture_wat, test_obj):
        """Test that WAT output is properly formatted and readable."""
        output = capture_wat(test_obj)
        # Output should be non-empty
        assert len(output.strip()) > 0
        # Should contain readable information
        lines = output.strip().split('\n')
        assert len(lines) > 0
        # Should not be just error messages
        assert not all("error" in line.lower() for line in lines)
    
    def test_none_handling(self, capture_wat):
        """Test WAT specifically with None."""
        output = capture_wat(None)
        assert "None" in output
        assert len(output.strip()) > 0


# Parametrized test for comprehensive type coverage
@pytest.mark.parametrize("obj,description", [
    # Basic types
    (42, "integer"),
    (3.14159, "float"),
    (2+3j, "complex number"),
    ("hello world", "string"),
    (b"binary data", "bytes"),
    
    # Collections
    ([1, 2, 3], "list"),
    ((1, 2, 3), "tuple"),
    ({"a": 1, "b": 2}, "dictionary"),
    ({1, 2, 3}, "set"),
    (frozenset([1, 2, 3]), "frozenset"),
    
    # Special values
    (True, "boolean true"),
    (False, "boolean false"),
    (None, "none type"),
    
    # Callables
    (len, "builtin function"),
    (lambda x: x, "lambda function"),
    
    # Iterables
    (range(5), "range object"),
    (iter([1, 2, 3]), "iterator"),
    
    # Others
    (slice(1, 10, 2), "slice object"),
    (Ellipsis, "ellipsis"),
])
def test_comprehensive_type_coverage(capture_wat, obj, description):
    """Comprehensive test covering many Python types."""
    output = capture_wat(obj)
    assert len(output.strip()) > 0, f"WAT should produce output for {description}"
    # The output should not be empty or just whitespace


if __name__ == "__main__":
    # Run tests with verbose output and show local variables on failure
    pytest.main([__file__, "-v", "--tb=short"])