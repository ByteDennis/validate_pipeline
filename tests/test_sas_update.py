"""
Advanced Pytest Tests for SAS to Python Translation Framework
Features: Fixtures, Parametrization, Custom Markers, Property-based Testing, Mocking
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings, example
from hypothesis.extra.pandas import data_frames, columns
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any
import warnings

# Import the SAS translation framework (assuming it's in the same directory)
# from sas_translator import SASTranslator, SASValidator, SASValidationResult, create_sample_data

# For testing purposes, we'll include the necessary classes here
# (In practice, you'd import from the main module)

# Custom pytest markers
pytestmark = [
    pytest.mark.sas_translation,
    pytest.mark.integration
]

# Custom markers defined in pytest.ini or conftest.py
pytest.mark.slow = pytest.mark.slow
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.regression = pytest.mark.regression
pytest.mark.performance = pytest.mark.performance


class TestDataFixtures:
    """Test data fixtures and factory methods"""
    
    @staticmethod
    def employee_data_factory(n_rows: int = 100, seed: int = 42) -> pd.DataFrame:
        """Factory for creating employee test data"""
        np.random.seed(seed)
        return pd.DataFrame({
            'id': range(1, n_rows + 1),
            'age': np.random.randint(18, 80, n_rows),
            'gender': np.random.choice(['M', 'F'], n_rows),
            'salary': np.random.normal(50000, 15000, n_rows),
            'department': np.random.choice(['Sales', 'Marketing', 'IT', 'HR'], n_rows),
            'hire_date': pd.date_range('2020-01-01', periods=n_rows, freq='D'),
            'performance_score': np.random.uniform(1, 5, n_rows)
        })
    
    @staticmethod
    def financial_data_factory(n_rows: int = 50, seed: int = 123) -> pd.DataFrame:
        """Factory for creating financial test data"""
        np.random.seed(seed)
        return pd.DataFrame({
            'account_id': range(1000, 1000 + n_rows),
            'balance': np.random.lognormal(8, 1, n_rows),
            'transaction_count': np.random.poisson(10, n_rows),
            'account_type': np.random.choice(['Checking', 'Savings', 'Credit'], n_rows),
            'open_date': pd.date_range('2019-01-01', periods=n_rows, freq='M'),
            'risk_score': np.random.beta(2, 5, n_rows) * 100
        })


# ============= FIXTURES =============

@pytest.fixture(scope="session")
def sample_employee_data():
    """Session-scoped fixture for employee data"""
    return TestDataFixtures.employee_data_factory(100, 42)


@pytest.fixture(scope="session")
def sample_financial_data():
    """Session-scoped fixture for financial data"""
    return TestDataFixtures.financial_data_factory(50, 123)


@pytest.fixture(scope="function")
def sas_translator():
    """Fresh SAS translator instance for each test"""
    from sas_translator import SASTranslator  # Assuming import works
    return SASTranslator(debug=True)


@pytest.fixture(scope="function")
def sas_validator():
    """Fresh SAS validator instance for each test"""
    from sas_translator import SASValidator  # Assuming import works
    return SASValidator(tolerance=1e-6)


@pytest.fixture(scope="function")
def temp_data_dir():
    """Temporary directory for test data files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(params=[10, 100, 1000], ids=["small", "medium", "large"])
def dataset_sizes(request):
    """Parametrized fixture for different dataset sizes"""
    return request.param


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Auto-used fixture to suppress pandas warnings during tests"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        yield


# ============= CUSTOM MARKERS AND DECORATORS =============

def requires_large_memory(func):
    """Decorator for tests requiring large amounts of memory"""
    return pytest.mark.skipif(
        os.environ.get('PYTEST_SKIP_MEMORY_INTENSIVE') == '1',
        reason="Skipping memory-intensive tests"
    )(func)


def slow_test(func):
    """Decorator for slow tests"""
    return pytest.mark.slow(func)


# ============= UNIT TESTS =============

@pytest.mark.unit
class TestSASTranslatorUnit:
    """Unit tests for SAS Translator components"""
    
    def test_translator_initialization(self, sas_translator):
        """Test translator initializes correctly"""
        assert isinstance(sas_translator.datasets, dict)
        assert len(sas_translator.datasets) == 0
        assert sas_translator.debug is True
        assert isinstance(sas_translator.operations_log, list)
    
    @pytest.mark.parametrize("debug_mode", [True, False])
    def test_debug_mode_toggle(self, debug_mode):
        """Test debug mode functionality"""
        from sas_translator import SASTranslator
        translator = SASTranslator(debug=debug_mode)
        assert translator.debug == debug_mode
    
    def test_dataset_storage(self, sas_translator, sample_employee_data):
        """Test dataset storage and retrieval"""
        sas_translator.datasets['test_data'] = sample_employee_data
        retrieved_data = sas_translator._get_dataframe('test_data')
        pd.testing.assert_frame_equal(retrieved_data, sample_employee_data)
    
    def test_operation_logging(self, sas_translator):
        """Test operation logging functionality"""
        initial_log_count = len(sas_translator.operations_log)
        sas_translator._log_operation("Test operation")
        assert len(sas_translator.operations_log) == initial_log_count + 1
        assert "Test operation" in sas_translator.operations_log[-1]


@pytest.mark.unit
@pytest.mark.parametrize("input_type,expected_type", [
    ("string_input", pd.DataFrame),
    (pd.DataFrame({'a': [1, 2, 3]}), pd.DataFrame),
])
def test_get_dataframe_method(sas_translator, input_type, expected_type):
    """Test _get_dataframe method with different input types"""
    if isinstance(input_type, str):
        # Test with string input (should return empty DataFrame if not found)
        result = sas_translator._get_dataframe(input_type)
        assert isinstance(result, expected_type)
        assert len(result) == 0
    else:
        # Test with DataFrame input
        result = sas_translator._get_dataframe(input_type)
        assert isinstance(result, expected_type)
        pd.testing.assert_frame_equal(result, input_type)


# ============= INTEGRATION TESTS =============

@pytest.mark.integration
class TestSASProcedures:
    """Integration tests for SAS procedure implementations"""
    
    def test_proc_freq_basic(self, sas_translator, sample_employee_data):
        """Test basic PROC FREQ functionality"""
        sas_translator.datasets['employees'] = sample_employee_data
        
        results = sas_translator.proc_freq(
            data='employees',
            variables=['gender', 'department']
        )
        
        assert 'gender' in results
        assert 'department' in results
        assert isinstance(results['gender'], pd.DataFrame)
        assert 'Frequency' in results['gender'].columns
        assert 'Percent' in results['gender'].columns
    
    def test_proc_freq_crosstabs(self, sas_translator, sample_employee_data):
        """Test PROC FREQ with cross-tabulation"""
        sas_translator.datasets['employees'] = sample_employee_data
        
        results = sas_translator.proc_freq(
            data='employees',
            variables=['gender'],
            tables=['gender*department']
        )
        
        assert 'gender*department' in results
        assert isinstance(results['gender*department'], pd.DataFrame)
    
    @pytest.mark.parametrize("stats", [
        ['mean', 'std'],
        ['min', 'max', 'count'],
        ['mean', 'std', 'min', 'max', 'count', 'median']
    ])
    def test_proc_means_different_stats(self, sas_translator, sample_employee_data, stats):
        """Test PROC MEANS with different statistics"""
        sas_translator.datasets['employees'] = sample_employee_data
        
        result = sas_translator.proc_means(
            data='employees',
            variables=['age', 'salary'],
            stats=stats
        )
        
        assert not result.empty
        # Check that requested stats are computed
        for stat in stats:
            if stat in ['mean', 'std', 'min', 'max', 'count']:
                assert stat in result.index or stat in result.columns
    
    @pytest.mark.parametrize("sort_vars,ascending", [
        (['age'], True),
        (['salary'], False),
        (['department', 'age'], [True, False]),
    ])
    def test_proc_sort_variations(self, sas_translator, sample_employee_data, sort_vars, ascending):
        """Test PROC SORT with different configurations"""
        sas_translator.datasets['employees'] = sample_employee_data
        
        result = sas_translator.proc_sort(
            data='employees',
            by=sort_vars,
            ascending=ascending,
            output_name='sorted_employees'
        )
        
        assert len(result) == len(sample_employee_data)
        assert 'sorted_employees' in sas_translator.datasets
        
        # Verify sorting
        if len(sort_vars) == 1:
            if ascending:
                assert result[sort_vars[0]].is_monotonic_increasing
            else:
                assert result[sort_vars[0]].is_monotonic_decreasing


# ============= PROPERTY-BASED TESTS =============

@pytest.mark.unit
class TestPropertyBased:
    """Property-based tests using Hypothesis"""
    
    @given(
        data_frames(
            columns=columns(['id', 'value'], dtype=int),
            rows=st.tuples(
                st.integers(min_value=1, max_value=1000),
                st.integers(min_value=0, max_value=100000)
            )
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_data_step_preserves_row_count(self, sas_translator, df):
        """Property: DATA step should preserve row count when no filtering"""
        if len(df) == 0:
            return  # Skip empty DataFrames
            
        result = sas_translator.data_step(
            input_data=df,
            output_name='test_output'
        )
        
        assert len(result) == len(df)
    
    @given(
        st.lists(
            st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
            min_size=1,
            max_size=100
        )
    )
    @settings(max_examples=30)
    def test_proc_freq_handles_any_categorical_data(self, sas_translator, categories):
        """Property: PROC FREQ should handle any categorical data"""
        df = pd.DataFrame({'category': categories})
        sas_translator.datasets['test_data'] = df
        
        results = sas_translator.proc_freq(
            data='test_data',
            variables=['category']
        )
        
        assert 'category' in results
        assert len(results['category']) == len(set(categories))
        assert results['category']['Frequency'].sum() == len(categories)
    
    @given(
        st.lists(
            st.floats(min_value=-1000000, max_value=1000000, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=1000
        )
    )
    @settings(max_examples=20)
    @example([1.0, 2.0, 3.0, 4.0, 5.0])  # Always test this specific case
    def test_proc_means_statistical_properties(self, sas_translator, numeric_data):
        """Property: PROC MEANS should maintain statistical relationships"""
        df = pd.DataFrame({'value': numeric_data})
        sas_translator.datasets['test_data'] = df
        
        result = sas_translator.proc_means(
            data='test_data',
            variables=['value'],
            stats=['mean', 'std', 'min', 'max', 'count']
        )
        
        # Basic statistical properties
        assert result.loc['min', 'value'] <= result.loc['mean', 'value']
        assert result.loc['mean', 'value'] <= result.loc['max', 'value']
        assert result.loc['count', 'value'] == len(numeric_data)
        assert result.loc['std', 'value'] >= 0


# ============= MOCK TESTS =============

@pytest.mark.unit
class TestMocking:
    """Tests using mocks for external dependencies"""
    
    @patch('pandas.DataFrame.to_csv')
    def test_data_export_mocked(self, mock_to_csv, sas_translator, sample_employee_data):
        """Test data export functionality with mocked file operations"""
        sas_translator.datasets['employees'] = sample_employee_data
        
        # Simulate export functionality (would be added to SASTranslator)
        df = sas_translator._get_dataframe('employees')
        df.to_csv('test_file.csv', index=False)
        
        mock_to_csv.assert_called_once_with('test_file.csv', index=False)
    
    def test_conditional_logic_with_mock_function(self, sas_translator, sample_employee_data):
        """Test conditional logic with mocked transformation function"""
        mock_transform = Mock(return_value=sample_employee_data.copy())
        
        result = sas_translator.data_step(
            input_data=sample_employee_data,
            output_name='test_output',
            operations=[mock_transform]
        )
        
        mock_transform.assert_called_once()
        assert len(result) == len(sample_employee_data)


# ============= PERFORMANCE TESTS =============

@pytest.mark.performance
@pytest.mark.slow
class TestPerformance:
    """Performance tests for SAS translation operations"""
    
    @requires_large_memory
    @pytest.mark.parametrize("n_rows", [1000, 10000, 100000])
    def test_large_dataset_performance(self, sas_translator, n_rows):
        """Test performance with large datasets"""
        import time
        
        large_data = TestDataFixtures.employee_data_factory(n_rows, 42)
        sas_translator.datasets['large_data'] = large_data
        
        start_time = time.time()
        results = sas_translator.proc_freq(
            data='large_data',
            variables=['department']
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Performance assertions (adjust thresholds as needed)
        assert processing_time < 10.0  # Should complete within 10 seconds
        assert 'department' in results
        assert len(results['department']) > 0
        
        # Log performance for monitoring
        print(f"Processing {n_rows} rows took {processing_time:.2f} seconds")
    
    @slow_test
    def test_memory_usage_stability(self, sas_translator):
        """Test memory usage doesn't grow excessively"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform multiple operations
        for i in range(100):
            data = TestDataFixtures.employee_data_factory(100, i)
            sas_translator.datasets[f'data_{i}'] = data
            
            results = sas_translator.proc_freq(
                data=f'data_{i}',
                variables=['department']
            )
            
            # Clean up to prevent excessive memory usage
            if i % 10 == 0:
                del sas_translator.datasets[f'data_{i}']
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory shouldn't increase by more than 100MB
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB"


# ============= REGRESSION TESTS =============

@pytest.mark.regression
class TestRegression:
    """Regression tests to catch breaking changes"""
    
    def test_known_proc_freq_output_format(self, sas_translator, sample_employee_data):
        """Regression test for PROC FREQ output format"""
        sas_translator.datasets['employees'] = sample_employee_data
        
        results = sas_translator.proc_freq(
            data='employees',
            variables=['gender']
        )
        
        # Expected format should remain consistent
        gender_result = results['gender']
        expected_columns = ['gender', 'Frequency', 'Percent']
        
        assert list(gender_result.columns) == expected_columns
        assert gender_result['Percent'].dtype in ['float64', 'float32']
        assert (gender_result['Percent'] >= 0).all()
        assert (gender_result['Percent'] <= 100).all()
    
    def test_proc_means_backward_compatibility(self, sas_translator, sample_employee_data):
        """Ensure PROC MEANS maintains backward compatibility"""
        sas_translator.datasets['employees'] = sample_employee_data
        
        # Test old-style call (should still work)
        result = sas_translator.proc_means(
            data='employees',
            variables=['age', 'salary']
        )
        
        # Should have default statistics
        default_stats = ['mean', 'std', 'min', 'max', 'count']
        for stat in default_stats:
            assert stat in result.index
    
    @pytest.mark.parametrize("merge_type", ['inner', 'left', 'right', 'outer'])
    def test_merge_operations_consistency(self, sas_translator, merge_type):
        """Test merge operations produce consistent results"""
        # Create reproducible datasets
        left_data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'value_left': ['A', 'B', 'C', 'D']
        })
        
        right_data = pd.DataFrame({
            'id': [2, 3, 4, 5],
            'value_right': ['X', 'Y', 'Z', 'W']
        })
        
        result = sas_translator.merge_datasets(
            left=left_data,
            right=right_data,
            by='id',
            how=merge_type
        )
        
        # Basic consistency checks
        assert 'id' in result.columns
        assert not result.empty or merge_type == 'inner'  # inner join might be empty
        
        # Specific checks based on merge type
        if merge_type == 'inner':
            expected_ids = {2, 3, 4}
            assert set(result['id']) == expected_ids
        elif merge_type == 'left':
            expected_ids = {1, 2, 3, 4}
            assert set(result['id']) == expected_ids


# ============= VALIDATION TESTS =============

@pytest.mark.integration
class TestSASValidator:
    """Tests for the SAS validation framework"""
    
    def test_dataframe_validation_success(self, sas_validator):
        """Test successful DataFrame validation"""
        df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df2 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        result = sas_validator.validate_dataframe(df1, df2, "identical_dataframes")
        
        assert result.passed is True
        assert result.test_name == "identical_dataframes"
        assert result.difference is None
    
    def test_dataframe_validation_failure(self, sas_validator):
        """Test DataFrame validation failure"""
        df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df2 = pd.DataFrame({'a': [1, 2, 4], 'b': [4, 5, 6]})  # Different value
        
        result = sas_validator.validate_dataframe(df1, df2, "different_dataframes")
        
        assert result.passed is False
        assert result.test_name == "different_dataframes"
        assert result.difference is not None
    
    @pytest.mark.parametrize("tolerance,val1,val2,should_pass", [
        (1e-6, 1.0, 1.0000001, True),
        (1e-6, 1.0, 1.000001, False),
        (1e-3, 1.0, 1.0005, True),
        (0, 1.0, 1.0, True),
    ])
    def test_numeric_validation_tolerance(self, tolerance, val1, val2, should_pass):
        """Test numeric validation with different tolerances"""
        validator = SASValidator(tolerance=tolerance)
        
        result = validator.validate_numeric(val1, val2, f"tolerance_test_{tolerance}")
        
        assert result.passed == should_pass
        assert result.tolerance == tolerance
    
    def test_validation_summary(self, sas_validator):
        """Test validation summary functionality"""
        # Add some test results
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'a': [1, 2, 3]})
        df3 = pd.DataFrame({'a': [1, 2, 4]})
        
        sas_validator.validate_dataframe(df1, df2, "pass_test_1")
        sas_validator.validate_dataframe(df1, df2, "pass_test_2")
        sas_validator.validate_dataframe(df1, df3, "fail_test_1")
        
        summary = sas_validator.get_summary()
        
        assert summary['total_tests'] == 3
        assert summary['passed_tests'] == 2
        assert summary['failed_tests'] == 1
        assert summary['pass_rate'] == 2/3
        assert 'fail_test_1' in summary['failed_test_names']


# ============= CUSTOM FIXTURES FOR COMPLEX SCENARIOS =============

@pytest.fixture
def complex_sas_scenario(sas_translator):
    """Fixture that sets up a complex SAS-like scenario"""
    # Employee master data
    employees = pd.DataFrame({
        'emp_id': range(1, 101),
        'first_name': [f'Employee_{i}' for i in range(1, 101)],
        'last_name': [f'Last_{i}' for i in range(1, 101)],
        'department': np.random.choice(['Sales', 'Marketing', 'IT', 'HR', 'Finance'], 100),
        'hire_date': pd.date_range('2020-01-01', periods=100, freq='D'),
        'salary': np.random.normal(50000, 15000, 100)
    })
    
    # Performance data
    performance = pd.DataFrame({
        'emp_id': np.random.choice(range(1, 101), 150),  # Some employees have multiple records
        'review_date': pd.date_range('2023-01-01', periods=150, freq='M'),
        'performance_score': np.random.uniform(1, 5, 150),
        'bonus_eligible': np.random.choice([True, False], 150)
    })
    
    # Department budget data
    budgets = pd.DataFrame({
        'department': ['Sales', 'Marketing', 'IT', 'HR', 'Finance'],
        'budget_2023': [500000, 300000, 800000, 200000, 400000],
        'budget_2024': [550000, 320000, 850000, 220000, 420000]
    })
    
    sas_translator.datasets['employees'] = employees
    sas_translator.datasets['performance'] = performance
    sas_translator.datasets['budgets'] = budgets
    
    return {
        'translator': sas_translator,
        'employees': employees,
        'performance': performance,
        'budgets': budgets
    }


@pytest.mark.integration
@pytest.mark.slow
def test_complex_sas_workflow(complex_sas_scenario):
    """Test complex SAS-like workflow with multiple operations"""
    scenario = complex_sas_scenario
    translator = scenario['translator']
    
    # Step 1: Merge employee and performance data
    emp_perf = translator.merge_datasets(
        left='employees',
        right='performance',
        by='emp_id',
        how='left',
        output_name='emp_performance'
    )
    
    # Step 2: Add calculated fields
    def add_calculations(df):
        # Calculate years of service
        df['years_service'] = (pd.Timestamp.now() - df['hire_date']).dt.days / 365.25
        
        # Performance categories
        conditions = [
            df['performance_score'] < 2,
            (df['performance_score'] >= 2) & (df['performance_score'] < 3.5),
            df['performance_score'] >= 3.5
        ]
        choices = ['Low', 'Average', 'High']
        df['performance_category'] = np.select(conditions, choices, default='Unknown')
        
        return df
    
    enhanced_data = translator.data_step(
        input_data='emp_performance',
        output_name='enhanced_employee_data',
        operations=[add_calculations]
    )
    
    # Step 3: Generate summary statistics by department
    dept_summary = translator.proc_means(
        data='enhanced_employee_data',
        variables=['salary', 'performance_score', 'years_service'],
        class_vars=['department'],
        stats=['mean', 'std', 'count']
    )
    
    # Step 4: Create frequency tables
    freq_results = translator.proc_freq(
        data='enhanced_employee_data',
        variables=['department', 'performance_category'],
        tables=['department*performance_category']
    )
    
    # Validation assertions
    assert len(enhanced_data) >= len(scenario['employees'])  # Should have at least as many rows
    assert 'years_service' in enhanced_data.columns
    assert 'performance_category' in enhanced_data.columns
    assert not dept_summary.empty
    assert 'department' in freq_results
    assert 'performance_category' in freq_results
    assert 'department*performance_category' in freq_results
    
    # Verify data quality
    assert enhanced_data['years_service'].notna().all()
    assert enhanced_data['performance_category'].isin(['Low', 'Average', 'High', 'Unknown']).all()


# ============= PARAMETERIZED EDGE CASE TESTS =============

@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases and error conditions"""
    
    @pytest.mark.parametrize("empty_data_scenario", [
        pd.DataFrame(),  # Completely empty
        pd.DataFrame({'col1': []}),  # Empty with columns
        pd.DataFrame({'col1': [None, None, None]}),  # All null values
    ])
    def test_empty_data_handling(self, sas_translator, empty_data_scenario):
        """Test handling of various empty data scenarios"""
        sas_translator.datasets['empty_data'] = empty_data_scenario
        
        # These operations should not crash
        freq_results = sas_translator.proc_freq(
            data='empty_data',
            variables=list(empty_data_scenario.columns) if not empty_data_scenario.empty else []
        )
        
        if not empty_data_scenario.empty and len(empty_data_scenario.columns) > 0:
            means_results = sas_translator.proc_means(
                data='empty_data',
                variables=list(empty_data_scenario.select_dtypes(include=[np.number]).columns)
            )
        
        # Should handle gracefully without errors
        assert isinstance(freq_results, dict)
    
    @pytest.mark.parametrize("invalid_column", [
        'nonexistent_column',
        '',
        'col with spaces',
        123,  # Non-string column name
    ])
    def test_invalid_column_handling(self, sas_translator, sample_employee_data, invalid_column):
        """Test handling of invalid column references"""
        sas_translator.datasets['test_data'] = sample_employee_data
        
        # Should handle invalid columns gracefully
        if isinstance(invalid_column, str) and invalid_column:
            freq_results = sas_translator.proc_freq(
                data='test_data',
                variables=[invalid_column]
            )
            
            # Should return empty results for invalid columns
            if invalid_column not in sample_employee_data.columns:
                assert invalid_column not in freq_results or len(freq_results[invalid_column]) == 0
    
    def test_circular_dataset_references(self, sas_translator):
        """Test handling of circular dataset references"""
        # This test ensures the system doesn't get stuck in infinite loops
        df1 = pd.DataFrame({'id': [1, 2, 3], 'ref': ['dataset2', 'dataset2', 'dataset2']})
        df2 = pd.DataFrame({'id': [1, 2, 3], 'ref': ['dataset1', 'dataset1', 'dataset1']})
        
        sas_translator.datasets['dataset1'] = df1
        sas_translator.datasets['dataset2'] = df2
        
        # Operations should complete without infinite recursion
        result = sas_translator.proc_freq(
            data='dataset1',
            variables=['ref']
        )
        
        assert 'ref' in result
        assert len(result['ref']) > 0


# ============= PYTEST CONFIGURATION AND HELPERS =============

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "regression: Regression tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "sas_translation: SAS translation specific tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle custom markers"""
    if config.getoption("--run-slow"):
        # Don't skip slow tests if explicitly requested
        return
    
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session", autouse=True)
def test_session_setup():
    """Session-wide setup for all tests"""
    print("\n" + "="*50)
    print("Starting SAS Translation Test Suite")
    print("="*50)
    
    yield
    
    print("\n" + "="*50)
    print("SAS Translation Test Suite Complete")
    print("="*50)


# ============= CUSTOM ASSERTIONS =============

def assert_dataframes_approximately_equal(df1: pd.DataFrame, df2: pd.DataFrame, 
                                        tolerance: float = 1e-6, 
                                        ignore_order: bool = False):
    """Custom assertion for approximate DataFrame equality"""
    if ignore_order:
        df1 = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
        df2 = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)
    
    # Check shapes
    assert df1.shape == df2.shape, f"DataFrame shapes differ: {df1.shape} vs {df2.shape}"
    
    # Check columns
    assert list(df1.columns) == list(df2.columns), f"Columns differ: {list(df1.columns)} vs {list(df2.columns)}"
    
    # Check numeric columns with tolerance
    for col in df1.columns:
        if pd.api.types.is_numeric_dtype(df1[col]):
            np.testing.assert_allclose(
                df1[col].fillna(0), df2[col].fillna(0), 
                rtol=tolerance, atol=tolerance,
                err_msg=f"Column {col} values differ beyond tolerance"
            )
        else:
            pd.testing.assert_series_equal(
                df1[col], df2[col], 
                check_names=True,
                check_exact=True
            )


# ============= EXAMPLE USAGE AND DOCUMENTATION =============

if __name__ == "__main__":
    """
    Example usage:
    
    # Run all tests
    pytest test_sas_translation.py -v
    
    # Run only unit tests
    pytest test_sas_translation.py -m unit -v
    
    # Run with slow tests
    pytest test_sas_translation.py --run-slow -v
    
    # Run specific test class
    pytest test_sas_translation.py::TestSASTranslatorUnit -v
    
    # Run with coverage
    pytest test_sas_translation.py --cov=sas_translator --cov-report=html
    
    # Run performance tests only
    pytest test_sas_translation.py -m performance -v
    
    # Generate test report
    pytest test_sas_translation.py --html=report.html --self-contained-html
    """
    
    # Example of running tests programmatically
    import sys
    
    # This would run the tests if the script is executed directly
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not slow"  # Skip slow tests by default
    ])
    
    sys.exit(exit_code)