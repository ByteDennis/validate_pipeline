"""
Comprehensive Pytest Tests for Oracle-Athena Data Comparator
Features: Mocking, Fixtures, Parametrization, Integration Tests
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import warnings

# Import the comparator classes (assuming they're in oracle_athena_comparator.py)
# from oracle_athena_comparator import (
#     DatabaseConfig, DatabaseConnector, DataComparator, 
#     ComparisonResult, create_sample_queries
# )

# For testing purposes, we'll include some basic classes here
# In practice, you'd import from the main module

# Test fixtures and sample data
@pytest.fixture
def sample_oracle_data():
    """Sample Oracle-like data"""
    return pd.DataFrame({
        'employee_id': [1, 2, 3, 4, 5],
        'first_name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'last_name': ['Doe', 'Smith', 'Johnson', 'Brown', 'Wilson'],
        'email': ['john.doe@company.com', 'jane.smith@company.com', 
                 'bob.johnson@company.com', 'alice.brown@company.com', 
                 'charlie.wilson@company.com'],
        'hire_date': pd.date_range('2020-01-01', periods=5, freq='M'),
        'salary': [50000.0, 55000.0, 60000.0, 45000.0, 52000.0],
        'department_id': [1, 2, 1, 3, 2]
    })


@pytest.fixture
def sample_athena_data():
    """Sample Athena-like data (identical to Oracle for matching tests)"""
    return pd.DataFrame({
        'employee_id': [1, 2, 3, 4, 5],
        'first_name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'last_name': ['Doe', 'Smith', 'Johnson', 'Brown', 'Wilson'],
        'email': ['john.doe@company.com', 'jane.smith@company.com', 
                 'bob.johnson@company.com', 'alice.brown@company.com', 
                 'charlie.wilson@company.com'],
        'hire_date': pd.date_range('2020-01-01', periods=5, freq='M'),
        'salary': [50000.0, 55000.0, 60000.0, 45000.0, 52000.0],
        'department_id': [1, 2, 1, 3, 2]
    })


@pytest.fixture
def sample_athena_data_with_differences():
    """Sample Athena data with differences for testing mismatch scenarios"""
    return pd.DataFrame({
        'employee_id': [1, 2, 3, 4, 6],  # Different ID: 6 instead of 5
        'first_name': ['John', 'Jane', 'Bob', 'Alice', 'Charles'],  # Different name
        'last_name': ['Doe', 'Smith', 'Johnson', 'Brown', 'Wilson'],
        'email': ['john.doe@company.com', 'jane.smith@company.com', 
                 'bob.johnson@company.com', 'alice.brown@company.com', 
                 'charles.wilson@company.com'],
        'hire_date': pd.date_range('2020-01-01', periods=5, freq='M'),
        'salary': [50000.0, 55000.0, 60000.0, 45000.0, 53000.0],  # Different salary
        'department_id': [1, 2, 1, 3, 2]
    })


@pytest.fixture
def sample_config():
    """Sample database configuration"""
    from oracle_athena_comparator import DatabaseConfig
    return DatabaseConfig(
        oracle_config={
            'user': 'test_user',
            'password': 'test_password',
            'dsn': 'test_host:1521/test_service'
        },
        athena_config={
            'aws_access_key_id': 'test_access_key',
            'aws_secret_access_key': 'test_secret_key',
            'region_name': 'us-east-1',
            's3_staging_dir': 's3://test-bucket/results/',
            'schema_name': 'test_schema'
        }
    )


@pytest.fixture
def mock_database_connector(sample_config):
    """Mocked database connector"""
    from oracle_athena_comparator import DatabaseConnector
    
    with patch.object(DatabaseConnector, 'connect_oracle'), \
         patch.object(DatabaseConnector, 'connect_athena'):
        connector = DatabaseConnector(sample_config)
        yield connector


@pytest.fixture
def mock_data_comparator(sample_config):
    """Mocked data comparator"""
    from oracle_athena_comparator import DataComparator
    
    with patch('oracle_athena_comparator.DatabaseConnector'):
        comparator = DataComparator(sample_config)
        yield comparator


@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


# ============= UNIT TESTS =============

@pytest.mark.unit
class TestDatabaseConfig:
    """Unit tests for DatabaseConfig"""
    
    def test_default_config_initialization(self):
        """Test default configuration initialization"""
        from oracle_athena_comparator import DatabaseConfig
        config = DatabaseConfig()
        
        assert 'user' in config.oracle_config
        assert 'aws_access_key_id' in config.athena_config
        assert config.oracle_config['encoding'] == 'UTF-8'
        assert config.athena_config['region_name'] == 'us-east-1'
    
    def test_custom_config_initialization(self, sample_config):
        """Test custom configuration initialization"""
        assert sample_config.oracle_config['user'] == 'test_user'
        assert sample_config.athena_config['aws_access_key_id'] == 'test_access_key'
    
    @pytest.mark.parametrize("config_type,expected_keys", [
        ('oracle_config', ['user', 'password', 'dsn']),
        ('athena_config', ['aws_access_key_id', 'aws_secret_access_key', 'region_name', 's3_staging_dir'])
    ])
    def test_required_config_keys(self, sample_config, config_type, expected_keys):
        """Test that required configuration keys are present"""
        config_dict = getattr(sample_config, config_type)
        for key in expected_keys:
            assert key in config_dict


@pytest.mark.unit
class TestComparisonResult:
    """Unit tests for ComparisonResult"""
    
    def test_comparison_result_initialization(self):
        """Test ComparisonResult initialization"""
        from oracle_athena_comparator import ComparisonResult
        
        result = ComparisonResult(
            table_name='test_table',
            oracle_count=100,
            athena_count=100,
            match_status='MATCH'
        )
        
        assert result.table_name == 'test_table'
        assert result.oracle_count == 100
        assert result.athena_count == 100
        assert result.match_status == 'MATCH'
        assert result.count_match is True
    
    def test_count_match_property(self):
        """Test count_match property logic"""
        from oracle_athena_comparator import ComparisonResult
        
        # Matching counts
        result_match = ComparisonResult('test', 100, 100, 'MATCH')
        assert result_match.count_match is True
        
        # Non-matching counts
        result_no_match = ComparisonResult('test', 100, 95, 'MISMATCH')
        assert result_no_match.count_match is False
    
    def test_summary_property(self):
        """Test summary property"""
        from oracle_athena_comparator import ComparisonResult
        
        result = ComparisonResult(
            table_name='test_table',
            oracle_count=100,
            athena_count=95,
            match_status='RECORD_COUNT_MISMATCH'
        )
        
        summary = result.summary
        assert summary['table_name'] == 'test_table'
        assert summary['oracle_count'] == 100
        assert summary['athena_count'] == 95
        assert summary['count_difference'] == 5
        assert summary['match_status'] == 'RECORD_COUNT_MISMATCH'
        assert 'timestamp' in summary


# ============= MOCK TESTS =============

@pytest.mark.unit
class TestDatabaseConnectorMocks:
    """Unit tests using mocks for DatabaseConnector"""
    
    @patch('oracledb.connect')
    def test_oracle_connection(self, mock_oracle_connect, sample_config):
        """Test Oracle database connection with mocking"""
        from oracle_athena_comparator import DatabaseConnector
        
        mock_connection = Mock()
        mock_oracle_connect.return_value = mock_connection
        mock_connection.ping.return_value = True
        
        connector = DatabaseConnector(sample_config)
        connection = connector.connect_oracle()
        
        mock_oracle_connect.assert_called_once_with(**sample_config.oracle_config)
        assert connection == mock_connection
    
    @patch('pyathena.connect')
    def test_athena_connection(self, mock_athena_connect, sample_config):
        """Test Athena database connection with mocking"""
        from oracle_athena_comparator import DatabaseConnector
        
        mock_connection = Mock()
        mock_athena_connect.return_value = mock_connection
        
        connector = DatabaseConnector(sample_config)
        connection = connector.connect_athena()
        
        mock_athena_connect.assert_called_once()
        assert connection == mock_connection
    
    @patch('pandas.read_sql')
    def test_oracle_query_execution(self, mock_read_sql, mock_database_connector, sample_oracle_data):
        """Test Oracle query execution"""
        mock_read_sql.return_value = sample_oracle_data
        
        result = mock_database_connector.execute_oracle_query("SELECT * FROM test_table")
        
        mock_read_sql.assert_called_once()
        pd.testing.assert_frame_equal(result, sample_oracle_data)
    
    def test_athena_query_execution(self, mock_database_connector, sample_athena_data):
        """Test Athena query execution with mocked cursor"""
        mock_cursor = Mock()
        mock_cursor.as_pandas.return_value = sample_athena_data
        
        mock_connection = Mock()
        mock_connection.cursor.return_value = mock_cursor
        
        with patch.object(mock_database_connector, 'connect_athena', return_value=mock_connection):
            result = mock_database_connector.execute_athena_query("SELECT * FROM test_table")
        
        mock_cursor.execute.assert_called_once_with("SELECT * FROM test_table")
        mock_cursor.as_pandas.assert_called_once()
        pd.testing.assert_frame_equal(result, sample_athena_data)


# ============= INTEGRATION TESTS =============

@pytest.mark.integration
class TestDataComparator:
    """Integration tests for DataComparator"""
    
    def test_identical_data_comparison(self, mock_data_comparator, sample_oracle_data, sample_athena_data):
        """Test comparison of identical datasets"""
        
        # Mock the database query methods
        mock_data_comparator.connector.execute_oracle_query = Mock(return_value=sample_oracle_data)
        mock_data_comparator.connector.execute_athena_query = Mock(return_value=sample_athena_data)
        
        result = mock_data_comparator.compare_tables(
            oracle_query="SELECT * FROM employees",
            athena_query="SELECT * FROM employees",
            table_name="employees",
            key_columns=['employee_id']
        )
        
        assert result.table_name == "employees"
        assert result.oracle_count == 5
        assert result.athena_count == 5
        assert result.match_status == "MATCH"
        assert result.count_match is True
    
    def test_different_data_comparison(self, mock_data_comparator, sample_oracle_data, sample_athena_data_with_differences):
        """Test comparison of different datasets"""
        
        mock_data_comparator.connector.execute_oracle_query = Mock(return_value=sample_oracle_data)
        mock_data_comparator.connector.execute_athena_query = Mock(return_value=sample_athena_data_with_differences)
        
        result = mock_data_comparator.compare_tables(
            oracle_query="SELECT * FROM employees",
            athena_query="SELECT * FROM employees",
            table_name="employees",
            key_columns=['employee_id']
        )
        
        assert result.table_name == "employees"
        assert result.oracle_count == 5
        assert result.athena_count == 5
        assert result.match_status != "MATCH"
        assert "differences" in result.differences
    
    @pytest.mark.parametrize("oracle_count,athena_count,expected_status", [
        (100, 100, "MATCH"),
        (100, 95, "RECORD_COUNT_MISMATCH"),
        (0, 100, "RECORD_COUNT_MISMATCH"),
        (100, 0, "RECORD_COUNT_MISMATCH")
    ])
    def test_count_comparison_scenarios(self, mock_data_comparator, oracle_count, athena_count, expected_status):
        """Test various count comparison scenarios"""
        
        # Create test data with specified counts
        oracle_data = pd.DataFrame({
            'id': range(oracle_count),
            'value': range(oracle_count)
        })
        
        athena_data = pd.DataFrame({
            'id': range(athena_count),
            'value': range(athena_count)
        })
        
        mock_data_comparator.connector.execute_oracle_query = Mock(return_value=oracle_data)
        mock_data_comparator.connector.execute_athena_query = Mock(return_value=athena_data)
        
        result = mock_data_comparator.compare_tables(
            oracle_query="SELECT * FROM test_table",
            athena_query="SELECT * FROM test_table",
            table_name="test_table",
            key_columns=['id']
        )
        
        assert result.oracle_count == oracle_count
        assert result.athena_count == athena_count
        
        if oracle_count != athena_count:
            assert "RECORD_COUNT_MISMATCH" in result.match_status or "MISMATCH" in result.match_status


@pytest.mark.integration
class TestSchemaComparison:
    """Tests for schema comparison functionality"""
    
    def test_schema_mismatch_detection(self, mock_data_comparator):
        """Test detection of schema mismatches"""
        
        oracle_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'oracle_only_column': [1, 2, 3]
        })
        
        athena_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'athena_only_column': [1, 2, 3]
        })
        
        mock_data_comparator.connector.execute_oracle_query = Mock(return_value=oracle_data)
        mock_data_comparator.connector.execute_athena_query = Mock(return_value=athena_data)
        
        result = mock_data_comparator.compare_tables(
            oracle_query="SELECT * FROM test_table",
            athena_query="SELECT * FROM test_table",
            table_name="test_table"
        )
        
        schema_diffs = result.differences['schema']
        assert 'oracle_only_column' in schema_diffs['oracle_only_columns']
        assert 'athena_only_column' in schema_diffs['athena_only_columns']
        assert set(schema_diffs['common_columns']) == {'id', 'name'}
    
    def test_data_type_comparison(self, mock_data_comparator):
        """Test data type comparison between datasets"""
        
        oracle_data = pd.DataFrame({
            'id': [1, 2, 3],
            'amount': [10.5, 20.5, 30.5],  # float
            'date_col': pd.date_range('2023-01-01', periods=3)
        })
        
        athena_data = pd.DataFrame({
            'id': ['1', '2', '3'],  # string instead of int
            'amount': [10.5, 20.5, 30.5],
            'date_col': pd.date_range('2023-01-01', periods=3)
        })
        
        mock_data_comparator.connector.execute_oracle_query = Mock(return_value=oracle_data)
        mock_data_comparator.connector.execute_athena_query = Mock(return_value=athena_data)
        
        result = mock_data_comparator.compare_tables(
            oracle_query="SELECT * FROM test_table",
            athena_query="SELECT * FROM test_table",
            table_name="test_table"
        )
        
        type_diffs = result.differences['data_types']
        assert 'id' in type_diffs
        assert 'int' in str(type_diffs['id']['oracle_type']).lower()
        assert 'object' in str(type_diffs['id']['athena_type']).lower()


# ============= STATISTICAL TESTS =============

@pytest.mark.unit
class TestStatisticalComparison:
    """Tests for statistical comparison functionality"""
    
    def test_numeric_statistics_comparison(self, mock_data_comparator):
        """Test statistical comparison of numeric columns"""
        
        np.random.seed(42)
        oracle_data = pd.DataFrame({
            'id': range(1000),
            'value': np.random.normal(100, 15, 1000)
        })
        
        athena_data = pd.DataFrame({
            'id': range(1000),
            'value': np.random.normal(100, 15, 1000)  # Same distribution
        })
        
        mock_data_comparator.connector.execute_oracle_query = Mock(return_value=oracle_data)
        mock_data_comparator.connector.execute_athena_query = Mock(return_value=athena_data)
        
        result = mock_data_comparator.compare_tables(
            oracle_query="SELECT * FROM test_table",
            athena_query="SELECT * FROM test_table",
            table_name="test_table",
            key_columns=['id']
        )
        
        numeric_stats = result.statistics['numeric_stats']
        assert 'value' in numeric_stats
        assert 'oracle_stats' in numeric_stats['value']
        assert 'athena_stats' in numeric_stats['value']
        assert 'ks_test' in numeric_stats['value']
        assert 'mannwhitney_test' in numeric_stats['value']
    
    def test_significantly_different_distributions(self, mock_data_comparator):
        """Test detection of significantly different distributions"""
        
        np.random.seed(42)
        oracle_data = pd.DataFrame({
            'id': range(100),
            'value': np.random.normal(100, 15, 100)  # Mean = 100
        })
        
        athena_data = pd.DataFrame({
            'id': range(100),
            'value': np.random.normal(120, 15, 100)  # Mean = 120 (significantly different)
        })
        
        mock_data_comparator.connector.execute_oracle_query = Mock(return_value=oracle_data)
        mock_data_comparator.connector.execute_athena_query = Mock(return_value=athena_data)
        
        result = mock_data_comparator.compare_tables(
            oracle_query="SELECT * FROM test_table",
            athena_query="SELECT * FROM test_table",
            table_name="test_table",
            key_columns=['id']
        )
        
        # Should detect statistical differences
        assert result.match_status == "STATISTICAL_DIFFERENCES"


# ============= ERROR HANDLING TESTS =============

@pytest.mark.unit
class TestErrorHandling:
    """Tests for error handling scenarios"""
    
    def test_oracle_connection_failure(self, sample_config):
        """Test handling of Oracle connection failures"""
        from oracle_athena_comparator import DatabaseConnector
        
        with patch('oracledb.connect', side_effect=Exception("Connection failed")):
            connector = DatabaseConnector(sample_config)
            
            with pytest.raises(Exception, match="Connection failed"):
                connector.connect_oracle()
    
    def test_athena_connection_failure(self, sample_config):
        """Test handling of Athena connection failures"""
        from oracle_athena_comparator import DatabaseConnector
        
        with patch('pyathena.connect', side_effect=Exception("Athena connection failed")):
            connector = DatabaseConnector(sample_config)
            
            with pytest.raises(Exception, match="Athena connection failed"):
                connector.connect_athena()
    
    def test_query_execution_failure(self, mock_data_comparator):
        """Test handling of query execution failures"""
        
        # Mock query failure
        mock_data_comparator.connector.execute_oracle_query = Mock(
            side_effect=Exception("Query execution failed")
        )
        mock_data_comparator.connector.execute_athena_query = Mock(return_value=pd.DataFrame())
        
        result = mock_data_comparator.compare_tables(
            oracle_query="SELECT * FROM non_existent_table",
            athena_query="SELECT * FROM test_table",
            table_name="test_table"
        )
        
        assert "ERROR" in result.match_status
        assert "Query execution failed" in result.match_status


# ============= REPORTING TESTS =============

@pytest.mark.integration
class TestReporting:
    """Tests for reporting functionality"""
    
    def test_comparison_report_generation(self, mock_data_comparator, temp_output_dir):
        """Test generation of comparison reports"""
        
        # Add some mock comparison results
        from oracle_athena_comparator import ComparisonResult
        
        result1 = ComparisonResult('table1', 100, 100, 'MATCH')
        result2 = ComparisonResult('table2', 95, 100, 'RECORD_COUNT_MISMATCH')
        
        mock_data_comparator.comparison_results = [result1, result2]
        
        output_file = temp_output_dir / 'test_report.json'
        report = mock_data_comparator.generate_comparison_report(str(output_file))
        
        assert output_file.exists()
        assert report['summary']['total_comparisons'] == 2
        assert report['summary']['matches'] == 1
        assert report['summary']['differences'] == 1
        
        # Verify file content
        with open(output_file) as f:
            file_content = json.load(f)
        
        assert file_content['summary']['total_comparisons'] == 2
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.subplots')
    def test_dashboard_creation(self, mock_subplots, mock_savefig, mock_data_comparator, temp_output_dir):
        """Test creation of comparison dashboard"""
        
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_axes = Mock()
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Add some mock results
        from oracle_athena_comparator import ComparisonResult
        mock_data_comparator.comparison_results = [
            ComparisonResult('table1', 100, 100, 'MATCH'),
            ComparisonResult('table2', 95, 100, 'RECORD_COUNT_MISMATCH')
        ]
        
        mock_data_comparator.create_comparison_dashboard(str(temp_output_dir))
        
        mock_savefig.assert_called_once()


# ============= PERFORMANCE TESTS =============

@pytest.mark.performance
@pytest.mark.slow
class TestPerformance:
    """Performance tests for the comparator"""
    
    def test_large_dataset_comparison_performance(self, mock_data_comparator):
        """Test performance with large datasets"""
        import time
        
        # Create large datasets
        n_rows = 10000
        oracle_data = pd.DataFrame({
            'id': range(n_rows),
            'value1': np.random.random(n_rows),
            'value2': np.random.randint(0, 100, n_rows),
            'category': np.random.choice(['A', 'B', 'C'], n_rows)
        })
        
        athena_data = oracle_data.copy()
        
        mock_data_comparator.connector.execute_oracle_query = Mock(return_value=oracle_data)
        mock_data_comparator.connector.execute_athena_query = Mock(return_value=athena_data)
        
        start_time = time.time()
        result = mock_data_comparator.compare_tables(
            oracle_query="SELECT * FROM large_table",
            athena_query="SELECT * FROM large_table",
            table_name="large_table",
            key_columns=['id']
        )
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertions
        assert execution_time < 30.0  # Should complete within 30 seconds
        assert result.oracle_count == n_rows
        assert result.athena_count == n_rows
        
        print(f"Large dataset comparison ({n_rows} rows) took {execution_time:.2f} seconds")
    
    def test_memory_efficiency(self, mock_data_comparator):
        """Test memory efficiency with multiple comparisons"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform multiple comparisons
        for i in range(10):
            data = pd.DataFrame({
                'id': range(1000),
                'value': np.random.random(1000)
            })
            
            mock_data_comparator.connector.execute_oracle_query = Mock(return_value=data)
            mock_data_comparator.connector.execute_athena_query = Mock(return_value=data)
            
            mock_data_comparator.compare_tables(
                oracle_query=f"SELECT * FROM table_{i}",
                athena_query=f"SELECT * FROM table_{i}",
                table_name=f"table_{i}",
                key_columns=['id']
            )
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory shouldn't increase excessively
        assert memory_increase < 50, f"Memory increased by {memory_increase:.2f}MB"


# ============= CUSTOM PYTEST CONFIGURATION =============

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle custom markers"""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Auto-used fixture to suppress warnings during tests"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        yield


# ============= EXAMPLE USAGE =============

if __name__ == "__main__":
    """
    Example usage:
    
    # Run all tests
    pytest test_oracle_athena.py -v
    
    # Run only unit tests
    pytest test_oracle_athena.py -m unit -v
    
    # Run with slow tests
    pytest test_oracle_athena.py --run-slow -v
    
    # Run integration tests only
    pytest test_oracle_athena.py -m integration -v
    
    # Run with coverage
    pytest test_oracle_athena.py --cov=oracle_athena_comparator --cov-report=html
    
    # Run performance tests only
    pytest test_oracle_athena.py -m performance -v
    """
    
    import sys
    
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not slow"
    ])
    
    sys.exit(exit_code)