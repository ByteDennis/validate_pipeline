#!/usr/bin/env python3
"""
Shared fixtures and configuration for pytest test suite.

This module provides common fixtures, test data generation utilities,
and shared configuration for testing the migration analysis system.
"""

import pytest
import pandas as pd
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import os
import sys

# Add source directory to path for imports
sys.path.insert(0, 'src')

# Import modules for type hinting and utilities
import utils
from atem_sisylana import PullStatus


class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def create_sample_table_list(num_tables: int = 5) -> pd.DataFrame:
        """
        Create a sample table list for testing.
        
        Args:
            num_tables: Number of tables to generate
            
        Returns:
            DataFrame with sample table information
        """
        tables = []
        for i in range(num_tables):
            service = f"service_{i % 3}"
            db = f"database_{i % 2}"
            table_name = f"test_table_{i:03d}"
            
            tables.append({
                'pcds_tbl': f'{service}.{table_name}',
                'aws_tbl': f'{db}.{table_name}',
                'col_map': f'{table_name}_mapping',
                'pcds_id': 'created_date' if i % 2 == 0 else 'updated_timestamp',
                'aws_id': 'created_date' if i % 2 == 0 else 'updated_timestamp',
                'group': f'Group_{i % 4}',
                'hydrate_only': 'Y' if i % 3 == 0 else 'N'
            })
        
        return pd.DataFrame(tables)
    
    @staticmethod
    def create_sample_column_mappings(table_names: list) -> dict:
        """
        Create sample column mappings for given table names.
        
        Args:
            table_names: List of table names to create mappings for
            
        Returns:
            Dictionary of column mappings
        """
        mappings = {}
        
        for table_name in table_names:
            mapping_name = f"{table_name}_mapping"
            mappings[mapping_name] = {
                'pcds2aws': {
                    'ID': 'id',
                    'CUSTOMER_ID': 'customer_id',
                    'FIRST_NAME': 'first_name',
                    'LAST_NAME': 'last_name',
                    'EMAIL_ADDRESS': 'email',
                    'PHONE_NUMBER': 'phone',
                    'CREATED_DATE': 'created_date',
                    'UPDATED_TIMESTAMP': 'updated_timestamp',
                    'STATUS': 'status',
                    'AMOUNT': 'amount',
                    'DESCRIPTION': 'description'
                }
            }
        
        return mappings
    
    @staticmethod
    def create_sample_pcds_schema(table_name: str, include_mismatches: bool = False) -> pd.DataFrame:
        """
        Create sample PCDS schema data.
        
        Args:
            table_name: Name of the table
            include_mismatches: Whether to include columns that will mismatch with AWS
            
        Returns:
            DataFrame with PCDS schema information
        """
        base_columns = [
            {'column_name': 'ID', 'data_type': 'NUMBER'},
            {'column_name': 'CUSTOMER_ID', 'data_type': 'NUMBER'},
            {'column_name': 'FIRST_NAME', 'data_type': 'VARCHAR2(50)'},
            {'column_name': 'LAST_NAME', 'data_type': 'VARCHAR2(50)'},
            {'column_name': 'EMAIL_ADDRESS', 'data_type': 'VARCHAR2(100)'},
            {'column_name': 'CREATED_DATE', 'data_type': 'DATE'},
            {'column_name': 'STATUS', 'data_type': 'VARCHAR2(20)'},
        ]
        
        if include_mismatches:
            base_columns.extend([
                {'column_name': 'MISMATCH_COL1', 'data_type': 'NUMBER'},
                {'column_name': 'MISMATCH_COL2', 'data_type': 'VARCHAR2(30)'},
                {'column_name': 'PCDS_ONLY_COL', 'data_type': 'CLOB'},
            ])
        
        df = pd.DataFrame(base_columns)
        
        # Add AWS column mappings
        aws_mapping = {
            'ID': 'id',
            'CUSTOMER_ID': 'customer_id',
            'FIRST_NAME': 'first_name',
            'LAST_NAME': 'last_name',
            'EMAIL_ADDRESS': 'email',
            'CREATED_DATE': 'created_date',
            'STATUS': 'status',
            'MISMATCH_COL1': 'mismatch_col1',
            'MISMATCH_COL2': 'mismatch_col2',
            'PCDS_ONLY_COL': pd.NA  # No AWS mapping
        }
        
        df['aws_colname'] = df['column_name'].map(aws_mapping)
        return df
    
    @staticmethod
    def create_sample_aws_schema(table_name: str, include_mismatches: bool = False) -> pd.DataFrame:
        """
        Create sample AWS schema data.
        
        Args:
            table_name: Name of the table
            include_mismatches: Whether to include columns that will mismatch with PCDS
            
        Returns:
            DataFrame with AWS schema information
        """
        base_columns = [
            {'column_name': 'id', 'data_type': 'bigint'},
            {'column_name': 'customer_id', 'data_type': 'bigint'},
            {'column_name': 'first_name', 'data_type': 'varchar(50)'},
            {'column_name': 'last_name', 'data_type': 'varchar(50)'},
            {'column_name': 'email', 'data_type': 'varchar(100)'},
            {'column_name': 'created_date', 'data_type': 'timestamp'},
            {'column_name': 'status', 'data_type': 'varchar(20)'},
        ]
        
        if include_mismatches:
            base_columns.extend([
                {'column_name': 'mismatch_col1', 'data_type': 'varchar(10)'},  # Type mismatch with PCDS NUMBER
                {'column_name': 'mismatch_col2', 'data_type': 'varchar(50)'},  # Length mismatch
                {'column_name': 'aws_only_col', 'data_type': 'integer'},       # AWS only column
            ])
        
        return pd.DataFrame(base_columns)
    
    @staticmethod
    def create_sample_date_data(table_name: str, date_column: str, 
                              start_date: str = '2023-01-01', 
                              num_days: int = 10, 
                              include_mismatches: bool = False,
                              platform: str = 'pcds') -> pd.DataFrame:
        """
        Create sample date distribution data.
        
        Args:
            table_name: Name of the table
            date_column: Name of the date column
            start_date: Start date for data generation
            num_days: Number of days to generate
            include_mismatches: Whether to include date mismatches
            platform: Platform type ('pcds' or 'aws')
            
        Returns:
            DataFrame with date distribution data
        """
        dates = pd.date_range(start_date, periods=num_days, freq='D')
        count_column = 'NROWS' if platform.lower() == 'pcds' else 'nrows'
        
        data = []
        for i, date in enumerate(dates):
            base_count = 100 + (i * 5)  # Varying counts
            
            # Add mismatches for specific dates if requested
            if include_mismatches and i in [2, 5, 7]:
                count = base_count + (10 if platform.lower() == 'pcds' else -10)
            else:
                count = base_count
            
            data.append({
                date_column: date.strftime('%Y-%m-%d'),
                count_column: count
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_sample_row_count_data(row_count: int) -> pd.DataFrame:
        """
        Create sample row count data.
        
        Args:
            row_count: Number of rows to report
            
        Returns:
            DataFrame with row count information
        """
        return pd.DataFrame([{'nrow': row_count}])


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data files."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def sample_config(test_data_dir):
    """
    Fixture providing a complete sample configuration.
    
    Returns:
        Dictionary containing sample configuration data
    """
    return {
        'input': {
            'table': str(test_data_dir / 'test_tables.xlsx'),
            'env': str(test_data_dir / '.env.test'),
            'name': 'pytest_migration_test',
            'step': 'automated_test',
            'range': [1, 10]
        },
        'output': {
            'folder': str(test_data_dir / 'output'),
            'csv': {
                'path': str(test_data_dir / 'output' / 'migration_results.csv'),
                'columns': [
                    'Consumer Loans Data Product',
                    'PCDS Table Details with DB Name',
                    'Tables delivered in AWS with DB Name',
                    'Hydrated Table only in AWS',
                    'PCDS Table Service Name',
                    'Status',
                    'Row UnMatch',
                    'Row UnMatch Details',
                    'Col Count Details',
                    'Time Span UnMatch',
                    'Time Span Variable',
                    'Time UnMatch Details',
                    'Column Type UnMatch',
                    'Type UnMatch Details',
                    'PCDS Unique Columns',
                    'AWS Unique Columns',
                    'Uncaptured Column Mappings'
                ]
            },
            'log': {
                'sink': str(test_data_dir / 'output' / 'test_analysis.log'),
                'level': 'DEBUG',
                'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}',
                'backtrace': True,
                'diagnose': True
            },
            'to_pkl': str(test_data_dir / 'output' / 'test_migration_analysis.pkl'),
            'next': {
                'file': str(test_data_dir / 'output' / 'next_processing_data.json'),
                'fields': {
                    'pcds_tbl': '',
                    'aws_tbl': '',
                    'pcds_id': '',
                    'aws_id': '',
                    'last_modified': '',
                    'pcds_cols': '',
                    'pcds_types': '',
                    'pcds_nrows': 0,
                    'aws_cols': '',
                    'aws_types': '',
                    'aws_nrows': 0,
                    'time_excludes': ''
                }
            },
            'to_s3': {
                'run': 's3://test-migration-bucket/pytest-runs'
            }
        },
        'column_maps': str(test_data_dir / 'test_column_mappings.json'),
        'match': {
            'drop_cols': ['audit_timestamp', 'temp_field', 'system_internal'],
            'add_cols': ['migration_id', 'etl_timestamp']
        }
    }


@pytest.fixture
def sample_table_list():
    """
    Fixture providing sample table list data.
    
    Returns:
        DataFrame with sample table information
    """
    return TestDataGenerator.create_sample_table_list(5)


@pytest.fixture
def sample_column_mappings(sample_table_list):
    """
    Fixture providing sample column mappings.
    
    Args:
        sample_table_list: Sample table list from fixture
        
    Returns:
        Dictionary of column mappings
    """
    table_names = [name.split('.')[-1] for name in sample_table_list['pcds_tbl']]
    return TestDataGenerator.create_sample_column_mappings(table_names)


@pytest.fixture
def mock_database_responses():
    """
    Fixture providing comprehensive mock database responses.
    
    Returns:
        Dictionary containing mock responses for various scenarios
    """
    return {
        'successful_table': {
            'pcds_schema': TestDataGenerator.create_sample_pcds_schema('successful_table'),
            'pcds_rowcount': TestDataGenerator.create_sample_row_count_data(1000),
            'pcds_dates': TestDataGenerator.create_sample_date_data(
                'successful_table', 'created_date', platform='pcds'
            ),
            'aws_schema': TestDataGenerator.create_sample_aws_schema('successful_table'),
            'aws_rowcount': TestDataGenerator.create_sample_row_count_data(1000),
            'aws_dates': TestDataGenerator.create_sample_date_data(
                'successful_table', 'created_date', platform='aws'
            )
        },
        'mismatch_table': {
            'pcds_schema': TestDataGenerator.create_sample_pcds_schema('mismatch_table', include_mismatches=True),
            'pcds_rowcount': TestDataGenerator.create_sample_row_count_data(1500),
            'pcds_dates': TestDataGenerator.create_sample_date_data(
                'mismatch_table', 'created_date', platform='pcds', include_mismatches=True
            ),
            'aws_schema': TestDataGenerator.create_sample_aws_schema('mismatch_table', include_mismatches=True),
            'aws_rowcount': TestDataGenerator.create_sample_row_count_data(1450),  # Row count mismatch
            'aws_dates': TestDataGenerator.create_sample_date_data(
                'mismatch_table', 'created_date', platform='aws', include_mismatches=True
            )
        },
        'empty_table': {
            'pcds_schema': pd.DataFrame(columns=['column_name', 'data_type', 'aws_colname']),
            'pcds_rowcount': TestDataGenerator.create_sample_row_count_data(0),
            'pcds_dates': pd.DataFrame(columns=['created_date', 'NROWS']),
            'aws_schema': pd.DataFrame(columns=['column_name', 'data_type']),
            'aws_rowcount': TestDataGenerator.create_sample_row_count_data(0),
            'aws_dates': pd.DataFrame(columns=['created_date', 'nrows'])
        }
    }


@pytest.fixture
def file_system_setup(test_data_dir, sample_config, sample_table_list, sample_column_mappings):
    """
    Fixture that sets up a complete file system for testing.
    
    Args:
        test_data_dir: Temporary directory from fixture
        sample_config: Sample configuration from fixture
        sample_table_list: Sample table list from fixture
        sample_column_mappings: Sample column mappings from fixture
        
    Returns:
        Dictionary with paths to created files
    """
    # Create output directory
    output_dir = test_data_dir / 'output'
    output_dir.mkdir(exist_ok=True)
    
    # Create Excel file with table list
    excel_file = test_data_dir / 'test_tables.xlsx'
    sample_table_list.to_excel(excel_file, index=False)
    
    # Create column mappings JSON file
    mappings_file = test_data_dir / 'test_column_mappings.json'
    with open(mappings_file, 'w') as f:
        json.dump(sample_column_mappings, f, indent=2)
    
    # Create environment file
    env_file = test_data_dir / '.env.test'
    env_content = """
# Test environment configuration
PCDS_HOST=test-pcds-host.example.com
PCDS_PORT=1521
PCDS_SERVICE=TESTPCDS
PCDS_USERNAME=test_pcds_user
PCDS_PASSWORD=test_pcds_password

AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=test_access_key
AWS_SECRET_ACCESS_KEY=test_secret_key
ATHENA_DATABASE=test_athena_db
ATHENA_S3_OUTPUT=s3://test-athena-results/

S3_BUCKET=test-migration-bucket
S3_PREFIX=pytest-migration-data
"""
    with open(env_file, 'w') as f:
        f.write(env_content.strip())
    
    # Create configuration file
    config_file = test_data_dir / 'test_config.json'
    with open(config_file, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    return {
        'config_file': config_file,
        'excel_file': excel_file,
        'mappings_file': mappings_file,
        'env_file': env_file,
        'output_dir': output_dir,
        'test_data_dir': test_data_dir
    }


@pytest.fixture
def mock_analyzer_dependencies():
    """
    Fixture providing mocked dependencies for MigrationAnalyzer.
    
    Returns:
        Dictionary of mock objects
    """
    with patch('atem_sisylana.Config') as mock_config, \
         patch('atem_sisylana.utils.MetaConfig') as mock_meta_config, \
         patch('atem_sisylana.utils.read_column_mapping') as mock_col_mapping, \
         patch('atem_sisylana.utils.read_input_excel') as mock_excel, \
         patch('atem_sisylana.load_dotenv') as mock_load_dotenv, \
         patch('os.makedirs') as mock_makedirs, \
         patch('os.remove') as mock_remove:
        
        yield {
            'config': mock_config,
            'meta_config': mock_meta_config,
            'column_mapping': mock_col_mapping,
            'excel': mock_excel,
            'load_dotenv': mock_load_dotenv,
            'makedirs': mock_makedirs,
            'remove': mock_remove
        }


@pytest.fixture
def mock_database_connector():
    """
    Fixture providing a mock database connector.
    
    Returns:
        Mock database connector object
    """
    with patch('atem_sisylana.utils.DatabaseConnector') as mock_connector_class:
        mock_connector_instance = Mock()
        mock_connector_class.return_value = mock_connector_instance
        
        # Default successful query response
        mock_connector_instance.query.return_value = pd.DataFrame([
            {'column_name': 'test_col', 'data_type': 'varchar(100)'}
        ])
        
        yield {
            'connector_class': mock_connector_class,
            'connector_instance': mock_connector_instance
        }


@pytest.fixture
def mock_aws_services():
    """
    Fixture providing mocked AWS services.
    
    Returns:
        Dictionary of mock AWS service objects
    """
    with patch('atem_sisylana.utils.start_run') as mock_start_run, \
         patch('atem_sisylana.utils.end_run') as mock_end_run, \
         patch('atem_sisylana.utils.aws_creds_renew') as mock_aws_creds, \
         patch('atem_sisylana.utils.s3_upload') as mock_s3_upload, \
         patch('atem_sisylana.utils.seperator') as mock_separator:
        
        yield {
            'start_run': mock_start_run,
            'end_run': mock_end_run,
            'aws_creds_renew': mock_aws_creds,
            's3_upload': mock_s3_upload,
            'separator': mock_separator
        }


@pytest.fixture(params=[
    PullStatus.SUCCESS,
    PullStatus.NONEXIST_PCDS,
    PullStatus.NONEXIST_AWS,
    PullStatus.NONDATE_PCDS,
    PullStatus.NONDATE_AWS,
    PullStatus.EMPTY_PCDS,
    PullStatus.EMPTY_AWS,
    PullStatus.NO_MAPPING
])
def pull_status_scenarios(request):
    """
    Parametrized fixture for testing different pull status scenarios.
    
    Args:
        request: Pytest request object containing the parameter
        
    Returns:
        PullStatus enum value for testing
    """
    return request.param


@pytest.fixture
def performance_test_data():
    """
    Fixture providing larger datasets for performance testing.
    
    Returns:
        Dictionary containing larger test datasets
    """
    # Generate larger table list
    large_table_list = TestDataGenerator.create_sample_table_list(100)
    
    # Generate larger schema data
    large_pcds_schema = pd.DataFrame([
        {
            'column_name': f'COLUMN_{i:04d}',
            'data_type': 'VARCHAR2(100)' if i % 2 == 0 else 'NUMBER',
            'aws_colname': f'column_{i:04d}'
        }
        for i in range(500)  # 500 columns
    ])
    
    large_aws_schema = pd.DataFrame([
        {
            'column_name': f'column_{i:04d}',
            'data_type': 'varchar(100)' if i % 2 == 0 else 'bigint'
        }
        for i in range(500)  # 500 columns
    ])
    
    # Generate larger date datasets
    large_date_data_pcds = pd.DataFrame([
        {
            'created_date': (datetime(2023, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d'),
            'NROWS': 1000 + (i * 10)
        }
        for i in range(365)  # Full year of data
    ])
    
    large_date_data_aws = pd.DataFrame([
        {
            'created_date': (datetime(2023, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d'),
            'nrows': 1000 + (i * 10) + (5 if i % 10 == 0 else 0)  # Some mismatches
        }
        for i in range(365)  # Full year of data
    ])
    
    return {
        'large_table_list': large_table_list,
        'large_pcds_schema': large_pcds_schema,
        'large_aws_schema': large_aws_schema,
        'large_date_data_pcds': large_date_data_pcds,
        'large_date_data_aws': large_date_data_aws
    }


@pytest.fixture
def error_scenarios():
    """
    Fixture providing various error scenarios for testing.
    
    Returns:
        Dictionary containing error scenario configurations
    """
    return {
        'database_connection_error': {
            'exception': utils.NONEXIST_TABLE("Connection failed"),
            'expected_status': PullStatus.NONEXIST_PCDS
        },
        'missing_date_column': {
            'exception': utils.NONEXIST_DATEVAR("Date column not found"),
            'expected_status': PullStatus.NONDATE_PCDS
        },
        'empty_result_set': {
            'result': pd.DataFrame(),
            'expected_status': PullStatus.EMPTY_PCDS
        },
        'malformed_data': {
            'result': pd.DataFrame([{'invalid_column': 'invalid_data'}]),
            'expected_status': None  # Should handle gracefully
        },
        'memory_error': {
            'exception': MemoryError("Insufficient memory"),
            'expected_status': None  # Should handle gracefully
        }
    }


@pytest.fixture(autouse=True)
def clean_test_environment():
    """
    Fixture that automatically cleans up test environment before and after tests.
    """
    # Setup: Clean any existing test artifacts
    test_artifacts = [
        'test_output',
        'test_analysis.pkl',
        'test_results.csv',
        'test_next_data.json'
    ]
    
    for artifact in test_artifacts:
        if os.path.exists(artifact):
            if os.path.isdir(artifact):
                shutil.rmtree(artifact)
            else:
                os.remove(artifact)
    
    yield  # Run the test
    
    # Teardown: Clean up after test
    for artifact in test_artifacts:
        if os.path.exists(artifact):
            try:
                if os.path.isdir(artifact):
                    shutil.rmtree(artifact)
                else:
                    os.remove(artifact)
            except (OSError, PermissionError):
                pass  # Ignore cleanup errors


@pytest.fixture
def integration_test_setup(file_system_setup, mock_database_responses):
    """
    Fixture providing complete setup for integration tests.
    
    Args:
        file_system_setup: File system setup from fixture
        mock_database_responses: Mock database responses from fixture
        
    Returns:
        Dictionary containing all integration test components
    """
    return {
        'files': file_system_setup,
        'mock_responses': mock_database_responses,
        'test_scenarios': [
            {
                'name': 'successful_migration',
                'tables': ['test_table_000', 'test_table_001'],
                'expected_status': PullStatus.SUCCESS,
                'response_key': 'successful_table'
            },
            {
                'name': 'mismatch_detection',
                'tables': ['test_table_002'],
                'expected_status': PullStatus.SUCCESS,
                'response_key': 'mismatch_table'
            },
            {
                'name': 'empty_table_handling',
                'tables': ['test_table_003'],
                'expected_status': PullStatus.EMPTY_PCDS,
                'response_key': 'empty_table'
            }
        ]
    }


# Custom pytest markers for categorizing tests
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "database: mark test as requiring database connection")
    config.addinivalue_line("markers", "aws: mark test as requiring AWS services")


# Custom assertion helpers
class TestAssertions:
    """Custom assertion helpers for migration analysis testing."""
    
    @staticmethod
    def assert_schema_comparison_valid(comparison_result):
        """
        Assert that a schema comparison result is valid.
        
        Args:
            comparison_result: MetaMerge object from schema comparison
        """
        assert hasattr(comparison_result, 'unique_pcds')
        assert hasattr(comparison_result, 'unique_aws')
        assert hasattr(comparison_result, 'col_mapping')
        assert hasattr(comparison_result, 'mismatches')
        assert hasattr(comparison_result, 'uncaptured')
        
        assert isinstance(comparison_result.unique_pcds, list)
        assert isinstance(comparison_result.unique_aws, list)
        assert isinstance(comparison_result.mismatches, str)
        assert isinstance(comparison_result.uncaptured, str)
    
    @staticmethod
    def assert_temporal_analysis_valid(temporal_result):
        """
        Assert that a temporal analysis result is valid.
        
        Args:
            temporal_result: Dictionary from temporal analysis
        """
        required_keys = [
            'Time Span UnMatch',
            'Time Span Variable',
            'Time UnMatch Details'
        ]
        
        for key in required_keys:
            assert key in temporal_result
        
        assert isinstance(temporal_result['Time Span UnMatch'], bool)
        assert isinstance(temporal_result['Time Span Variable'], str)
        assert isinstance(temporal_result['Time UnMatch Details'], str)
    
    @staticmethod
    def assert_metadata_analysis_valid(metadata_result):
        """
        Assert that a metadata analysis result is valid.
        
        Args:
            metadata_result: Dictionary from metadata analysis
        """
        required_keys = [
            'Row UnMatch',
            'Row UnMatch Details',
            'Col Count Details',
            'Column Type UnMatch',
            'Type UnMatch Details',
            'PCDS Unique Columns',
            'AWS Unique Columns',
            'Uncaptured Column Mappings'
        ]
        
        for key in required_keys:
            assert key in metadata_result
        
        assert isinstance(metadata_result['Row UnMatch'], bool)
        assert isinstance(metadata_result['Column Type UnMatch'], bool)
    
    @staticmethod
    def assert_csv_output_valid(csv_file_path, expected_columns):
        """
        Assert that CSV output file is valid.
        
        Args:
            csv_file_path: Path to CSV file
            expected_columns: List of expected column names
        """
        assert os.path.exists(csv_file_path)
        
        df = pd.read_csv(csv_file_path)
        assert not df.empty
        
        for col in expected_columns:
            assert col in df.columns
    
    @staticmethod
    def assert_pickle_output_valid(pickle_file_path):
        """
        Assert that pickle output file is valid.
        
        Args:
            pickle_file_path: Path to pickle file
        """
        assert os.path.exists(pickle_file_path)
        
        import pickle
        with open(pickle_file_path, 'rb') as f:
            data = pickle.load(f)
        
        assert isinstance(data, dict)
        assert len(data) > 0


# Make test assertions available globally
pytest.test_assertions = TestAssertions()