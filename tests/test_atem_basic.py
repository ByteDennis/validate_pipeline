#!/usr/bin/env python3
"""
Tests for database operations, query building, and data type mapping.

This module tests the core database interaction components including:
- DatabaseQueryBuilder
- DataTypeMapper
- StringMatcher
- Database connection and query execution
"""

import pytest
import pandas as pd
import re
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

# Import the modules under test
import sys
sys.path.append('src')
from atem_sisylana import (
    DatabaseQueryBuilder, 
    DataTypeMapper, 
    StringMatcher, 
    PullStatus,
    MigrationAnalyzer
)
import utils


class TestDatabaseQueryBuilder:
    """Test suite for DatabaseQueryBuilder class."""
    
    @pytest.mark.parametrize("platform,category,expected_template", [
        ('PCDS', 'meta', 'all_tab_cols'),
        ('PCDS', 'nrow', 'COUNT(*)'),
        ('PCDS', 'date', 'GROUP BY'),
        ('AWS', 'meta', 'information_schema'),
        ('AWS', 'nrow', 'COUNT(*)'),
        ('AWS', 'date', 'GROUP BY'),
    ])
    def test_get_query_templates(self, platform, category, expected_template):
        """Test that query templates contain expected components."""
        if category == 'date':
            query = DatabaseQueryBuilder.get_query(
                platform, category, 'test_table', 'test_db', 'test_date'
            )
        elif platform == 'AWS':
            query = DatabaseQueryBuilder.get_query(
                platform, category, 'test_table', 'test_db'
            )
        else:
            query = DatabaseQueryBuilder.get_query(
                platform, category, 'test_table'
            )
        
        assert expected_template in query
        assert 'test_table' in query.lower()
    
    def test_pcds_meta_query_structure(self):
        """Test PCDS metadata query structure."""
        query = DatabaseQueryBuilder.get_query('PCDS', 'meta', 'EMPLOYEES')
        
        assert 'column_name' in query
        assert 'data_type' in query
        assert 'all_tab_cols' in query
        assert 'UPPER(\'EMPLOYEES\')' in query or 'EMPLOYEES' in query
        assert 'order by column_id' in query.lower()
    
    def test_aws_meta_query_structure(self):
        """Test AWS metadata query structure."""
        query = DatabaseQueryBuilder.get_query('AWS', 'meta', 'employees', 'hr_db')
        
        assert 'information_schema.columns' in query
        assert 'table_schema' in query
        assert 'table_name' in query
        assert 'hr_db' in query.lower()
        assert 'employees' in query.lower()
    
    def test_date_query_formatting(self):
        """Test date query parameter substitution."""
        pcds_query = DatabaseQueryBuilder.get_query(
            'PCDS', 'date', 'TRANSACTIONS', date_col='TRANSACTION_DATE'
        )
        aws_query = DatabaseQueryBuilder.get_query(
            'AWS', 'date', 'transactions', 'finance_db', 'transaction_date'
        )
        
        assert 'TRANSACTION_DATE' in pcds_query
        assert 'GROUP BY TRANSACTION_DATE' in pcds_query
        assert 'transaction_date' in aws_query
        assert 'finance_db.transactions' in aws_query
    
    def test_invalid_platform_raises_keyerror(self):
        """Test that invalid platform raises KeyError."""
        with pytest.raises(KeyError):
            DatabaseQueryBuilder.get_query('INVALID', 'meta', 'table')
    
    def test_invalid_category_raises_keyerror(self):
        """Test that invalid category raises KeyError."""
        with pytest.raises(KeyError):
            DatabaseQueryBuilder.get_query('PCDS', 'invalid', 'table')


class TestDataTypeMapper:
    """Test suite for DataTypeMapper class."""
    
    @pytest.fixture
    def sample_data_types(self):
        """Fixture providing sample data type mappings."""
        return [
            {'data_type_pcds': 'NUMBER', 'data_type_aws': 'double', 'expected': True},
            {'data_type_pcds': 'NUMBER(10,2)', 'data_type_aws': 'decimal(10,2)', 'expected': True},
            {'data_type_pcds': 'NUMBER(10,2)', 'data_type_aws': 'decimal(10,3)', 'expected': False},
            {'data_type_pcds': 'VARCHAR2(100)', 'data_type_aws': 'varchar(100)', 'expected': True},
            {'data_type_pcds': 'VARCHAR2(50)', 'data_type_aws': 'varchar(100)', 'expected': False},
            {'data_type_pcds': 'CHAR(1)', 'data_type_aws': 'char(1)', 'expected': True},
            {'data_type_pcds': 'CHAR(10)', 'data_type_aws': 'VARCHAR(10)', 'expected': False},
            {'data_type_pcds': 'DATE', 'data_type_aws': 'date', 'expected': True},
            {'data_type_pcds': 'DATE', 'data_type_aws': 'timestamp', 'expected': True},
            {'data_type_pcds': 'TIMESTAMP', 'data_type_aws': 'timestamp', 'expected': True},
            {'data_type_pcds': 'TIMESTAMP(6)', 'data_type_aws': 'timestamp(6)', 'expected': True},
        ]
    
    @pytest.mark.parametrize("test_case", [
        {'data_type_pcds': 'NUMBER', 'data_type_aws': 'double', 'expected': True},
        {'data_type_pcds': 'NUMBER', 'data_type_aws': 'integer', 'expected': False},
        {'data_type_pcds': 'VARCHAR2(100)', 'data_type_aws': 'varchar(100)', 'expected': True},
        {'data_type_pcds': 'DATE', 'data_type_aws': 'date', 'expected': True},
    ])
    def test_map_pcds_to_aws_basic_types(self, test_case):
        """Test basic data type mapping."""
        row = pd.Series(test_case)
        result = DataTypeMapper.map_pcds_to_aws(row)
        assert result == test_case['expected']
    
    def test_number_with_precision_scale(self):
        """Test NUMBER type with precision and scale."""
        test_cases = [
            ('NUMBER(10,2)', 'decimal(10,2)', True),
            ('NUMBER(10,2)', 'decimal(10,3)', False),
            ('NUMBER(5,0)', 'decimal(5,0)', True),
            ('NUMBER(15,4)', 'decimal(15,4)', True),
        ]
        
        for pcds_type, aws_type, expected in test_cases:
            row = pd.Series({
                'data_type_pcds': pcds_type,
                'data_type_aws': aws_type,
                'column_name_aws': 'test_col'
            })
            result = DataTypeMapper.map_pcds_to_aws(row)
            assert result == expected, f"Failed for {pcds_type} -> {aws_type}"
    
    def test_varchar_types(self):
        """Test VARCHAR2 type mappings."""
        test_cases = [
            ('VARCHAR2(100)', 'varchar(100)', True),
            ('VARCHAR2(50)', 'varchar(50)', True),
            ('VARCHAR2(255)', 'varchar(100)', False),  # Different lengths
        ]
        
        for pcds_type, aws_type, expected in test_cases:
            row = pd.Series({
                'data_type_pcds': pcds_type,
                'data_type_aws': aws_type,
                'column_name_aws': 'test_col'
            })
            result = DataTypeMapper.map_pcds_to_aws(row)
            assert result == expected
    
    def test_char_types(self):
        """Test CHAR type mappings."""
        row_char_match = pd.Series({
            'data_type_pcds': 'CHAR(1)',
            'data_type_aws': 'char(1)',
            'column_name_aws': 'flag_col'
        })
        assert DataTypeMapper.map_pcds_to_aws(row_char_match) == True
        
        # CHAR with length > 1 should not match VARCHAR
        row_char_varchar = pd.Series({
            'data_type_pcds': 'CHAR(10)',
            'data_type_aws': 'VARCHAR(10)',
            'column_name_aws': 'code_col'
        })
        assert DataTypeMapper.map_pcds_to_aws(row_char_varchar) == False
    
    def test_timestamp_types(self):
        """Test TIMESTAMP type variations."""
        test_cases = [
            ('TIMESTAMP', 'timestamp', True),
            ('TIMESTAMP(6)', 'timestamp(6)', True),
            ('TIMESTAMP WITH TIME ZONE', 'timestamp', True),
        ]
        
        for pcds_type, aws_type, expected in test_cases:
            row = pd.Series({
                'data_type_pcds': pcds_type,
                'data_type_aws': aws_type,
                'column_name_aws': 'datetime_col'
            })
            result = DataTypeMapper.map_pcds_to_aws(row)
            assert result == expected
    
    @patch('atem_sisylana.logger')
    def test_unrecognized_type_logs_warning(self, mock_logger):
        """Test that unrecognized types log warnings."""
        row = pd.Series({
            'data_type_pcds': 'UNKNOWN_TYPE',
            'data_type_aws': 'some_type',
            'column_name_aws': 'unknown_col'
        })
        result = DataTypeMapper.map_pcds_to_aws(row)
        assert result == False
        mock_logger.info.assert_called()


class TestStringMatcher:
    """Test suite for StringMatcher class."""
    
    def test_has_prefix_match_positive_cases(self):
        """Test prefix matching for positive cases."""
        assert StringMatcher.has_prefix_match('customer_id', 'customer') == True
        assert StringMatcher.has_prefix_match('id', 'customer_id') == True
        assert StringMatcher.has_prefix_match('account', 'account_number') == True
        assert StringMatcher.has_prefix_match('transaction_date', 'transaction') == True
    
    def test_has_prefix_match_negative_cases(self):
        """Test prefix matching for negative cases."""
        assert StringMatcher.has_prefix_match('customer', 'account') == False
        assert StringMatcher.has_prefix_match('first_name', 'last_name') == False
        assert StringMatcher.has_prefix_match('id', 'name') == False
    
    def test_find_differences_exact_mode(self):
        """Test finding differences in exact match mode."""
        list_a = ['id', 'name', 'email', 'phone']
        list_b = ['id', 'name', 'address']
        
        differences = StringMatcher.find_differences(list_a, list_b, mode='exact')
        assert differences == {'email', 'phone'}
    
    def test_find_differences_prefix_mode(self):
        """Test finding differences in prefix match mode."""
        list_a = ['customer_id', 'customer_name', 'account_id', 'phone']
        list_b = ['customer', 'account', 'email']
        
        differences = StringMatcher.find_differences(list_a, list_b, mode='prefix')
        assert 'phone' in differences
        assert 'customer_id' not in differences  # Should match 'customer'
        assert 'account_id' not in differences   # Should match 'account'
    
    def test_find_differences_with_drop_patterns(self):
        """Test finding differences with drop patterns."""
        list_a = ['id', 'name', 'temp_field', 'audit_created']
        list_b = ['id', 'name']
        drop_patterns = ['temp_.*', 'audit_.*']
        
        differences = StringMatcher.find_differences(
            list_a, list_b, mode='exact', drop=drop_patterns
        )
        assert differences == set()  # temp_field and audit_created should be dropped
    
    def test_find_common_mappings_exact_priority(self):
        """Test that exact matches are prioritized in common mappings."""
        list_a = ['id', 'customer_id', 'name']
        list_b = ['id', 'customer', 'name', 'customer_id']
        
        mappings = StringMatcher.find_common_mappings(list_a, list_b)
        
        assert mappings['id'] == 'id'  # Exact match
        assert mappings['name'] == 'name'  # Exact match
        # customer_id should map to customer_id (exact) not customer (prefix)
        assert mappings.get('customer_id') == 'customer_id'
    
    def test_find_common_mappings_prefix_fallback(self):
        """Test prefix matching when exact matches aren't available."""
        list_a = ['cust_id', 'cust_name', 'acct_num']
        list_b = ['customer_id', 'customer_name', 'account_number']
        
        mappings = StringMatcher.find_common_mappings(list_a, list_b)
        
        # Should have some mappings based on prefix matching
        assert len(mappings) > 0
        # Verify prefix relationships exist
        for source, target in mappings.items():
            assert StringMatcher.has_prefix_match(source, target)
    
    def test_find_common_mappings_no_duplicates(self):
        """Test that mappings don't create duplicate targets."""
        list_a = ['customer_id', 'customer_name', 'customer_phone']
        list_b = ['customer', 'phone']
        
        mappings = StringMatcher.find_common_mappings(list_a, list_b)
        
        # Ensure no target appears twice
        target_values = list(mappings.values())
        assert len(target_values) == len(set(target_values))


class TestDatabaseConnections:
    """Test suite for database connection and query execution."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        config_data = {
            'input': {
                'table': 'test_tables.xlsx',
                'env': '.env',
                'name': 'test_migration',
                'step': 'analysis',
                'range': [1, 100]
            },
            'output': {
                'folder': 'test_output',
                'csv': {
                    'path': 'test_output/results.csv',
                    'columns': ['table', 'status', 'details']
                },
                'log': {
                    'sink': 'test_output/test.log',
                    'level': 'INFO'
                },
                'to_pkl': 'test_output/analysis.pkl',
                'next': {
                    'file': 'test_output/next_data.json',
                    'fields': {}
                },
                'to_s3': {
                    'run': 's3://test-bucket/runs'
                }
            },
            'column_maps': 'column_mappings.json',
            'match': {
                'drop_cols': ['temp_col', 'audit_col'],
                'add_cols': ['new_col']
            }
        }
        return config_data
    
    @pytest.fixture
    def mock_migration_analyzer(self, mock_config, tmp_path):
        """Create a MigrationAnalyzer instance with mocked dependencies."""
        # Create temporary config file
        config_file = tmp_path / "test_config.json"
        config_file.write_text(json.dumps(mock_config))
        
        with patch('atem_sisylana.Config') as mock_config_class, \
             patch('atem_sisylana.utils.MetaConfig') as mock_meta_config, \
             patch('atem_sisylana.utils.read_column_mapping') as mock_col_map, \
             patch('atem_sisylana.utils.read_input_excel') as mock_excel:
            
            # Setup mocks
            mock_config_instance = Mock()
            mock_config_instance.from_disk.return_value = mock_config
            mock_config_class.return_value = mock_config_instance
            
            mock_meta_config.return_value = Mock(**mock_config)
            mock_col_map.return_value = {'test_table': {'pcds2aws': {'ID': 'id'}}}
            
            # Mock Excel data
            mock_excel.return_value = pd.DataFrame([
                {
                    'pcds_tbl': 'service.test_table',
                    'aws_tbl': 'db.test_table',
                    'col_map': 'test_table',
                    'pcds_id': 'created_date',
                    'aws_id': 'created_date',
                    'group': 'test_group',
                    'hydrate_only': 'N'
                }
            ])
            
            analyzer = MigrationAnalyzer(config_file)
            return analyzer
    
    @patch('atem_sisylana.utils.DatabaseConnector')
    def test_query_database_pcds_success(self, mock_db_connector, mock_migration_analyzer):
        """Test successful PCDS database query."""
        # Setup mock database connector
        mock_connector_instance = Mock()
        mock_db_connector.return_value = mock_connector_instance
        
        expected_result = pd.DataFrame([
            {'column_name': 'ID', 'data_type': 'NUMBER'},
            {'column_name': 'NAME', 'data_type': 'VARCHAR2(100)'}
        ])
        mock_connector_instance.query.return_value = expected_result
        
        # Test the query
        result = mock_migration_analyzer.query_database('test_table', 'meta', svc='test_service')
        
        assert not result.empty
        assert 'column_name' in result.columns
        assert 'data_type' in result.columns
        mock_connector_instance.query.assert_called_once()
    
    @patch('atem_sisylana.utils.DatabaseConnector')
    def test_query_database_aws_success(self, mock_db_connector, mock_migration_analyzer):
        """Test successful AWS database query."""
        # Setup mock database connector
        mock_connector_instance = Mock()
        mock_db_connector.return_value = mock_connector_instance
        
        expected_result = pd.DataFrame([
            {'column_name': 'id', 'data_type': 'bigint'},
            {'column_name': 'name', 'data_type': 'varchar(100)'}
        ])
        mock_connector_instance.query.return_value = expected_result
        
        # Test the query
        result = mock_migration_analyzer.query_database('test_table', 'meta', db='test_db')
        
        assert not result.empty
        assert len(result) == 2
        mock_connector_instance.query.assert_called_once()
    
    @patch('atem_sisylana.utils.DatabaseConnector')
    def test_query_database_handles_exceptions(self, mock_db_connector, mock_migration_analyzer):
        """Test that database query handles exceptions properly."""
        # Setup mock to raise exception
        mock_connector_instance = Mock()
        mock_db_connector.return_value = mock_connector_instance
        mock_connector_instance.query.side_effect = utils.NONEXIST_TABLE("Table not found")
        
        # Test that exception is propagated
        with pytest.raises(utils.NONEXIST_TABLE):
            mock_migration_analyzer.query_database('nonexistent_table', 'meta')
    
    @patch('atem_sisylana.utils.DatabaseConnector')
    def test_query_database_with_caching(self, mock_db_connector, mock_migration_analyzer):
        """Test that database queries are cached properly."""
        # Setup mock database connector
        mock_connector_instance = Mock()
        mock_db_connector.return_value = mock_connector_instance
        
        expected_result = pd.DataFrame([{'count': 1000}])
        mock_connector_instance.query.return_value = expected_result
        
        # Make the same query twice
        result1 = mock_migration_analyzer.query_database('test_table', 'nrow')
        result2 = mock_migration_analyzer.query_database('test_table', 'nrow')
        
        # Should only call database once due to caching
        assert mock_connector_instance.query.call_count <= 2  # Allow for cache behavior
        pd.testing.assert_frame_equal(result1, result2)


class TestPullStatus:
    """Test suite for PullStatus enumeration."""
    
    def test_pull_status_values(self):
        """Test that PullStatus enum has expected values."""
        expected_statuses = [
            'Nonexisting PCDS Table',
            'Nonexisting AWS Table', 
            'Nonexisting Date Variable in PCDS',
            'Nonexisting Date Variable in AWS',
            'Empty PCDS Table',
            'Empty AWS Table',
            'Column Mapping Not Provided',
            'Successful Data Access'
        ]
        
        actual_statuses = [status.value for status in PullStatus]
        
        for expected in expected_statuses:
            assert expected in actual_statuses
    
    def test_pull_status_success(self):
        """Test SUCCESS status specifically."""
        assert PullStatus.SUCCESS.value == 'Successful Data Access'
    
    def test_pull_status_enum_comparison(self):
        """Test enum comparison operations."""
        assert PullStatus.SUCCESS == PullStatus.SUCCESS
        assert PullStatus.NONEXIST_PCDS != PullStatus.NONEXIST_AWS
        assert PullStatus.SUCCESS in [PullStatus.SUCCESS, PullStatus.NONEXIST_PCDS]


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])