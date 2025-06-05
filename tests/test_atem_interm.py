#!/usr/bin/env python3
"""
Tests for data processing, schema comparison, and temporal analysis.

This module tests the core data processing components including:
- Schema merging and comparison
- Temporal data analysis
- Metadata processing
- Column mapping and validation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path

# Import the modules under test
import sys
sys.path.append('src')
from atem_sisylana import MigrationAnalyzer, PullStatus
import utils


class TestSchemaComparison:
    """Test suite for schema comparison and merging functionality."""
    
    @pytest.fixture
    def sample_pcds_schema(self):
        """Sample PCDS schema data."""
        return pd.DataFrame([
            {'column_name': 'CUSTOMER_ID', 'data_type': 'NUMBER', 'aws_colname': 'customer_id'},
            {'column_name': 'FIRST_NAME', 'data_type': 'VARCHAR2(50)', 'aws_colname': 'first_name'},
            {'column_name': 'LAST_NAME', 'data_type': 'VARCHAR2(50)', 'aws_colname': 'last_name'},
            {'column_name': 'EMAIL', 'data_type': 'VARCHAR2(100)', 'aws_colname': 'email'},
            {'column_name': 'CREATED_DATE', 'data_type': 'DATE', 'aws_colname': 'created_date'},
            {'column_name': 'PCDS_ONLY_COL', 'data_type': 'VARCHAR2(20)', 'aws_colname': pd.NA},
        ])
    
    @pytest.fixture
    def sample_aws_schema(self):
        """Sample AWS schema data."""
        return pd.DataFrame([
            {'column_name': 'customer_id', 'data_type': 'bigint'},
            {'column_name': 'first_name', 'data_type': 'varchar(50)'},
            {'column_name': 'last_name', 'data_type': 'varchar(50)'},
            {'column_name': 'email', 'data_type': 'varchar(100)'},
            {'column_name': 'created_date', 'data_type': 'timestamp'},
            {'column_name': 'aws_only_col', 'data_type': 'varchar(30)'},
        ])
    
    @pytest.fixture
    def migration_analyzer_instance(self, tmp_path):
        """Create a MigrationAnalyzer instance for testing."""
        config_data = {
            'input': {
                'table': 'test.xlsx',
                'env': '.env',
                'name': 'test',
                'step': 'test',
                'range': [1, 10]
            },
            'output': {
                'folder': str(tmp_path),
                'csv': {'path': str(tmp_path / 'results.csv'), 'columns': []},
                'log': {'sink': str(tmp_path / 'test.log'), 'level': 'INFO'},
                'to_pkl': str(tmp_path / 'test.pkl'),
                'next': {'file': str(tmp_path / 'next.json'), 'fields': {}},
                'to_s3': {'run': 's3://test/run'}
            },
            'column_maps': 'test_maps.json',
            'match': {'drop_cols': ['audit_col'], 'add_cols': ['new_col']}
        }
        
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))
        
        with patch('atem_sisylana.Config') as mock_config, \
             patch('atem_sisylana.utils.MetaConfig') as mock_meta_config, \
             patch('atem_sisylana.utils.read_column_mapping') as mock_col_map, \
             patch('atem_sisylana.utils.read_input_excel') as mock_excel, \
             patch('atem_sisylana.load_dotenv'), \
             patch('os.makedirs'), \
             patch('os.remove'):
            
            mock_config.return_value.from_disk.return_value = config_data
            mock_meta_config.return_value = Mock(**config_data)
            mock_col_map.return_value = {}
            mock_excel.return_value = pd.DataFrame()
            
            analyzer = MigrationAnalyzer(config_file)
            analyzer.SEP = '; '
            analyzer.mismatch_data = {}
            analyzer.next_data = {}
            
            return analyzer
    
    def test_merge_and_compare_schemas_basic(self, migration_analyzer_instance, 
                                           sample_pcds_schema, sample_aws_schema):
        """Test basic schema merging and comparison."""
        result = migration_analyzer_instance.merge_and_compare_schemas(
            sample_pcds_schema, sample_aws_schema, PullStatus.SUCCESS
        )
        
        # Check that unique columns are identified
        assert 'PCDS_ONLY_COL' in result.unique_pcds
        assert 'aws_only_col' in result.unique_aws
        
        # Check that type comparison is performed
        assert result.col_mapping is not None
        assert 'type_match' in result.col_mapping.columns
    
    def test_merge_and_compare_schemas_no_mapping(self, migration_analyzer_instance,
                                                sample_pcds_schema, sample_aws_schema):
        """Test schema comparison when no mapping is provided."""
        # Remove aws_colname mappings
        pcds_schema_no_mapping = sample_pcds_schema.copy()
        pcds_schema_no_mapping['aws_colname'] = pd.NA
        
        result = migration_analyzer_instance.merge_and_compare_schemas(
            pcds_schema_no_mapping, sample_aws_schema, PullStatus.NO_MAPPING
        )
        
        assert result.col_mapping is None
        assert result.mismatches == ''
        assert len(result.uncaptured) > 0  # Should have automatic mappings
    
    def test_automatic_column_mapping(self, migration_analyzer_instance):
        """Test automatic column mapping when explicit mapping is missing."""
        pcds_schema = pd.DataFrame([
            {'column_name': 'CUSTOMER_ID', 'data_type': 'NUMBER', 'aws_colname': pd.NA},
            {'column_name': 'FIRST_NAME', 'data_type': 'VARCHAR2(50)', 'aws_colname': pd.NA},
            {'column_name': 'ACCOUNT_NUM', 'data_type': 'VARCHAR2(20)', 'aws_colname': pd.NA},
        ])
        
        aws_schema = pd.DataFrame([
            {'column_name': 'customer_id', 'data_type': 'bigint'},
            {'column_name': 'first_name', 'data_type': 'varchar(50)'},
            {'column_name': 'account_number', 'data_type': 'varchar(20)'},
        ])
        
        result = migration_analyzer_instance.merge_and_compare_schemas(
            pcds_schema, aws_schema, PullStatus.SUCCESS
        )
        
        # Should find automatic mappings
        assert 'uncaptured' in result._asdict()
        assert len(result.uncaptured) > 0
    
    def test_type_mismatch_detection(self, migration_analyzer_instance):
        """Test detection of data type mismatches."""
        pcds_schema = pd.DataFrame([
            {'column_name': 'ID', 'data_type': 'NUMBER', 'aws_colname': 'id'},
            {'column_name': 'NAME', 'data_type': 'VARCHAR2(50)', 'aws_colname': 'name'},
        ])
        
        aws_schema = pd.DataFrame([
            {'column_name': 'id', 'data_type': 'varchar(10)'},  # Intentional mismatch
            {'column_name': 'name', 'data_type': 'varchar(100)'},  # Different length
        ])
        
        result = migration_analyzer_instance.merge_and_compare_schemas(
            pcds_schema, aws_schema, PullStatus.SUCCESS
        )
        
        # Should detect type mismatches
        assert len(result.mismatches) > 0
        assert 'NUMBER->varchar(10)' in result.mismatches or 'VARCHAR2(50)->varchar(100)' in result.mismatches


class TestMetadataProcessing:
    """Test suite for metadata processing functionality."""
    
    @pytest.fixture
    def mock_database_responses(self):
        """Mock database response data."""
        return {
            'pcds_schema': pd.DataFrame([
                {'column_name': 'ID', 'data_type': 'NUMBER'},
                {'column_name': 'NAME', 'data_type': 'VARCHAR2(100)'},
                {'column_name': 'CREATED_DATE', 'data_type': 'DATE'}
            ]),
            'pcds_rowcount': pd.DataFrame([{'nrow': 1000}]),
            'aws_schema': pd.DataFrame([
                {'column_name': 'id', 'data_type': 'bigint'},
                {'column_name': 'name', 'data_type': 'varchar(100)'},
                {'column_name': 'created_date', 'data_type': 'timestamp'}
            ]),
            'aws_rowcount': pd.DataFrame([{'nrow': 995}])
        }
    
    @pytest.fixture
    def analyzer_with_mocks(self, tmp_path, mock_database_responses):
        """Migration analyzer with mocked database calls."""
        config_data = {
            'input': {
                'table': 'test.xlsx',
                'env': '.env',
                'name': 'test',
                'step': 'test',
                'range': [1, 10]
            },
            'output': {
                'folder': str(tmp_path),
                'csv': {'path': str(tmp_path / 'results.csv'), 'columns': []},
                'log': {'sink': str(tmp_path / 'test.log'), 'level': 'INFO'},
                'to_pkl': str(tmp_path / 'test.pkl'),
                'next': {'file': str(tmp_path / 'next.json'), 'fields': {}},
                'to_s3': {'run': 's3://test/run'}
            },
            'column_maps': 'test_maps.json',
            'match': {'drop_cols': [], 'add_cols': []}
        }
        
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))
        
        with patch('atem_sisylana.Config') as mock_config, \
             patch('atem_sisylana.utils.MetaConfig') as mock_meta_config, \
             patch('atem_sisylana.utils.read_column_mapping') as mock_col_map, \
             patch('atem_sisylana.utils.read_input_excel') as mock_excel, \
             patch('atem_sisylana.load_dotenv'), \
             patch('os.makedirs'), \
             patch('os.remove'):
            
            mock_config.return_value.from_disk.return_value = config_data
            mock_meta_config.return_value = Mock(**config_data)
            mock_col_map.return_value = {'test_table': {'pcds2aws': {'ID': 'id', 'NAME': 'name'}}}
            mock_excel.return_value = pd.DataFrame()
            
            analyzer = MigrationAnalyzer(config_file)
            analyzer.SEP = '; '
            return analyzer
    
    @patch('atem_sisylana.MigrationAnalyzer.query_database')
    def test_process_pcds_metadata_success(self, mock_query, analyzer_with_mocks, mock_database_responses):
        """Test successful PCDS metadata processing."""
        # Setup mock responses
        mock_query.side_effect = [
            mock_database_responses['pcds_schema'],
            mock_database_responses['pcds_rowcount']
        ]
        
        metadata, has_mapping = analyzer_with_mocks.process_pcds_metadata(
            'service.test_table', 'test_table'
        )
        
        assert 'column' in metadata
        assert 'row' in metadata
        assert 'svc' in metadata
        assert has_mapping == True
        assert 'aws_colname' in metadata['column'].columns
    
    @patch('atem_sisylana.MigrationAnalyzer.query_database')
    def test_process_pcds_metadata_table_not_found(self, mock_query, analyzer_with_mocks):
        """Test PCDS metadata processing when table doesn't exist."""
        mock_query.side_effect = utils.NONEXIST_TABLE("Table not found")
        
        with pytest.raises(utils.NONEXIST_TABLE):
            analyzer_with_mocks.process_pcds_metadata('service.nonexistent_table', 'test_table')
    
    @patch('atem_sisylana.MigrationAnalyzer.query_database')
    def test_process_aws_metadata_success(self, mock_query, analyzer_with_mocks, mock_database_responses):
        """Test successful AWS metadata processing."""
        mock_query.side_effect = [
            mock_database_responses['aws_schema'],
            mock_database_responses['aws_rowcount']
        ]
        
        metadata = analyzer_with_mocks.process_aws_metadata('database.test_table')
        
        assert 'column' in metadata
        assert 'row' in metadata
        assert len(metadata['column']) == 3
        assert metadata['row'].iloc[0]['nrow'] == 995
    
    @patch('atem_sisylana.MigrationAnalyzer.query_database')
    def test_analyze_metadata_differences(self, mock_query, analyzer_with_mocks, mock_database_responses):
        """Test metadata difference analysis."""
        pcds_meta = {
            'column': mock_database_responses['pcds_schema'].copy(),
            'row': mock_database_responses['pcds_rowcount']
        }
        pcds_meta['column']['aws_colname'] = ['id', 'name', 'created_date']
        
        aws_meta = {
            'column': mock_database_responses['aws_schema'],
            'row': mock_database_responses['aws_rowcount']
        }
        
        result = analyzer_with_mocks.analyze_metadata_differences(pcds_meta, aws_meta)
        
        assert 'Row UnMatch' in result
        assert 'Row UnMatch Details' in result
        assert result['Row UnMatch'] == True  # 1000 vs 995
        assert 'PCDS(1000) : AWS(995)' == result['Row UnMatch Details']
    
    def test_clean_column_lists(self, analyzer_with_mocks):
        """Test column list cleaning functionality."""
        analyzer_with_mocks.config.match = Mock()
        analyzer_with_mocks.config.match.drop_cols = ['AUDIT_COL', 'TEMP_COL']
        analyzer_with_mocks.config.match.add_cols = ['system_col']
        
        pcds_unique = 'ID; NAME; AUDIT_COL; TEMP_COL; ACTIVE_FLAG'
        aws_unique = 'email; phone; system_col; status'
        
        cleaned_pcds, cleaned_aws = analyzer_with_mocks.clean_column_lists(pcds_unique, aws_unique)
        
        assert 'AUDIT_COL' not in cleaned_pcds
        assert 'TEMP_COL' not in cleaned_pcds
        assert 'system_col' not in cleaned_aws
        assert 'ID' in cleaned_pcds
        assert 'email' in cleaned_aws


class TestTemporalAnalysis:
    """Test suite for temporal data analysis functionality."""
    
    @pytest.fixture
    def sample_pcds_date_data(self):
        """Sample PCDS date distribution data."""
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        return pd.DataFrame({
            'created_date': dates,
            'NROWS': [100, 95, 110, 105, 98, 102, 88, 92, 106, 99]
        })
    
    @pytest.fixture
    def sample_aws_date_data(self):
        """Sample AWS date distribution data."""
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        return pd.DataFrame({
            'created_date': dates,
            'nrows': [100, 95, 110, 105, 90, 102, 88, 92, 106, 99]  # Day 5 has mismatch
        })
    
    @pytest.fixture
    def analyzer_for_temporal(self, tmp_path):
        """Analyzer instance for temporal testing."""
        config_data = {
            'input': {'name': 'test', 'step': 'test'},
            'output': {
                'folder': str(tmp_path),
                'next': {'fields': {}}
            }
        }
        
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))
        
        with patch('atem_sisylana.Config') as mock_config, \
             patch('atem_sisylana.utils.MetaConfig') as mock_meta_config, \
             patch('atem_sisylana.utils.read_column_mapping'), \
             patch('atem_sisylana.utils.read_input_excel'), \
             patch('atem_sisylana.load_dotenv'), \
             patch('os.makedirs'), \
             patch('os.remove'):
            
            mock_config.return_value.from_disk.return_value = config_data
            mock_meta_config.return_value = Mock(**config_data)
            
            analyzer = MigrationAnalyzer(config_file)
            analyzer.SEP = '; '
            analyzer.mismatch_data = {}
            analyzer.next_data = {}
            
            return analyzer
    
    def test_analyze_temporal_differences_with_mismatches(self, analyzer_for_temporal,
                                                        sample_pcds_date_data, sample_aws_date_data):
        """Test temporal analysis when there are data mismatches."""
        result = analyzer_for_temporal.analyze_temporal_differences(
            sample_pcds_date_data, 'created_date',
            sample_aws_date_data, 'created_date'
        )
        
        assert 'Time Span UnMatch' in result
        assert result['Time Span UnMatch'] == True
        assert 'Time Span Variable' in result
        assert 'Time UnMatch Details' in result
        assert len(result['Time UnMatch Details']) > 0
    
    def test_analyze_temporal_differences_no_mismatches(self, analyzer_for_temporal):
        """Test temporal analysis when data matches perfectly."""
        dates = pd.date_range('2023-01-01', '2023-01-05', freq='D')
        pcds_data = pd.DataFrame({
            'date_col': dates,
            'NROWS': [100, 95, 110, 105, 98]
        })
        aws_data = pd.DataFrame({
            'date_col': dates,
            'nrows': [100, 95, 110, 105, 98]  # Same counts
        })
        
        result = analyzer_for_temporal.analyze_temporal_differences(
            pcds_data, 'date_col', aws_data, 'date_col'
        )
        
        assert result['Time Span UnMatch'] == False
        assert result['Time UnMatch Details'] == ''
    
    def test_convert_date_format(self, analyzer_for_temporal):
        """Test date format conversion functionality."""
        df = pd.DataFrame({
            'date_col': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
        })
        
        analyzer_for_temporal.convert_date_format(df, 'date_col', 'timestamp')
        
        # Check that dates are converted to string format
        assert df['date_col'].dtype == 'object'
        assert df['date_col'].iloc[0] == '2023-01-01'
    
    @patch('atem_sisylana.utils.get_datesort')
    def test_temporal_analysis_stores_excludes(self, mock_get_datesort, analyzer_for_temporal):
        """Test that temporal analysis stores time excludes properly."""
        mock_get_datesort.return_value = ['2023-01-05', '2023-01-08']
        
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        pcds_data = pd.DataFrame({'date_col': dates, 'NROWS': [100] * 10})
        aws_data = pd.DataFrame({'date_col': dates, 'nrows': [95] * 10})  # All mismatched
        
        analyzer_for_temporal.analyze_temporal_differences(
            pcds_data, 'date_col', aws_data, 'date_col'
        )
        
        assert 'time_excludes' in analyzer_for_temporal.next_data
        assert analyzer_for_temporal.next_data['time_excludes'] == '2023-01-05; 2023-01-08'


class TestDataProcessingEdgeCases:
    """Test suite for edge cases in data processing."""
    
    @pytest.fixture
    def minimal_analyzer(self, tmp_path):
        """Minimal analyzer setup for edge case testing."""
        config_data = {
            'input': {'name': 'test'},
            'output': {'folder': str(tmp_path)},
            'match': {}
        }
        
        with patch('atem_sisylana.Config') as mock_config, \
             patch('atem_sisylana.utils.MetaConfig') as mock_meta_config, \
             patch('atem_sisylana.utils.read_column_mapping'), \
             patch('atem_sisylana.utils.read_input_excel'), \
             patch('atem_sisylana.load_dotenv'), \
             patch('os.makedirs'), \
             patch('os.remove'):
            
            mock_config.return_value.from_disk.return_value = config_data
            mock_meta_config.return_value = Mock(**config_data)
            
            analyzer = MigrationAnalyzer(tmp_path / "config.json")
            analyzer.SEP = '; '
            analyzer.config.match = Mock(drop_cols=[], add_cols=[])
            return analyzer
    
    def test_empty_schema_handling(self, minimal_analyzer):
        """Test handling of empty schemas."""
        empty_pcds = pd.DataFrame(columns=['column_name', 'data_type', 'aws_colname'])
        empty_aws = pd.DataFrame(columns=['column_name', 'data_type'])
        
        result = minimal_analyzer.merge_and_compare_schemas(
            empty_pcds, empty_aws, PullStatus.SUCCESS
        )
        
        assert len(result.unique_pcds) == 0
        assert len(result.unique_aws) == 0
        assert result.col_mapping is not None
        assert len(result.col_mapping) == 0
    
    def test_single_column_schema(self, minimal_analyzer):
        """Test handling of single column schemas."""
        single_pcds = pd.DataFrame([
            {'column_name': 'ID', 'data_type': 'NUMBER', 'aws_colname': 'id'}
        ])
        single_aws = pd.DataFrame([
            {'column_name': 'id', 'data_type': 'bigint'}
        ])
        
        result = minimal_analyzer.merge_and_compare_schemas(
            single_pcds, single_aws, PullStatus.SUCCESS
        )
        
        assert len(result.unique_pcds) == 0
        assert len(result.unique_aws) == 0
        assert len(result.col_mapping) == 1
    
    def test_null_handling_in_mappings(self, minimal_analyzer):
        """Test handling of null values in column mappings."""
        pcds_with_nulls = pd.DataFrame([
            {'column_name': 'ID', 'data_type': 'NUMBER', 'aws_colname': 'id'},
            {'column_name': 'NAME', 'data_type': 'VARCHAR2(50)', 'aws_colname': None},
            {'column_name': 'EMAIL', 'data_type': 'VARCHAR2(100)', 'aws_colname': pd.NA}
        ])
        aws_schema = pd.DataFrame([
            {'column_name': 'id', 'data_type': 'bigint'},
            {'column_name': 'name', 'data_type': 'varchar(50)'},
            {'column_name': 'email', 'data_type': 'varchar(100)'}
        ])
        
        result = minimal_analyzer.merge_and_compare_schemas(
            pcds_with_nulls, aws_schema, PullStatus.SUCCESS
        )
        
        # Should handle nulls gracefully and potentially find automatic mappings
        assert result is not None
        assert len(result.unique_pcds) >= 0
        assert len(result.unique_aws) >= 0
    
    def test_duplicate_column_names(self, minimal_analyzer):
        """Test handling of duplicate column names."""
        duplicate_aws = pd.DataFrame([
            {'column_name': 'id', 'data_type': 'bigint'},
            {'column_name': 'id', 'data_type': 'varchar(10)'},  # Duplicate
            {'column_name': 'name', 'data_type': 'varchar(50)'}
        ])
        
        pcds_schema = pd.DataFrame([
            {'column_name': 'ID', 'data_type': 'NUMBER', 'aws_colname': 'id'}
        ])
        
        # Should handle duplicates without crashing
        result = minimal_analyzer.merge_and_compare_schemas(
            pcds_schema, duplicate_aws, PullStatus.SUCCESS
        )
        
        assert result is not None
    
    def test_special_characters_in_column_names(self, minimal_analyzer):
        """Test handling of special characters in column names."""
        special_pcds = pd.DataFrame([
            {'column_name': 'CUSTOMER$ID', 'data_type': 'NUMBER', 'aws_colname': 'customer_id'},
            {'column_name': 'FIRST#NAME', 'data_type': 'VARCHAR2(50)', 'aws_colname': 'first_name'},
            {'column_name': 'EMAIL@ADDR', 'data_type': 'VARCHAR2(100)', 'aws_colname': 'email_addr'}
        ])
        
        special_aws = pd.DataFrame([
            {'column_name': 'customer_id', 'data_type': 'bigint'},
            {'column_name': 'first_name', 'data_type': 'varchar(50)'},
            {'column_name': 'email_addr', 'data_type': 'varchar(100)'}
        ])
        
        result = minimal_analyzer.merge_and_compare_schemas(
            special_pcds, special_aws, PullStatus.SUCCESS
        )
        
        # Should handle special characters without issues
        assert result is not None
        assert len(result.col_mapping) == 3
    
    def test_very_long_column_names(self, minimal_analyzer):
        """Test handling of very long column names."""
        long_name = 'A' * 128  # Very long column name
        long_pcds = pd.DataFrame([
            {'column_name': long_name, 'data_type': 'VARCHAR2(50)', 'aws_colname': 'long_col'}
        ])
        
        long_aws = pd.DataFrame([
            {'column_name': 'long_col', 'data_type': 'varchar(50)'}
        ])
        
        result = minimal_analyzer.merge_and_compare_schemas(
            long_pcds, long_aws, PullStatus.SUCCESS
        )
        
        assert result is not None
        assert len(result.col_mapping) == 1
    
    def test_case_sensitivity_handling(self, minimal_analyzer):
        """Test case sensitivity in column matching."""
        mixed_case_pcds = pd.DataFrame([
            {'column_name': 'Customer_ID', 'data_type': 'NUMBER', 'aws_colname': pd.NA},
            {'column_name': 'FIRST_name', 'data_type': 'VARCHAR2(50)', 'aws_colname': pd.NA}
        ])
        
        mixed_case_aws = pd.DataFrame([
            {'column_name': 'customer_id', 'data_type': 'bigint'},
            {'column_name': 'first_name', 'data_type': 'varchar(50)'}
        ])
        
        result = minimal_analyzer.merge_and_compare_schemas(
            mixed_case_pcds, mixed_case_aws, PullStatus.SUCCESS
        )
        
        # Should find automatic mappings despite case differences
        assert result is not None
        assert len(result.uncaptured) > 0


class TestDataIntegrity:
    """Test suite for data integrity and validation."""
    
    @pytest.fixture
    def integrity_analyzer(self, tmp_path):
        """Analyzer setup for integrity testing."""
        config_data = {
            'input': {'name': 'integrity_test'},
            'output': {'folder': str(tmp_path), 'next': {'fields': {}}}
        }
        
        with patch('atem_sisylana.Config') as mock_config, \
             patch('atem_sisylana.utils.MetaConfig') as mock_meta_config, \
             patch('atem_sisylana.utils.read_column_mapping'), \
             patch('atem_sisylana.utils.read_input_excel'), \
             patch('atem_sisylana.load_dotenv'), \
             patch('os.makedirs'), \
             patch('os.remove'):
            
            mock_config.return_value.from_disk.return_value = config_data
            mock_meta_config.return_value = Mock(**config_data)
            
            analyzer = MigrationAnalyzer(tmp_path / "config.json")
            analyzer.SEP = '; '
            analyzer.next_data = {}
            return analyzer
    
    def test_date_format_consistency(self, integrity_analyzer):
        """Test that date formats are handled consistently."""
        # Test with different date formats
        mixed_dates = pd.DataFrame({
            'date_col': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'count': [100, 95, 110]
        })
        mixed_dates['date_col'] = pd.to_datetime(mixed_dates['date_col'])
        
        # Convert to string format
        integrity_analyzer.convert_date_format(mixed_dates, 'date_col', 'timestamp')
        
        # All dates should be in consistent string format
        assert all(isinstance(d, str) for d in mixed_dates['date_col'])
        assert all(len(d) == 10 for d in mixed_dates['date_col'])  # YYYY-MM-DD format
    
    def test_row_count_validation(self, integrity_analyzer):
        """Test row count validation logic."""
        pcds_meta = {
            'column': pd.DataFrame([{'column_name': 'ID', 'aws_colname': 'id'}]),
            'row': pd.DataFrame([{'nrow': 1000}])
        }
        aws_meta = {
            'column': pd.DataFrame([{'column_name': 'id', 'data_type': 'bigint'}]),
            'row': pd.DataFrame([{'nrow': 1000}])
        }
        
        result = integrity_analyzer.analyze_metadata_differences(pcds_meta, aws_meta)
        
        assert result['Row UnMatch'] == False
        assert 'PCDS(1000) : AWS(1000)' == result['Row UnMatch Details']
    
    def test_column_count_reporting(self, integrity_analyzer):
        """Test column count reporting accuracy."""
        pcds_meta = {
            'column': pd.DataFrame([
                {'column_name': 'ID', 'aws_colname': 'id'},
                {'column_name': 'NAME', 'aws_colname': 'name'},
                {'column_name': 'EMAIL', 'aws_colname': 'email'}
            ]),
            'row': pd.DataFrame([{'nrow': 100}])
        }
        aws_meta = {
            'column': pd.DataFrame([
                {'column_name': 'id', 'data_type': 'bigint'},
                {'column_name': 'name', 'data_type': 'varchar(100)'}
            ]),
            'row': pd.DataFrame([{'nrow': 100}])
        }
        
        result = integrity_analyzer.analyze_metadata_differences(pcds_meta, aws_meta)
        
        assert 'PCDS(3) : AWS(2)' == result['Col Count Details']


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])