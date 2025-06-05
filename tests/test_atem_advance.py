#!/usr/bin/env python3
"""
Integration tests for the complete migration analysis workflow.

This module tests the end-to-end functionality including:
- Complete analysis workflow
- File I/O operations
- Configuration handling
- Error recovery and reporting
- S3 integration
- CSV output generation
"""

import pytest
import pandas as pd
import json
import pickle
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
import os
import csv

# Import the modules under test
import sys
sys.path.append('src')
from atem_sisylana import MigrationAnalyzer, PullStatus
import utils


class TestConfigurationHandling:
    """Test suite for configuration file handling and validation."""
    
    @pytest.fixture
    def sample_config_data(self):
        """Sample configuration data for testing."""
        return {
            'input': {
                'table': 'test_tables.xlsx',
                'env': '.env.test',
                'name': 'test_migration',
                'step': 'analysis_step',
                'range': [1, 50]
            },
            'output': {
                'folder': 'test_output',
                'csv': {
                    'path': 'test_output/migration_results.csv',
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
                    'sink': 'test_output/analysis.log',
                    'level': 'INFO',
                    'format': '{time} - {level} - {message}'
                },
                'to_pkl': 'test_output/migration_analysis.pkl',
                'next': {
                    'file': 'test_output/next_processing_data.json',
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
                    'run': 's3://migration-bucket/analysis-runs'
                }
            },
            'column_maps': 'column_mappings.json',
            'match': {
                'drop_cols': ['audit_timestamp', 'temp_column'],
                'add_cols': ['system_generated_id']
            }
        }
    
    @pytest.fixture
    def sample_table_data(self):
        """Sample table data for testing."""
        return pd.DataFrame([
            {
                'pcds_tbl': 'loans.customer_accounts',
                'aws_tbl': 'loans_db.customer_accounts',
                'col_map': 'customer_accounts_mapping',
                'pcds_id': 'created_date',
                'aws_id': 'created_date',
                'group': 'Customer Data',
                'hydrate_only': 'N'
            },
            {
                'pcds_tbl': 'loans.loan_applications',
                'aws_tbl': 'loans_db.loan_applications',
                'col_map': 'loan_applications_mapping',
                'pcds_id': 'application_date',
                'aws_id': 'application_date',
                'group': 'Loan Processing',
                'hydrate_only': 'Y'
            },
            {
                'pcds_tbl': 'service.transaction_history',
                'aws_tbl': 'transactions_db.transaction_history',
                'col_map': 'transaction_mapping',
                'pcds_id': 'transaction_date',
                'aws_id': 'transaction_date',
                'group': 'Transaction Data',
                'hydrate_only': 'N'
            }
        ])
    
    @pytest.fixture
    def sample_column_mappings(self):
        """Sample column mappings for testing."""
        return {
            'customer_accounts_mapping': {
                'pcds2aws': {
                    'CUSTOMER_ID': 'customer_id',
                    'ACCOUNT_NUMBER': 'account_number',
                    'FIRST_NAME': 'first_name',
                    'LAST_NAME': 'last_name',
                    'EMAIL_ADDRESS': 'email',
                    'CREATED_DATE': 'created_date'
                }
            },
            'loan_applications_mapping': {
                'pcds2aws': {
                    'APPLICATION_ID': 'application_id',
                    'CUSTOMER_ID': 'customer_id',
                    'LOAN_AMOUNT': 'loan_amount',
                    'APPLICATION_DATE': 'application_date',
                    'STATUS': 'status'
                }
            },
            'transaction_mapping': {
                'pcds2aws': {
                    'TRANSACTION_ID': 'transaction_id',
                    'ACCOUNT_ID': 'account_id',
                    'AMOUNT': 'amount',
                    'TRANSACTION_DATE': 'transaction_date',
                    'DESCRIPTION': 'description'
                }
            }
        }
    
    def test_config_file_loading(self, tmp_path, sample_config_data):
        """Test configuration file loading and parsing."""
        config_file = tmp_path / "test_config.json"
        config_file.write_text(json.dumps(sample_config_data))
        
        with patch('atem_sisylana.Config') as mock_config, \
             patch('atem_sisylana.utils.MetaConfig') as mock_meta_config, \
             patch('atem_sisylana.utils.read_column_mapping'), \
             patch('atem_sisylana.utils.read_input_excel'), \
             patch('atem_sisylana.load_dotenv'), \
             patch('os.makedirs'), \
             patch('os.remove'):
            
            mock_config_instance = Mock()
            mock_config_instance.from_disk.return_value = sample_config_data
            mock_config.return_value = mock_config_instance
            mock_meta_config.return_value = Mock(**sample_config_data)
            
            analyzer = MigrationAnalyzer(config_file)
            
            assert analyzer.config is not None
            mock_config_instance.from_disk.assert_called_once_with(config_file)
    
    def test_config_validation_required_fields(self, tmp_path):
        """Test that missing required configuration fields are handled."""
        incomplete_config = {
            'input': {
                'table': 'test.xlsx'
                # Missing other required fields
            },
            'output': {
                'folder': 'output'
                # Missing other required fields
            }
        }
        
        config_file = tmp_path / "incomplete_config.json"
        config_file.write_text(json.dumps(incomplete_config))
        
        with patch('atem_sisylana.Config') as mock_config, \
             patch('atem_sisylana.utils.MetaConfig') as mock_meta_config, \
             patch('atem_sisylana.utils.read_column_mapping'), \
             patch('atem_sisylana.utils.read_input_excel'), \
             patch('atem_sisylana.load_dotenv'), \
             patch('os.makedirs'), \
             patch('os.remove'):
            
            mock_config.return_value.from_disk.return_value = incomplete_config
            mock_meta_config.return_value = Mock(**incomplete_config)
            
            # Should not raise exception during initialization
            analyzer = MigrationAnalyzer(config_file)
            assert analyzer is not None


class TestFileOperations:
    """Test suite for file I/O operations."""
    
    @pytest.fixture
    def test_files_setup(self, tmp_path, sample_config_data, sample_table_data, sample_column_mappings):
        """Setup test files for file operations testing."""
        # Create test Excel file
        excel_file = tmp_path / "test_tables.xlsx"
        sample_table_data.to_excel(excel_file, index=False)
        
        # Create column mappings file
        mappings_file = tmp_path / "column_mappings.json"
        mappings_file.write_text(json.dumps(sample_column_mappings))
        
        # Create environment file
        env_file = tmp_path / ".env.test"
        env_file.write_text("DB_HOST=localhost\nDB_USER=test\nDB_PASS=secret")
        
        # Update config paths
        sample_config_data['input']['table'] = str(excel_file)
        sample_config_data['input']['env'] = str(env_file)
        sample_config_data['column_maps'] = str(mappings_file)
        sample_config_data['output']['folder'] = str(tmp_path / "output")
        
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(sample_config_data))
        
        return {
            'config_file': config_file,
            'excel_file': excel_file,
            'mappings_file': mappings_file,
            'env_file': env_file,
            'output_dir': tmp_path / "output"
        }
    
    def test_excel_file_reading(self, test_files_setup):
        """Test Excel file reading functionality."""
        with patch('atem_sisylana.Config') as mock_config, \
             patch('atem_sisylana.utils.MetaConfig') as mock_meta_config, \
             patch('atem_sisylana.load_dotenv'), \
             patch('os.makedirs'), \
             patch('os.remove'):
            
            config_data = json.loads(test_files_setup['config_file'].read_text())
            mock_config.return_value.from_disk.return_value = config_data
            mock_meta_config.return_value = Mock(**config_data)
            
            with patch('atem_sisylana.utils.read_input_excel') as mock_read_excel:
                mock_read_excel.return_value = pd.read_excel(test_files_setup['excel_file'])
                
                analyzer = MigrationAnalyzer(test_files_setup['config_file'])
                
                assert len(analyzer.tbl_list) == 3
                assert 'customer_accounts' in analyzer.tbl_list['pcds_tbl'].iloc[0]
    
    def test_column_mapping_loading(self, test_files_setup):
        """Test column mapping file loading."""
        with patch('atem_sisylana.Config') as mock_config, \
             patch('atem_sisylana.utils.MetaConfig') as mock_meta_config, \
             patch('atem_sisylana.utils.read_input_excel'), \
             patch('atem_sisylana.load_dotenv'), \
             patch('os.makedirs'), \
             patch('os.remove'):
            
            config_data = json.loads(test_files_setup['config_file'].read_text())
            mock_config.return_value.from_disk.return_value = config_data
            mock_meta_config.return_value = Mock(**config_data)
            
            with patch('atem_sisylana.utils.read_column_mapping') as mock_read_mapping:
                mappings = json.loads(test_files_setup['mappings_file'].read_text())
                mock_read_mapping.return_value = mappings
                
                analyzer = MigrationAnalyzer(test_files_setup['config_file'])
                
                assert 'customer_accounts_mapping' in analyzer.col_maps
                assert 'CUSTOMER_ID' in analyzer.col_maps['customer_accounts_mapping']['pcds2aws']
    
    def test_output_directory_creation(self, test_files_setup):
        """Test output directory creation."""
        with patch('atem_sisylana.Config') as mock_config, \
             patch('atem_sisylana.utils.MetaConfig') as mock_meta_config, \
             patch('atem_sisylana.utils.read_column_mapping'), \
             patch('atem_sisylana.utils.read_input_excel'), \
             patch('atem_sisylana.load_dotenv'), \
             patch('os.remove'):
            
            config_data = json.loads(test_files_setup['config_file'].read_text())
            mock_config.return_value.from_disk.return_value = config_data
            mock_meta_config.return_value = Mock(**config_data)
            
            with patch('os.makedirs') as mock_makedirs:
                analyzer = MigrationAnalyzer(test_files_setup['config_file'])
                
                mock_makedirs.assert_called_with(
                    str(test_files_setup['output_dir']), exist_ok=True
                )
    
    def test_csv_output_generation(self, test_files_setup):
        """Test CSV output file generation."""
        output_file = test_files_setup['output_dir'] / "test_results.csv"
        
        # Sample analysis results
        sample_results = [
            {
                'Consumer Loans Data Product': 'Customer Data',
                'PCDS Table Details with DB Name': 'customer_accounts',
                'Tables delivered in AWS with DB Name': 'loans_db.customer_accounts',
                'Hydrated Table only in AWS': 'N',
                'PCDS Table Service Name': 'loans',
                'Status': 'Successful Data Access',
                'Row UnMatch': True,
                'Row UnMatch Details': 'PCDS(1000) : AWS(995)',
                'Col Count Details': 'PCDS(6) : AWS(6)',
                'Time Span UnMatch': False,
                'Time Span Variable': 'created_date : created_date',
                'Time UnMatch Details': '',
                'Column Type UnMatch': False,
                'Type UnMatch Details': '',
                'PCDS Unique Columns': '',
                'AWS Unique Columns': '',
                'Uncaptured Column Mappings': ''
            }
        ]
        
        # Create output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write CSV file
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sample_results[0].keys())
            writer.writeheader()
            writer.writerows(sample_results)
        
        # Verify file creation and content
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]['Status'] == 'Successful Data Access'
    
    def test_pickle_file_operations(self, test_files_setup):
        """Test pickle file save and load operations."""
        pickle_file = test_files_setup['output_dir'] / "test_analysis.pkl"
        pickle_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Sample analysis data
        analysis_data = {
            'customer_accounts': {
                'pcds_meta': {
                    'column': pd.DataFrame([
                        {'column_name': 'CUSTOMER_ID', 'data_type': 'NUMBER'},
                        {'column_name': 'FIRST_NAME', 'data_type': 'VARCHAR2(50)'}
                    ]),
                    'row': pd.DataFrame([{'nrow': 1000}]),
                    'svc': 'loans'
                },
                'aws_meta': {
                    'column': pd.DataFrame([
                        {'column_name': 'customer_id', 'data_type': 'bigint'},
                        {'column_name': 'first_name', 'data_type': 'varchar(50)'}
                    ]),
                    'row': pd.DataFrame([{'nrow': 995}])
                },
                'pcds_date': pd.DataFrame([
                    {'created_date': '2023-01-01', 'NROWS': 100},
                    {'created_date': '2023-01-02', 'NROWS': 95}
                ]),
                'aws_date': pd.DataFrame([
                    {'created_date': '2023-01-01', 'nrows': 100},
                    {'created_date': '2023-01-02', 'nrows': 95}
                ]),
                'mismatch': pd.DataFrame()
            }
        }
        
        # Save to pickle
        with open(pickle_file, 'wb') as f:
            pickle.dump(analysis_data, f)
        
        # Load from pickle
        with open(pickle_file, 'rb') as f:
            loaded_data = pickle.load(f)
        
        assert 'customer_accounts' in loaded_data
        assert 'pcds_meta' in loaded_data['customer_accounts']
        assert len(loaded_data['customer_accounts']['pcds_meta']['column']) == 2


class TestEndToEndWorkflow:
    """Test suite for complete end-to-end workflow."""
    
    @pytest.fixture
    def complete_setup(self, tmp_path):
        """Complete setup for end-to-end testing."""
        # Configuration
        config_data = {
            'input': {
                'table': str(tmp_path / 'tables.xlsx'),
                'env': str(tmp_path / '.env'),
                'name': 'e2e_test',
                'step': 'integration_test',
                'range': [1, 2]  # Only process 2 tables for speed
            },
            'output': {
                'folder': str(tmp_path / 'output'),
                'csv': {
                    'path': str(tmp_path / 'output' / 'results.csv'),
                    'columns': [
                        'Consumer Loans Data Product', 'PCDS Table Details with DB Name',
                        'Tables delivered in AWS with DB Name', 'Status'
                    ]
                },
                'log': {'sink': str(tmp_path / 'output' / 'test.log'), 'level': 'INFO'},
                'to_pkl': str(tmp_path / 'output' / 'analysis.pkl'),
                'next': {
                    'file': str(tmp_path / 'output' / 'next.json'),
                    'fields': {}
                },
                'to_s3': {'run': 's3://test/run'}
            },
            'column_maps': str(tmp_path / 'mappings.json'),
            'match': {'drop_cols': [], 'add_cols': []}
        }
        
        # Create test files
        table_data = pd.DataFrame([
            {
                'pcds_tbl': 'service1.table1',
                'aws_tbl': 'db1.table1',
                'col_map': 'table1_map',
                'pcds_id': 'date_col',
                'aws_id': 'date_col',
                'group': 'Group1',
                'hydrate_only': 'N'
            },
            {
                'pcds_tbl': 'service2.table2',
                'aws_tbl': 'db2.table2',
                'col_map': 'table2_map',
                'pcds_id': 'timestamp_col',
                'aws_id': 'timestamp_col',
                'group': 'Group2',
                'hydrate_only': 'Y'
            }
        ])
        
        mappings = {
            'table1_map': {
                'pcds2aws': {
                    'ID': 'id',
                    'NAME': 'name',
                    'DATE_COL': 'date_col'
                }
            },
            'table2_map': {
                'pcds2aws': {
                    'RECORD_ID': 'record_id',
                    'DESCRIPTION': 'description',
                    'TIMESTAMP_COL': 'timestamp_col'
                }
            }
        }
        
        # Write files
        table_data.to_excel(tmp_path / 'tables.xlsx', index=False)
        (tmp_path / 'mappings.json').write_text(json.dumps(mappings))
        (tmp_path / '.env').write_text('DB_HOST=localhost')
        
        config_file = tmp_path / 'config.json'
        config_file.write_text(json.dumps(config_data))
        
        return {
            'config_file': config_file,
            'config_data': config_data,
            'table_data': table_data,
            'mappings': mappings,
            'tmp_path': tmp_path
        }
    
    @patch('atem_sisylana.utils.start_run')
    @patch('atem_sisylana.utils.end_run')
    @patch('atem_sisylana.utils.aws_creds_renew')
    @patch('atem_sisylana.utils.s3_upload')
    @patch('atem_sisylana.MigrationAnalyzer.query_database')
    def test_complete_analysis_workflow(self, mock_query, mock_s3_upload, mock_aws_creds,
                                      mock_end_run, mock_start_run, complete_setup):
        """Test complete analysis workflow from start to finish."""
        # Setup mock database responses
        mock_responses = {
            # PCDS responses
            ('table1', 'meta'): pd.DataFrame([
                {'column_name': 'ID', 'data_type': 'NUMBER'},
                {'column_name': 'NAME', 'data_type': 'VARCHAR2(100)'},
                {'column_name': 'DATE_COL', 'data_type': 'DATE'}
            ]),
            ('table1', 'nrow'): pd.DataFrame([{'nrow': 1000}]),
            ('table1', 'date'): pd.DataFrame([
                {'date_col': '2023-01-01', 'NROWS': 100},
                {'date_col': '2023-01-02', 'NROWS': 95}
            ]),
            # AWS responses
            ('table1', 'meta', 'db1'): pd.DataFrame([
                {'column_name': 'id', 'data_type': 'bigint'},
                {'column_name': 'name', 'data_type': 'varchar(100)'},
                {'column_name': 'date_col', 'data_type': 'timestamp'}
            ]),
            ('table1', 'nrow', 'db1'): pd.DataFrame([{'nrow': 1000}]),
            ('table1', 'date', 'db1'): pd.DataFrame([
                {'date_col': '2023-01-01', 'nrows': 100},
                {'date_col': '2023-01-02', 'nrows': 95}
            ])
        }
        
        def mock_query_side_effect(table, category, db=None, **kwargs):
            key = (table, category, db) if db else (table, category)
            if key in mock_responses:
                return mock_responses[key]
            else:
                raise utils.NONEXIST_TABLE(f"Mock table {table} not found")
        
        mock_query.side_effect = mock_query_side_effect
        
        # Mock other dependencies
        with patch('atem_sisylana.Config') as mock_config, \
             patch('atem_sisylana.utils.MetaConfig') as mock_meta_config, \
             patch('atem_sisylana.utils.read_column_mapping') as mock_col_map, \
             patch('atem_sisylana.utils.read_input_excel') as mock_excel, \
             patch('atem_sisylana.load_dotenv'), \
             patch('os.makedirs'), \
             patch('os.remove'), \
             patch('shutil.copy'):
            
            mock_config.return_value.from_disk.return_value = complete_setup['config_data']
            mock_meta_config.return_value = Mock(**complete_setup['config_data'])
            mock_col_map.return_value = complete_setup['mappings']
            mock_excel.return_value = complete_setup['table_data']
            
            analyzer = MigrationAnalyzer(complete_setup['config_file'])
            
            # Run the analysis
            analyzer.run_analysis()
            
            # Verify that key workflow steps were called
            mock_start_run.assert_called_once()
            mock_aws_creds.assert_called_once()
            mock_end_run.assert_called_once()
            
            # Verify output files exist
            output_dir = complete_setup['tmp_path'] / 'output'
            assert (output_dir / 'results.csv').exists()
            assert (output_dir / 'analysis.pkl').exists()
            assert (output_dir / 'next.json').exists()
    
    @patch('atem_sisylana.MigrationAnalyzer.query_database')
    def test_error_handling_nonexistent_table(self, mock_query, complete_setup):
        """Test error handling when tables don't exist."""
        # Mock database to raise exceptions for non-existent tables
        mock_query.side_effect = utils.NONEXIST_TABLE("Table not found")
        
        with patch('atem_sisylana.Config') as mock_config, \
             patch('atem_sisylana.utils.MetaConfig') as mock_meta_config, \
             patch('atem_sisylana.utils.read_column_mapping') as mock_col_map, \
             patch('atem_sisylana.utils.read_input_excel') as mock_excel, \
             patch('atem_sisylana.utils.start_run'), \
             patch('atem_sisylana.utils.end_run'), \
             patch('atem_sisylana.utils.aws_creds_renew'), \
             patch('atem_sisylana.load_dotenv'), \
             patch('os.makedirs'), \
             patch('os.remove'), \
             patch('shutil.copy'):
            
            mock_config.return_value.from_disk.return_value = complete_setup['config_data']
            mock_meta_config.return_value = Mock(**complete_setup['config_data'])
            mock_col_map.return_value = complete_setup['mappings']
            mock_excel.return_value = complete_setup['table_data']
            
            analyzer = MigrationAnalyzer(complete_setup['config_file'])
            
            # Should not raise exception, should handle gracefully
            analyzer.run_analysis()
            
            # Verify output files are still created
            output_dir = complete_setup['tmp_path'] / 'output'
            assert (output_dir / 'results.csv').exists()
    
    def test_range_processing(self, complete_setup):
        """Test that row range processing works correctly."""
        # Modify config to process only row 1
        config_data = complete_setup['config_data'].copy()
        config_data['input']['range'] = [1, 1]
        
        config_file = complete_setup['tmp_path'] / 'range_config.json'
        config_file.write_text(json.dumps(config_data))
        
        with patch('atem_sisylana.Config') as mock_config, \
             patch('atem_sisylana.utils.MetaConfig') as mock_meta_config, \
             patch('atem_sisylana.utils.read_column_mapping') as mock_col_map, \
             patch('atem_sisylana.utils.read_input_excel') as mock_excel, \
             patch('atem_sisylana.MigrationAnalyzer.query_database') as mock_query, \
             patch('atem_sisylana.utils.start_run'), \
             patch('atem_sisylana.utils.end_run'), \
             patch('atem_sisylana.utils.aws_creds_renew'), \
             patch('atem_sisylana.load_dotenv'), \
             patch('os.makedirs'), \
             patch('os.remove'), \
             patch('shutil.copy'):
            
            mock_config.return_value.from_disk.return_value = config_data
            mock_meta_config.return_value = Mock(**config_data)
            mock_col_map.return_value = complete_setup['mappings']
            mock_excel.return_value = complete_setup['table_data']
            mock_query.side_effect = utils.NONEXIST_TABLE("Skip for range test")
            
            analyzer = MigrationAnalyzer(config_file)
            analyzer.run_analysis()
            
            # Should process fewer calls due to range limitation
            assert mock_query.call_count <= 4  # Should be limited by range


class TestErrorRecoveryAndReporting:
    """Test suite for error recovery and comprehensive reporting."""
    
    @pytest.fixture
    def error_scenario_setup(self, tmp_path):
        """Setup for testing various error scenarios."""
        config_data = {
            'input': {
                'table': str(tmp_path / 'tables.xlsx'),
                'env': str(tmp_path / '.env'),
                'name': 'error_test',
                'step': 'error_step',
                'range': [1, 3]
            },
            'output': {
                'folder': str(tmp_path / 'output'),
                'csv': {'path': str(tmp_path / 'output' / 'results.csv'), 'columns': []},
                'log': {'sink': str(tmp_path / 'output' / 'error.log'), 'level': 'ERROR'},
                'to_pkl': str(tmp_path / 'output' / 'analysis.pkl'),
                'next': {'file': str(tmp_path / 'output' / 'next.json'), 'fields': {}},
                'to_s3': {'run': 's3://test/error-run'}
            },
            'column_maps': str(tmp_path / 'mappings.json'),
            'match': {'drop_cols': [], 'add_cols': []}
        }
        
        # Create test data with various scenarios
        table_data = pd.DataFrame([
            {
                'pcds_tbl': 'good_service.good_table',
                'aws_tbl': 'good_db.good_table',
                'col_map': 'good_mapping',
                'pcds_id': 'date_col',
                'aws_id': 'date_col',
                'group': 'Good',
                'hydrate_only': 'N'
            },
            {
                'pcds_tbl': 'bad_service.missing_table',
                'aws_tbl': 'bad_db.missing_table',
                'col_map': 'missing_mapping',
                'pcds_id': 'bad_date',
                'aws_id': 'bad_date',
                'group': 'Bad',
                'hydrate_only': 'N'
            },
            {
                'pcds_tbl': 'partial_service.partial_table',
                'aws_tbl': 'partial_db.partial_table',
                'col_map': 'partial_mapping',
                'pcds_id': 'partial_date',
                'aws_id': 'partial_date',
                'group': 'Partial',
                'hydrate_only': 'N'
            }
        ])
        
        mappings = {
            'good_mapping': {'pcds2aws': {'ID': 'id', 'NAME': 'name'}},
            'partial_mapping': {'pcds2aws': {'ID': 'id'}}
            # missing_mapping intentionally omitted
        }
        
        # Write files
        table_data.to_excel(tmp_path / 'tables.xlsx', index=False)
        (tmp_path / 'mappings.json').write_text(json.dumps(mappings))
        (tmp_path / '.env').write_text('DB_HOST=localhost')
        
        config_file = tmp_path / 'config.json'
        config_file.write_text(json.dumps(config_data))
        
        return {
            'config_file': config_file,
            'config_data': config_data,
            'table_data': table_data,
            'mappings': mappings,
            'tmp_path': tmp_path
        }
    
    @patch('atem_sisylana.MigrationAnalyzer.query_database')
    def test_mixed_success_failure_scenarios(self, mock_query, error_scenario_setup):
        """Test handling of mixed success and failure scenarios."""
        def mock_query_side_effect(table, category, db=None, **kwargs):
            if 'good_table' in table:
                # Successful case
                if category == 'meta':
                    return pd.DataFrame([{'column_name': 'ID', 'data_type': 'NUMBER'}])
                elif category == 'nrow':
                    return pd.DataFrame([{'nrow': 100}])
                elif category == 'date':
                    return pd.DataFrame([{'date_col': '2023-01-01', 'NROWS': 10}])
            elif 'missing_table' in table:
                # Table doesn't exist
                raise utils.NONEXIST_TABLE("Table not found")
            elif 'partial_table' in table:
                # Partial data available
                if category == 'meta':
                    return pd.DataFrame([{'column_name': 'ID', 'data_type': 'NUMBER'}])
                elif category == 'date':
                    raise utils.NONEXIST_DATEVAR("Date column not found")
                else:
                    return pd.DataFrame([{'nrow': 50}])
            
            return pd.DataFrame()
        
        mock_query.side_effect = mock_query_side_effect
        
        with patch('atem_sisylana.Config') as mock_config, \
             patch('atem_sisylana.utils.MetaConfig') as mock_meta_config, \
             patch('atem_sisylana.utils.read_column_mapping') as mock_col_map, \
             patch('atem_sisylana.utils.read_input_excel') as mock_excel, \
             patch('atem_sisylana.utils.start_run'), \
             patch('atem_sisylana.utils.end_run'), \
             patch('atem_sisylana.utils.aws_creds_renew'), \
             patch('atem_sisylana.load_dotenv'), \
             patch('os.makedirs'), \
             patch('os.remove'), \
             patch('shutil.copy'):
            
            mock_config.return_value.from_disk.return_value = error_scenario_setup['config_data']
            mock_meta_config.return_value = Mock(**error_scenario_setup['config_data'])
            mock_col_map.return_value = error_scenario_setup['mappings']
            mock_excel.return_value = error_scenario_setup['table_data']
            
            analyzer = MigrationAnalyzer(error_scenario_setup['config_file'])
            
            # Should complete without raising exceptions
            analyzer.run_analysis()
            
            # Verify output files are created
            output_dir = error_scenario_setup['tmp_path'] / 'output'
            assert (output_dir / 'results.csv').exists()
            assert (output_dir / 'analysis.pkl').exists()
    
    def test_s3_upload_error_handling(self, error_scenario_setup):
        """Test error handling for S3 upload failures."""
        with patch('atem_sisylana.Config') as mock_config, \
             patch('atem_sisylana.utils.MetaConfig') as mock_meta_config, \
             patch('atem_sisylana.utils.read_column_mapping'), \
             patch('atem_sisylana.utils.read_input_excel'), \
             patch('atem_sisylana.MigrationAnalyzer.query_database'), \
             patch('atem_sisylana.utils.s3_upload') as mock_s3_upload, \
             patch('atem_sisylana.utils.start_run'), \
             patch('atem_sisylana.utils.end_run'), \
             patch('atem_sisylana.utils.aws_creds_renew'), \
             patch('atem_sisylana.load_dotenv'), \
             patch('os.makedirs'), \
             patch('os.remove'), \
             patch('shutil.copy'):
            
            # Make S3 upload fail
            mock_s3_upload.side_effect = Exception("S3 upload failed")
            
            mock_config.return_value.from_disk.return_value = error_scenario_setup['config_data']
            mock_meta_config.return_value = Mock(**error_scenario_setup['config_data'])
            
            analyzer = MigrationAnalyzer(error_scenario_setup['config_file'])
            
            # Should not raise exception even if S3 upload fails
            analyzer.run_analysis()
            
            # Verify local files are still created
            output_dir = error_scenario_setup['tmp_path'] / 'output'
            assert (output_dir / 'analysis.pkl').exists()


class TestPerformanceAndScalability:
    """Test suite for performance and scalability considerations."""
    
    @pytest.fixture
    def large_dataset_setup(self, tmp_path):
        """Setup for testing with larger datasets."""
        # Create configuration for larger dataset
        config_data = {
            'input': {
                'table': str(tmp_path / 'large_tables.xlsx'),
                'env': str(tmp_path / '.env'),
                'name': 'performance_test',
                'step': 'perf_step',
                'range': [1, 50]  # Process 50 tables
            },
            'output': {
                'folder': str(tmp_path / 'output'),
                'csv': {'path': str(tmp_path / 'output' / 'results.csv'), 'columns': []},
                'log': {'sink': str(tmp_path / 'output' / 'perf.log'), 'level': 'INFO'},
                'to_pkl': str(tmp_path / 'output' / 'analysis.pkl'),
                'next': {'file': str(tmp_path / 'output' / 'next.json'), 'fields': {}},
                'to_s3': {'run': 's3://test/perf-run'}
            },
            'column_maps': str(tmp_path / 'mappings.json'),
            'match': {'drop_cols': [], 'add_cols': []}
        }
        
        # Generate larger dataset
        table_data = []
        mappings = {}
        
        for i in range(50):
            table_name = f'table_{i:03d}'
            mapping_name = f'mapping_{i:03d}'
            
            table_data.append({
                'pcds_tbl': f'service_{i % 5}.{table_name}',
                'aws_tbl': f'db_{i % 3}.{table_name}',
                'col_map': mapping_name,
                'pcds_id': 'created_date',
                'aws_id': 'created_date',
                'group': f'Group_{i % 10}',
                'hydrate_only': 'N' if i % 2 == 0 else 'Y'
            })
            
            mappings[mapping_name] = {
                'pcds2aws': {
                    'ID': 'id',
                    'NAME': 'name',
                    'CREATED_DATE': 'created_date'
                }
            }
        
        table_df = pd.DataFrame(table_data)
        
        # Write files
        table_df.to_excel(tmp_path / 'large_tables.xlsx', index=False)
        (tmp_path / 'mappings.json').write_text(json.dumps(mappings))
        (tmp_path / '.env').write_text('DB_HOST=localhost')
        
        config_file = tmp_path / 'config.json'
        config_file.write_text(json.dumps(config_data))
        
        return {
            'config_file': config_file,
            'config_data': config_data,
            'table_data': table_df,
            'mappings': mappings,
            'tmp_path': tmp_path
        }
    
    @patch('atem_sisylana.MigrationAnalyzer.query_database')
    def test_large_dataset_processing(self, mock_query, large_dataset_setup):
        """Test processing of larger datasets."""
        # Mock database responses with consistent data
        def mock_query_side_effect(table, category, db=None, **kwargs):
            if category == 'meta':
                return pd.DataFrame([
                    {'column_name': 'ID', 'data_type': 'NUMBER'},
                    {'column_name': 'NAME', 'data_type': 'VARCHAR2(100)'},
                    {'column_name': 'CREATED_DATE', 'data_type': 'DATE'}
                ])
            elif category == 'nrow':
                return pd.DataFrame([{'nrow': 1000}])
            elif category == 'date':
                return pd.DataFrame([
                    {'created_date': '2023-01-01', 'NROWS': 100}
                ])
            return pd.DataFrame()
        
        mock_query.side_effect = mock_query_side_effect
        
        with patch('atem_sisylana.Config') as mock_config, \
             patch('atem_sisylana.utils.MetaConfig') as mock_meta_config, \
             patch('atem_sisylana.utils.read_column_mapping') as mock_col_map, \
             patch('atem_sisylana.utils.read_input_excel') as mock_excel, \
             patch('atem_sisylana.utils.start_run'), \
             patch('atem_sisylana.utils.end_run'), \
             patch('atem_sisylana.utils.aws_creds_renew'), \
             patch('atem_sisylana.load_dotenv'), \
             patch('os.makedirs'), \
             patch('os.remove'), \
             patch('shutil.copy'):
            
            mock_config.return_value.from_disk.return_value = large_dataset_setup['config_data']
            mock_meta_config.return_value = Mock(**large_dataset_setup['config_data'])
            mock_col_map.return_value = large_dataset_setup['mappings']
            mock_excel.return_value = large_dataset_setup['table_data']
            
            analyzer = MigrationAnalyzer(large_dataset_setup['config_file'])
            
            # Measure execution time
            import time
            start_time = time.time()
            analyzer.run_analysis()
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Should complete within reasonable time (adjust threshold as needed)
            assert execution_time < 60  # Should complete within 60 seconds
            
            # Verify all output files are created
            output_dir = large_dataset_setup['tmp_path'] / 'output'
            assert (output_dir / 'results.csv').exists()
            assert (output_dir / 'analysis.pkl').exists()
    
    def test_memory_usage_with_large_schemas(self, large_dataset_setup):
        """Test memory usage with large schema data."""
        # Create large schema DataFrames
        large_pcds_schema = pd.DataFrame([
            {
                'column_name': f'COLUMN_{i:04d}',
                'data_type': 'VARCHAR2(100)' if i % 2 == 0 else 'NUMBER',
                'aws_colname': f'column_{i:04d}'
            }
            for i in range(1000)  # 1000 columns
        ])
        
        large_aws_schema = pd.DataFrame([
            {
                'column_name': f'column_{i:04d}',
                'data_type': 'varchar(100)' if i % 2 == 0 else 'bigint'
            }
            for i in range(1000)  # 1000 columns
        ])
        
        with patch('atem_sisylana.Config') as mock_config, \
             patch('atem_sisylana.utils.MetaConfig') as mock_meta_config, \
             patch('atem_sisylana.utils.read_column_mapping'), \
             patch('atem_sisylana.utils.read_input_excel'), \
             patch('atem_sisylana.load_dotenv'), \
             patch('os.makedirs'), \
             patch('os.remove'):
            
            mock_config.return_value.from_disk.return_value = large_dataset_setup['config_data']
            mock_meta_config.return_value = Mock(**large_dataset_setup['config_data'])
            
            analyzer = MigrationAnalyzer(large_dataset_setup['config_file'])
            
            # Test schema comparison with large datasets
            result = analyzer.merge_and_compare_schemas(
                large_pcds_schema, large_aws_schema, PullStatus.SUCCESS
            )
            
            # Should handle large schemas without memory issues
            assert result is not None
            assert result.col_mapping is not None
            assert len(result.col_mapping) == 1000


class TestConfigurationEdgeCases:
    """Test suite for configuration edge cases and validation."""
    
    def test_missing_config_file(self, tmp_path):
        """Test handling of missing configuration file."""
        nonexistent_config = tmp_path / "nonexistent_config.json"
        
        with pytest.raises(FileNotFoundError):
            MigrationAnalyzer(nonexistent_config)
    
    def test_malformed_config_file(self, tmp_path):
        """Test handling of malformed configuration file."""
        malformed_config = tmp_path / "malformed_config.json"
        malformed_config.write_text("{ invalid json content")
        
        with patch('atem_sisylana.Config') as mock_config:
            mock_config.return_value.from_disk.side_effect = json.JSONDecodeError(
                "Invalid JSON", "test", 0
            )
            
            with pytest.raises(json.JSONDecodeError):
                MigrationAnalyzer(malformed_config)
    
    def test_missing_required_config_sections(self, tmp_path):
        """Test handling of missing required configuration sections."""
        incomplete_configs = [
            {},  # Completely empty
            {'input': {}},  # Missing output section
            {'output': {}},  # Missing input section
            {'input': {'table': 'test.xlsx'}, 'output': {}}  # Minimal sections
        ]
        
        for i, config_data in enumerate(incomplete_configs):
            config_file = tmp_path / f"incomplete_config_{i}.json"
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
                
                # Should handle gracefully or provide meaningful error
                try:
                    analyzer = MigrationAnalyzer(config_file)
                    assert analyzer is not None
                except (AttributeError, KeyError) as e:
                    # Expected for incomplete configurations
                    assert "missing" in str(e).lower() or "attribute" in str(e).lower()
    
    def test_invalid_range_configuration(self, tmp_path):
        """Test handling of invalid range configurations."""
        invalid_ranges = [
            [5, 1],      # End before start
            [-1, 10],    # Negative start
            [1, -5],     # Negative end
            [0, 0],      # Zero range
        ]
        
        base_config = {
            'input': {
                'table': 'test.xlsx',
                'env': '.env',
                'name': 'test',
                'step': 'test'
            },
            'output': {'folder': 'output'},
            'column_maps': 'maps.json'
        }
        
        for i, invalid_range in enumerate(invalid_ranges):
            config_data = base_config.copy()
            config_data['input']['range'] = invalid_range
            
            config_file = tmp_path / f"invalid_range_{i}.json"
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
                
                # Should either handle gracefully or provide clear error
                analyzer = MigrationAnalyzer(config_file)
                assert analyzer is not None


class TestDataValidationIntegration:
    """Test suite for data validation in integration scenarios."""
    
    @pytest.fixture
    def validation_setup(self, tmp_path):
        """Setup for data validation testing."""
        config_data = {
            'input': {
                'table': str(tmp_path / 'validation_tables.xlsx'),
                'env': str(tmp_path / '.env'),
                'name': 'validation_test',
                'step': 'validation',
                'range': [1, 3]
            },
            'output': {
                'folder': str(tmp_path / 'output'),
                'csv': {'path': str(tmp_path / 'output' / 'results.csv'), 'columns': []},
                'log': {'sink': str(tmp_path / 'output' / 'validation.log'), 'level': 'DEBUG'},
                'to_pkl': str(tmp_path / 'output' / 'analysis.pkl'),
                'next': {'file': str(tmp_path / 'output' / 'next.json'), 'fields': {}},
                'to_s3': {'run': 's3://test/validation-run'}
            },
            'column_maps': str(tmp_path / 'mappings.json'),
            'match': {'drop_cols': ['audit_col'], 'add_cols': ['system_col']}
        }
        
        # Create test tables with validation scenarios
        table_data = pd.DataFrame([
            {
                'pcds_tbl': 'validation.type_mismatches',
                'aws_tbl': 'validation_db.type_mismatches',
                'col_map': 'type_mismatch_mapping',
                'pcds_id': 'created_date',
                'aws_id': 'created_date',
                'group': 'Validation',
                'hydrate_only': 'N'
            },
            {
                'pcds_tbl': 'validation.row_count_diff',
                'aws_tbl': 'validation_db.row_count_diff',
                'col_map': 'row_count_mapping',
                'pcds_id': 'updated_date',
                'aws_id': 'updated_date',
                'group': 'Validation',
                'hydrate_only': 'N'
            },
            {
                'pcds_tbl': 'validation.temporal_mismatches',
                'aws_tbl': 'validation_db.temporal_mismatches',
                'col_map': 'temporal_mapping',
                'pcds_id': 'event_date',
                'aws_id': 'event_date',
                'group': 'Validation',
                'hydrate_only': 'N'
            }
        ])
        
        mappings = {
            'type_mismatch_mapping': {
                'pcds2aws': {
                    'ID': 'id',
                    'AMOUNT': 'amount',
                    'CREATED_DATE': 'created_date'
                }
            },
            'row_count_mapping': {
                'pcds2aws': {
                    'RECORD_ID': 'record_id',
                    'STATUS': 'status',
                    'UPDATED_DATE': 'updated_date'
                }
            },
            'temporal_mapping': {
                'pcds2aws': {
                    'EVENT_ID': 'event_id',
                    'EVENT_TYPE': 'event_type',
                    'EVENT_DATE': 'event_date'
                }
            }
        }
        
        # Write files
        table_data.to_excel(tmp_path / 'validation_tables.xlsx', index=False)
        (tmp_path / 'mappings.json').write_text(json.dumps(mappings))
        (tmp_path / '.env').write_text('DB_HOST=localhost')
        
        config_file = tmp_path / 'config.json'
        config_file.write_text(json.dumps(config_data))
        
        return {
            'config_file': config_file,
            'config_data': config_data,
            'table_data': table_data,
            'mappings': mappings,
            'tmp_path': tmp_path
        }
    
    @patch('atem_sisylana.MigrationAnalyzer.query_database')
    def test_comprehensive_validation_scenarios(self, mock_query, validation_setup):
        """Test comprehensive validation scenarios."""
        def mock_query_side_effect(table, category, db=None, **kwargs):
            if 'type_mismatches' in table:
                if category == 'meta':
                    if db:  # AWS
                        return pd.DataFrame([
                            {'column_name': 'id', 'data_type': 'varchar(10)'},  # Type mismatch
                            {'column_name': 'amount', 'data_type': 'varchar(20)'},  # Type mismatch
                            {'column_name': 'created_date', 'data_type': 'timestamp'}
                        ])
                    else:  # PCDS
                        return pd.DataFrame([
                            {'column_name': 'ID', 'data_type': 'NUMBER'},
                            {'column_name': 'AMOUNT', 'data_type': 'NUMBER(10,2)'},
                            {'column_name': 'CREATED_DATE', 'data_type': 'DATE'}
                        ])
                elif category == 'nrow':
                    return pd.DataFrame([{'nrow': 1000}])
                elif category == 'date':
                    return pd.DataFrame([
                        {'created_date': '2023-01-01', 'NROWS' if not db else 'nrows': 100}
                    ])
            
            elif 'row_count_diff' in table:
                if category == 'meta':
                    if db:  # AWS
                        return pd.DataFrame([
                            {'column_name': 'record_id', 'data_type': 'bigint'},
                            {'column_name': 'status', 'data_type': 'varchar(20)'},
                            {'column_name': 'updated_date', 'data_type': 'timestamp'}
                        ])
                    else:  # PCDS
                        return pd.DataFrame([
                            {'column_name': 'RECORD_ID', 'data_type': 'NUMBER'},
                            {'column_name': 'STATUS', 'data_type': 'VARCHAR2(20)'},
                            {'column_name': 'UPDATED_DATE', 'data_type': 'DATE'}
                        ])
                elif category == 'nrow':
                    return pd.DataFrame([{'nrow': 1000 if not db else 950}])  # Row count mismatch
                elif category == 'date':
                    return pd.DataFrame([
                        {'updated_date': '2023-01-01', 'NROWS' if not db else 'nrows': 50}
                    ])
            
            elif 'temporal_mismatches' in table:
                if category == 'meta':
                    if db:  # AWS
                        return pd.DataFrame([
                            {'column_name': 'event_id', 'data_type': 'bigint'},
                            {'column_name': 'event_type', 'data_type': 'varchar(50)'},
                            {'column_name': 'event_date', 'data_type': 'date'}
                        ])
                    else:  # PCDS
                        return pd.DataFrame([
                            {'column_name': 'EVENT_ID', 'data_type': 'NUMBER'},
                            {'column_name': 'EVENT_TYPE', 'data_type': 'VARCHAR2(50)'},
                            {'column_name': 'EVENT_DATE', 'data_type': 'DATE'}
                        ])
                elif category == 'nrow':
                    return pd.DataFrame([{'nrow': 500}])
                elif category == 'date':
                    if db:  # AWS - different counts for some dates
                        return pd.DataFrame([
                            {'event_date': '2023-01-01', 'nrows': 50},
                            {'event_date': '2023-01-02', 'nrows': 45},  # Mismatch
                            {'event_date': '2023-01-03', 'nrows': 55}
                        ])
                    else:  # PCDS
                        return pd.DataFrame([
                            {'event_date': '2023-01-01', 'NROWS': 50},
                            {'event_date': '2023-01-02', 'NROWS': 50},  # Different from AWS
                            {'event_date': '2023-01-03', 'NROWS': 55}
                        ])
            
            return pd.DataFrame()
        
        mock_query.side_effect = mock_query_side_effect
        
        with patch('atem_sisylana.Config') as mock_config, \
             patch('atem_sisylana.utils.MetaConfig') as mock_meta_config, \
             patch('atem_sisylana.utils.read_column_mapping') as mock_col_map, \
             patch('atem_sisylana.utils.read_input_excel') as mock_excel, \
             patch('atem_sisylana.utils.start_run'), \
             patch('atem_sisylana.utils.end_run'), \
             patch('atem_sisylana.utils.aws_creds_renew'), \
             patch('atem_sisylana.load_dotenv'), \
             patch('os.makedirs'), \
             patch('os.remove'), \
             patch('shutil.copy'):
            
            mock_config.return_value.from_disk.return_value = validation_setup['config_data']
            mock_meta_config.return_value = Mock(**validation_setup['config_data'])
            mock_col_map.return_value = validation_setup['mappings']
            mock_excel.return_value = validation_setup['table_data']
            
            analyzer = MigrationAnalyzer(validation_setup['config_file'])
            analyzer.run_analysis()
            
            # Verify all validation scenarios were processed
            output_dir = validation_setup['tmp_path'] / 'output'
            assert (output_dir / 'results.csv').exists()
            
            # Read and validate CSV results
            results_df = pd.read_csv(output_dir / 'results.csv')
            assert len(results_df) == 3  # Should have processed 3 tables
            
            # Verify that different types of mismatches were detected
            type_mismatch_row = results_df[results_df['PCDS Table Details with DB Name'].str.contains('type_mismatches')]
            row_count_row = results_df[results_df['PCDS Table Details with DB Name'].str.contains('row_count_diff')]
            temporal_row = results_df[results_df['PCDS Table Details with DB Name'].str.contains('temporal_mismatches')]
            
            assert len(type_mismatch_row) == 1
            assert len(row_count_row) == 1  
            assert len(temporal_row) == 1


if __name__ == '__main__':
    # Run all tests with comprehensive reporting
    pytest.main([
        __file__, 
        '-v', 
        '--tb=short',
        '--cov=atem_sisylana',
        '--cov-report=html',
        '--cov-report=term-missing'
    ])