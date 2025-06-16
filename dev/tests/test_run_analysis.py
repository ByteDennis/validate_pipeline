import pytest
from unittest.mock import patch, MagicMock
from utils.types import PullStatus, NONEXIST_TABLE, NONEXIST_DATEVAR

def test_no_mapping():
    analyzer = MigrationAnalyzer('config.cfg')
    analyzer.col_maps = {}  # Empty mapping
    
    table_info = {'pcds_tbl': 'svc.table1', 'aws_tbl': 'db.table1', 'col_map': 'missing_key'}
    result = analyzer.analyze_table(table_info)
    
    assert result['status'] == PullStatus.NO_MAPPING

def test_nonexist_pcds():
    with patch('src.analyzer.get_table_metadata') as mock_meta:
        mock_meta.side_effect = NONEXIST_TABLE("Table not found")
        
        analyzer = MigrationAnalyzer('config.cfg')
        analyzer.col_maps = {'test_map': {}}
        
        table_info = {'pcds_tbl': 'svc.missing_table', 'aws_tbl': 'db.table1', 'col_map': 'test_map'}
        result = analyzer.analyze_table(table_info)
        
        assert result['status'] == PullStatus.NONEXIST_PCDS

def test_empty_pcds():
    with patch('src.analyzer.get_table_metadata') as mock_meta, \
         patch('src.analyzer.get_table_row_count') as mock_rows:
        
        mock_meta.return_value = pd.DataFrame()
        mock_rows.return_value = pd.DataFrame({'nrow': [0]})  # Empty table
        
        analyzer = MigrationAnalyzer('config.cfg')
        analyzer.col_maps = {'test_map': {}}
        
        table_info = {'pcds_tbl': 'svc.empty_table', 'aws_tbl': 'db.table1', 'col_map': 'test_map'}
        result = analyzer.analyze_table(table_info)
        
        assert result['status'] == PullStatus.EMPTY_PCDS

def test_nonexist_aws():
    with patch('src.analyzer.get_table_metadata') as mock_meta, \
         patch('src.analyzer.get_table_row_count') as mock_rows:
        
        # PCDS calls succeed, AWS fails
        mock_meta.side_effect = [pd.DataFrame(), NONEXIST_TABLE("AWS table not found")]
        mock_rows.return_value = pd.DataFrame({'nrow': [100]})
        
        analyzer = MigrationAnalyzer('config.cfg')
        analyzer.col_maps = {'test_map': {}}
        
        table_info = {'pcds_tbl': 'svc.table1', 'aws_tbl': 'db.missing_table', 'col_map': 'test_map'}
        result = analyzer.analyze_table(table_info)
        
        assert result['status'] == PullStatus.NONEXIST_AWS

def test_empty_aws():
    with patch('src.analyzer.get_table_metadata') as mock_meta, \
         patch('src.analyzer.get_table_row_count') as mock_rows:
        
        mock_meta.return_value = pd.DataFrame()
        mock_rows.side_effect = [
            pd.DataFrame({'nrow': [100]}),  # PCDS has data
            pd.DataFrame({'nrow': [0]})     # AWS is empty
        ]
        
        analyzer = MigrationAnalyzer('config.cfg')
        analyzer.col_maps = {'test_map': {}}
        
        table_info = {'pcds_tbl': 'svc.table1', 'aws_tbl': 'db.empty_table', 'col_map': 'test_map'}
        result = analyzer.analyze_table(table_info)
        
        assert result['status'] == PullStatus.EMPTY_AWS

def test_nondate_pcds():
    with patch('src.analyzer.get_table_metadata') as mock_meta, \
         patch('src.analyzer.get_table_row_count') as mock_rows, \
         patch('src.analyzer.get_date_distribution') as mock_date:
        
        mock_meta.return_value = pd.DataFrame()
        mock_rows.return_value = pd.DataFrame({'nrow': [100]})
        mock_date.side_effect = NONEXIST_DATEVAR("Date column not found")
        
        analyzer = MigrationAnalyzer('config.cfg')
        analyzer.col_maps = {'test_map': {}}
        
        table_info = {
            'pcds_tbl': 'svc.table1', 
            'aws_tbl': 'db.table1', 
            'col_map': 'test_map',
            'pcds_date_col': 'missing_date_col'
        }
        result = analyzer.analyze_table(table_info)
        
        assert result['status'] == PullStatus.NONDATE_PCDS

def test_success():
    with patch('src.analyzer.get_table_metadata') as mock_meta, \
         patch('src.analyzer.get_table_row_count') as mock_rows:
        
        mock_meta.return_value = pd.DataFrame({'column_name': ['col1'], 'data_type': ['VARCHAR']})
        mock_rows.return_value = pd.DataFrame({'nrow': [100]})
        
        analyzer = MigrationAnalyzer('config.cfg')
        analyzer.col_maps = {'test_map': {}}
        analyzer._compare_schemas = MagicMock(return_value={})
        
        table_info = {'pcds_tbl': 'svc.table1', 'aws_tbl': 'db.table1', 'col_map': 'test_map'}
        result = analyzer.analyze_table(table_info)
        
        assert result['status'] == PullStatus.SUCCESS