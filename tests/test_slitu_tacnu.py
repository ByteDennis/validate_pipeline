import pytest
import pandas as pd
import pickle
import tempfile
import os
from unittest.mock import patch, mock_open
from mismatch_processor import DataMismatchProcessor


@pytest.fixture
def sample_csv_content():
    """Sample CSV content for testing."""
    return """tbl,mismatched date
a,2025-02-02; 2025-04-04
b,2025-02-02; 2025-04-01"""


@pytest.fixture
def sample_data_dict():
    """Sample pickle data for testing."""
    pcds_a = pd.DataFrame({
        'date': ['2025-01-01', '2025-02-02', '2025-04-04'],
        'number of record': [24, 14, 30]
    })
    aws_a = pd.DataFrame({
        'date': ['2025-01-01', '2025-02-02', '2025-04-04'],
        'number of record': [24, 32, 50]
    })
    pcds_b = pd.DataFrame({
        'date': ['2025-01-01', '2025-02-02', '2025-04-01'],
        'number of record': [20, 30, 1]
    })
    aws_b = pd.DataFrame({
        'date': ['2025-01-01', '2025-02-02', '2025-04-01'],
        'number of record': [20, 10, 0]
    })
    
    return {
        'a': {'pcds': pcds_a, 'aws': aws_a},
        'b': {'pcds': pcds_b, 'aws': aws_b}
    }


@pytest.fixture
def temp_files(sample_csv_content, sample_data_dict):
    """Create temporary files for testing."""
    # Create temporary CSV file
    csv_fd, csv_path = tempfile.mkstemp(suffix='.csv')
    with os.fdopen(csv_fd, 'w') as f:
        f.write(sample_csv_content)
    
    # Create temporary pickle file
    pkl_fd, pkl_path = tempfile.mkstemp(suffix='.pkl')
    with os.fdopen(pkl_fd, 'wb') as f:
        pickle.dump(sample_data_dict, f)
    
    yield csv_path, pkl_path
    
    # Cleanup
    os.unlink(csv_path)
    os.unlink(pkl_path)


class TestDataMismatchProcessor:
    
    def test_init(self):
        """Test processor initialization."""
        processor = DataMismatchProcessor('test.csv', 'test.pkl')
        assert processor.csv_path == 'test.csv'
        assert processor.pkl_path == 'test.pkl'
        assert processor.mismatched_dates_df is None
        assert processor.data_dict is None
    
    def test_load_data(self, temp_files):
        """Test data loading functionality."""
        csv_path, pkl_path = temp_files
        processor = DataMismatchProcessor(csv_path, pkl_path)
        processor.load_data()
        
        assert processor.mismatched_dates_df is not None
        assert processor.data_dict is not None
        assert len(processor.mismatched_dates_df) == 2
        assert 'a' in processor.data_dict
        assert 'b' in processor.data_dict
    
    def test_parse_dates(self, temp_files):
        """Test date parsing functionality."""
        csv_path, pkl_path = temp_files
        processor = DataMismatchProcessor(csv_path, pkl_path)
        
        # Test normal case
        dates = processor._parse_dates("2025-02-02; 2025-04-04")
        assert dates == ['2025-02-02', '2025-04-04']
        
        # Test single date
        dates = processor._parse_dates("2025-02-02")
        assert dates == ['2025-02-02']
        
        # Test empty string
        dates = processor._parse_dates("")
        assert dates == []
        
        # Test NaN
        dates = processor._parse_dates(pd.NA)
        assert dates == []
    
    def test_get_record_count(self, temp_files, sample_data_dict):
        """Test record count retrieval."""
        csv_path, pkl_path = temp_files
        processor = DataMismatchProcessor(csv_path, pkl_path)
        processor.data_dict = sample_data_dict
        
        pcds_a = sample_data_dict['a']['pcds']
        
        # Test existing date
        count = processor._get_record_count(pcds_a, '2025-02-02')
        assert count == 14
        
        # Test non-existing date
        count = processor._get_record_count(pcds_a, '2025-12-31')
        assert count == 0
    
    def test_check_mismatch(self, temp_files, sample_data_dict):
        """Test mismatch checking logic."""
        csv_path, pkl_path = temp_files
        processor = DataMismatchProcessor(csv_path, pkl_path)
        processor.data_dict = sample_data_dict
        
        # Test mismatch case
        has_mismatch, display = processor._check_mismatch('a', '2025-02-02')
        assert has_mismatch is True
        assert display == "Yes (14 : 32)"
        
        # Test no mismatch case (assuming we modify data)
        # Create a case where counts match
        sample_data_dict['a']['aws'].loc[
            sample_data_dict['a']['aws']['date'] == '2025-01-01', 
            'number of record'
        ] = 24
        sample_data_dict['a']['pcds'].loc[
            sample_data_dict['a']['pcds']['date'] == '2025-01-01', 
            'number of record'
        ] = 24
        
        has_mismatch, display = processor._check_mismatch('a', '2025-01-01')
        assert has_mismatch is False
        assert display == "No"
        
        # Test non-existing table
        has_mismatch, display = processor._check_mismatch('nonexistent', '2025-02-02')
        assert has_mismatch is False
        assert display == "No"
    
    def test_generate_matrix(self, temp_files):
        """Test matrix generation."""
        csv_path, pkl_path = temp_files
        processor = DataMismatchProcessor(csv_path, pkl_path)
        processor.load_data()
        
        result_df = processor.generate_matrix()
        
        # Check structure
        assert 'tbl' in result_df.columns
        assert len(result_df) == 2  # Two tables: a, b
        assert result_df['tbl'].tolist() == ['a', 'b']
        
        # Check that date columns exist
        expected_dates = ['2025-02-02', '2025-04-01', '2025-04-04']
        for date in expected_dates:
            assert date in result_df.columns
    
    def test_generate_matrix_without_loaded_data(self, temp_files):
        """Test matrix generation without loaded data raises error."""
        csv_path, pkl_path = temp_files
        processor = DataMismatchProcessor(csv_path, pkl_path)
        
        with pytest.raises(ValueError, match="Data not loaded"):
            processor.generate_matrix()
    
    def test_process_complete_pipeline(self, temp_files):
        """Test complete processing pipeline."""
        csv_path, pkl_path = temp_files
        processor = DataMismatchProcessor(csv_path, pkl_path)
        
        result_df = processor.process()
        
        # Verify the result structure
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2
        assert 'tbl' in result_df.columns
        
        # Verify specific results based on sample data
        # Table 'a' on 2025-02-02 should be "Yes (14 : 32)"
        row_a = result_df[result_df['tbl'] == 'a'].iloc[0]
        assert row_a['2025-02-02'] == "Yes (14 : 32)"
        
        # Table 'a' on 2025-04-04 should be "Yes (30 : 50)"
        assert row_a['2025-04-04'] == "Yes (30 : 50)"
    
    def test_file_not_found_error(self):
        """Test handling of non-existent files."""
        processor = DataMismatchProcessor('nonexistent.csv', 'nonexistent.pkl')
        
        with pytest.raises(FileNotFoundError):
            processor.load_data()
    
    @pytest.fixture
    def malformed_data_dict(self):
        """Sample data with missing keys for testing edge cases."""
        return {
            'a': {'pcds': pd.DataFrame({'date': ['2025-01-01'], 'number of record': [10]})},
            'b': {}
        }
    
    def test_missing_aws_data(self, temp_files, malformed_data_dict):
        """Test handling of missing AWS data."""
        csv_path, pkl_path = temp_files
        processor = DataMismatchProcessor(csv_path, pkl_path)
        processor.mismatched_dates_df = pd.read_csv(csv_path)
        processor.data_dict = malformed_data_dict
        
        # Should return "No" when AWS data is missing
        has_mismatch, display = processor._check_mismatch('a', '2025-01-01')
        assert has_mismatch is False
        assert display == "No"
        
        # Should return "No" when table data is completely empty
        has_mismatch, display = processor._check_mismatch('b', '2025-01-01')
        assert has_mismatch is False
        assert display == "No"


# Integration test
def test_integration_example_data(temp_files):
    """Test with the exact example data from the problem description."""
    csv_path, pkl_path = temp_files
    
    # Create processor and run complete pipeline
    processor = DataMismatchProcessor(csv_path, pkl_path)
    result = processor.process()
    
    # Print result for manual verification
    print("\nGenerated Matrix:")
    print(result.to_string(index=False))
    
    # Basic assertions
    assert len(result) == 2
    assert result['tbl'].tolist() == ['a', 'b']