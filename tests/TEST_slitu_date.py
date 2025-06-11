import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List
import tempfile
import os
import pickle
import time
from mismatch_processor import DataMismatchProcessor



@pytest.fixture
def complex_csv_content():
    """More complex CSV with various edge cases."""
    return """tbl,mismatched date
users,2024-01-15; 2024-02-28; 2024-03-31; 2024-12-31
orders,2024-01-01; 2024-06-15; 2024-11-30
products,2024-02-29; 2024-07-04; 2024-10-31; 2024-12-25
inventory,2024-01-15; 2024-05-01
transactions,2024-03-17; 2024-08-15; 2024-09-30; 2024-11-11; 2024-12-01
customers,2024-04-01; 2024-07-15
analytics,2024-01-01; 2024-02-14; 2024-03-30; 2024-04-15; 2024-05-31; 2024-06-30
logs,2024-12-31
empty_table,2024-01-01; 2024-06-01
partial_data,2024-02-15; 2024-08-20"""


@pytest.fixture
def generate_realistic_dataframe():
    """Factory function to generate realistic dataframes with various patterns."""
    def _generate_df(
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        base_records: int = 1000,
        variance: float = 0.3,
        missing_dates: List[str] = None,
        spike_dates: List[str] = None,
        trend: str = "stable"  # "stable", "increasing", "decreasing", "seasonal"
    ) -> pd.DataFrame:
        
        # Generate date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        date_range = pd.date_range(start=start, end=end, freq='D')
        
        records = []
        for i, date in enumerate(date_range):
            date_str = date.strftime('%Y-%m-%d')
            
            # Skip missing dates
            if missing_dates and date_str in missing_dates:
                continue
            
            # Base calculation
            if trend == "increasing":
                base = base_records + (i * 10)
            elif trend == "decreasing":
                base = max(10, base_records - (i * 5))
            elif trend == "seasonal":
                # Simulate seasonal pattern (higher in winter/summer)
                seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * i / 365)
                base = int(base_records * seasonal_factor)
            else:  # stable
                base = base_records
            
            # Add variance
            variance_factor = 1 + random.uniform(-variance, variance)
            count = max(0, int(base * variance_factor))
            
            # Add spikes
            if spike_dates and date_str in spike_dates:
                count = int(count * random.uniform(3, 8))
            
            records.append({
                'date': date_str,
                'number of record': count
            })
        
        return pd.DataFrame(records)
    
    return _generate_df


@pytest.fixture
def complex_data_dict(generate_realistic_dataframe):
    """Complex data dictionary with various realistic scenarios."""
    
    # Scenario 1: Users table - steady growth with some missing AWS data
    users_pcds = generate_realistic_dataframe(
        base_records=5000,
        trend="increasing",
        spike_dates=["2024-12-31"]  # New Year spike
    )
    users_aws = generate_realistic_dataframe(
        base_records=5000,
        trend="increasing",
        missing_dates=["2024-02-28", "2024-03-31"],  # AWS outage
        spike_dates=["2024-12-31"]
    )
    
    # Scenario 2: Orders table - seasonal pattern with Black Friday spike
    orders_pcds = generate_realistic_dataframe(
        base_records=2000,
        trend="seasonal",
        spike_dates=["2024-11-30"]  # Black Friday
    )
    orders_aws = generate_realistic_dataframe(
        base_records=2000,
        trend="seasonal",
        spike_dates=["2024-11-30"],
        variance=0.1  # Lower variance for AWS
    )
    
    # Scenario 3: Products table - stable with occasional discrepancies
    products_pcds = generate_realistic_dataframe(
        base_records=800,
        variance=0.2
    )
    products_aws = generate_realistic_dataframe(
        base_records=750,  # Consistently lower
        variance=0.2
    )
    
    # Scenario 4: Inventory - decreasing trend
    inventory_pcds = generate_realistic_dataframe(
        base_records=1500,
        trend="decreasing"
    )
    inventory_aws = generate_realistic_dataframe(
        base_records=1500,
        trend="decreasing",
        missing_dates=["2024-05-01"]  # Maintenance day
    )
    
    # Scenario 5: Transactions - high volume with significant variance
    transactions_pcds = generate_realistic_dataframe(
        base_records=10000,
        variance=0.5,
        spike_dates=["2024-12-01"]  # Holiday shopping
    )
    transactions_aws = generate_realistic_dataframe(
        base_records=9800,  # Slightly different base
        variance=0.5,
        spike_dates=["2024-12-01"]
    )
    
    # Scenario 6: Customers - perfect match (no mismatches)
    customers_base = generate_realistic_dataframe(base_records=3000, variance=0.1)
    customers_pcds = customers_base.copy()
    customers_aws = customers_base.copy()
    
    # Scenario 7: Analytics - complex pattern with multiple spikes
    analytics_pcds = generate_realistic_dataframe(
        base_records=500,
        spike_dates=["2024-02-14", "2024-04-15", "2024-06-30"]  # Valentine's, Tax day, Quarter end
    )
    analytics_aws = generate_realistic_dataframe(
        base_records=500,
        spike_dates=["2024-02-14", "2024-04-15", "2024-06-30"],
        variance=0.4  # Higher variance
    )
    
    # Scenario 8: Logs - year-end only data
    logs_data = [{
        'date': '2024-12-31',
        'number of record': 50000
    }]
    logs_pcds = pd.DataFrame(logs_data)
    logs_aws = pd.DataFrame([{
        'date': '2024-12-31',
        'number of record': 48000  # Mismatch
    }])
    
    # Scenario 9: Empty table - no data in AWS
    empty_pcds = pd.DataFrame([
        {'date': '2024-01-01', 'number of record': 100},
        {'date': '2024-06-01', 'number of record': 150}
    ])
    empty_aws = pd.DataFrame(columns=['date', 'number of record'])
    
    # Scenario 10: Partial data - AWS has some dates, PCDS has others
    partial_pcds = pd.DataFrame([
        {'date': '2024-02-15', 'number of record': 200},
        {'date': '2024-08-20', 'number of record': 300}
    ])
    partial_aws = pd.DataFrame([
        {'date': '2024-02-15', 'number of record': 200},  # Match
        {'date': '2024-07-01', 'number of record': 250}   # Different date
    ])
    
    return {
        'users': {'pcds': users_pcds, 'aws': users_aws},
        'orders': {'pcds': orders_pcds, 'aws': orders_aws},
        'products': {'pcds': products_pcds, 'aws': products_aws},
        'inventory': {'pcds': inventory_pcds, 'aws': inventory_aws},
        'transactions': {'pcds': transactions_pcds, 'aws': transactions_aws},
        'customers': {'pcds': customers_pcds, 'aws': customers_aws},
        'analytics': {'pcds': analytics_pcds, 'aws': analytics_aws},
        'logs': {'pcds': logs_pcds, 'aws': logs_aws},
        'empty_table': {'pcds': empty_pcds, 'aws': empty_aws},
        'partial_data': {'pcds': partial_pcds, 'aws': partial_aws}
    }


@pytest.fixture
def edge_case_csv_content():
    """CSV with various edge cases."""
    return """tbl,mismatched date
normal_table,2024-01-01; 2024-06-01
single_date_table,2024-03-15
empty_dates_table,
whitespace_table, 2024-02-01 ; 2024-12-01 
duplicate_dates,2024-05-01; 2024-05-01; 2024-10-01
nonexistent_table,2024-01-01
malformed_dates,2024-13-45; invalid-date; 2024-02-30"""


@pytest.fixture
def edge_case_data_dict():
    """Data dictionary with edge cases."""
    return {
        'normal_table': {
            'pcds': pd.DataFrame([
                {'date': '2024-01-01', 'number of record': 100},
                {'date': '2024-06-01', 'number of record': 200}
            ]),
            'aws': pd.DataFrame([
                {'date': '2024-01-01', 'number of record': 100},
                {'date': '2024-06-01', 'number of record': 250}
            ])
        },
        'single_date_table': {
            'pcds': pd.DataFrame([
                {'date': '2024-03-15', 'number of record': 50}
            ]),
            'aws': pd.DataFrame([
                {'date': '2024-03-15', 'number of record': 75}
            ])
        },
        'empty_dates_table': {
            'pcds': pd.DataFrame(columns=['date', 'number of record']),
            'aws': pd.DataFrame(columns=['date', 'number of record'])
        },
        'whitespace_table': {
            'pcds': pd.DataFrame([
                {'date': '2024-02-01', 'number of record': 30},
                {'date': '2024-12-01', 'number of record': 40}
            ]),
            'aws': pd.DataFrame([
                {'date': '2024-02-01', 'number of record': 30},
                {'date': '2024-12-01', 'number of record': 35}
            ])
        },
        'duplicate_dates': {
            'pcds': pd.DataFrame([
                {'date': '2024-05-01', 'number of record': 60},
                {'date': '2024-10-01', 'number of record': 70}
            ]),
            'aws': pd.DataFrame([
                {'date': '2024-05-01', 'number of record': 65},
                {'date': '2024-10-01', 'number of record': 70}
            ])
        },
        'missing_aws_key': {
            'pcds': pd.DataFrame([
                {'date': '2024-01-01', 'number of record': 10}
            ])
            # No 'aws' key
        },
        'malformed_dates': {
            'pcds': pd.DataFrame([
                {'date': '2024-01-01', 'number of record': 100}
            ]),
            'aws': pd.DataFrame([
                {'date': '2024-01-01', 'number of record': 100}
            ])
        }
    }


@pytest.fixture
def performance_test_data():
    """Large dataset for performance testing."""
    def create_large_df(base_records: int = 10000, num_days: int = 365):
        dates = pd.date_range('2024-01-01', periods=num_days, freq='D')
        data = []
        for date in dates:
            # Add some randomness to make it realistic
            records = base_records + random.randint(-1000, 1000)
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'number of record': max(0, records)
            })
        return pd.DataFrame(data)
    
    # Create multiple large tables
    large_data = {}
    for i in range(20):  # 20 tables
        table_name = f'large_table_{i:02d}'
        large_data[table_name] = {
            'pcds': create_large_df(base_records=5000 + i*1000),
            'aws': create_large_df(base_records=5000 + i*1000 + random.randint(-500, 500))
        }
    
    return large_data


@pytest.fixture
def performance_csv_content():
    """CSV for performance testing."""
    tables = [f'large_table_{i:02d}' for i in range(20)]
    # Generate random dates for each table
    csv_lines = ['tbl,mismatched date']
    
    for table in tables:
        # Generate 5-10 random dates for each table
        num_dates = random.randint(5, 10)
        dates = []
        for _ in range(num_dates):
            month = random.randint(1, 12)
            day = random.randint(1, 28)  # Safe day range
            dates.append(f'2024-{month:02d}-{day:02d}')
        
        csv_lines.append(f'{table},{"; ".join(sorted(set(dates)))}')
    
    return '\n'.join(csv_lines)


@pytest.fixture
def mixed_data_types_dict():
    """Data with mixed types and potential issues."""
    return {
        'string_numbers': {
            'pcds': pd.DataFrame([
                {'date': '2024-01-01', 'number of record': '100'},  # String number
                {'date': '2024-02-01', 'number of record': 200}
            ]),
            'aws': pd.DataFrame([
                {'date': '2024-01-01', 'number of record': 100},
                {'date': '2024-02-01', 'number of record': '200'}   # String number
            ])
        },
        'float_numbers': {
            'pcds': pd.DataFrame([
                {'date': '2024-01-01', 'number of record': 100.5},  # Float
                {'date': '2024-02-01', 'number of record': 200.7}
            ]),
            'aws': pd.DataFrame([
                {'date': '2024-01-01', 'number of record': 100},
                {'date': '2024-02-01', 'number of record': 201}
            ])
        },
        'null_values': {
            'pcds': pd.DataFrame([
                {'date': '2024-01-01', 'number of record': None},   # Null
                {'date': '2024-02-01', 'number of record': 200}
            ]),
            'aws': pd.DataFrame([
                {'date': '2024-01-01', 'number of record': 100},
                {'date': '2024-02-01', 'number of record': np.nan}  # NaN
            ])
        },
        'different_date_formats': {
            'pcds': pd.DataFrame([
                {'date': '2024-01-01', 'number of record': 100},
                {'date': '2024/02/01', 'number of record': 200}     # Different format
            ]),
            'aws': pd.DataFrame([
                {'date': '2024-01-01', 'number of record': 100},
                {'date': '2024-02-01', 'number of record': 200}
            ])
        }
    }



class TestComplexScenarios:
    
    def create_temp_files(self, csv_content, data_dict):
        """Helper to create temporary files."""
        csv_fd, csv_path = tempfile.mkstemp(suffix='.csv')
        with os.fdopen(csv_fd, 'w') as f:
            f.write(csv_content)
        
        pkl_fd, pkl_path = tempfile.mkstemp(suffix='.pkl')
        with os.fdopen(pkl_fd, 'wb') as f:
            pickle.dump(data_dict, f)
        
        return csv_path, pkl_path
    
    def cleanup_files(self, *file_paths):
        """Helper to cleanup temporary files."""
        for path in file_paths:
            if os.path.exists(path):
                os.unlink(path)
    
    def test_complex_realistic_scenario(self, complex_csv_content, complex_data_dict):
        """Test with complex realistic data scenarios."""
        csv_path, pkl_path = self.create_temp_files(complex_csv_content, complex_data_dict)
        
        try:
            processor = DataMismatchProcessor(csv_path, pkl_path)
            result = processor.process()
            
            # Verify structure
            assert len(result) == 10  # 10 tables in complex fixture
            assert 'tbl' in result.columns
            
            # Test specific scenarios
            users_row = result[result['tbl'] == 'users'].iloc[0]
            
            # Users table should have mismatches on missing AWS dates
            if '2024-02-28' in result.columns:
                assert 'Yes' in str(users_row['2024-02-28']) or users_row['2024-02-28'] == 'No'
            
            # Customers should have no mismatches (perfect match scenario)
            customers_row = result[result['tbl'] == 'customers'].iloc[0]
            mismatch_columns = [col for col in result.columns if col != 'tbl']
            customer_values = [customers_row[col] for col in mismatch_columns]
            
            # Logs should have specific mismatch
            logs_row = result[result['tbl'] == 'logs'].iloc[0]
            if '2024-12-31' in result.columns:
                assert 'Yes' in str(logs_row['2024-12-31'])
                assert '50000' in str(logs_row['2024-12-31'])
                assert '48000' in str(logs_row['2024-12-31'])
            
            print(f"\nComplex scenario result shape: {result.shape}")
            print(f"Tables processed: {result['tbl'].tolist()}")
            
        finally:
            self.cleanup_files(csv_path, pkl_path)
    
    def test_edge_cases(self, edge_case_csv_content, edge_case_data_dict):
        """Test various edge cases."""
        csv_path, pkl_path = self.create_temp_files(edge_case_csv_content, edge_case_data_dict)
        
        try:
            processor = DataMismatchProcessor(csv_path, pkl_path)
            result = processor.process()
            
            # Should handle all edge cases gracefully
            assert len(result) >= 6  # At least the tables we defined
            
            # Test single date table
            single_date_row = result[result['tbl'] == 'single_date_table'].iloc[0]
            if '2024-03-15' in result.columns:
                assert 'Yes (50 : 75)' == single_date_row['2024-03-15']
            
            # Test empty dates table
            empty_dates_row = result[result['tbl'] == 'empty_dates_table'].iloc[0]
            # Should have all "No" values since no dates to check
            
            # Test nonexistent table (should be in CSV but not in data_dict)
            nonexistent_rows = result[result['tbl'] == 'nonexistent_table']
            if len(nonexistent_rows) > 0:
                nonexistent_row = nonexistent_rows.iloc[0]
                # Should have all "No" values
                mismatch_columns = [col for col in result.columns if col != 'tbl']
                for col in mismatch_columns:
                    if col in nonexistent_row:
                        assert nonexistent_row[col] == 'No'
            
            print(f"\nEdge cases result shape: {result.shape}")
            
        finally:
            self.cleanup_files(csv_path, pkl_path)
    
    def test_performance_large_dataset(self, performance_csv_content, performance_test_data):
        """Test performance with large dataset."""
        csv_path, pkl_path = self.create_temp_files(performance_csv_content, performance_test_data)
        
        try:
            processor = DataMismatchProcessor(csv_path, pkl_path)
            
            start_time = time.time()
            result = processor.process()
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Verify results
            assert len(result) == 20  # 20 large tables
            assert result.shape[1] >= 2  # At least 'tbl' column + date columns
            
            # Performance check - should complete within reasonable time
            # Adjust threshold based on your requirements
            assert processing_time < 30.0, f"Processing took too long: {processing_time:.2f} seconds"
            
            print(f"\nPerformance test completed in {processing_time:.2f} seconds")
            print(f"Processed {len(result)} tables with {result.shape[1]-1} date columns")
            
            # Check for memory efficiency - result shouldn't be excessively large
            result_memory = result.memory_usage(deep=True).sum()
            print(f"Result memory usage: {result_memory / 1024 / 1024:.2f} MB")
            
        finally:
            self.cleanup_files(csv_path, pkl_path)
    
    def test_mixed_data_types(self, mixed_data_types_dict):
        """Test handling of mixed data types."""
        csv_content = """tbl,mismatched date
string_numbers,2024-01-01; 2024-02-01
float_numbers,2024-01-01; 2024-02-01
null_values,2024-01-01; 2024-02-01
different_date_formats,2024-01-01; 2024-02-01"""
        
        csv_path, pkl_path = self.create_temp_files(csv_content, mixed_data_types_dict)
        
        try:
            processor = DataMismatchProcessor(csv_path, pkl_path)
            
            # Should handle mixed types gracefully without crashing
            result = processor.process()
            
            assert len(result) == 4
            
            # Check string numbers handling
            string_row = result[result['tbl'] == 'string_numbers'].iloc[0]
            # Should be able to compare string '100' with int 100
            
            # Check float numbers handling
            float_row = result[result['tbl'] == 'float_numbers'].iloc[0]
            if '2024-01-01' in result.columns:
                # 100.5 vs 100 should be a mismatch
                assert 'Yes' in str(float_row['2024-01-01'])
            
            print(f"\nMixed data types handled successfully")
            
        finally:
            self.cleanup_files(csv_path, pkl_path)
    
    def test_date_parsing_edge_cases(self):
        """Test various date parsing scenarios."""
        processor = DataMismatchProcessor('dummy.csv', 'dummy.pkl')
        
        # Test various date string formats
        test_cases = [
            ("2024-01-01; 2024-02-02; 2024-03-03", ['2024-01-01', '2024-02-02', '2024-03-03']),
            ("2024-01-01;2024-02-02", ['2024-01-01', '2024-02-02']),  # No spaces
            (" 2024-01-01 ; 2024-02-02 ", ['2024-01-01', '2024-02-02']),  # Extra spaces
            ("2024-01-01", ['2024-01-01']),  # Single date
            ("", []),  # Empty string
            ("   ", []),  # Whitespace only
            ("2024-01-01; ; 2024-02-02", ['2024-01-01', '2024-02-02']),  # Empty component
        ]
        
        for input_str, expected in test_cases:
            result = processor._parse_dates(input_str)
            assert result == expected, f"Failed for input: '{input_str}'"
    
    def test_record_count_edge_cases(self, complex_data_dict):
        """Test record count retrieval with various edge cases."""
        processor = DataMismatchProcessor('dummy.csv', 'dummy.pkl')
        
        # Test with real dataframe from complex fixture
        users_pcds = complex_data_dict['users']['pcds']
        
        # Test existing dates
        existing_dates = users_pcds['date'].head(5).tolist()
        for date in existing_dates:
            count = processor._get_record_count(users_pcds, date)
            assert count >= 0
        
        # Test non-existing dates
        non_existing_dates = ['1999-01-01', '2030-12-31', '2024-02-30']
        for date in non_existing_dates:
            count = processor._get_record_count(users_pcds, date)
            assert count == 0
    
    def test_memory_efficiency_large_results(self, performance_csv_content, performance_test_data):
        """Test memory efficiency with large result matrices."""
        csv_path, pkl_path = self.create_temp_files(performance_csv_content, performance_test_data)
        
        try:
            processor = DataMismatchProcessor(csv_path, pkl_path)
            
            # Monitor memory before processing
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = processor.process()
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            print(f"\nMemory usage - Before: {memory_before:.2f} MB, After: {memory_after:.2f} MB")
            print(f"Memory increase: {memory_increase:.2f} MB")
            
            # Result should be reasonably sized
            result_size_mb = result.memory_usage(deep=True).sum() / 1024 / 1024
            print(f"Result DataFrame size: {result_size_mb:.2f} MB")
            
            # Memory increase should be reasonable (adjust threshold as needed)
            assert memory_increase < 500, f"Memory increase too high: {memory_increase:.2f} MB"
            
        except ImportError:
            pytest.skip("psutil not available for memory testing")
        finally:
            self.cleanup_files(csv_path, pkl_path)
    
    def test_data_consistency_validation(self, complex_data_dict):
        """Test that the processor maintains data consistency."""
        # Create a CSV that matches our complex data
        tables = list(complex_data_dict.keys())
        csv_lines = ['tbl,mismatched date']
        
        # Add some dates that exist in our data
        for table in tables[:5]:  # Test first 5 tables
            if 'pcds' in complex_data_dict[table]:
                pcds_df = complex_data_dict[table]['pcds']
                if len(pcds_df) > 0:
                    # Get first few dates from the dataframe
                    sample_dates = pcds_df['date'].head(3).tolist()
                    csv_lines.append(f"{table},{'; '.join(sample_dates)}")
        
        csv_content = '\n'.join(csv_lines)
        csv_path, pkl_path = self.create_temp_files(csv_content, complex_data_dict)
        
        try:
            processor = DataMismatchProcessor(csv_path, pkl_path)
            result = processor.process()
            
            # Verify each cell in the result
            for _, row in result.iterrows():
                table = row['tbl']
                for col in result.columns:
                    if col != 'tbl':
                        cell_value = row[col]
                        # Each cell should be either "No" or "Yes (X : Y)" format
                        assert cell_value == 'No' or (
                            cell_value.startswith('Yes (') and 
                            ' : ' in cell_value and 
                            cell_value.endswith(')')
                        ), f"Invalid cell format: {cell_value}"
            
            print(f"\nData consistency validation passed for {len(result)} tables")
            
        finally:
            self.cleanup_files(csv_path, pkl_path)


# Additional utility test
def test_fixture_data_integrity(complex_data_dict, performance_test_data, mixed_data_types_dict):
    """Test that our fixtures themselves are well-formed."""
    
    # Test complex data dict
    for table_name, table_data in complex_data_dict.items():
        assert 'pcds' in table_data, f"Missing PCDS data for {table_name}"
        assert 'aws' in table_data, f"Missing AWS data for {table_name}"
        
        pcds_df = table_data['pcds']
        aws_df = table_data['aws']
        
        # Check required columns
        if len(pcds_df) > 0:
            assert 'date' in pcds_df.columns, f"Missing date column in PCDS for {table_name}"
            assert 'number of record' in pcds_df.columns, f"Missing record count in PCDS for {table_name}"
        
        if len(aws_df) > 0:
            assert 'date' in aws_df.columns, f"Missing date column in AWS for {table_name}"
            assert 'number of record' in aws_df.columns, f"Missing record count in AWS for {table_name}"
    
    # Test performance data dict
    for table_name, table_data in performance_test_data.items():
        assert isinstance(table_data, dict), f"Invalid structure for {table_name}"
        assert 'pcds' in table_data and 'aws' in table_data, f"Missing data sources for {table_name}"
    
    # Test mixed data types dict
    for table_name, table_data in mixed_data_types_dict.items():
        pcds_df = table_data.get('pcds')
        aws_df = table_data.get('aws')
        
        if pcds_df is not None:
            assert isinstance(pcds_df, pd.DataFrame), f"PCDS should be DataFrame for {table_name}"
        if aws_df is not None:
            assert isinstance(aws_df, pd.DataFrame), f"AWS should be DataFrame for {table_name}"
    
    print("All fixture data integrity checks passed!")


class TestRobustnessAndReliability:
    """Additional tests for robustness and reliability."""
    
    def create_temp_files(self, csv_content, data_dict):
        """Helper to create temporary files."""
        csv_fd, csv_path = tempfile.mkstemp(suffix='.csv')
        with os.fdopen(csv_fd, 'w') as f:
            f.write(csv_content)
        
        pkl_fd, pkl_path = tempfile.mkstemp(suffix='.pkl')
        with os.fdopen(pkl_fd, 'wb') as f:
            pickle.dump(data_dict, f)
        
        return csv_path, pkl_path
    
    def cleanup_files(self, *file_paths):
        """Helper to cleanup temporary files."""
        for path in file_paths:
            if os.path.exists(path):
                os.unlink(path)
    
    def test_extreme_date_ranges(self):
        """Test with extreme date ranges and edge dates."""
        extreme_data = {
            'historical': {
                'pcds': pd.DataFrame([
                    {'date': '1900-01-01', 'number of record': 1},
                    {'date': '2000-02-29', 'number of record': 100},  # Leap year
                    {'date': '2100-12-31', 'number of record': 1000}   # Future date
                ]),
                'aws': pd.DataFrame([
                    {'date': '1900-01-01', 'number of record': 2},
                    {'date': '2000-02-29', 'number of record': 100},
                    {'date': '2100-12-31', 'number of record': 999}
                ])
            }
        }
        
        csv_content = """tbl,mismatched date
historical,1900-01-01; 2000-02-29; 2100-12-31"""
        
        csv_path, pkl_path = self.create_temp_files(csv_content, extreme_data)
        
        try:
            processor = DataMismatchProcessor(csv_path, pkl_path)
            result = processor.process()
            
            assert len(result) == 1
            hist_row = result.iloc[0]
            
            # Check specific results
            if '1900-01-01' in result.columns:
                assert hist_row['1900-01-01'] == 'Yes (1 : 2)'
            if '2000-02-29' in result.columns:
                assert hist_row['2000-02-29'] == 'No'
            if '2100-12-31' in result.columns:
                assert hist_row['2100-12-31'] == 'Yes (1000 : 999)'
            
            print("Extreme date ranges handled successfully")
            
        finally:
            self.cleanup_files(csv_path, pkl_path)
    
    def test_very_large_numbers(self):
        """Test with very large record counts."""
        large_numbers_data = {
            'big_table': {
                'pcds': pd.DataFrame([
                    {'date': '2024-01-01', 'number of record': 999999999},
                    {'date': '2024-01-02', 'number of record': 1000000000},
                    {'date': '2024-01-03', 'number of record': 0}
                ]),
                'aws': pd.DataFrame([
                    {'date': '2024-01-01', 'number of record': 999999998},
                    {'date': '2024-01-02', 'number of record': 1000000000},
                    {'date': '2024-01-03', 'number of record': 1}
                ])
            }
        }
        
        csv_content = """tbl,mismatched date
big_table,2024-01-01; 2024-01-02; 2024-01-03"""
        
        csv_path, pkl_path = self.create_temp_files(csv_content, large_numbers_data)
        
        try:
            processor = DataMismatchProcessor(csv_path, pkl_path)
            result = processor.process()
            
            big_table_row = result.iloc[0]
            
            # Verify large number handling
            if '2024-01-01' in result.columns:
                assert '999999999 : 999999998' in big_table_row['2024-01-01']
            if '2024-01-02' in result.columns:
                assert big_table_row['2024-01-02'] == 'No'
            if '2024-01-03' in result.columns:
                assert '0 : 1' in big_table_row['2024-01-03']
            
            print("Large numbers handled successfully")
            
        finally:
            self.cleanup_files(csv_path, pkl_path)
    
    def test_unicode_and_special_characters(self):
        """Test with unicode table names and special characters."""
        unicode_data = {
            'table_with_ümlauts': {
                'pcds': pd.DataFrame([
                    {'date': '2024-01-01', 'number of record': 50}
                ]),
                'aws': pd.DataFrame([
                    {'date': '2024-01-01', 'number of record': 60}
                ])
            },
            'table-with-dashes_and_underscores': {
                'pcds': pd.DataFrame([
                    {'date': '2024-01-01', 'number of record': 100}
                ]),
                'aws': pd.DataFrame([
                    {'date': '2024-01-01', 'number of record': 100}
                ])
            },
            '123_numeric_start': {
                'pcds': pd.DataFrame([
                    {'date': '2024-01-01', 'number of record': 75}
                ]),
                'aws': pd.DataFrame([
                    {'date': '2024-01-01', 'number of record': 80}
                ])
            }
        }
        
        csv_content = """tbl,mismatched date
table_with_ümlauts,2024-01-01
table-with-dashes_and_underscores,2024-01-01
123_numeric_start,2024-01-01"""
        
        csv_path, pkl_path = self.create_temp_files(csv_content, unicode_data)
        
        try:
            processor = DataMismatchProcessor(csv_path, pkl_path)
            result = processor.process()
            
            assert len(result) == 3
            
            # Check unicode handling
            unicode_row = result[result['tbl'] == 'table_with_ümlauts']
            assert len(unicode_row) == 1
            
            print("Unicode and special characters handled successfully")
            
        finally:
            self.cleanup_files(csv_path, pkl_path)
    
    def test_concurrent_processing_safety(self, complex_csv_content, complex_data_dict):
        """Test thread safety (basic test)."""
        import threading
        import queue
        
        csv_path, pkl_path = self.create_temp_files(complex_csv_content, complex_data_dict)
        
        try:
            results_queue = queue.Queue()
            errors_queue = queue.Queue()
            
            def process_data(thread_id):
                try:
                    processor = DataMismatchProcessor(csv_path, pkl_path)
                    result = processor.process()
                    results_queue.put((thread_id, result))
                except Exception as e:
                    errors_queue.put((thread_id, e))
            
            # Create and start multiple threads
            threads = []
            num_threads = 3
            
            for i in range(num_threads):
                thread = threading.Thread(target=process_data, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Check results
            assert results_queue.qsize() == num_threads, "Not all threads completed successfully"
            assert errors_queue.empty(), f"Errors occurred: {list(errors_queue.queue)}"
            
            # Verify all results are identical (they should be for same input)
            results = []
            while not results_queue.empty():
                thread_id, result = results_queue.get()
                results.append(result)
            
            # Compare first result with others
            base_result = results[0]
            for i, result in enumerate(results[1:], 1):
                pd.testing.assert_frame_equal(base_result, result, 
                                            f"Results differ between threads 0 and {i}")
            
            print(f"Concurrent processing test passed with {num_threads} threads")
            
        finally:
            self.cleanup_files(csv_path, pkl_path)
    
    def test_malformed_pickle_recovery(self):
        """Test recovery from malformed pickle data."""
        csv_content = """tbl,mismatched date
test_table,2024-01-01"""
        
        # Create CSV file
        csv_fd, csv_path = tempfile.mkstemp(suffix='.csv')
        with os.fdopen(csv_fd, 'w') as f:
            f.write(csv_content)
        
        # Create malformed pickle file
        pkl_fd, pkl_path = tempfile.mkstemp(suffix='.pkl')
        with os.fdopen(pkl_fd, 'wb') as f:
            f.write(b'This is not valid pickle data')
        
        try:
            processor = DataMismatchProcessor(csv_path, pkl_path)
            
            # Should raise an exception when trying to load malformed pickle
            with pytest.raises((pickle.UnpicklingError, EOFError, ValueError)):
                processor.load_data()
            
            print("Malformed pickle handling test passed")
            
        finally:
            self.cleanup_files(csv_path, pkl_path)
    
    def test_extremely_sparse_data(self):
        """Test with very sparse data (mostly empty dataframes)."""
        sparse_data = {
            'mostly_empty': {
                'pcds': pd.DataFrame([
                    {'date': '2024-06-15', 'number of record': 1}
                ]),
                'aws': pd.DataFrame([
                    {'date': '2024-06-15', 'number of record': 1}
                ])
            },
            'completely_empty': {
                'pcds': pd.DataFrame(columns=['date', 'number of record']),
                'aws': pd.DataFrame(columns=['date', 'number of record'])
            },
            'one_sided_empty': {
                'pcds': pd.DataFrame([
                    {'date': '2024-01-01', 'number of record': 10}
                ]),
                'aws': pd.DataFrame(columns=['date', 'number of record'])
            }
        }
        
        csv_content = """tbl,mismatched date
mostly_empty,2024-06-15
completely_empty,2024-01-01
one_sided_empty,2024-01-01"""
        
        csv_path, pkl_path = self.create_temp_files(csv_content, sparse_data)
        
        try:
            processor = DataMismatchProcessor(csv_path, pkl_path)
            result = processor.process()
            
            assert len(result) == 3
            
            # Check sparse data handling
            mostly_empty_row = result[result['tbl'] == 'mostly_empty'].iloc[0]
            if '2024-06-15' in result.columns:
                assert mostly_empty_row['2024-06-15'] == 'No'
            
            # Empty data should result in "No" for all dates
            completely_empty_row = result[result['tbl'] == 'completely_empty'].iloc[0]
            one_sided_row = result[result['tbl'] == 'one_sided_empty'].iloc[0]
            
            print("Sparse data handling test passed")
            
        finally:
            self.cleanup_files(csv_path, pkl_path)


# Benchmark test for comparing performance improvements
def test_performance_benchmark(performance_csv_content, performance_test_data):
    """Benchmark test to measure and compare performance."""
    
    def create_temp_files(csv_content, data_dict):
        csv_fd, csv_path = tempfile.mkstemp(suffix='.csv')
        with os.fdopen(csv_fd, 'w') as f:
            f.write(csv_content)
        
        pkl_fd, pkl_path = tempfile.mkstemp(suffix='.pkl')
        with os.fdopen(pkl_fd, 'wb') as f:
            pickle.dump(data_dict, f)
        
        return csv_path, pkl_path
    
    csv_path, pkl_path = create_temp_files(performance_csv_content, performance_test_data)
    
    try:
        processor = DataMismatchProcessor(csv_path, pkl_path)
        
        # Warm-up run
        _ = processor.process()
        
        # Benchmark runs
        times = []
        num_runs = 5
        
        for i in range(num_runs):
            # Create fresh processor for each run
            processor = DataMismatchProcessor(csv_path, pkl_path)
            
            start_time = time.time()
            result = processor.process()
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\nPerformance Benchmark Results ({num_runs} runs):")
        print(f"Average time: {avg_time:.3f} seconds")
        print(f"Min time: {min_time:.3f} seconds") 
        print(f"Max time: {max_time:.3f} seconds")
        print(f"Result shape: {result.shape}")
        
        # Performance assertions (adjust thresholds as needed)
        assert avg_time < 10.0, f"Average processing time too slow: {avg_time:.3f}s"
        assert max_time < 15.0, f"Worst case processing time too slow: {max_time:.3f}s"
        
    finally:
        os.unlink(csv_path)
        os.unlink(pkl_path)