import pytest
import pandas as pd
import numpy as np
from proc_sql import ProcSQL  # Assuming the class is saved as proc_sql.py


@pytest.fixture
def sample_employees_data():
    """Sample employees dataset for testing."""
    return pd.DataFrame({
        'employee_id': [1, 2, 3, 4, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'HR'],
        'salary': [75000, 55000, 80000, 65000, 70000, 50000],
        'age': [28, 35, 42, 31, 26, 45],
        'hire_date': pd.to_datetime(['2020-01-15', '2019-03-22', '2018-07-10', 
                                   '2021-02-28', '2022-05-12', '2017-11-03'])
    })


@pytest.fixture
def sample_departments_data():
    """Sample departments dataset for testing."""
    return pd.DataFrame({
        'department': ['IT', 'HR', 'Finance', 'Marketing'],
        'manager': ['John', 'Sarah', 'Mike', 'Lisa'],
        'budget': [500000, 200000, 300000, 250000],
        'location': ['Building A', 'Building B', 'Building A', 'Building C']
    })


@pytest.fixture
def sample_projects_data():
    """Sample projects dataset for testing."""
    return pd.DataFrame({
        'project_id': [101, 102, 103, 104],
        'employee_id': [1, 2, 1, 3],
        'project_name': ['Web App', 'HR System', 'Database', 'Analytics'],
        'hours': [120, 80, 150, 200]
    })


@pytest.fixture
def proc_sql_instance(sample_employees_data, sample_departments_data, sample_projects_data):
    """ProcSQL instance with sample data loaded."""
    instance = ProcSQL()
    instance.add_dataset('employees', sample_employees_data)
    instance.add_dataset('departments', sample_departments_data)
    instance.add_dataset('projects', sample_projects_data)
    return instance


class TestProcSQLInitialization:
    """Test ProcSQL initialization and basic setup."""
    
    def test_init_empty(self):
        """Test initialization without datasets."""
        proc_sql = ProcSQL()
        assert proc_sql.datasets == {}
        assert proc_sql.query_history == []
    
    def test_init_with_datasets(self, sample_employees_data):
        """Test initialization with datasets."""
        datasets = {'employees': sample_employees_data}
        proc_sql = ProcSQL(datasets)
        assert 'employees' in proc_sql.datasets
        assert len(proc_sql.datasets['employees']) == 6
    
    def test_add_dataset(self, sample_employees_data):
        """Test adding dataset to instance."""
        proc_sql = ProcSQL()
        proc_sql.add_dataset('test_data', sample_employees_data)
        assert 'test_data' in proc_sql.datasets
        assert len(proc_sql.datasets['test_data']) == 6


class TestProcSQLSelect:
    """Test SELECT functionality."""
    
    def test_select_all_columns(self, proc_sql_instance):
        """Test selecting all columns."""
        result = proc_sql_instance.select('employees')
        assert len(result) == 6
        assert len(result.columns) == 6
    
    @pytest.mark.parametrize("columns,expected_cols", [
        (['name', 'salary'], 2),
        (['employee_id'], 1),
        (['name', 'department', 'salary'], 3),
    ])
    def test_select_specific_columns(self, proc_sql_instance, columns, expected_cols):
        """Test selecting specific columns."""
        result = proc_sql_instance.select('employees', columns=columns)
        assert len(result.columns) == expected_cols
        assert all(col in result.columns for col in columns)
    
    @pytest.mark.parametrize("where_clause,expected_rows", [
        ("salary > 60000", 4),
        ("department == 'IT'", 3),
        ("age < 30", 2),
        ("salary >= 70000", 3),
    ])
    def test_select_with_where(self, proc_sql_instance, where_clause, expected_rows):
        """Test SELECT with WHERE clauses."""
        result = proc_sql_instance.select('employees', where=where_clause)
        assert len(result) == expected_rows
    
    def test_select_with_order_by(self, proc_sql_instance):
        """Test SELECT with ORDER BY."""
        result = proc_sql_instance.select('employees', order_by=['salary DESC'])
        assert result.iloc[0]['salary'] == 80000  # Highest salary first
        assert result.iloc[-1]['salary'] == 50000  # Lowest salary last
    
    def test_select_with_limit(self, proc_sql_instance):
        """Test SELECT with LIMIT."""
        result = proc_sql_instance.select('employees', limit=3)
        assert len(result) == 3
    
    def test_select_nonexistent_dataset(self, proc_sql_instance):
        """Test selecting from non-existent dataset."""
        with pytest.raises(ValueError, match="Dataset 'nonexistent' not found"):
            proc_sql_instance.select('nonexistent')
    
    def test_select_nonexistent_column(self, proc_sql_instance):
        """Test selecting non-existent column."""
        with pytest.raises(ValueError, match="Columns not found"):
            proc_sql_instance.select('employees', columns=['nonexistent_col'])


class TestProcSQLGroupBy:
    """Test GROUP BY functionality."""
    
    def test_group_by_single_column(self, proc_sql_instance):
        """Test GROUP BY with single column."""
        result = proc_sql_instance.group_by(
            'employees', 
            group_cols=['department'], 
            agg_funcs={'salary': ['mean', 'count']}
        )
        assert len(result) == 3  # IT, HR, Finance
        assert 'salary_mean' in result.columns
        assert 'salary_count' in result.columns
    
    @pytest.mark.parametrize("group_cols,agg_funcs,expected_rows", [
        (['department'], {'salary': 'sum'}, 3),
        (['department'], {'age': ['min', 'max']}, 3),
        (['department'], {'employee_id': 'count'}, 3),
    ])
    def test_group_by_various_aggregations(self, proc_sql_instance, group_cols, agg_funcs, expected_rows):
        """Test GROUP BY with various aggregation functions."""
        result = proc_sql_instance.group_by('employees', group_cols, agg_funcs)
        assert len(result) == expected_rows
    
    def test_group_by_with_having(self, proc_sql_instance):
        """Test GROUP BY with HAVING clause."""
        result = proc_sql_instance.group_by(
            'employees',
            group_cols=['department'],
            agg_funcs={'salary': 'mean'},
            having='salary_mean > 60000'
        )
        assert len(result) == 2  # Only IT and Finance departments


class TestProcSQLJoins:
    """Test JOIN functionality."""
    
    @pytest.mark.parametrize("join_type,expected_rows", [
        ('inner', 3),  # Only departments that exist in both tables
        ('left', 6),   # All employees
        ('right', 4),  # All departments
        ('outer', 7),  # All records from both tables
    ])
    def test_different_join_types(self, proc_sql_instance, join_type, expected_rows):
        """Test different types of joins."""
        result = proc_sql_instance.join(
            'employees', 'departments', 
            join_type=join_type, 
            on='department'
        )
        assert len(result) == expected_rows
    
    def test_join_with_different_column_names(self, proc_sql_instance):
        """Test join with different column names."""
        result = proc_sql_instance.join(
            'employees', 'projects',
            join_type='inner',
            left_on='employee_id',
            right_on='employee_id'
        )
        assert len(result) == 4  # 4 project assignments
    
    def test_join_nonexistent_dataset(self, proc_sql_instance):
        """Test join with non-existent dataset."""
        with pytest.raises(ValueError, match="Dataset 'nonexistent' not found"):
            proc_sql_instance.join('employees', 'nonexistent', on='id')


class TestProcSQLUnion:
    """Test UNION functionality."""
    
    def test_union_datasets(self, proc_sql_instance):
        """Test UNION of datasets."""
        # Create a second employees dataset
        additional_employees = pd.DataFrame({
            'employee_id': [7, 8],
            'name': ['Grace', 'Henry'],
            'department': ['Marketing', 'IT'],
            'salary': [60000, 75000],
            'age': [29, 33],
            'hire_date': pd.to_datetime(['2023-01-15', '2023-02-20'])
        })
        proc_sql_instance.add_dataset('new_employees', additional_employees)
        
        result = proc_sql_instance.union(['employees', 'new_employees'])
        assert len(result) == 8  # 6 + 2 new employees
    
    def test_union_all(self, proc_sql_instance):
        """Test UNION ALL (including duplicates)."""
        proc_sql_instance.add_dataset('employees_copy', proc_sql_instance.datasets['employees'])
        
        result = proc_sql_instance.union(['employees', 'employees_copy'], all_records=True)
        assert len(result) == 12  # 6 * 2 (all duplicates included)
    
    def test_union_empty_list(self, proc_sql_instance):
        """Test UNION with empty dataset list."""
        with pytest.raises(ValueError, match="Must specify at least one dataset"):
            proc_sql_instance.union([])


class TestProcSQLStatistics:
    """Test statistical functionality."""
    
    def test_calculate_stats_all_numeric(self, proc_sql_instance):
        """Test calculating stats for all numeric columns."""
        result = proc_sql_instance.calculate_stats('employees')
        assert 'salary' in result.index
        assert 'age' in result.index
        assert 'employee_id' in result.index
        assert 'mean' in result.columns
        assert 'std' in result.columns
    
    def test_calculate_stats_specific_columns(self, proc_sql_instance):
        """Test calculating stats for specific columns."""
        result = proc_sql_instance.calculate_stats('employees', columns=['salary'])
        assert len(result) == 1
        assert 'salary' in result.index
    
    def test_describe_dataset(self, proc_sql_instance):
        """Test dataset description functionality."""
        desc = proc_sql_instance.describe_dataset('employees')
        assert desc['name'] == 'employees'
        assert desc['observations'] == 6
        assert desc['variables'] == 6
        assert 'column_info' in desc


class TestProcSQLUtilities:
    """Test utility functions."""
    
    def test_list_datasets(self, proc_sql_instance):
        """Test listing available datasets."""
        datasets = proc_sql_instance.list_datasets()
        assert 'employees' in datasets
        assert 'departments' in datasets
        assert 'projects' in datasets
        assert len(datasets) == 3
    
    def test_drop_dataset(self, proc_sql_instance):
        """Test dropping a dataset."""
        initial_count = len(proc_sql_instance.list_datasets())
        proc_sql_instance.drop_dataset('projects')
        assert len(proc_sql_instance.list_datasets()) == initial_count - 1
        assert 'projects' not in proc_sql_instance.list_datasets()
    
    def test_drop_nonexistent_dataset(self, proc_sql_instance):
        """Test dropping non-existent dataset (should not raise error)."""
        with pytest.warns(UserWarning):
            proc_sql_instance.drop_dataset('nonexistent')


class TestProcSQLCreateTable:
    """Test CREATE TABLE functionality."""
    
    def test_create_table_from_select(self, proc_sql_instance):
        """Test creating table from SELECT results."""
        result = proc_sql_instance.create_table(
            'high_earners',
            {
                'dataset_name': 'employees',
                'columns': ['name', 'salary'],
                'where': 'salary > 60000'
            }
        )
        assert 'high_earners' in proc_sql_instance.list_datasets()
        assert len(result) == 4
        assert len(result.columns) == 2


class TestProcSQLComplexQueries:
    """Test complex query scenarios."""
    
    def test_complex_where_clause(self, proc_sql_instance):
        """Test complex WHERE clauses with multiple conditions."""
        result = proc_sql_instance.select(
            'employees',
            where="(salary > 60000) & (age < 40)"
        )
        assert len(result) == 3  # Alice, Charlie, Diana
    
    def test_chained_operations(self, proc_sql_instance):
        """Test chaining multiple operations."""
        # First create a filtered dataset
        proc_sql_instance.create_table(
            'it_employees',
            {
                'dataset_name': 'employees',
                'where': "department == 'IT'"
            }
        )
        
        # Then perform aggregation on it
        result = proc_sql_instance.group_by(
            'it_employees',
            group_cols=['department'],
            agg_funcs={'salary': ['mean', 'count']}
        )
        
        assert len(result) == 1
        assert result.iloc[0]['salary_count'] == 3


# Fixtures for error testing
@pytest.fixture
def invalid_data():
    """Sample data that might cause errors."""
    return pd.DataFrame({
        'col1': [1, 2, None, 4],
        'col2': ['a', 'b', 'c', 'd']
    })


class TestProcSQLErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_where_clause(self, proc_sql_instance):
        """Test handling of invalid WHERE clauses."""
        with pytest.raises(ValueError, match="Invalid WHERE clause"):
            proc_sql_instance.select('employees', where="invalid syntax here")
    
    def test_unsupported_aggregation_function(self, proc_sql_instance):
        """Test handling of unsupported aggregation functions."""
        with pytest.raises(ValueError, match="Unsupported aggregation function"):
            proc_sql_instance.group_by(
                'employees',
                group_cols=['department'],
                agg_funcs={'salary': 'unsupported_func'}
            )
    
    @pytest.mark.parametrize("invalid_input", [
        {'left_dataset': 'nonexistent', 'right_dataset': 'employees'},
        {'left_dataset': 'employees', 'right_dataset': 'nonexistent'},
    ])
    def test_join_error_handling(self, proc_sql_instance, invalid_input):
        """Test error handling in join operations."""
        with pytest.raises(ValueError, match="not found"):
            proc_sql_instance.join(**invalid_input, on='id')


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])