"""
Test file for pandas advanced functions
Each test is concise (≤5 lines) and focused on one advanced feature
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestPandas1:
    """Test pandas advanced functionality with concise tests"""
    
    def test_pivot_table(self):
        """Test pivot table aggregation"""
        df = pd.DataFrame({'A': ['foo', 'foo', 'bar'], 'B': [1, 2, 3], 'C': [10, 20, 30]})
        result = df.pivot_table(values='C', index='A', aggfunc='sum')
        assert result.loc['foo', 'C'] == 30
    
    def test_melt_unpivot(self):
        """Test melting DataFrame from wide to long format"""
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        result = df.melt(id_vars=['A'], value_vars=['B', 'C'])
        assert len(result) == 4 and 'variable' in result.columns
    
    def test_groupby_transform(self):
        """Test groupby with transform for broadcasting"""
        df = pd.DataFrame({'group': ['A', 'A', 'B'], 'value': [1, 2, 3]})
        df['pct_of_group'] = df['value'] / df.groupby('group')['value'].transform('sum')
        assert df.loc[0, 'pct_of_group'] == 1/3
    
    def test_window_rolling(self):
        """Test rolling window calculations"""
        df = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
        result = df['values'].rolling(window=3).mean()
        assert result.iloc[2] == 2.0
    
    def test_categorical_data(self):
        """Test categorical data type optimization"""
        df = pd.DataFrame({'category': ['A', 'B', 'A', 'C'] * 1000})
        df['category'] = df['category'].astype('category')
        assert df['category'].dtype.name == 'category'
    
    def test_multi_index(self):
        """Test MultiIndex DataFrame operations"""
        idx = pd.MultiIndex.from_product([['A', 'B'], [1, 2]], names=['letter', 'number'])
        df = pd.DataFrame({'value': [10, 20, 30, 40]}, index=idx)
        assert df.loc[('A', 1), 'value'] == 10
    
    def test_query_method(self):
        """Test DataFrame query with string expressions"""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = df.query('A > 1 and B < 6')
        assert len(result) == 1 and result.iloc[0]['A'] == 2
    
    def test_apply_lambda(self):
        """Test apply with lambda functions"""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        result = df.apply(lambda row: row['x'] * row['y'], axis=1)
        assert result.tolist() == [4, 10, 18]
    
    def test_cross_tabulation(self):
        """Test cross-tabulation (crosstab)"""
        df = pd.DataFrame({'A': ['foo', 'foo', 'bar'], 'B': ['one', 'two', 'one']})
        result = pd.crosstab(df['A'], df['B'])
        assert result.loc['foo', 'one'] == 1
    
    def test_string_methods(self):
        """Test vectorized string operations"""
        df = pd.DataFrame({'text': ['Hello World', 'PANDAS rocks', 'Data Science']})
        result = df['text'].str.lower().str.contains('data')
        assert result.iloc[2] == True
    
    def test_datetime_operations(self):
        """Test datetime indexing and operations"""
        dates = pd.date_range('2023-01-01', periods=3, freq='D')
        df = pd.DataFrame({'value': [1, 2, 3]}, index=dates)
        assert df.loc['2023-01-02', 'value'] == 2
    
    def test_merge_complex(self):
        """Test complex merge operations"""
        df1 = pd.DataFrame({'key': ['A', 'B'], 'value1': [1, 2]})
        df2 = pd.DataFrame({'key': ['A', 'C'], 'value2': [3, 4]})
        result = df1.merge(df2, on='key', how='outer')
        assert len(result) == 3
    
    def test_resampling(self):
        """Test time series resampling"""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        df = pd.DataFrame({'value': range(10)}, index=dates)
        result = df.resample('3D').sum()
        assert len(result) == 4
    
    def test_explode_lists(self):
        """Test exploding list-like columns"""
        df = pd.DataFrame({'A': [1, 2], 'B': [[1, 2], [3, 4]]})
        result = df.explode('B')
        assert len(result) == 4
    
    def test_eval_expressions(self):
        """Test eval for efficient calculations"""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df.eval('C = A + B', inplace=True)
        assert df['C'].tolist() == [5, 7, 9]
    
    def test_pipe_chaining(self):
        """Test method chaining with pipe"""
        df = pd.DataFrame({'A': [1, 2, 3]})
        result = df.pipe(lambda x: x * 2).pipe(lambda x: x + 1)
        assert result['A'].tolist() == [3, 5, 7]
    
    def test_assign_method(self):
        """Test assign for adding columns functionally"""
        df = pd.DataFrame({'A': [1, 2, 3]})
        result = df.assign(B=lambda x: x['A'] * 2, C=10)
        assert result['B'].tolist() == [2, 4, 6] and all(result['C'] == 10)
    
    def test_cut_binning(self):
        """Test cut for binning continuous data"""
        df = pd.DataFrame({'score': [10, 25, 45, 78, 95]})
        df['grade'] = pd.cut(df['score'], bins=[0, 30, 60, 100], labels=['F', 'C', 'A'])
        assert df['grade'].iloc[0] == 'F'
    
    def test_qcut_quantiles(self):
        """Test qcut for quantile-based binning"""
        df = pd.DataFrame({'value': range(100)})
        df['quartile'] = pd.qcut(df['value'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        assert df['quartile'].iloc[10] == 'Q1'
    
    def test_rank_method(self):
        """Test ranking with different methods"""
        df = pd.DataFrame({'score': [85, 90, 85, 95]})
        df['rank'] = df['score'].rank(method='min')
        assert df['rank'].iloc[0] == 1.0
    
    def test_nlargest_nsmallest(self):
        """Test nlargest and nsmallest methods"""
        df = pd.DataFrame({'A': [1, 5, 3, 9, 2]})
        result = df.nlargest(2, 'A')
        assert result['A'].tolist() == [9, 5]
    
    def test_stack_unstack(self):
        """Test stack/unstack for reshaping"""
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        stacked = df.stack()
        assert stacked.loc[(0, 'A')] == 1
    
    def test_expanding_window(self):
        """Test expanding window calculations"""
        df = pd.DataFrame({'value': [1, 2, 3, 4]})
        result = df['value'].expanding().sum()
        assert result.iloc[2] == 6
    
    def test_interpolate_missing(self):
        """Test interpolation for missing values"""
        df = pd.DataFrame({'value': [1, np.nan, 3]})
        result = df['value'].interpolate()
        assert result.iloc[1] == 2.0
    
    def test_factorize_encoding(self):
        """Test factorize for label encoding"""
        df = pd.DataFrame({'category': ['A', 'B', 'A', 'C']})
        codes, uniques = pd.factorize(df['category'])
        assert codes[0] == codes[2] and len(uniques) == 3
    
    def test_where_method(self):
        """Test where method for conditional selection"""
        df = pd.DataFrame({'A': [1, 2, 3, 4]})
        result = df['A'].where(df['A'] > 2, 0)
        assert result.tolist() == [0, 0, 3, 4]
    
    def test_mask_method(self):
        """Test mask method (inverse of where)"""
        df = pd.DataFrame({'A': [1, 2, 3, 4]})
        result = df['A'].mask(df['A'] <= 2, 0)
        assert result.tolist() == [0, 0, 3, 4]
    
    def test_sample_method(self):
        """Test random sampling"""
        df = pd.DataFrame({'A': range(100)})
        result = df.sample(n=5, random_state=42)
        assert len(result) == 5
    
    def test_groupby_agg_multiple(self):
        """Test groupby with multiple aggregations"""
        df = pd.DataFrame({'group': ['A', 'A', 'B'], 'value': [1, 2, 3]})
        result = df.groupby('group')['value'].agg(['sum', 'mean'])
        assert result.loc['A', 'sum'] == 3
    
    def test_transform_zscore(self):
        """Test transform for z-score normalization"""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        df['zscore'] = (df['value'] - df['value'].mean()) / df['value'].std()
        assert abs(df['zscore'].mean()) < 1e-10
    
    def test_memory_usage(self):
        """Test memory usage optimization"""
        df = pd.DataFrame({'int_col': [1, 2, 3]})
        original_memory = df.memory_usage(deep=True).sum()
        df['int_col'] = df['int_col'].astype('int8')
        assert df.memory_usage(deep=True).sum() <= original_memory


if __name__ == "__main__":
    pytest.main([__file__, "-v"])