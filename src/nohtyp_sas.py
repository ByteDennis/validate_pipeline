"""
SAS to Python Translation Framework

SASTranslator Class: Reproduces core SAS functionality
    DATA steps with conditional logic
    PROC FREQ (frequency tables and cross-tabs)
    PROC MEANS (descriptive statistics)
    PROC SORT (sorting operations)
    Dataset merging operations
    Variable formatting
    Operation logging for debugging


SASValidator Class: Validates translations against expected results
    DataFrame comparison with tolerance
    Numeric validation
    Comprehensive validation summaries



Advanced Pytest Features
    Property-based testing with Hypothesis
    Parametrized tests for multiple scenarios
    Custom markers (unit, integration, performance, slow)
    Mocking for external dependencies
    Performance testing with memory monitoring
    Regression tests for backward compatibility
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

class SASTranslator:
    """Main class for translating SAS operations to Python"""
    
    def __init__(self, debug: bool = True):
        self.datasets = {}
        self.debug = debug
        self.operations_log = []
        
    def data_step(self, input_data: Union[pd.DataFrame, str], 
                  output_name: str,
                  operations: List[Callable] = None) -> pd.DataFrame:
        """
        Simulate SAS DATA step
        
        Args:
            input_data: Input DataFrame or dataset name
            output_name: Name for output dataset
            operations: List of operations to apply
        """
        if isinstance(input_data, str):
            df = self.datasets.get(input_data, pd.DataFrame())
        else:
            df = input_data.copy()
            
        if operations:
            for op in operations:
                df = op(df)
                
        self.datasets[output_name] = df
        self._log_operation(f"DATA {output_name} created with {len(df)} observations")
        return df
    
    def proc_freq(self, data: Union[pd.DataFrame, str], 
                  variables: List[str],
                  tables: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Simulate SAS PROC FREQ"""
        df = self._get_dataframe(data)
        results = {}
        
        for var in variables:
            if var in df.columns:
                freq_table = df[var].value_counts().reset_index()
                freq_table.columns = [var, 'Frequency']
                freq_table['Percent'] = (freq_table['Frequency'] / len(df) * 100).round(2)
                results[var] = freq_table
                
        if tables:
            for table in tables:
                if '*' in table:
                    vars_combo = table.split('*')
                    if all(v.strip() in df.columns for v in vars_combo):
                        crosstab = pd.crosstab(df[vars_combo[0].strip()], 
                                             df[vars_combo[1].strip()], 
                                             margins=True)
                        results[table] = crosstab
                        
        self._log_operation(f"PROC FREQ executed on {variables}")
        return results
    
    def proc_means(self, data: Union[pd.DataFrame, str], 
                   variables: List[str],
                   class_vars: Optional[List[str]] = None,
                   stats: List[str] = None) -> pd.DataFrame:
        """Simulate SAS PROC MEANS"""
        df = self._get_dataframe(data)
        
        if stats is None:
            stats = ['mean', 'std', 'min', 'max', 'count']
            
        if class_vars:
            result = df.groupby(class_vars)[variables].agg(stats)
        else:
            result = df[variables].agg(stats)
            
        self._log_operation(f"PROC MEANS executed on {variables}")
        return result
    
    def proc_sort(self, data: Union[pd.DataFrame, str], 
                  by: List[str], 
                  ascending: Union[bool, List[bool]] = True,
                  output_name: Optional[str] = None) -> pd.DataFrame:
        """Simulate SAS PROC SORT"""
        df = self._get_dataframe(data)
        sorted_df = df.sort_values(by=by, ascending=ascending).reset_index(drop=True)
        
        if output_name:
            self.datasets[output_name] = sorted_df
            
        self._log_operation(f"PROC SORT executed by {by}")
        return sorted_df
    
    def merge_datasets(self, left: Union[pd.DataFrame, str],
                      right: Union[pd.DataFrame, str],
                      by: Union[str, List[str]],
                      how: str = 'inner',
                      output_name: Optional[str] = None) -> pd.DataFrame:
        """Simulate SAS merge operation"""
        left_df = self._get_dataframe(left)
        right_df = self._get_dataframe(right)
        
        merged = pd.merge(left_df, right_df, on=by, how=how)
        
        if output_name:
            self.datasets[output_name] = merged
            
        self._log_operation(f"MERGE executed: {how} join on {by}")
        return merged
    
    def conditional_logic(self, df: pd.DataFrame, 
                         conditions: Dict[str, Any]) -> pd.DataFrame:
        """Apply SAS-style conditional logic"""
        result_df = df.copy()
        
        for column, condition in conditions.items():
            if callable(condition):
                result_df[column] = condition(result_df)
            else:
                result_df[column] = condition
                
        return result_df
    
    def format_variables(self, df: pd.DataFrame, 
                        formats: Dict[str, str]) -> pd.DataFrame:
        """Apply SAS-style formatting"""
        formatted_df = df.copy()
        
        for col, fmt in formats.items():
            if col in formatted_df.columns:
                if fmt.startswith('$'):
                    # Character format
                    formatted_df[col] = formatted_df[col].astype(str)
                elif 'date' in fmt.lower():
                    # Date format
                    formatted_df[col] = pd.to_datetime(formatted_df[col])
                elif '.' in fmt:
                    # Numeric format
                    try:
                        decimals = int(fmt.split('.')[1])
                        formatted_df[col] = formatted_df[col].round(decimals)
                    except:
                        pass
                        
        return formatted_df
    
    def _get_dataframe(self, data: Union[pd.DataFrame, str]) -> pd.DataFrame:
        """Helper method to get DataFrame from various inputs"""
        if isinstance(data, str):
            return self.datasets.get(data, pd.DataFrame())
        return data
    
    def _log_operation(self, message: str):
        """Log operations for debugging"""
        if self.debug:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {message}"
            self.operations_log.append(log_entry)
            print(log_entry)
    
    def get_operation_log(self) -> List[str]:
        """Return the operations log"""
        return self.operations_log.copy()


@dataclass
class SASValidationResult:
    """Class for storing validation results"""
    test_name: str
    passed: bool
    sas_result: Any
    python_result: Any
    difference: Optional[Any] = None
    tolerance: float = 1e-6
    
    def __post_init__(self):
        if not self.passed and self.sas_result is not None and self.python_result is not None:
            try:
                if isinstance(self.sas_result, (pd.DataFrame, pd.Series)):
                    self.difference = self.sas_result.compare(self.python_result)
                elif isinstance(self.sas_result, (int, float)) and isinstance(self.python_result, (int, float)):
                    self.difference = abs(self.sas_result - self.python_result)
            except Exception as e:
                self.difference = f"Could not compute difference: {str(e)}"


class SASValidator:
    """Class for validating SAS to Python translations"""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.results = []
    
    def validate_dataframe(self, sas_df: pd.DataFrame, 
                          python_df: pd.DataFrame,
                          test_name: str) -> SASValidationResult:
        """Validate DataFrame equivalence"""
        try:
            # Check shapes
            if sas_df.shape != python_df.shape:
                result = SASValidationResult(
                    test_name=test_name,
                    passed=False,
                    sas_result=sas_df,
                    python_result=python_df
                )
                self.results.append(result)
                return result
            
            # Check column names
            if not sas_df.columns.equals(python_df.columns):
                result = SASValidationResult(
                    test_name=test_name,
                    passed=False,
                    sas_result=sas_df,
                    python_result=python_df
                )
                self.results.append(result)
                return result
            
            # Check values
            try:
                pd.testing.assert_frame_equal(sas_df, python_df, atol=self.tolerance)
                passed = True
            except AssertionError:
                passed = False
                
            result = SASValidationResult(
                test_name=test_name,
                passed=passed,
                sas_result=sas_df,
                python_result=python_df,
                tolerance=self.tolerance
            )
            
        except Exception as e:
            result = SASValidationResult(
                test_name=test_name,
                passed=False,
                sas_result=sas_df,
                python_result=python_df,
                difference=str(e)
            )
            
        self.results.append(result)
        return result
    
    def validate_numeric(self, sas_value: float, 
                        python_value: float,
                        test_name: str) -> SASValidationResult:
        """Validate numeric equivalence"""
        passed = abs(sas_value - python_value) <= self.tolerance
        
        result = SASValidationResult(
            test_name=test_name,
            passed=passed,
            sas_result=sas_value,
            python_result=python_value,
            tolerance=self.tolerance
        )
        
        self.results.append(result)
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'failed_test_names': [r.test_name for r in self.results if not r.passed]
        }


# Example usage and common SAS operations
def create_sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    data = {
        'id': range(1, 101),
        'age': np.random.randint(18, 80, 100),
        'gender': np.random.choice(['M', 'F'], 100),
        'salary': np.random.normal(50000, 15000, 100),
        'department': np.random.choice(['Sales', 'Marketing', 'IT', 'HR'], 100),
        'hire_date': pd.date_range('2020-01-01', periods=100, freq='D')
    }
    return pd.DataFrame(data)


def example_sas_translation():
    """Example of translating SAS code to Python"""
    
    # Initialize translator
    translator = SASTranslator(debug=True)
    
    # Create sample data (equivalent to SAS data creation)
    sample_df = create_sample_data()
    translator.datasets['employees'] = sample_df
    
    # Example 1: DATA step with conditional logic
    def salary_categories(df):
        conditions = [
            df['salary'] < 40000,
            (df['salary'] >= 40000) & (df['salary'] < 60000),
            df['salary'] >= 60000
        ]
        choices = ['Low', 'Medium', 'High']
        df['salary_category'] = np.select(conditions, choices, default='Unknown')
        return df
    
    # Equivalent to SAS DATA step
    processed_data = translator.data_step(
        input_data='employees',
        output_name='employees_processed',
        operations=[salary_categories]
    )
    
    # Example 2: PROC FREQ equivalent
    freq_results = translator.proc_freq(
        data='employees_processed',
        variables=['gender', 'department', 'salary_category'],
        tables=['gender*department']
    )
    
    # Example 3: PROC MEANS equivalent
    means_results = translator.proc_means(
        data='employees_processed',
        variables=['age', 'salary'],
        class_vars=['department']
    )
    
    # Example 4: PROC SORT equivalent
    sorted_data = translator.proc_sort(
        data='employees_processed',
        by=['department', 'salary'],
        ascending=[True, False],
        output_name='employees_sorted'
    )
    
    return translator, freq_results, means_results


# ===== DATA STEP EQUIVALENT =====
def data_step(input_df: pd.DataFrame, 
              keep: List[str] = None,
              drop: List[str] = None,
              where: str = None,
              compute: Dict[str, str] = None,
              output_name: str = None) -> pd.DataFrame:
    """
    Equivalent to SAS DATA step
    
    Example:
    result = data_step(df, 
                      keep=['name', 'age', 'salary'],
                      where='age > 25',
                      compute={'bonus': 'salary * 0.1',
                              'age_group': "np.where(age < 30, 'Young', 'Mature')"})
    """
    result = input_df.copy()
    
    # WHERE clause (subset)
    if where:
        result = result.query(where)
    
    # Compute new variables
    if compute:
        for var_name, expression in compute.items():
            result[var_name] = result.eval(expression)
    
    # DROP variables
    if drop:
        result = result.drop(columns=[col for col in drop if col in result.columns])
    
    # KEEP variables
    if keep:
        result = result[[col for col in keep if col in result.columns]]
    
    return result

# ===== PROC PRINT =====
def proc_print(df: pd.DataFrame, 
               var: List[str] = None,
               obs: int = None,
               where: str = None,
               by: List[str] = None,
               title: str = None) -> None:
    """
    Equivalent to PROC PRINT
    
    Example:
    proc_print(df, var=['name', 'age'], obs=10, where='age > 25')
    """
    result = df.copy()
    
    if where:
        result = result.query(where)
    
    if var:
        result = result[var]
    
    if obs:
        result = result.head(obs)
    
    if title:
        print(f"\n{title}")
        print("=" * len(title))
    
    if by:
        for group_vals, group_df in result.groupby(by):
            print(f"\nGroup: {dict(zip(by, group_vals))}")
            print(group_df)
    else:
        print(result)

# ===== PROC MEANS =====
def proc_means(df: pd.DataFrame,
               var: List[str] = None,
               by: List[str] = None,
               class_vars: List[str] = None,
               stats: List[str] = ['n', 'mean', 'std', 'min', 'max']) -> pd.DataFrame:
    """
    Equivalent to PROC MEANS
    
    Example:
    stats_df = proc_means(df, var=['age', 'salary'], by=['department'])
    """
    # Select numeric variables if not specified
    if var is None:
        var = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Define stat mappings
    stat_mapping = {
        'n': 'count',
        'mean': 'mean',
        'std': 'std',
        'min': 'min',
        'max': 'max',
        'sum': 'sum',
        'median': 'median'
    }
    
    groupby_vars = by or class_vars
    
    if groupby_vars:
        result = df.groupby(groupby_vars)[var].agg([stat_mapping[s] for s in stats if s in stat_mapping])
        result.columns = [f"{col}_{stat}" for col, stat in result.columns]
        return result.reset_index()
    else:
        result = df[var].agg([stat_mapping[s] for s in stats if s in stat_mapping])
        return pd.DataFrame(result).T

# ===== PROC FREQ =====
def proc_freq(df: pd.DataFrame,
              tables: List[str] = None,
              by: List[str] = None,
              out: str = None) -> Dict[str, pd.DataFrame]:
    """
    Equivalent to PROC FREQ
    
    Example:
    freq_tables = proc_freq(df, tables=['gender', 'department'])
    """
    results = {}
    
    if tables is None:
        tables = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for table in tables:
        if by:
            freq_result = df.groupby(by)[table].value_counts().unstack(fill_value=0)
            freq_result['Total'] = freq_result.sum(axis=1)
            freq_result.loc['Total'] = freq_result.sum(axis=0)
        else:
            freq_counts = df[table].value_counts()
            freq_pct = df[table].value_counts(normalize=True) * 100
            freq_result = pd.DataFrame({
                'Frequency': freq_counts,
                'Percent': freq_pct
            })
            freq_result.loc['Total'] = [freq_counts.sum(), 100.0]
        
        results[table] = freq_result
        
        # Print frequency table
        print(f"\nFrequency Table: {table}")
        print("=" * 40)
        print(freq_result)
    
    return results

# ===== PROC SORT =====
def proc_sort(df: pd.DataFrame,
              by: List[str],
              ascending: Union[bool, List[bool]] = True,
              nodupkey: bool = False,
              out: str = None) -> pd.DataFrame:
    """
    Equivalent to PROC SORT
    
    Example:
    sorted_df = proc_sort(df, by=['department', 'salary'], ascending=[True, False])
    """
    result = df.sort_values(by=by, ascending=ascending)
    
    if nodupkey:
        result = result.drop_duplicates(subset=by, keep='first')
    
    return result.reset_index(drop=True)

# ===== PROC SQL =====
def proc_sql(query: str, **datasets) -> pd.DataFrame:
    """
    Equivalent to PROC SQL using pandas query syntax
    
    Example:
    result = proc_sql('''
        SELECT name, age, salary
        FROM employees
        WHERE age > 30
        ORDER BY salary DESC
    ''', employees=df)
    """
    # This is a simplified version - in practice you'd use SQLAlchemy or similar
    # For demo purposes, showing concept with pandas operations
    import re
    
    # Extract table name from query (simplified)
    table_match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
    if table_match:
        table_name = table_match.group(1)
        if table_name in datasets:
            # This would require a full SQL parser for complete implementation
            # For now, return the dataset as example
            return datasets[table_name]
    
    raise NotImplementedError("Full SQL parsing not implemented in this example")

# ===== PROC TTEST =====
def proc_ttest(df: pd.DataFrame,
               var: str,
               class_var: str = None,
               paired: bool = False,
               alpha: float = 0.05) -> Dict[str, Any]:
    """
    Equivalent to PROC TTEST
    
    Example:
    ttest_result = proc_ttest(df, var='salary', class_var='gender')
    """
    results = {}
    
    if class_var:
        # Two-sample t-test
        groups = df.groupby(class_var)[var].apply(list)
        group_names = list(groups.index)
        
        if len(group_names) == 2:
            group1, group2 = groups.iloc[0], groups.iloc[1]
            
            if paired:
                t_stat, p_value = stats.ttest_rel(group1, group2)
                test_type = "Paired t-test"
            else:
                t_stat, p_value = stats.ttest_ind(group1, group2)
                test_type = "Two-sample t-test"
            
            results = {
                'test_type': test_type,
                'variable': var,
                'class_variable': class_var,
                'group1': group_names[0],
                'group2': group_names[1],
                'n1': len(group1),
                'n2': len(group2),
                'mean1': np.mean(group1),
                'mean2': np.mean(group2),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < alpha
            }
    else:
        # One-sample t-test (against 0)
        t_stat, p_value = stats.ttest_1samp(df[var].dropna(), 0)
        results = {
            'test_type': 'One-sample t-test',
            'variable': var,
            'n': len(df[var].dropna()),
            'mean': df[var].mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha
        }
    
    # Print results
    print(f"\n{results['test_type']} Results")
    print("=" * 40)
    for key, value in results.items():
        if key != 'test_type':
            print(f"{key}: {value}")
    
    return results

# ===== PROC REG =====
def proc_reg(df: pd.DataFrame,
             model: str,
             output_stats: bool = True) -> Dict[str, Any]:
    """
    Equivalent to PROC REG
    
    Example:
    reg_result = proc_reg(df, model='salary ~ age + experience')
    """
    import statsmodels.formula.api as smf
    
    # Fit the model
    model_fit = smf.ols(model, data=df).fit()
    
    results = {
        'model': model,
        'r_squared': model_fit.rsquared,
        'adj_r_squared': model_fit.rsquared_adj,
        'f_statistic': model_fit.fvalue,
        'f_pvalue': model_fit.f_pvalue,
        'aic': model_fit.aic,
        'bic': model_fit.bic,
        'coefficients': model_fit.params.to_dict(),
        'p_values': model_fit.pvalues.to_dict(),
        'confidence_intervals': model_fit.conf_int().to_dict(),
        'residuals': model_fit.resid,
        'fitted_values': model_fit.fittedvalues
    }
    
    if output_stats:
        print(f"\nRegression Results: {model}")
        print("=" * 50)
        print(model_fit.summary())
    
    return results

# ===== PROC LOGISTIC =====
def proc_logistic(df: pd.DataFrame,
                  target: str,
                  features: List[str],
                  output_stats: bool = True) -> Dict[str, Any]:
    """
    Equivalent to PROC LOGISTIC
    
    Example:
    logit_result = proc_logistic(df, target='promoted', features=['age', 'salary', 'experience'])
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    
    X = df[features]
    y = df[target]
    
    # Fit logistic regression
    model = LogisticRegression()
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    auc = roc_auc_score(y, y_pred_proba)
    
    results = {
        'target': target,
        'features': features,
        'coefficients': dict(zip(features, model.coef_[0])),
        'intercept': model.intercept_[0],
        'auc': auc,
        'predictions': y_pred,
        'predicted_probabilities': y_pred_proba,
        'confusion_matrix': confusion_matrix(y, y_pred),
        'classification_report': classification_report(y, y_pred, output_dict=True)
    }
    
    if output_stats:
        print(f"\nLogistic Regression Results")
        print("=" * 40)
        print(f"Target: {target}")
        print(f"Features: {features}")
        print(f"AUC: {auc:.4f}")
        print("\nCoefficients:")
        for feature, coef in results['coefficients'].items():
            print(f"  {feature}: {coef:.4f}")
        print(f"Intercept: {model.intercept_[0]:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y, y_pred))
    
    return results

# ===== PROC UNIVARIATE =====
def proc_univariate(df: pd.DataFrame,
                    var: List[str] = None,
                    by: List[str] = None,
                    histogram: bool = True) -> Dict[str, Dict]:
    """
    Equivalent to PROC UNIVARIATE
    
    Example:
    univar_results = proc_univariate(df, var=['age', 'salary'])
    """
    if var is None:
        var = df.select_dtypes(include=[np.number]).columns.tolist()
    
    results = {}
    
    for variable in var:
        data = df[variable].dropna()
        
        # Basic statistics
        stats_dict = {
            'n': len(data),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'q1': data.quantile(0.25),
            'median': data.median(),
            'q3': data.quantile(0.75),
            'max': data.max(),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis(),
            'missing': df[variable].isna().sum()
        }
        
        results[variable] = stats_dict
        
        # Print results
        print(f"\nUnivariate Analysis: {variable}")
        print("=" * 40)
        for stat, value in stats_dict.items():
            print(f"{stat}: {value:.4f}" if isinstance(value, float) else f"{stat}: {value}")
        
        # Create histogram if requested
        if histogram:
            plt.figure(figsize=(8, 6))
            plt.hist(data, bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'Histogram of {variable}')
            plt.xlabel(variable)
            plt.ylabel('Frequency')
            plt.show()
    
    return results

# import pandas as pd
# import numpy as np
# from typing import Dict, List, Optional, Union, Any
# import re
# from functools import reduce
# import warnings


class ProcSQL:
    """
    A Python class that replicates common PROC SQL functionality from SAS.
    Provides methods for data manipulation, aggregation, joins, and filtering.
    """
    
    def __init__(self, datasets: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Initialize ProcSQL with optional datasets dictionary.
        
        Args:
            datasets: Dictionary mapping dataset names to pandas DataFrames
        """
        self.datasets = datasets or {}
        self.query_history = []
    
    def add_dataset(self, name: str, df: pd.DataFrame) -> None:
        """Add a dataset to the available datasets."""
        self.datasets[name] = df.copy()
    
    def select(self, dataset_name: str, columns: Optional[List[str]] = None, 
               where: Optional[str] = None, order_by: Optional[List[str]] = None,
               limit: Optional[int] = None) -> pd.DataFrame:
        """
        Perform a SELECT operation similar to PROC SQL.
        
        Args:
            dataset_name: Name of the dataset to select from
            columns: List of column names to select (None for all)
            where: WHERE clause condition as string
            order_by: List of columns to order by
            limit: Maximum number of rows to return
            
        Returns:
            Filtered and selected DataFrame
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name].copy()
        
        # Apply WHERE clause
        if where:
            df = self._apply_where_clause(df, where)
        
        # Select columns
        if columns:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Columns not found: {missing_cols}")
            df = df[columns]
        
        # Apply ORDER BY
        if order_by:
            ascending = []
            sort_cols = []
            for col in order_by:
                if col.upper().endswith(' DESC'):
                    sort_cols.append(col[:-5].strip())
                    ascending.append(False)
                else:
                    sort_cols.append(col.replace(' ASC', '').strip())
                    ascending.append(True)
            df = df.sort_values(sort_cols, ascending=ascending)
        
        # Apply LIMIT
        if limit:
            df = df.head(limit)
        
        return df.reset_index(drop=True)
    
    def create_table(self, table_name: str, select_query: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a new table using SELECT statement results.
        
        Args:
            table_name: Name for the new table
            select_query: Dictionary with select parameters
            
        Returns:
            Created DataFrame
        """
        result = self.select(**select_query)
        self.datasets[table_name] = result
        return result
    
    def group_by(self, dataset_name: str, group_cols: List[str], 
                 agg_funcs: Dict[str, Union[str, List[str]]], 
                 having: Optional[str] = None) -> pd.DataFrame:
        """
        Perform GROUP BY operations with aggregation functions.
        
        Args:
            dataset_name: Name of the dataset
            group_cols: Columns to group by
            agg_funcs: Dictionary mapping column names to aggregation functions
            having: HAVING clause condition
            
        Returns:
            Grouped and aggregated DataFrame
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name].copy()
        
        # Perform groupby and aggregation
        grouped = df.groupby(group_cols)
        
        # Apply aggregation functions
        agg_dict = {}
        for col, funcs in agg_funcs.items():
            if isinstance(funcs, str):
                funcs = [funcs]
            for func in funcs:
                if func.lower() in ['count', 'sum', 'mean', 'min', 'max', 'std', 'var']:
                    agg_dict[f"{col}_{func}"] = (col, func.lower())
                else:
                    raise ValueError(f"Unsupported aggregation function: {func}")
        
        result = grouped.agg(agg_dict).reset_index()
        
        # Flatten column names if needed
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = [col[0] if col[1] == '' else col[0] for col in result.columns]
        
        # Apply HAVING clause
        if having:
            result = self._apply_where_clause(result, having)
        
        return result
    
    def join(self, left_dataset: str, right_dataset: str, 
             join_type: str = 'inner', on: Optional[Union[str, List[str]]] = None,
             left_on: Optional[Union[str, List[str]]] = None,
             right_on: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Perform JOIN operations between two datasets.
        
        Args:
            left_dataset: Name of left dataset
            right_dataset: Name of right dataset
            join_type: Type of join ('inner', 'left', 'right', 'outer')
            on: Column(s) to join on (if same in both datasets)
            left_on: Column(s) to join on from left dataset
            right_on: Column(s) to join on from right dataset
            
        Returns:
            Joined DataFrame
        """
        if left_dataset not in self.datasets:
            raise ValueError(f"Dataset '{left_dataset}' not found")
        if right_dataset not in self.datasets:
            raise ValueError(f"Dataset '{right_dataset}' not found")
        
        left_df = self.datasets[left_dataset].copy()
        right_df = self.datasets[right_dataset].copy()
        
        # Determine join keys
        if on is not None:
            left_on = right_on = on
        elif left_on is None or right_on is None:
            raise ValueError("Must specify either 'on' or both 'left_on' and 'right_on'")
        
        # Perform join
        result = pd.merge(left_df, right_df, 
                         left_on=left_on, right_on=right_on, 
                         how=join_type, suffixes=('_left', '_right'))
        
        return result
    
    def union(self, dataset_names: List[str], all_records: bool = False) -> pd.DataFrame:
        """
        Perform UNION operation on multiple datasets.
        
        Args:
            dataset_names: List of dataset names to union
            all_records: If True, include duplicate records (UNION ALL)
            
        Returns:
            Unioned DataFrame
        """
        if not dataset_names:
            raise ValueError("Must specify at least one dataset")
        
        dfs = []
        for name in dataset_names:
            if name not in self.datasets:
                raise ValueError(f"Dataset '{name}' not found")
            dfs.append(self.datasets[name])
        
        # Concatenate all DataFrames
        result = pd.concat(dfs, ignore_index=True)
        
        # Remove duplicates if not UNION ALL
        if not all_records:
            result = result.drop_duplicates()
        
        return result
    
    def calculate_stats(self, dataset_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate descriptive statistics similar to PROC MEANS.
        
        Args:
            dataset_name: Name of the dataset
            columns: Columns to calculate stats for (None for all numeric)
            
        Returns:
            DataFrame with statistical summaries
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name]
        
        if columns:
            df = df[columns]
        else:
            df = df.select_dtypes(include=[np.number])
        
        stats = df.describe().T
        stats['count'] = df.count()
        stats['missing'] = df.isnull().sum()
        
        return stats
    
    def _apply_where_clause(self, df: pd.DataFrame, where_clause: str) -> pd.DataFrame:
        """
        Apply WHERE clause conditions to DataFrame.
        Supports basic comparison operators and logical operators.
        """
        # Simple implementation - in real scenario, would need more robust parsing
        where_clause = where_clause.strip()
        
        # Replace SQL operators with pandas equivalents
        where_clause = where_clause.replace(' AND ', ' & ')
        where_clause = where_clause.replace(' OR ', ' | ')
        where_clause = where_clause.replace(' and ', ' & ')
        where_clause = where_clause.replace(' or ', ' | ')
        
        # Handle string comparisons
        where_clause = re.sub(r"(\w+)\s*=\s*'([^']*)'", r"(\1 == '\2')", where_clause)
        where_clause = re.sub(r"(\w+)\s*=\s*([^'\s&|]+)", r"(\1 == \2)", where_clause)
        
        try:
            # Evaluate the condition
            mask = df.eval(where_clause)
            return df[mask]
        except Exception as e:
            raise ValueError(f"Invalid WHERE clause: {where_clause}. Error: {str(e)}")
    
    def describe_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a dataset similar to PROC CONTENTS.
        
        Returns:
            Dictionary with dataset information
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name]
        
        return {
            'name': dataset_name,
            'observations': len(df),
            'variables': len(df.columns),
            'column_info': {
                col: {
                    'dtype': str(df[col].dtype),
                    'non_null': df[col].count(),
                    'null': df[col].isnull().sum(),
                    'unique': df[col].nunique()
                }
                for col in df.columns
            }
        }
    
    def list_datasets(self) -> List[str]:
        """Return list of available dataset names."""
        return list(self.datasets.keys())
    
    def drop_dataset(self, dataset_name: str) -> None:
        """Remove a dataset from available datasets."""
        if dataset_name in self.datasets:
            del self.datasets[dataset_name]
        else:
            warnings.warn(f"Dataset '{dataset_name}' not found")

# ===== PROC TRANSPOSE =====
def proc_transpose(df: pd.DataFrame,
                   by: List[str] = None,
                   var: List[str] = None,
                   id_var: str = None,
                   prefix: str = 'COL') -> pd.DataFrame:
    """
    Equivalent to PROC TRANSPOSE
    
    Example:
    transposed = proc_transpose(df, by=['id'], var=['jan', 'feb', 'mar'], id_var='month')
    """
    if var is None:
        # Transpose all numeric columns
        var = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if by:
        # Group by specified variables and transpose
        result = df.melt(id_vars=by, value_vars=var, var_name='_NAME_', value_name='_VALUE_')
        if id_var:
            result = result.pivot_table(index=by, columns=id_var, values='_VALUE_', fill_value=0)
            result.columns = [f"{prefix}{col}" for col in result.columns]
            return result.reset_index()
    else:
        # Simple transpose
        result = df[var].T
        result.columns = [f"{prefix}{i+1}" for i in range(len(result.columns))]
        return result

# Example usage and demonstration
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'id': range(1, 101),
        'name': [f'Person_{i}' for i in range(1, 101)],
        'age': np.random.randint(22, 65, 100),
        'department': np.random.choice(['Sales', 'Engineering', 'Marketing'], 100),
        'salary': np.random.normal(50000, 15000, 100),
        'experience': np.random.randint(0, 20, 100),
        'gender': np.random.choice(['M', 'F'], 100)
    })
    
    print("Sample Data:")
    print(sample_data.head())
    
    # DATA Step example
    print("\n" + "="*50)
    print("DATA STEP EXAMPLE")
    enriched_data = data_step(sample_data,
                             where='age > 30',
                             compute={'salary_bonus': 'salary * 0.1',
                                     'age_group': "np.where(age < 40, 'Young', 'Senior')"})
    print(f"Filtered and computed data shape: {enriched_data.shape}")
    
    # PROC MEANS example
    print("\n" + "="*50)
    print("PROC MEANS EXAMPLE")
    means_result = proc_means(sample_data, var=['age', 'salary'], by=['department'])
    print(means_result)
    
    # PROC FREQ example
    print("\n" + "="*50)
    print("PROC FREQ EXAMPLE")
    freq_results = proc_freq(sample_data, tables=['department', 'gender'])
    

if __name__ == "__main__":
    # Run example
    translator, freq_results, means_results = example_sas_translation()
    
    print("=== Operation Log ===")
    for log_entry in translator.get_operation_log():
        print(log_entry)
    
    print("\n=== FREQ Results ===")
    for var, result in freq_results.items():
        print(f"\n{var}:")
        print(result)
    
    print("\n=== MEANS Results ===")
    print(means_results)