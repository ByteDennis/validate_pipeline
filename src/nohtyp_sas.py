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
from pathlib import Path
import logging
from datetime import datetime
import warnings

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