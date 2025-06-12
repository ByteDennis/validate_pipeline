"""
Oracle DB vs AWS Athena Data Comparator
Comprehensive tool for comparing data between Oracle Database and AWS Athena

Core Functionality

Database Connections: Handles both Oracle (oracledb) and AWS Athena (pyathena) connections
Data Comparison:

Row count comparison
Schema comparison (column names, data types)
Statistical comparison for numeric data
Record-level comparison with key matching
Value-level differences detection


Advanced Comparisons:

Statistical Tests: Kolmogorov-Smirnov and Mann-Whitney U tests
Tolerance-based Numeric Comparison: Handles floating-point precision differences
Data Profiling: Automatic statistics for all numeric columns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import warnings
import json
import hashlib

# Database connections
import oracledb
import boto3
import pyathena
from pyathena import connect
from pyathena.pandas.cursor import PandasCursor

# For data profiling and comparison
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


@dataclass
class DatabaseConfig:
    """Configuration for database connections"""
    oracle_config: Dict[str, Any] = field(default_factory=dict)
    athena_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Default Oracle configuration
        if not self.oracle_config:
            self.oracle_config = {
                'user': 'your_oracle_user',
                'password': 'your_oracle_password',
                'dsn': 'your_oracle_host:1521/your_service_name',
                'encoding': 'UTF-8'
            }
        
        # Default Athena configuration
        if not self.athena_config:
            self.athena_config = {
                'aws_access_key_id': 'your_access_key',
                'aws_secret_access_key': 'your_secret_key',
                'region_name': 'us-east-1',
                's3_staging_dir': 's3://your-athena-bucket/results/',
                'schema_name': 'default'
            }


@dataclass
class ComparisonResult:
    """Results of data comparison between Oracle and Athena"""
    table_name: str
    oracle_count: int
    athena_count: int
    match_status: str
    differences: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    sample_differences: Optional[pd.DataFrame] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def count_match(self) -> bool:
        return self.oracle_count == self.athena_count
    
    @property
    def summary(self) -> Dict[str, Any]:
        return {
            'table_name': self.table_name,
            'oracle_count': self.oracle_count,
            'athena_count': self.athena_count,
            'count_difference': abs(self.oracle_count - self.athena_count),
            'match_status': self.match_status,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat()
        }


class DatabaseConnector:
    """Handles connections to Oracle and Athena databases"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.oracle_conn = None
        self.athena_conn = None
        self.logger = logging.getLogger(__name__)
    
    def connect_oracle(self) -> oracledb.Connection:
        """Connect to Oracle database"""
        try:
            if self.oracle_conn is None or not self.oracle_conn.ping():
                self.oracle_conn = oracledb.connect(**self.config.oracle_config)
                self.logger.info("Connected to Oracle database")
        except Exception as e:
            self.logger.error(f"Failed to connect to Oracle: {e}")
            raise
        return self.oracle_conn
    
    def connect_athena(self) -> pyathena.Connection:
        """Connect to AWS Athena"""
        try:
            if self.athena_conn is None:
                self.athena_conn = connect(
                    aws_access_key_id=self.config.athena_config['aws_access_key_id'],
                    aws_secret_access_key=self.config.athena_config['aws_secret_access_key'],
                    s3_staging_dir=self.config.athena_config['s3_staging_dir'],
                    region_name=self.config.athena_config['region_name'],
                    schema_name=self.config.athena_config['schema_name'],
                    cursor_class=PandasCursor
                )
                self.logger.info("Connected to AWS Athena")
        except Exception as e:
            self.logger.error(f"Failed to connect to Athena: {e}")
            raise
        return self.athena_conn
    
    def execute_oracle_query(self, query: str) -> pd.DataFrame:
        """Execute query on Oracle and return DataFrame"""
        conn = self.connect_oracle()
        try:
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            self.logger.error(f"Oracle query failed: {e}")
            raise
    
    def execute_athena_query(self, query: str) -> pd.DataFrame:
        """Execute query on Athena and return DataFrame"""
        conn = self.connect_athena()
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            df = cursor.as_pandas()
            return df
        except Exception as e:
            self.logger.error(f"Athena query failed: {e}")
            raise
    
    def close_connections(self):
        """Close all database connections"""
        if self.oracle_conn:
            self.oracle_conn.close()
            self.logger.info("Oracle connection closed")
        
        if self.athena_conn:
            self.athena_conn.close()
            self.logger.info("Athena connection closed")


class DataComparator:
    """Main class for comparing data between Oracle and Athena"""
    
    def __init__(self, config: DatabaseConfig, tolerance: float = 1e-6):
        self.connector = DatabaseConnector(config)
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)
        self.comparison_results = []
    
    def compare_tables(self, 
                      oracle_query: str, 
                      athena_query: str,
                      table_name: str,
                      key_columns: Optional[List[str]] = None,
                      compare_columns: Optional[List[str]] = None,
                      sample_size: Optional[int] = None) -> ComparisonResult:
        """
        Compare data between Oracle and Athena tables
        
        Args:
            oracle_query: SQL query for Oracle
            athena_query: SQL query for Athena
            table_name: Name identifier for the comparison
            key_columns: Columns to use for joining/matching records
            compare_columns: Specific columns to compare (if None, compares all)
            sample_size: Number of sample records to show differences
        """
        start_time = datetime.now()
        
        try:
            # Execute queries
            self.logger.info(f"Executing Oracle query for {table_name}")
            oracle_df = self.connector.execute_oracle_query(oracle_query)
            
            self.logger.info(f"Executing Athena query for {table_name}")
            athena_df = self.connector.execute_athena_query(athena_query)
            
            # Normalize column names (Oracle might return uppercase)
            oracle_df.columns = oracle_df.columns.str.lower()
            athena_df.columns = athena_df.columns.str.lower()
            
            # Basic count comparison
            oracle_count = len(oracle_df)
            athena_count = len(athena_df)
            
            # Detailed comparison
            comparison_result = self._detailed_comparison(
                oracle_df, athena_df, table_name, 
                key_columns, compare_columns, sample_size
            )
            
            # Update with counts and timing
            comparison_result.oracle_count = oracle_count
            comparison_result.athena_count = athena_count
            comparison_result.execution_time = (datetime.now() - start_time).total_seconds()
            
            self.comparison_results.append(comparison_result)
            return comparison_result
            
        except Exception as e:
            self.logger.error(f"Comparison failed for {table_name}: {e}")
            return ComparisonResult(
                table_name=table_name,
                oracle_count=0,
                athena_count=0,
                match_status=f"ERROR: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _detailed_comparison(self, 
                           oracle_df: pd.DataFrame, 
                           athena_df: pd.DataFrame,
                           table_name: str,
                           key_columns: Optional[List[str]] = None,
                           compare_columns: Optional[List[str]] = None,
                           sample_size: Optional[int] = None) -> ComparisonResult:
        """Perform detailed comparison between DataFrames"""
        
        differences = {}
        statistics = {}
        sample_differences = None
        
        # Schema comparison
        oracle_cols = set(oracle_df.columns)
        athena_cols = set(athena_df.columns)
        
        differences['schema'] = {
            'oracle_only_columns': list(oracle_cols - athena_cols),
            'athena_only_columns': list(athena_cols - oracle_cols),
            'common_columns': list(oracle_cols & athena_cols)
        }
        
        # If no common columns, return early
        if not differences['schema']['common_columns']:
            return ComparisonResult(
                table_name=table_name,
                oracle_count=len(oracle_df),
                athena_count=len(athena_df),
                match_status="SCHEMA_MISMATCH",
                differences=differences,
                statistics=statistics
            )
        
        # Focus on common columns
        common_cols = differences['schema']['common_columns']
        oracle_common = oracle_df[common_cols].copy()
        athena_common = athena_df[common_cols].copy()
        
        # Data type comparison
        differences['data_types'] = self._compare_data_types(oracle_common, athena_common)
        
        # Statistical comparison for numeric columns
        statistics['numeric_stats'] = self._compare_numeric_statistics(oracle_common, athena_common)
        
        # Record-level comparison
        if key_columns and all(col in common_cols for col in key_columns):
            record_comparison = self._compare_records(
                oracle_common, athena_common, key_columns, compare_columns
            )
            differences.update(record_comparison['differences'])
            statistics.update(record_comparison['statistics'])
            sample_differences = record_comparison.get('sample_differences')
        
        # Determine overall match status
        match_status = self._determine_match_status(differences, statistics)
        
        return ComparisonResult(
            table_name=table_name,
            oracle_count=len(oracle_df),
            athena_count=len(athena_df),
            match_status=match_status,
            differences=differences,
            statistics=statistics,
            sample_differences=sample_differences
        )
    
    def _compare_data_types(self, oracle_df: pd.DataFrame, athena_df: pd.DataFrame) -> Dict[str, Any]:
        """Compare data types between DataFrames"""
        oracle_types = oracle_df.dtypes.to_dict()
        athena_types = athena_df.dtypes.to_dict()
        
        type_differences = {}
        for col in oracle_types:
            if col in athena_types:
                if str(oracle_types[col]) != str(athena_types[col]):
                    type_differences[col] = {
                        'oracle_type': str(oracle_types[col]),
                        'athena_type': str(athena_types[col])
                    }
        
        return type_differences
    
    def _compare_numeric_statistics(self, oracle_df: pd.DataFrame, athena_df: pd.DataFrame) -> Dict[str, Any]:
        """Compare statistical measures for numeric columns"""
        numeric_stats = {}
        
        oracle_numeric = oracle_df.select_dtypes(include=[np.number])
        athena_numeric = athena_df.select_dtypes(include=[np.number])
        
        common_numeric = list(set(oracle_numeric.columns) & set(athena_numeric.columns))
        
        for col in common_numeric:
            oracle_col = oracle_numeric[col].dropna()
            athena_col = athena_numeric[col].dropna()
            
            # Basic statistics
            oracle_stats = oracle_col.describe()
            athena_stats = athena_col.describe()
            
            # Statistical tests
            try:
                # Kolmogorov-Smirnov test for distribution comparison
                ks_stat, ks_pvalue = stats.ks_2samp(oracle_col, athena_col)
                
                # Mann-Whitney U test for median comparison
                mw_stat, mw_pvalue = stats.mannwhitneyu(oracle_col, athena_col, alternative='two-sided')
                
                numeric_stats[col] = {
                    'oracle_stats': oracle_stats.to_dict(),
                    'athena_stats': athena_stats.to_dict(),
                    'ks_test': {'statistic': ks_stat, 'pvalue': ks_pvalue},
                    'mannwhitney_test': {'statistic': mw_stat, 'pvalue': mw_pvalue},
                    'mean_difference': abs(oracle_stats['mean'] - athena_stats['mean']),
                    'std_difference': abs(oracle_stats['std'] - athena_stats['std'])
                }
            except Exception as e:
                numeric_stats[col] = {
                    'oracle_stats': oracle_stats.to_dict(),
                    'athena_stats': athena_stats.to_dict(),
                    'error': str(e)
                }
        
        return numeric_stats
    
    def _compare_records(self, 
                        oracle_df: pd.DataFrame, 
                        athena_df: pd.DataFrame,
                        key_columns: List[str],
                        compare_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare records between DataFrames using key columns"""
        
        if compare_columns is None:
            compare_columns = [col for col in oracle_df.columns if col not in key_columns]
        
        # Sort both DataFrames by key columns for consistent comparison
        oracle_sorted = oracle_df.sort_values(key_columns).reset_index(drop=True)
        athena_sorted = athena_df.sort_values(key_columns).reset_index(drop=True)
        
        # Create composite keys for matching
        oracle_keys = oracle_sorted[key_columns].apply(
            lambda x: hashlib.md5(str(tuple(x)).encode()).hexdigest(), axis=1
        )
        athena_keys = athena_sorted[key_columns].apply(
            lambda x: hashlib.md5(str(tuple(x)).encode()).hexdigest(), axis=1
        )
        
        oracle_sorted['_key'] = oracle_keys
        athena_sorted['_key'] = athena_keys
        
        # Find matching and non-matching records
        oracle_key_set = set(oracle_keys)
        athena_key_set = set(athena_keys)
        
        common_keys = oracle_key_set & athena_key_set
        oracle_only_keys = oracle_key_set - athena_key_set
        athena_only_keys = athena_key_set - oracle_key_set
        
        # Compare values for common records
        value_differences = []
        if common_keys:
            oracle_common = oracle_sorted[oracle_sorted['_key'].isin(common_keys)]
            athena_common = athena_sorted[athena_sorted['_key'].isin(common_keys)]
            
            # Merge on key for comparison
            merged = pd.merge(
                oracle_common, athena_common, 
                on='_key', suffixes=('_oracle', '_athena')
            )
            
            # Compare each column
            for col in compare_columns:
                oracle_col = f"{col}_oracle"
                athena_col = f"{col}_athena"
                
                if oracle_col in merged.columns and athena_col in merged.columns:
                    # Handle numeric comparisons with tolerance
                    if pd.api.types.is_numeric_dtype(merged[oracle_col]):
                        diff_mask = abs(merged[oracle_col] - merged[athena_col]) > self.tolerance
                    else:
                        diff_mask = merged[oracle_col] != merged[athena_col]
                    
                    if diff_mask.any():
                        differences_subset = merged[diff_mask][
                            key_columns + [oracle_col, athena_col]
                        ].head(100)  # Limit sample size
                        
                        value_differences.append({
                            'column': col,
                            'different_count': diff_mask.sum(),
                            'sample_differences': differences_subset
                        })
        
        return {
            'differences': {
                'oracle_only_records': len(oracle_only_keys),
                'athena_only_records': len(athena_only_keys),
                'common_records': len(common_keys),
                'value_differences': value_differences
            },
            'statistics': {
                'total_oracle_records': len(oracle_df),
                'total_athena_records': len(athena_df),
                'match_rate': len(common_keys) / max(len(oracle_df), len(athena_df)) if max(len(oracle_df), len(athena_df)) > 0 else 0
            },
            'sample_differences': pd.concat([vd['sample_differences'] for vd in value_differences]) if value_differences else None
        }
    
    def _determine_match_status(self, differences: Dict[str, Any], statistics: Dict[str, Any]) -> str:
        """Determine overall match status based on differences and statistics"""
        
        # Check for schema mismatches
        if differences.get('schema', {}).get('oracle_only_columns') or \
           differences.get('schema', {}).get('athena_only_columns'):
            return "SCHEMA_MISMATCH"
        
        # Check for record count mismatches
        if differences.get('oracle_only_records', 0) > 0 or \
           differences.get('athena_only_records', 0) > 0:
            return "RECORD_COUNT_MISMATCH"
        
        # Check for value differences
        if differences.get('value_differences'):
            total_differences = sum(vd['different_count'] for vd in differences['value_differences'])
            if total_differences > 0:
                return "VALUE_DIFFERENCES"
        
        # Check statistical differences for numeric columns
        numeric_stats = statistics.get('numeric_stats', {})
        for col_stats in numeric_stats.values():
            if isinstance(col_stats, dict) and 'ks_test' in col_stats:
                if col_stats['ks_test']['pvalue'] < 0.05:  # Significant difference
                    return "STATISTICAL_DIFFERENCES"
        
        return "MATCH"
    
    def generate_comparison_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive comparison report"""
        
        report = {
            'summary': {
                'total_comparisons': len(self.comparison_results),
                'matches': len([r for r in self.comparison_results if r.match_status == "MATCH"]),
                'differences': len([r for r in self.comparison_results if r.match_status != "MATCH"]),
                'errors': len([r for r in self.comparison_results if "ERROR" in r.match_status]),
                'total_execution_time': sum(r.execution_time for r in self.comparison_results)
            },
            'detailed_results': [result.summary for result in self.comparison_results],
            'timestamp': datetime.now().isoformat()
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Report saved to {output_file}")
        
        return report
    
    def create_comparison_dashboard(self, output_dir: str = "comparison_results"):
        """Create visual dashboard for comparison results"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Summary statistics
        statuses = [r.match_status for r in self.comparison_results]
        status_counts = pd.Series(statuses).value_counts()
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Status distribution
        status_counts.plot(kind='bar', ax=axes[0, 0], title='Comparison Status Distribution')
        axes[0, 0].set_ylabel('Count')
        
        # Execution time distribution
        execution_times = [r.execution_time for r in self.comparison_results]
        axes[0, 1].hist(execution_times, bins=20, title='Execution Time Distribution')
        axes[0, 1].set_xlabel('Execution Time (seconds)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Record count comparison
        oracle_counts = [r.oracle_count for r in self.comparison_results]
        athena_counts = [r.athena_count for r in self.comparison_results]
        
        axes[1, 0].scatter(oracle_counts, athena_counts, alpha=0.6)
        axes[1, 0].plot([0, max(max(oracle_counts), max(athena_counts))], 
                       [0, max(max(oracle_counts), max(athena_counts))], 'r--', alpha=0.8)
        axes[1, 0].set_xlabel('Oracle Record Count')
        axes[1, 0].set_ylabel('Athena Record Count')
        axes[1, 0].set_title('Record Count Comparison')
        
        # Match rate over time
        timestamps = [r.timestamp for r in self.comparison_results]
        match_rates = [1 if r.match_status == "MATCH" else 0 for r in self.comparison_results]
        
        axes[1, 1].plot(timestamps, match_rates, 'o-', alpha=0.7)
        axes[1, 1].set_xlabel('Timestamp')
        axes[1, 1].set_ylabel('Match (1) / No Match (0)')
        axes[1, 1].set_title('Match Rate Over Time')
        
        plt.tight_layout()
        plt.savefig(output_path / 'comparison_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Dashboard saved to {output_path / 'comparison_dashboard.png'}")
    
    def close(self):
        """Close database connections"""
        self.connector.close_connections()


# Example usage and helper functions
def create_sample_queries():
    """Create sample queries for testing"""
    
    # Sample Oracle queries
    oracle_queries = {
        'employees': """
            SELECT employee_id, first_name, last_name, email, hire_date, salary, department_id
            FROM employees
            WHERE hire_date >= TO_DATE('2020-01-01', 'YYYY-MM-DD')
            ORDER BY employee_id
        """,
        'departments': """
            SELECT department_id, department_name, manager_id, location_id
            FROM departments
            ORDER BY department_id
        """,
        'sales_summary': """
            SELECT 
                TRUNC(order_date, 'MM') as month,
                SUM(total_amount) as total_sales,
                COUNT(*) as order_count,
                AVG(total_amount) as avg_order_value
            FROM orders 
            WHERE order_date >= TO_DATE('2023-01-01', 'YYYY-MM-DD')
            GROUP BY TRUNC(order_date, 'MM')
            ORDER BY month
        """
    }
    
    # Sample Athena queries (adjusted for Presto SQL syntax)
    athena_queries = {
        'employees': """
            SELECT employee_id, first_name, last_name, email, hire_date, salary, department_id
            FROM employees
            WHERE hire_date >= DATE('2020-01-01')
            ORDER BY employee_id
        """,
        'departments': """
            SELECT department_id, department_name, manager_id, location_id
            FROM departments
            ORDER BY department_id
        """,
        'sales_summary': """
            SELECT 
                DATE_TRUNC('month', order_date) as month,
                SUM(total_amount) as total_sales,
                COUNT(*) as order_count,
                AVG(total_amount) as avg_order_value
            FROM orders 
            WHERE order_date >= DATE('2023-01-01')
            GROUP BY DATE_TRUNC('month', order_date)
            ORDER BY month
        """
    }
    
    return oracle_queries, athena_queries


def main():
    """Main function demonstrating usage"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Database configuration
    config = DatabaseConfig(
        oracle_config={
            'user': 'your_oracle_user',
            'password': 'your_oracle_password',
            'dsn': 'your_oracle_host:1521/your_service_name'
        },
        athena_config={
            'aws_access_key_id': 'your_access_key',
            'aws_secret_access_key': 'your_secret_key',
            'region_name': 'us-east-1',
            's3_staging_dir': 's3://your-athena-bucket/results/',
            'schema_name': 'your_schema'
        }
    )
    
    # Initialize comparator
    comparator = DataComparator(config, tolerance=1e-6)
    
    # Get sample queries
    oracle_queries, athena_queries = create_sample_queries()
    
    try:
        # Compare each table
        for table_name in oracle_queries:
            print(f"\n=== Comparing {table_name} ===")
            
            result = comparator.compare_tables(
                oracle_query=oracle_queries[table_name],
                athena_query=athena_queries[table_name],
                table_name=table_name,
                key_columns=['employee_id'] if table_name == 'employees' else ['department_id'] if table_name == 'departments' else ['month'],
                sample_size=100
            )
            
            print(f"Status: {result.match_status}")
            print(f"Oracle records: {result.oracle_count}")
            print(f"Athena records: {result.athena_count}")
            print(f"Execution time: {result.execution_time:.2f}s")
            
            if result.match_status != "MATCH":
                print(f"Differences found: {result.differences}")
    
        # Generate report
        report = comparator.generate_comparison_report('comparison_report.json')
        print(f"\n=== Summary ===")
        print(f"Total comparisons: {report['summary']['total_comparisons']}")
        print(f"Matches: {report['summary']['matches']}")
        print(f"Differences: {report['summary']['differences']}")
        print(f"Errors: {report['summary']['errors']}")
        
        # Create dashboard
        comparator.create_comparison_dashboard()
        
    finally:
        # Clean up
        comparator.close()


if __name__ == "__main__":
    main()