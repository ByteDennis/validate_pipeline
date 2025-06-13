#!/usr/bin/env python3
"""
Data Alignment Script

Aligns PCDS and AWS data by removing unmatched columns and rows based on 
migration analysis results and statistics comparison outcomes.
"""

import pandas as pd
import awswrangler as wr
import boto3
import json
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from multiprocessing import Pool

def normalize_column(col_data, numeric_tol=1e-6, ignore_case=True, null_vals=['', 'NULL', 'N/A']):
    """Normalize a single column for comparison"""
    col = col_data.copy()
    
    # Handle nulls
    col = col.replace(null_vals, np.nan)
    
    # Try numeric conversion
    if col.dtype == 'object':
        numeric_col = pd.to_numeric(col, errors='ignore')
        if not numeric_col.equals(col):
            return numeric_col
    
    # String normalization
    if col.dtype == 'object':
        col = col.astype(str).str.strip()
        if ignore_case:
            col = col.str.lower()
        col = col.replace('nan', np.nan)
    
    return col

def compare_column_chunk(args):
    """Compare a chunk of column data - for parallelization"""
    col1_chunk, col2_chunk, col_name, numeric_tol = args
    
    differences = []
    for idx, (val1, val2) in enumerate(zip(col1_chunk, col2_chunk)):
        if pd.isna(val1) and pd.isna(val2):
            continue
        if pd.isna(val1) or pd.isna(val2):
            differences.append((idx, val1, val2))
            continue
            
        # Numeric comparison
        if pd.api.types.is_numeric_dtype(type(val1)) or pd.api.types.is_numeric_dtype(type(val2)):
            try:
                if abs(float(val1) - float(val2)) > numeric_tol:
                    differences.append((idx, val1, val2))
            except (ValueError, TypeError):
                if str(val1) != str(val2):
                    differences.append((idx, val1, val2))
        else:
            if str(val1) != str(val2):
                differences.append((idx, val1, val2))
    
    return col_name, differences

def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, 
                      numeric_tol: float = 1e-6,
                      ignore_case: bool = True,
                      key_columns: Optional[List[str]] = None,
                      n_jobs: int = 1,
                      chunk_size: int = 10000) -> Dict[str, Any]:
    """
    Fast DataFrame comparison with parallelism
    
    Args:
        df1, df2: DataFrames to compare
        numeric_tol: Tolerance for numeric differences
        ignore_case: Ignore case for string comparison
        key_columns: Columns to sort by
        n_jobs: Number of parallel processes (1 = no parallelism)
        chunk_size: Size of chunks for parallel processing
    
    Returns:
        Dict with comparison results
    """
    
    # Basic checks
    if df1.shape != df2.shape:
        return {'equal': False, 'error': f'Shape mismatch: {df1.shape} vs {df2.shape}'}
    
    common_cols = list(set(df1.columns) & set(df2.columns))
    if not common_cols:
        return {'equal': False, 'error': 'No common columns'}
    
    # Sort if key columns provided
    if key_columns:
        available_keys = [col for col in key_columns if col in common_cols]
        if available_keys:
            df1 = df1.sort_values(available_keys).reset_index(drop=True)
            df2 = df2.sort_values(available_keys).reset_index(drop=True)
    
    # Normalize columns
    df1_norm = df1[common_cols].copy()
    df2_norm = df2[common_cols].copy()
    
    for col in common_cols:
        df1_norm[col] = normalize_column(df1_norm[col], numeric_tol, ignore_case)
        df2_norm[col] = normalize_column(df2_norm[col], numeric_tol, ignore_case)
    
    all_differences = {}
    
    if n_jobs == 1:
        # Sequential processing
        for col in common_cols:
            _, diffs = compare_column_chunk((df1_norm[col], df2_norm[col], col, numeric_tol))
            if diffs:
                all_differences[col] = diffs
    else:
        # Parallel processing
        tasks = []
        for col in common_cols:
            col1_data = df1_norm[col].values
            col2_data = df2_norm[col].values
            
            # Split into chunks
            for i in range(0, len(col1_data), chunk_size):
                chunk1 = col1_data[i:i+chunk_size]
                chunk2 = col2_data[i:i+chunk_size]
                tasks.append((chunk1, chunk2, f"{col}_chunk_{i}", numeric_tol))
        
        # Process in parallel
        with Pool(n_jobs) as pool:
            chunk_results = pool.map(compare_column_chunk, tasks)
        
        # Aggregate results
        for chunk_name, diffs in chunk_results:
            if diffs:
                col_name = chunk_name.split('_chunk_')[0]
                if col_name not in all_differences:
                    all_differences[col_name] = []
                
                # Adjust indices for chunk offset
                chunk_offset = int(chunk_name.split('_chunk_')[1])
                adjusted_diffs = [(idx + chunk_offset, val1, val2) for idx, val1, val2 in diffs]
                all_differences[col_name].extend(adjusted_diffs)
    
    # Summary
    total_diffs = sum(len(diffs) for diffs in all_differences.values())
    total_cells = len(df1_norm) * len(common_cols)
    
    return {
        'equal': total_diffs == 0,
        'total_differences': total_diffs,
        'total_cells': total_cells,
        'match_percentage': ((total_cells - total_diffs) / total_cells * 100) if total_cells > 0 else 0,
        'differences_by_column': all_differences,
        'columns_compared': common_cols
    }

def quick_diff_summary(result: Dict) -> str:
    """Generate quick summary of differences"""
    if result['equal']:
        return "✓ DataFrames are identical"
    
    summary = [f"✗ Found {result['total_differences']} differences ({result['match_percentage']:.1f}% match)"]
    
    for col, diffs in result['differences_by_column'].items():
        summary.append(f"  {col}: {len(diffs)} differences")
        # Show first few differences
        for i, (idx, val1, val2) in enumerate(diffs[:3]):
            summary.append(f"    Row {idx}: '{val1}' vs '{val2}'")
        if len(diffs) > 3:
            summary.append(f"    ... and {len(diffs) - 3} more")
    
    return '\n'.join(summary)

# Convenience functions
def fast_compare(df1: pd.DataFrame, df2: pd.DataFrame, **kwargs) -> bool:
    """Quick boolean check if DataFrames are equal"""
    return compare_dataframes(df1, df2, **kwargs)['equal']

def diff_report(df1: pd.DataFrame, df2: pd.DataFrame, **kwargs) -> str:
    """Quick difference report"""
    result = compare_dataframes(df1, df2, **kwargs)
    return quick_diff_summary(result)

# Example usage
if __name__ == "__main__":
    # Sample data
    df1 = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'David'],
        'value': [1.0, 2.0, 3.0, 4.0]
    })
    
    df2 = pd.DataFrame({
        'id': ['1', '2', '3', '4'],
        'name': [' alice ', 'BOB', 'Charlie', 'David'],
        'value': [1.001, 2.0, 3.0, 4.0]
    })
    
    # Quick comparison
    print("Are equal:", fast_compare(df1, df2, numeric_tol=0.01))
    
    # Detailed report
    print("\nDetailed report:")
    print(diff_report(df1, df2, numeric_tol=0.01, key_columns=['id']))
    
    # Parallel comparison (for large datasets)
    # result = compare_dataframes(df1, df2, n_jobs=4, chunk_size=5000)

class DataAligner:
    """Aligns data between PCDS and AWS based on migration analysis results."""
    
    def __init__(self, boto3_session: Optional[boto3.Session] = None):
        self.session = boto3_session or boto3.Session()
    
    def load_migration_metadata(self, analysis_pickle_path: str, 
                               stats_comparison_path: str) -> Tuple[Dict, Dict]:
        """Load migration analysis and statistics comparison results."""
        import pickle
        
        # Load migration analysis
        with open(analysis_pickle_path, 'rb') as f:
            migration_data = pickle.load(f)
        
        # Load statistics comparison
        with open(stats_comparison_path, 'r') as f:
            stats_data = json.load(f)
        
        return migration_data, stats_data
    
    def get_aligned_columns(self, table_name: str, migration_data: Dict, 
                           stats_data: Dict) -> Tuple[List[str], Dict[str, str]]:
        """Get columns that should be included in aligned comparison."""
        table_meta = migration_data.get(table_name, {})
        
        # Get column mapping from migration analysis
        pcds_meta = table_meta.get('pcds_meta', {})
        column_df = pcds_meta.get('column', pd.DataFrame())
        
        if column_df.empty:
            return [], {}
        
        # Build initial column mapping
        column_mapping = {}
        for _, row in column_df.iterrows():
            pcds_col = row['column_name']
            aws_col = row.get('aws_colname')
            if pd.notna(aws_col) and aws_col != '':
                column_mapping[pcds_col] = aws_col
        
        # Filter out columns with statistical mismatches
        mismatched_columns = set(stats_data.get('mismatched_columns', []))
        
        # Remove mismatched columns from mapping
        aligned_mapping = {
            pcds_col: aws_col 
            for pcds_col, aws_col in column_mapping.items()
            if aws_col not in mismatched_columns
        }
        
        aligned_columns = list(aligned_mapping.values())
        
        print(f"Table {table_name}: {len(column_mapping)} total columns, "
              f"{len(aligned_columns)} aligned columns after removing {len(mismatched_columns)} mismatched")
        
        return aligned_columns, aligned_mapping
    
    def get_aligned_date_filter(self, table_name: str, migration_data: Dict) -> Optional[str]:
        """Get date filter to exclude mismatched time periods."""
        table_meta = migration_data.get(table_name, {})
        
        # Get time exclusions from migration analysis
        next_data_file = "migration_next_data.json"  # This should be from your migration analysis
        try:
            with open(next_data_file, 'r') as f:
                next_data = json.load(f)
            
            table_next = next_data.get(table_name, {})
            time_excludes = table_next.get('time_excludes', '')
            
            if time_excludes:
                excluded_dates = time_excludes.split('; ')
                # Create NOT IN clause for excluded dates
                date_list = "', '".join(excluded_dates)
                return f"DATE(_upload_date) NOT IN ('{date_list}')"
            
        except FileNotFoundError:
            print("Warning: Migration next data file not found for date filtering")
        
        return None
    
    def load_aligned_s3_data(self, s3_path: str, aligned_columns: List[str],
                            upload_cutoff: str, date_filter: Optional[str] = None) -> pd.DataFrame:
        """Load S3 data with column and date alignment."""
        # Build column selection
        columns_sql = ', '.join([f's."{col}"' for col in aligned_columns])
        
        # Build WHERE clause
        where_clauses = [f"s._upload_timestamp <= '{upload_cutoff}'"]
        if date_filter:
            where_clauses.append(date_filter.replace('_upload_date', 's._upload_date'))
        
        where_sql = ' AND '.join(where_clauses)
        
        query = f"""
        SELECT {columns_sql}
        FROM s3object s 
        WHERE {where_sql}
        """
        
        try:
            df = wr.s3.select_query(
                sql=query,
                path=s3_path,
                input_serialization="Parquet",
                boto3_session=self.session
            )
            print(f"Loaded {len(df)} aligned rows from S3")
            return df
        except Exception as e:
            print(f"Error loading S3 data: {e}")
            return pd.DataFrame()
    
    def load_aligned_athena_data(self, database: str, table: str, 
                                aligned_columns: List[str], upload_cutoff: str,
                                date_filter: Optional[str] = None) -> pd.DataFrame:
        """Load Athena data with column and date alignment."""
        # Build column selection
        columns_sql = ', '.join([f'"{col}"' for col in aligned_columns])
        
        # Build WHERE clause
        where_clauses = [f"_upload_timestamp <= TIMESTAMP '{upload_cutoff}'"]
        if date_filter:
            where_clauses.append(date_filter)
        
        where_sql = ' AND '.join(where_clauses)
        
        query = f"""
        SELECT {columns_sql}
        FROM {database}.{table}
        WHERE {where_sql}
        ORDER BY {aligned_columns[0] if aligned_columns else '_upload_timestamp'}
        """
        
        try:
            df = wr.athena.read_sql_query(
                sql=query,
                database=database,
                boto3_session=self.session
            )
            print(f"Loaded {len(df)} aligned rows from Athena")
            return df
        except Exception as e:
            print(f"Error loading Athena data: {e}")
            return pd.DataFrame()
    
    def align_and_compare_table(self, table_name: str, s3_path: str,
                               athena_database: str, athena_table: str,
                               migration_data: Dict, stats_data: Dict,
                               upload_cutoff: str) -> Dict[str, any]:
        """Perform complete alignment and comparison for a table."""
        print(f"\n=== Aligning data for {table_name} ===")
        
        # Get aligned columns
        aligned_columns, column_mapping = self.get_aligned_columns(
            table_name, migration_data, stats_data
        )
        
        if not aligned_columns:
            return {'status': 'failed', 'reason': 'No aligned columns found'}
        
        # Get date filter for time alignment
        date_filter = self.get_aligned_date_filter(table_name, migration_data)
        if date_filter:
            print(f"Applying date filter: {date_filter}")
        
        # Load aligned data from both sources
        s3_df = self.load_aligned_s3_data(
            s3_path, aligned_columns, upload_cutoff, date_filter
        )
        
        athena_df = self.load_aligned_athena_data(
            athena_database, athena_table, aligned_columns, upload_cutoff, date_filter
        )
        
        if s3_df.empty or athena_df.empty:
            return {'status': 'failed', 'reason': 'Empty datasets after alignment'}
        
        # Perform final comparison
        comparison_result = self._compare_aligned_data(s3_df, athena_df, table_name)
        
        return {
            'status': 'success',
            'table_name': table_name,
            'aligned_columns': aligned_columns,
            'column_mapping': column_mapping,
            'date_filter_applied': date_filter is not None,
            's3_row_count': len(s3_df),
            'athena_row_count': len(athena_df),
            'comparison_result': comparison_result
        }
    
    def _compare_aligned_data(self, s3_df: pd.DataFrame, athena_df: pd.DataFrame,
                             table_name: str) -> Dict[str, any]:
        """Compare aligned datasets and return summary."""
        # Basic shape comparison
        shape_match = s3_df.shape == athena_df.shape
        
        # Row count comparison
        row_count_match = len(s3_df) == len(athena_df)
        
        # Column-wise comparison for common columns
        common_columns = set(s3_df.columns) & set(athena_df.columns)
        column_comparisons = {}
        
        for col in common_columns:
            try:
                if pd.api.types.is_numeric_dtype(s3_df[col]):
                    # Numeric comparison with tolerance
                    values_match = pd.testing.assert_series_equal(
                        s3_df[col].sort_values().reset_index(drop=True),
                        athena_df[col].sort_values().reset_index(drop=True),
                        check_exact=False, rtol=1e-5, check_names=False
                    ) is None
                else:
                    # Exact comparison for non-numeric
                    values_match = (
                        s3_df[col].sort_values().reset_index(drop=True).equals(
                            athena_df[col].sort_values().reset_index(drop=True)
                        )
                    )
                
                column_comparisons[col] = {'match': True, 'values_match': values_match}
                
            except Exception as e:
                column_comparisons[col] = {'match': False, 'error': str(e)}
        
        # Calculate overall match percentage
        successful_comparisons = sum(1 for comp in column_comparisons.values() 
                                   if comp.get('match', False) and comp.get('values_match', False))
        match_percentage = (successful_comparisons / len(common_columns)) * 100 if common_columns else 0
        
        return {
            'shape_match': shape_match,
            'row_count_match': row_count_match,
            'common_columns_count': len(common_columns),
            'successful_matches': successful_comparisons,
            'match_percentage': match_percentage,
            'column_details': column_comparisons
        }


def main():
    """Example usage of data alignment."""
    from migration_uploader import UploadMetadataReader
    
    # Configuration
    metadata_file = "upload_metadata_20250603_143015.json"
    analysis_pickle = "migration_analysis.pkl"
    stats_comparison = "table_stats_comparison.json"
    athena_database = "your_database"
    s3_bucket = "your-migration-bucket"
    
    # Initialize aligner
    aligner = DataAligner()
    
    # Load metadata
    upload_metadata = UploadMetadataReader.load_upload_metadata(metadata_file)
    migration_data, stats_data = aligner.load_migration_metadata(
        analysis_pickle, stats_comparison
    )
    
    # Process each table
    alignment_results = {}
    
    for table_name, upload_info in upload_metadata.uploads.items():
        s3_path = f"s3://{s3_bucket}/{upload_info['s3_path']}"
        upload_cutoff = upload_info['upload_timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            result = aligner.align_and_compare_table(
                table_name=table_name,
                s3_path=s3_path,
                athena_database=athena_database,
                athena_table=table_name,
                migration_data=migration_data,
                stats_data=stats_data,
                upload_cutoff=upload_cutoff
            )
            
            alignment_results[table_name] = result
            
            if result['status'] == 'success':
                comp = result['comparison_result']
                print(f"\n{table_name} Alignment Results:")
                print(f"  Row count match: {comp['row_count_match']}")
                print(f"  Match percentage: {comp['match_percentage']:.1f}%")
                print(f"  Aligned columns: {len(result['aligned_columns'])}")
            else:
                print(f"\n{table_name} Alignment Failed: {result['reason']}")
        
        except Exception as e:
            print(f"Error aligning {table_name}: {e}")
            alignment_results[table_name] = {'status': 'error', 'error': str(e)}
    
    # Save alignment summary
    with open("data_alignment_results.json", 'w') as f:
        json.dump(alignment_results, f, indent=2, default=str)
    
    print("\nAlignment complete. Results saved to data_alignment_results.json")


if __name__ == '__main__':
    main()