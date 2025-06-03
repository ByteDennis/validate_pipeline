#!/usr/bin/env python3
"""
Column-by-Column Statistics Comparison Script

Compares statistical measures between uploaded PCDS data in S3 and migrated data in Athena
to validate data integrity during migration process.
"""

import pandas as pd
import awswrangler as wr
import boto3
from typing import Dict, Any, Tuple, Optional
import json
from pathlib import Path


class StatisticsComparator:
    """Compares column statistics between S3 (PCDS) and Athena (migrated) data."""
    
    def __init__(self, boto3_session: Optional[boto3.Session] = None):
        self.session = boto3_session or boto3.Session()
    
    def get_s3_data_stats(self, s3_path: str, upload_cutoff: str) -> pd.DataFrame:
        """Get statistics from S3 data with upload cutoff filter."""
        query = f"""
        SELECT * FROM s3object s 
        WHERE s._upload_timestamp <= '{upload_cutoff}'
        """
        
        df = wr.s3.select_query(
            sql=query,
            path=s3_path,
            input_serialization="Parquet",
            boto3_session=self.session
        )
        return self._calculate_column_stats(df, 'S3_PCDS')
    
    def get_athena_data_stats(self, database: str, table: str, upload_cutoff: str) -> pd.DataFrame:
        """Get statistics from Athena table with upload cutoff filter."""
        query = f"""
        SELECT * FROM {database}.{table}
        WHERE _upload_timestamp <= TIMESTAMP '{upload_cutoff}'
        """
        
        df = wr.athena.read_sql_query(
            sql=query,
            database=database,
            boto3_session=self.session
        )
        return self._calculate_column_stats(df, 'Athena_AWS')
    
    def _calculate_column_stats(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Calculate comprehensive column statistics."""
        stats_list = []
        
        for col in df.columns:
            if col.startswith('_upload'):  # Skip metadata columns
                continue
                
            col_data = df[col]
            dtype = str(col_data.dtype)
            
            # Basic stats
            stats = {
                'column_name': col,
                'source': source,
                'data_type': dtype,
                'row_count': len(col_data),
                'null_count': col_data.isnull().sum(),
                'null_percentage': (col_data.isnull().sum() / len(col_data)) * 100,
                'unique_count': col_data.nunique(),
                'unique_percentage': (col_data.nunique() / len(col_data)) * 100
            }
            
            # Numeric stats
            if pd.api.types.is_numeric_dtype(col_data):
                stats.update({
                    'min_value': col_data.min() if not col_data.empty else None,
                    'max_value': col_data.max() if not col_data.empty else None,
                    'mean_value': col_data.mean() if not col_data.empty else None,
                    'std_value': col_data.std() if not col_data.empty else None,
                    'median_value': col_data.median() if not col_data.empty else None
                })
            
            # String stats
            elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
                non_null_data = col_data.dropna().astype(str)
                stats.update({
                    'min_length': non_null_data.str.len().min() if not non_null_data.empty else None,
                    'max_length': non_null_data.str.len().max() if not non_null_data.empty else None,
                    'avg_length': non_null_data.str.len().mean() if not non_null_data.empty else None
                })
            
            # Date stats
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                stats.update({
                    'min_date': col_data.min() if not col_data.empty else None,
                    'max_date': col_data.max() if not col_data.empty else None,
                    'date_range_days': (col_data.max() - col_data.min()).days if not col_data.empty else None
                })
            
            stats_list.append(stats)
        
        return pd.DataFrame(stats_list)
    
    def compare_statistics(self, s3_stats: pd.DataFrame, athena_stats: pd.DataFrame) -> pd.DataFrame:
        """Compare statistics between S3 and Athena data."""
        # Merge on column name
        comparison = pd.merge(
            s3_stats, athena_stats,
            on='column_name', suffixes=('_s3', '_athena'),
            how='outer', indicator=True
        )
        
        # Calculate differences for numeric columns
        numeric_cols = ['row_count', 'null_count', 'unique_count', 'min_value', 'max_value', 'mean_value']
        
        for col in numeric_cols:
            s3_col = f'{col}_s3'
            athena_col = f'{col}_athena'
            
            if s3_col in comparison.columns and athena_col in comparison.columns:
                comparison[f'{col}_diff'] = comparison[athena_col] - comparison[s3_col]
                comparison[f'{col}_match'] = (
                    abs(comparison[f'{col}_diff'].fillna(0)) < 0.001
                )
        
        # Overall match flag
        comparison['stats_match'] = (
            (comparison['row_count_match'].fillna(True)) &
            (comparison['null_count_match'].fillna(True)) &
            (comparison['unique_count_match'].fillna(True))
        )
        
        return comparison
    
    def generate_comparison_report(self, table_name: str, s3_path: str, 
                                 athena_database: str, athena_table: str,
                                 upload_cutoff: str) -> Dict[str, Any]:
        """Generate complete comparison report."""
        print(f"Comparing statistics for {table_name}...")
        
        # Get statistics from both sources
        s3_stats = self.get_s3_data_stats(s3_path, upload_cutoff)
        athena_stats = self.get_athena_data_stats(athena_database, athena_table, upload_cutoff)
        
        # Compare statistics
        comparison = self.compare_statistics(s3_stats, athena_stats)
        
        # Summary metrics
        total_columns = len(comparison)
        matching_columns = comparison['stats_match'].sum()
        mismatched_columns = comparison[~comparison['stats_match']]['column_name'].tolist()
        
        report = {
            'table_name': table_name,
            'total_columns': total_columns,
            'matching_columns': matching_columns,
            'mismatch_percentage': ((total_columns - matching_columns) / total_columns) * 100,
            'mismatched_columns': mismatched_columns,
            'detailed_comparison': comparison.to_dict('records')
        }
        
        return report


def main():
    """Example usage of statistics comparison."""
    from migration_uploader import UploadMetadataReader
    
    # Configuration
    metadata_file = "upload_metadata_20250603_143015.json"
    s3_bucket = "your-migration-bucket"
    athena_database = "your_database"
    
    # Initialize comparator
    comparator = StatisticsComparator()
    
    # Load upload metadata for cutoff time
    metadata = UploadMetadataReader.load_upload_metadata(metadata_file)
    
    # Compare each migrated table
    for table_name, upload_info in metadata.uploads.items():
        s3_path = f"s3://{s3_bucket}/{upload_info['s3_path']}"
        upload_cutoff = upload_info['upload_timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            report = comparator.generate_comparison_report(
                table_name=table_name,
                s3_path=s3_path,
                athena_database=athena_database,
                athena_table=table_name,
                upload_cutoff=upload_cutoff
            )
            
            print(f"\n=== {table_name} Statistics Comparison ===")
            print(f"Total columns: {report['total_columns']}")
            print(f"Matching columns: {report['matching_columns']}")
            print(f"Mismatch percentage: {report['mismatch_percentage']:.2f}%")
            
            if report['mismatched_columns']:
                print(f"Mismatched columns: {', '.join(report['mismatched_columns'])}")
            
            # Save detailed report
            with open(f"{table_name}_stats_comparison.json", 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
        except Exception as e:
            print(f"Failed to compare {table_name}: {e}")


if __name__ == '__main__':
    main()