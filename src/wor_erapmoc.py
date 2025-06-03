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
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path


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
            print(f"Warning: Migration next data file not found for date filtering")
        
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
    
    print(f"\nAlignment complete. Results saved to data_alignment_results.json")


if __name__ == '__main__':
    main()