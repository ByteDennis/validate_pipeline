#!/usr/bin/env python3
'''
Data Alignment Component
Simplified version of wor_erapmoc.py
'''

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from loguru import logger
from confection import Config

from utils.aws import S3Utils, execute_comparison_query
from utils.types import MetaConfig, AlignmentResult

class DataAligner:
    '''Data alignment component for final validation.'''
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.s3_utils = S3Utils()
        
    def _load_config(self) -> MetaConfig:
        config_dict = Config().from_disk(self.config_path)
        return MetaConfig(**config_dict)
    
    def align_table_data(self, table_name: str, upload_metadata: Dict[str, Any],
                        stats_comparison: Dict[str, Any],
                        filter_mismatched: bool = True) -> AlignmentResult:
        '''Align data for a single table.'''
        logger.info(f'Aligning data for {table_name}')
        
        upload_info = upload_metadata['uploads'][table_name]
        comparison_info = stats_comparison.get(table_name, {})
        
        # Get aligned columns (remove mismatched if requested)
        aligned_columns = self._get_aligned_columns(
            comparison_info, filter_mismatched
        )
        
        if not aligned_columns:
            return AlignmentResult(
                table_name=table_name,
                status='failed',
                aligned_columns=[],
                s3_row_count=0,
                athena_row_count=0,
                match_percentage=0.0,
                comparison_details={'error': 'No aligned columns found'}
            )
        
        # Load data with aligned columns only
        s3_df = self._load_aligned_s3_data(
            upload_info['s3_path'], aligned_columns
        )
        
        athena_df = self._load_aligned_athena_data(
            table_name, aligned_columns, upload_info['upload_timestamp']
        )
        
        # Compare aligned data
        comparison_result = self._compare_aligned_data(s3_df, athena_df)
        
        return AlignmentResult(
            table_name=table_name,
            status='success',
            aligned_columns=aligned_columns,
            s3_row_count=len(s3_df),
            athena_row_count=len(athena_df),
            match_percentage=comparison_result['match_percentage'],
            comparison_details=comparison_result
        )
    
    def _get_aligned_columns(self, comparison_info: Dict[str, Any],
                           filter_mismatched: bool) -> List[str]:
        '''Get list of aligned columns.'''
        if not comparison_info or 'detailed_results' not in comparison_info:
            return []
        
        detailed_results = comparison_info['detailed_results']
        
        if filter_mismatched:
            # Only include matching columns
            aligned = [
                result['column_name'] 
                for result in detailed_results 
                if result.get('stats_match', False)
            ]
        else:
            # Include all columns
            aligned = [result['column_name'] for result in detailed_results]
        
        return aligned
    
    def _load_aligned_s3_data(self, s3_path: str, columns: List[str]) -> pd.DataFrame:
        '''Load S3 data with only aligned columns.'''
        df = self.s3_utils.read_dataframe(s3_path)
        
        # Select only aligned columns that exist
        available_columns = [col for col in columns if col in df.columns]
        return df[available_columns]
    
    def _load_aligned_athena_data(self, table_name: str, columns: List[str],
                                upload_cutoff: str) -> pd.DataFrame:
        '''Load Athena data with only aligned columns.'''
        # Build column selection query
        column_sql = ', '.join(f'"{col}"' for col in columns)
        
        query = f'''
            SELECT {column_sql}
            FROM migration_database.{table_name}
            WHERE _upload_timestamp <= TIMESTAMP '{upload_cutoff}'
        '''
        
        from utils.aws import AthenaUtils
        athena_utils = AthenaUtils()
        return athena_utils.execute_query(query, 'migration_database')
    
    def _compare_aligned_data(self, s3_df: pd.DataFrame, 
                            athena_df: pd.DataFrame) -> Dict[str, Any]:
        '''Compare aligned datasets.'''
        # Basic comparison
        row_count_match = len(s3_df) == len(athena_df)
        shape_match = s3_df.shape == athena_df.shape
        
        # Column-wise comparison
        common_columns = set(s3_df.columns) & set(athena_df.columns)
        successful_matches = 0
        
        for col in common_columns:
            try:
                if pd.api.types.is_numeric_dtype(s3_df[col]):
                    # Numeric comparison with tolerance
                    if len(s3_df) == len(athena_df):
                        s3_sorted = s3_df[col].sort_values().reset_index(drop=True)
                        athena_sorted = athena_df[col].sort_values().reset_index(drop=True)
                        
                        if pd.testing.assert_series_equal(
                            s3_sorted, athena_sorted, 
                            check_exact=False, rtol=1e-5, check_names=False
                        ) is None:
                            successful_matches += 1
                else:
                    # Exact comparison for non-numeric
                    if len(s3_df) == len(athena_df):
                        s3_sorted = s3_df[col].sort_values().reset_index(drop=True)
                        athena_sorted = athena_df[col].sort_values().reset_index(drop=True)
                        
                        if s3_sorted.equals(athena_sorted):
                            successful_matches += 1
                            
            except Exception:
                # Comparison failed for this column
                pass
        
        match_percentage = (successful_matches / len(common_columns)) * 100 if common_columns else 0
        
        return {
            'shape_match': shape_match,
            'row_count_match': row_count_match,
            'common_columns': len(common_columns),
            'successful_matches': successful_matches,
            'match_percentage': match_percentage
        }
    
    def align_all_tables(self, metadata_file: str, stats_comparison_file: str,
                        table_filter: Optional[List[str]] = None,
                        filter_mismatched: bool = True) -> str:
        '''Align all tables.'''
        logger.info('Starting data alignment')
        
        # Load metadata and comparison results
        with open(metadata_file, 'r') as f:
            upload_metadata = json.load(f)
        
        with open(stats_comparison_file, 'r') as f:
            stats_comparison = json.load(f)
        
        # Filter tables
        tables_to_align = list(upload_metadata['uploads'].keys())
        if table_filter:
            tables_to_align = [t for t in tables_to_align if t in table_filter]
        
        alignment_results = {}
        
        # Align each table
        for table_name in tables_to_align:
            try:
                result = self.align_table_data(
                    table_name, upload_metadata, stats_comparison, filter_mismatched
                )
                alignment_results[table_name] = result
                
            except Exception as e:
                logger.error(f'Alignment failed for {table_name}: {e}')
                alignment_results[table_name] = AlignmentResult(
                    table_name=table_name,
                    status='failed',
                    aligned_columns=[],
                    s3_row_count=0,
                    athena_row_count=0,
                    match_percentage=0.0,
                    comparison_details={'error': str(e)}
                )
        
        # Save results
        results_file = self._save_alignment_results(alignment_results)
        
        logger.info(f'Data alignment complete. Results saved to {results_file}')
        return results_file
    
    def _save_alignment_results(self, results: Dict[str, Any]) -> str:
        '''Save alignment results to file.'''
        output_path = Path(self.config.output.folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / 'alignment_results.json'
        
        # Convert to JSON-serializable format
        json_results = {}
        for table_name, result in results.items():
            if hasattr(result, '__dict__'):
                json_results[table_name] = result.__dict__
            else:
                json_results[table_name] = result
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        return str(results_file)