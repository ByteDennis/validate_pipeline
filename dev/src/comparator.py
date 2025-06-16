#!/usr/bin/env python3
'''
Statistics Comparison Component
Simplified version of sats_erapmoc.py
'''

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from loguru import logger
from confection import Config

from utils.aws import S3Utils, execute_comparison_query
from utils.database import DatabaseConnector
from utils.data import StatisticsCalculator, DataProcessor
from utils.types import MetaConfig, StatisticsComparison

class StatisticsComparator:
    '''Statistics comparison component.'''
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.stats_calc = StatisticsCalculator()
        self.s3_utils = S3Utils()
        
    def _load_config(self) -> MetaConfig:
        config_dict = Config().from_disk(self.config_path)
        return MetaConfig(**config_dict)
    
    def compare_table_statistics(self, table_name: str, 
                                upload_metadata: Dict[str, Any]) -> StatisticsComparison:
        '''Compare statistics for a single table.'''
        logger.info(f'Comparing statistics for {table_name}')
        
        upload_info = upload_metadata['uploads'][table_name]
        upload_cutoff = upload_info['upload_timestamp']
        
        # Get PCDS data from S3
        s3_path = upload_info['s3_path']
        pcds_df = self.s3_utils.read_dataframe(s3_path)
        
        # Get AWS data from Athena
        aws_df = execute_comparison_query(
            table_name, 'migration_database', upload_cutoff
        )
        
        # Calculate statistics
        pcds_stats = self.stats_calc.calculate_column_stats(pcds_df, 'PCDS')
        aws_stats = self.stats_calc.calculate_column_stats(aws_df, 'AWS')
        
        # Compare statistics
        comparison = self.stats_calc.compare_statistics(pcds_stats, aws_stats)
        
        # Find mismatched columns
        mismatched = comparison[~comparison['stats_match']]['column_name'].tolist()
        
        return StatisticsComparison(
            table_name=table_name,
            total_columns=len(comparison),
            matching_columns=comparison['stats_match'].sum(),
            mismatched_columns=mismatched,
            mismatch_percentage=(len(mismatched) / len(comparison)) * 100,
            detailed_results=comparison.to_dict('records')
        )
    
    def compare_all_tables(self, metadata_file: str,
                          table_filter: Optional[List[str]] = None,
                          output_dir: Optional[str] = None) -> str:
        '''Compare statistics for all tables.'''
        logger.info('Starting statistics comparison')
        
        # Load upload metadata
        with open(metadata_file, 'r') as f:
            upload_metadata = json.load(f)
        
        # Filter tables
        tables_to_compare = list(upload_metadata['uploads'].keys())
        if table_filter:
            tables_to_compare = [t for t in tables_to_compare if t in table_filter]
        
        comparison_results = {}
        
        # Compare each table
        for table_name in tables_to_compare:
            try:
                result = self.compare_table_statistics(table_name, upload_metadata)
                comparison_results[table_name] = result
                
            except Exception as e:
                logger.error(f'Comparison failed for {table_name}: {e}')
                comparison_results[table_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Save results
        results_file = self._save_comparison_results(comparison_results, output_dir)
        
        logger.info(f'Statistics comparison complete. Results saved to {results_file}')
        return results_file
    
    def _save_comparison_results(self, results: Dict[str, Any], 
                               output_dir: Optional[str] = None) -> str:
        '''Save comparison results to file.'''
        output_path = Path(output_dir) if output_dir else Path(self.config.output.folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / 'stats_comparison_results.json'
        
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