#!/usr/bin/env python3
'''
Migration Analysis Component
Refactored from atem_sisylana.py with simplified structure.
'''

import os
import pickle
import json
from pathlib import Path
from datetime import datetime as dt
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from loguru import logger
from confection import Config
from tqdm import tqdm

from utils.types import MetaConfig, PullStatus, MetaMerge
from utils.database import DatabaseConnector, QueryBuilder, get_table_metadata, get_table_row_count, get_date_distribution
from utils.common import read_excel_input, read_column_mapping, write_csv_report, find_common_mappings, Timer

class MigrationAnalyzer:
    '''Main migration analysis component.'''
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.col_maps = read_column_mapping(self.config.column_maps.file)
        self.table_list = self._load_table_list()
        
    def _load_config(self) -> MetaConfig:
        '''Load configuration.'''
        config_dict = Config().from_disk(self.config_path)
        return MetaConfig(**config_dict)
    
    def _load_table_list(self) -> pd.DataFrame:
        '''Load table list from Excel.'''
        return read_excel_input(
            self.config.input.table.file,
            sheet_name=self.config.input.table.sheet,
            skip_rows=self.config.input.table.skip_rows
        )
    
    def analyze_table(self, table_info: Dict[str, Any]) -> Dict[str, Any]:
        '''Analyze a single table.'''
        pcds_table = table_info['pcds_tbl']
        aws_table = table_info['aws_tbl']
        
        logger.info(f'Analyzing {pcds_table} -> {aws_table}')
        
        result = {
            'pcds_table': pcds_table,
            'aws_table': aws_table,
            'status': PullStatus.SUCCESS,
            'metadata_comparison': {},
            'row_comparison': {},
            'date_comparison': {}
        }
        
        try:
            # Check if column mapping exists
            col_map_key = table_info.get('col_map')
            if not col_map_key or col_map_key not in self.col_maps:
                result['status'] = PullStatus.NO_MAPPING
                result['error'] = f'Column mapping not provided for {col_map_key}'
                return result
            
            # Get PCDS metadata
            try:
                pcds_meta = get_table_metadata('PCDS', pcds_table.split('.')[1], 
                                            svc=pcds_table.split('.')[0])
            except PullStatus.NONEXIST_TABLE:
                result['status'] = PullStatus.NONEXIST_PCDS
                result['error'] = f'PCDS table {pcds_table} does not exist'
                return result
            
            # Get PCDS row count
            try:
                pcds_rows = get_table_row_count('PCDS', pcds_table.split('.')[1],
                                            svc=pcds_table.split('.')[0])
                pcds_row_count = int(pcds_rows.iloc[0]['nrow'])
                
                if pcds_row_count == 0:
                    result['status'] = PullStatus.EMPTY_PCDS
                    result['error'] = f'PCDS table {pcds_table} is empty'
                    return result
                    
            except Exception as e:
                result['status'] = PullStatus.NONEXIST_PCDS
                result['error'] = f'Failed to get PCDS row count: {e}'
                return result
            
            # Get AWS metadata
            try:
                aws_db, aws_tbl = aws_table.split('.', 1)
                aws_meta = get_table_metadata('AWS', aws_tbl, db=aws_db)
            except PullStatus.NONEXIST_TABLE:
                result['status'] = PullStatus.NONEXIST_AWS
                result['error'] = f'AWS table {aws_table} does not exist'
                return result
            
            # Get AWS row count
            try:
                aws_rows = get_table_row_count('AWS', aws_tbl, db=aws_db)
                aws_row_count = int(aws_rows.iloc[0]['nrow'])
                
                if aws_row_count == 0:
                    result['status'] = PullStatus.EMPTY_AWS
                    result['error'] = f'AWS table {aws_table} is empty'
                    return result
                    
            except Exception as e:
                result['status'] = PullStatus.NONEXIST_AWS
                result['error'] = f'Failed to get AWS row count: {e}'
                return result
            
            # Check for date variables if specified
            pcds_date_col = table_info.get('pcds_date_col')
            aws_date_col = table_info.get('aws_date_col')
            
            if pcds_date_col:
                try:
                    pcds_date_dist = get_date_distribution('PCDS', pcds_table.split('.')[1], 
                                                        pcds_date_col, svc=pcds_table.split('.')[0])
                    result['date_comparison']['pcds_dates'] = pcds_date_dist
                except PullStatus.NONEXIST_DATEVAR:
                    result['status'] = PullStatus.NONDATE_PCDS
                    result['error'] = f'Date column {pcds_date_col} not found in PCDS table'
                    return result
            
            if aws_date_col:
                try:
                    aws_date_dist = get_date_distribution('AWS', aws_tbl, aws_date_col, db=aws_db)
                    result['date_comparison']['aws_dates'] = aws_date_dist
                except PullStatus.NONEXIST_DATEVAR:
                    result['status'] = PullStatus.NONDATE_AWS
                    result['error'] = f'Date column {aws_date_col} not found in AWS table'
                    return result
            
            # If we get here, everything is successful
            result['metadata_comparison'] = self._compare_schemas(
                pcds_meta, aws_meta, table_info['col_map']
            )
            
            result['row_comparison'] = {
                'pcds_rows': pcds_row_count,
                'aws_rows': aws_row_count,
                'match': pcds_row_count == aws_row_count
            }
            
            result['pcds_metadata'] = {
                'columns': pcds_meta,
                'row_count': pcds_rows
            }
            result['aws_metadata'] = {
                'columns': aws_meta,
                'row_count': aws_rows
            }
            
        except Exception as e:
            logger.error(f'Unexpected error analyzing {pcds_table}: {e}')
            result['status'] = PullStatus.NONEXIST_PCDS  # Default fallback
            result['error'] = str(e)
        
        return result
    
    def _compare_schemas(self, pcds_df: pd.DataFrame, aws_df: pd.DataFrame,
                        col_map_key: str) -> Dict[str, Any]:
        '''Compare PCDS and AWS schemas.'''
        # Get column mappings
        column_mappings = self.col_maps.get(col_map_key.lower(), {})
        
        # Apply mappings
        pcds_df['aws_colname'] = pcds_df['column_name'].map(column_mappings)
        
        # Find unmapped columns
        unmapped_pcds = pcds_df[pcds_df['aws_colname'].isna()]['column_name'].tolist()
        unmapped_aws = aws_df[~aws_df['column_name'].isin(pcds_df['aws_colname'])]['column_name'].tolist()
        
        # Find automatic mappings
        auto_mappings = find_common_mappings(unmapped_pcds, unmapped_aws)
        
        return {
            'pcds_columns': len(pcds_df),
            'aws_columns': len(aws_df), 
            'mapped_columns': len(column_mappings),
            'unmapped_pcds': unmapped_pcds,
            'unmapped_aws': unmapped_aws,
            'auto_mappings': auto_mappings,
            'column_mappings': column_mappings
        }
    
    def run_analysis(self, table_filter: Optional[List[str]] = None,
                    output_override: Optional[str] = None,
                    row_range: Optional[Tuple[int, int]] = None) -> str:
        '''Run complete analysis.'''
        logger.info('Starting migration analysis')
        
        # Filter tables
        tables_to_analyze = self.table_list
        if table_filter:
            tables_to_analyze = tables_to_analyze[
                tables_to_analyze['pcds_tbl'].isin(table_filter)
            ]
        
        if row_range:
            start_row, end_row = row_range
            tables_to_analyze = tables_to_analyze.iloc[start_row-1:end_row]
        
        # Analyze each table
        results = {}
        
        for _, table_info in tqdm(tables_to_analyze.iterrows(), 
                                desc='Analyzing tables',
                                total=len(tables_to_analyze)):
            
            table_name = table_info['pcds_tbl'].split('.')[1]
            results[table_name] = self.analyze_table(table_info.to_dict())
        
        # Save results
        output_dir = Path(output_override) if output_override else Path(self.config.output.folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pickle_file = output_dir / self.config.output.to_pkl
        with open(pickle_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Generate summary report
        self._generate_summary_report(results, output_dir)
        
        logger.info(f'Analysis complete. Results saved to {pickle_file}')
        return str(pickle_file)
    
    def _generate_summary_report(self, results: Dict[str, Any], output_dir: Path):
        '''Generate CSV summary report.'''
        report_data = []
        
        for table_name, result in results.items():
            row = {
                'Table': table_name,
                'Status': result['status'].value if hasattr(result['status'], 'value') else result['status'],
                'PCDS_Columns': result.get('metadata_comparison', {}).get('pcds_columns', 0),
                'AWS_Columns': result.get('metadata_comparison', {}).get('aws_columns', 0),
                'Mapped_Columns': result.get('metadata_comparison', {}).get('mapped_columns', 0),
                'PCDS_Rows': result.get('row_comparison', {}).get('pcds_rows', 0),
                'AWS_Rows': result.get('row_comparison', {}).get('aws_rows', 0),
                'Row_Match': result.get('row_comparison', {}).get('match', False),
                'Error': result.get('error', '')
            }
            report_data.append(row)
        
        report_file = output_dir / 'analysis_summary.csv'
        write_csv_report(report_data, str(report_file))
        logger.info(f'Summary report saved to {report_file}')