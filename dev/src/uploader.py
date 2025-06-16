#!/usr/bin/env python3
'''
Data Upload Component  
Simplified version of atad_daolpu.py
'''

import pickle
import json
from pathlib import Path
from datetime import datetime as dt
from typing import Dict, Any, List, Optional

import pandas as pd
from loguru import logger
from confection import Config

from utils.aws import AWSManager, S3Utils, upload_migration_data, create_athena_table_for_migration
from utils.database import DatabaseConnector
from utils.data import DataProcessor, add_migration_metadata
from utils.types import MetaConfig, UploadMetadata

class DataUploader:
    '''Data upload component for migrating PCDS data to AWS.'''
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.aws_manager = AWSManager()
        self.data_processor = DataProcessor()
        self.upload_metadata = []
        
    def _load_config(self) -> MetaConfig:
        config_dict = Config().from_disk(self.config_path)
        return MetaConfig(**config_dict)
    
    def upload_table_data(self, table_name: str, analysis_result: Dict[str, Any],
                         partition_strategy: str = 'date') -> Dict[str, Any]:
        '''Upload data for a single table.'''
        logger.info(f'Uploading data for {table_name}')
        
        # Extract data from PCDS
        pcds_table = analysis_result['pcds_table']
        service, table = pcds_table.split('.', 1)
        
        # Get column mappings from analysis
        column_mappings = analysis_result['metadata_comparison']['column_mappings']
        
        # Extract data
        pcds_connector = DatabaseConnector('PCDS')
        query = f'SELECT * FROM {table}'
        df = pcds_connector.query(query, svc=service)
        
        # Process data
        df_processed = self.data_processor.process_pcds_data(
            df, {}, column_mappings  # TODO: Add column types from analysis
        )
        
        # Add migration metadata
        df_with_meta = add_migration_metadata(df_processed, table_name)
        
        # Upload to S3
        upload_info = upload_migration_data(
            df_with_meta, 
            table_name,
            self.config.aws.s3_bucket,
            partition_strategy
        )
        
        # Create Athena table
        create_athena_table_for_migration(
            table_name,
            upload_info['s3_path'],
            'migration_database',  # TODO: Get from config
            df_with_meta.head(1)
        )
        
        # Record metadata
        metadata = UploadMetadata(
            table_name=table_name,
            upload_timestamp=upload_info['upload_timestamp'].isoformat(),
            upload_date=upload_info['upload_timestamp'].strftime('%Y-%m-%d'),
            s3_path=upload_info['s3_path'],
            row_count=upload_info['row_count'],
            file_size_mb=upload_info.get('estimated_size_mb')
        )
        
        self.upload_metadata.append(metadata)
        
        return {
            'status': 'success',
            'table_name': table_name,
            'upload_info': upload_info,
            'metadata': metadata
        }
    
    def run_migration(self, analysis_output_path: str,
                     table_filter: Optional[List[str]] = None,
                     partition_strategy: str = 'date') -> str:
        '''Run complete data migration.'''
        logger.info('Starting data migration')
        
        # Load analysis results
        with open(analysis_output_path, 'rb') as f:
            analysis_results = pickle.load(f)
        
        # Filter tables
        if table_filter:
            analysis_results = {
                k: v for k, v in analysis_results.items() 
                if k in table_filter
            }
        
        upload_results = {}
        
        # Upload each table
        for table_name, analysis_result in analysis_results.items():
            try:
                result = self.upload_table_data(
                    table_name, analysis_result, partition_strategy
                )
                upload_results[table_name] = result
                
            except Exception as e:
                logger.error(f'Upload failed for {table_name}: {e}')
                upload_results[table_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Save upload metadata
        metadata_file = self._save_upload_metadata()
        
        logger.info(f'Migration complete. Metadata saved to {metadata_file}')
        return metadata_file
    
    def _save_upload_metadata(self) -> str:
        '''Save upload metadata to file.'''
        timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
        metadata_file = Path(self.config.output.folder) / f'upload_metadata_{timestamp}.json'
        
        metadata_dict = {
            'upload_session': {
                'start_time': dt.now().isoformat(),
                'total_tables': len(self.upload_metadata)
            },
            'uploads': {
                meta.table_name: {
                    'upload_timestamp': meta.upload_timestamp,
                    'upload_date': meta.upload_date,
                    's3_path': meta.s3_path,
                    'row_count': meta.row_count,
                    'file_size_mb': meta.file_size_mb
                }
                for meta in self.upload_metadata
            }
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        return str(metadata_file)