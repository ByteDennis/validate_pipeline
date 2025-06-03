#!/usr/bin/env python3
"""
PCDS to AWS Data Migration Uploader

This tool leverages the output from MigrationAnalyzer to extract data from PCDS tables,
compress it to Parquet format, and upload to AWS S3 using awswrangler. It handles
data type conversions, compression optimization, and batch processing for large datasets.

Features:
- Reads MigrationAnalyzer output for table metadata
- Extracts data from PCDS tables with pagination
- Converts data types according to mapping rules
- Compresses data to Parquet format for efficient storage
- Uploads to AWS S3 with proper partitioning
- Handles large datasets with chunking and memory management
- Comprehensive logging and error handling

Dependencies:
- awswrangler
- pandas
- pyarrow
- boto3

Author: Migration Team
Date: 2025
"""

import os
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime as dt
from typing import Dict, List, Optional, Tuple, Any, Iterator
from io import BytesIO
import tempfile
import shutil

import pandas as pd
import awswrangler as wr
import boto3
from loguru import logger
from confection import Config
from tqdm import tqdm

import utils  # Assuming the same utils module from MigrationAnalyzer


class DataTypeConverter:
    """Handles data type conversions between PCDS and AWS formats."""
    
    PCDS_TO_PANDAS_MAPPING = {
        'NUMBER': 'float64',
        'VARCHAR2': 'string',
        'CHAR': 'string',
        'DATE': 'datetime64[ns]',
        'TIMESTAMP': 'datetime64[ns]',
        'CLOB': 'string',
        'BLOB': 'object'
    }
    
    @classmethod
    def convert_pcds_column_types(cls, df: pd.DataFrame, 
                                 column_types: Dict[str, str]) -> pd.DataFrame:
        """
        Convert PCDS data types to pandas-compatible types.
        
        Args:
            df: DataFrame to convert
            column_types: Mapping of column names to PCDS data types
            
        Returns:
            DataFrame with converted types
        """
        df_converted = df.copy()
        
        for column, pcds_type in column_types.items():
            if column not in df_converted.columns:
                continue
                
            try:
                # Handle NUMBER types
                if pcds_type.startswith('NUMBER'):
                    if ',' in pcds_type:  # Has decimal places
                        df_converted[column] = pd.to_numeric(
                            df_converted[column], errors='coerce'
                        ).astype('float64')
                    else:  # Integer
                        df_converted[column] = pd.to_numeric(
                            df_converted[column], errors='coerce'
                        ).astype('Int64')
                
                # Handle string types
                elif pcds_type.startswith(('VARCHAR2', 'CHAR', 'CLOB')):
                    df_converted[column] = df_converted[column].astype('string')
                
                # Handle date/timestamp types
                elif pcds_type in ('DATE', 'TIMESTAMP') or pcds_type.startswith('TIMESTAMP'):
                    df_converted[column] = pd.to_datetime(
                        df_converted[column], errors='coerce'
                    )
                
                # Handle boolean types (if any)
                elif pcds_type.upper() in ('BOOLEAN', 'BOOL'):
                    df_converted[column] = df_converted[column].astype('boolean')
                    
            except Exception as e:
                logger.warning(f"Failed to convert column {column} of type {pcds_type}: {e}")
                
        return df_converted
    
    @classmethod
    def optimize_dtypes_for_parquet(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize data types for efficient Parquet storage.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            DataFrame with optimized types
        """
        df_optimized = df.copy()
        
        for column in df_optimized.columns:
            col_data = df_optimized[column]
            
            # Optimize integer columns
            if pd.api.types.is_integer_dtype(col_data):
                # Try to downcast to smaller integer types
                df_optimized[column] = pd.to_numeric(col_data, downcast='integer')
            
            # Optimize float columns
            elif pd.api.types.is_float_dtype(col_data):
                # Try to downcast to smaller float types
                df_optimized[column] = pd.to_numeric(col_data, downcast='float')
            
            # Convert object columns to category if beneficial
            elif col_data.dtype == 'object':
                unique_ratio = col_data.nunique() / len(col_data)
                if unique_ratio < 0.5 and col_data.nunique() < 1000:
                    df_optimized[column] = col_data.astype('category')
        
        return df_optimized


class PCDSDataExtractor:
    """Handles data extraction from PCDS databases."""
    
    def __init__(self, chunk_size: int = 50000, max_memory_mb: int = 512):
        """
        Initialize the data extractor.
        
        Args:
            chunk_size: Number of rows to fetch per chunk
            max_memory_mb: Maximum memory usage in MB before forcing compression
        """
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb
        self.db_connector = utils.DatabaseConnector('PCDS')
    
    def get_table_row_count(self, table_name: str, service: Optional[str] = None) -> int:
        """
        Get total row count for a table.
        
        Args:
            table_name: Name of the table
            service: PCDS service name
            
        Returns:
            Total number of rows
        """
        query = f"SELECT COUNT(*) as total_rows FROM {table_name}"
        result = self.db_connector.query(query, svc=service)
        return int(result.iloc[0]['total_rows'])
    
    def extract_table_data_chunked(self, table_name: str, 
                                  columns: Optional[List[str]] = None,
                                  where_clause: Optional[str] = None,
                                  service: Optional[str] = None) -> Iterator[pd.DataFrame]:
        """
        Extract table data in chunks to manage memory usage.
        
        Args:
            table_name: Name of the table to extract
            columns: List of columns to extract (None for all)
            where_clause: Optional WHERE clause for filtering
            service: PCDS service name
            
        Yields:
            DataFrames containing chunks of the table data
        """
        # Get total row count
        total_rows = self.get_table_row_count(table_name, service)
        
        # Build base query
        column_list = '*' if columns is None else ', '.join(columns)
        base_query = f"SELECT {column_list} FROM {table_name}"
        
        if where_clause:
            base_query += f" WHERE {where_clause}"
        
        # Extract data in chunks
        for offset in range(0, total_rows, self.chunk_size):
            query = f"{base_query} OFFSET {offset} ROWS FETCH NEXT {self.chunk_size} ROWS ONLY"
            
            try:
                chunk_df = self.db_connector.query(query, svc=service)
                if not chunk_df.empty:
                    yield chunk_df
                else:
                    break
            except Exception as e:
                logger.error(f"Failed to extract chunk at offset {offset}: {e}")
                break
    
    def extract_table_data_paginated(self, table_name: str, 
                                   date_column: str,
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None,
                                   columns: Optional[List[str]] = None,
                                   service: Optional[str] = None) -> Iterator[pd.DataFrame]:
        """
        Extract table data using date-based pagination.
        
        Args:
            table_name: Name of the table to extract
            date_column: Date column for pagination
            start_date: Start date for extraction (YYYY-MM-DD)
            end_date: End date for extraction (YYYY-MM-DD)
            columns: List of columns to extract
            service: PCDS service name
            
        Yields:
            DataFrames containing date-partitioned data
        """
        # Get date range if not provided
        if not start_date or not end_date:
            date_query = f"SELECT MIN({date_column}) as min_date, MAX({date_column}) as max_date FROM {table_name}"
            date_result = self.db_connector.query(date_query, svc=service)
            start_date = start_date or date_result.iloc[0]['min_date'].strftime('%Y-%m-%d')
            end_date = end_date or date_result.iloc[0]['max_date'].strftime('%Y-%m-%d')
        
        # Build column list
        column_list = '*' if columns is None else ', '.join(columns)
        
        # Extract data by date ranges (daily chunks)
        current_date = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        
        while current_date <= end_date_dt:
            date_str = current_date.strftime('%Y-%m-%d')
            next_date_str = (current_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            
            query = f"""
            SELECT {column_list} 
            FROM {table_name} 
            WHERE {date_column} >= DATE '{date_str}' 
            AND {date_column} < DATE '{next_date_str}'
            """
            
            try:
                chunk_df = self.db_connector.query(query, svc=service)
                if not chunk_df.empty:
                    yield chunk_df
            except Exception as e:
                logger.warning(f"Failed to extract data for date {date_str}: {e}")
            
            current_date += pd.Timedelta(days=1)


class UploadMetadata:
    """Stores metadata about upload operations."""
    
    def __init__(self):
        self.uploads = {}
        self.upload_start_time = None
        self.upload_end_time = None
    
    def start_upload_session(self):
        """Mark the start of an upload session."""
        self.upload_start_time = dt.now()
        logger.info(f"Upload session started at {self.upload_start_time}")
    
    def end_upload_session(self):
        """Mark the end of an upload session."""
        self.upload_end_time = dt.now()
        logger.info(f"Upload session ended at {self.upload_end_time}")
    
    def add_upload_record(self, table_name: str, s3_path: str, 
                         row_count: int, file_size_mb: float = None):
        """
        Record metadata for a completed upload.
        
        Args:
            table_name: Name of the uploaded table
            s3_path: S3 path where data was uploaded
            row_count: Number of rows uploaded
            file_size_mb: Size of uploaded data in MB
        """
        upload_time = dt.now()
        self.uploads[table_name] = {
            'upload_timestamp': upload_time,
            'upload_date': upload_time.strftime('%Y-%m-%d'),
            'upload_datetime_str': upload_time.strftime('%Y-%m-%d %H:%M:%S'),
            's3_path': s3_path,
            'row_count': row_count,
            'file_size_mb': file_size_mb,
            'session_start': self.upload_start_time,
            'session_end': None  # Will be updated when session ends
        }
        logger.info(f"Recorded upload for {table_name} at {upload_time}")
    
    def get_upload_cutoff_time(self, table_name: str = None) -> Optional[dt]:
        """
        Get the upload cutoff time for data consistency.
        
        Args:
            table_name: Specific table name, or None for session start time
            
        Returns:
            Datetime representing the cutoff for consistent data queries
        """
        if table_name and table_name in self.uploads:
            return self.uploads[table_name]['upload_timestamp']
        return self.upload_start_time
    
    def get_upload_cutoff_date_str(self, table_name: str = None) -> str:
        """
        Get upload cutoff date as string for SQL queries.
        
        Args:
            table_name: Specific table name, or None for session start time
            
        Returns:
            Date string in YYYY-MM-DD format
        """
        cutoff_time = self.get_upload_cutoff_time(table_name)
        return cutoff_time.strftime('%Y-%m-%d') if cutoff_time else dt.now().strftime('%Y-%m-%d')
    
    def save_metadata_to_file(self, file_path: str):
        """Save upload metadata to JSON file."""
        metadata_dict = {
            'session_start': self.upload_start_time.isoformat() if self.upload_start_time else None,
            'session_end': self.upload_end_time.isoformat() if self.upload_end_time else None,
            'uploads': {}
        }
        
        for table_name, upload_info in self.uploads.items():
            metadata_dict['uploads'][table_name] = {
                'upload_timestamp': upload_info['upload_timestamp'].isoformat(),
                'upload_date': upload_info['upload_date'],
                'upload_datetime_str': upload_info['upload_datetime_str'],
                's3_path': upload_info['s3_path'],
                'row_count': upload_info['row_count'],
                'file_size_mb': upload_info['file_size_mb'],
                'session_start': upload_info['session_start'].isoformat() if upload_info['session_start'] else None
            }
        
        with open(file_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        logger.info(f"Upload metadata saved to {file_path}")
    
    def load_metadata_from_file(self, file_path: str):
        """Load upload metadata from JSON file."""
        with open(file_path, 'r') as f:
            metadata_dict = json.load(f)
        
        self.upload_start_time = dt.fromisoformat(metadata_dict['session_start']) if metadata_dict['session_start'] else None
        self.upload_end_time = dt.fromisoformat(metadata_dict['session_end']) if metadata_dict['session_end'] else None
        
        for table_name, upload_info in metadata_dict['uploads'].items():
            self.uploads[table_name] = {
                'upload_timestamp': dt.fromisoformat(upload_info['upload_timestamp']),
                'upload_date': upload_info['upload_date'],
                'upload_datetime_str': upload_info['upload_datetime_str'],
                's3_path': upload_info['s3_path'],
                'row_count': upload_info['row_count'],
                'file_size_mb': upload_info['file_size_mb'],
                'session_start': dt.fromisoformat(upload_info['session_start']) if upload_info['session_start'] else None
            }
        
        logger.info(f"Upload metadata loaded from {file_path}")


class AWSUploader:
    """Handles data upload to AWS S3 using awswrangler."""
    
    def __init__(self, s3_bucket: str, aws_profile: Optional[str] = None):
        """
        Initialize AWS uploader.
        
        Args:
            s3_bucket: S3 bucket name for uploads
            aws_profile: AWS profile name (optional)
        """
        self.s3_bucket = s3_bucket
        self.aws_profile = aws_profile
        self.upload_metadata = UploadMetadata()
        
        # Initialize boto3 session
        if aws_profile:
            self.boto3_session = boto3.Session(profile_name=aws_profile)
        else:
            self.boto3_session = boto3.Session()
    
    def upload_dataframe_to_s3(self, df: pd.DataFrame, 
                              s3_path: str,
                              table_name: str,
                              partition_cols: Optional[List[str]] = None,
                              compression: str = 'snappy',
                              mode: str = 'overwrite') -> Tuple[str, Dict[str, Any]]:
        """
        Upload DataFrame to S3 as Parquet with metadata tracking.
        
        Args:
            df: DataFrame to upload
            s3_path: S3 path (without s3:// prefix)
            table_name: Name of the table being uploaded
            partition_cols: Columns to partition by
            compression: Compression algorithm
            mode: Write mode ('overwrite', 'append', 'overwrite_partitions')
            
        Returns:
            Tuple of (S3 URI, upload metadata dict)
        """
        s3_uri = f"s3://{self.s3_bucket}/{s3_path}"
        upload_start_time = dt.now()
        
        # Add upload timestamp columns to the DataFrame
        df_with_metadata = df.copy()
        df_with_metadata['_upload_timestamp'] = upload_start_time
        df_with_metadata['_upload_date'] = upload_start_time.strftime('%Y-%m-%d')
        df_with_metadata['_upload_batch_id'] = upload_start_time.strftime('%Y%m%d_%H%M%S')
        
        try:
            # Calculate approximate file size before upload
            memory_usage = df_with_metadata.memory_usage(deep=True).sum()
            estimated_size_mb = memory_usage / (1024 * 1024)
            
            result = wr.s3.to_parquet(
                df=df_with_metadata,
                path=s3_uri,
                partition_cols=partition_cols,
                compression=compression,
                mode=mode,
                boto3_session=self.boto3_session,
                sanitize_columns=True,
                max_rows_by_file=100000  # Control file size
            )
            
            upload_end_time = dt.now()
            upload_duration = (upload_end_time - upload_start_time).total_seconds()
            
            # Record upload metadata
            upload_info = {
                'upload_start': upload_start_time,
                'upload_end': upload_end_time,
                'upload_duration_seconds': upload_duration,
                'row_count': len(df),
                'estimated_size_mb': estimated_size_mb,
                's3_uri': s3_uri,
                'files_written': len(result.get('paths', [])) if isinstance(result, dict) else 1
            }
            
            # Store in metadata tracker
            self.upload_metadata.add_upload_record(
                table_name=table_name,
                s3_path=s3_path,
                row_count=len(df),
                file_size_mb=estimated_size_mb
            )
            
            logger.info(f"Successfully uploaded {len(df)} rows to {s3_uri} in {upload_duration:.2f}s")
            return s3_uri, upload_info
            
        except Exception as e:
            logger.error(f"Failed to upload to {s3_uri}: {e}")
            raise
    
    def create_athena_table(self, table_name: str, 
                           s3_path: str,
                           database: str,
                           df_sample: pd.DataFrame,
                           partition_cols: Optional[List[str]] = None) -> None:
        """
        Create Athena table pointing to S3 data.
        
        Args:
            table_name: Name of the table to create
            s3_path: S3 path containing the data
            database: Athena database name
            df_sample: Sample DataFrame for schema inference
            partition_cols: Partition columns
        """
        s3_uri = f"s3://{self.s3_bucket}/{s3_path}"
        
        try:
            wr.catalog.create_parquet_table(
                database=database,
                table=table_name,
                path=s3_uri,
                columns_types=wr.catalog.extract_athena_types(df_sample),
                partitions_types={col: 'string' for col in (partition_cols or [])},
                boto3_session=self.boto3_session,
                mode='overwrite'
            )
            
            logger.info(f"Created Athena table {database}.{table_name}")
            
        except Exception as e:
            logger.error(f"Failed to create Athena table {table_name}: {e}")
            raise


class MigrationUploader:
    """Main class for orchestrating PCDS to AWS data migration."""
    
    def __init__(self, config_path: Path):
        """
        Initialize the migration uploader.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config().from_disk(config_path)
        self.migration_config = utils.MigrationConfig(**self.config)
        
        # Initialize components
        self.extractor = PCDSDataExtractor(
            chunk_size=getattr(self.migration_config, 'chunk_size', 50000),
            max_memory_mb=getattr(self.migration_config, 'max_memory_mb', 512)
        )
        
        self.uploader = AWSUploader(
            s3_bucket=self.migration_config.aws.s3_bucket,
            aws_profile=getattr(self.migration_config.aws, 'profile', None)
        )
        
        self.converter = DataTypeConverter()
        
        # Initialize upload metadata tracker
        self.upload_metadata = UploadMetadata()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = getattr(self.migration_config, 'logging', {})
        if log_config:
            logger.add(**log_config)
        
        logger.info('Migration Uploader initialized')
        logger.info(f'Configuration: {self.migration_config}')
    
    def load_migration_analysis_results(self, analysis_output_path: str) -> Dict[str, Any]:
        """
        Load results from MigrationAnalyzer.
        
        Args:
            analysis_output_path: Path to MigrationAnalyzer output (pickle file)
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            with open(analysis_output_path, 'rb') as fp:
                analysis_data = pickle.load(fp)
            
            logger.info(f"Loaded analysis data for {len(analysis_data)} tables")
            return analysis_data
            
        except Exception as e:
            logger.error(f"Failed to load analysis results: {e}")
            raise
    
    def get_table_migration_info(self, table_name: str, 
                                analysis_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract migration information for a specific table.
        
        Args:
            table_name: Name of the table
            analysis_data: Analysis results from MigrationAnalyzer
            
        Returns:
            Dictionary containing table migration metadata
        """
        if table_name not in analysis_data:
            logger.warning(f"Table {table_name} not found in analysis data")
            return None
        
        table_data = analysis_data[table_name]
        
        # Extract column information
        pcds_meta = table_data.get('pcds_meta', {})
        aws_meta = table_data.get('aws_meta', {})
        
        if not pcds_meta or not aws_meta:
            logger.warning(f"Incomplete metadata for table {table_name}")
            return None
        
        # Build column mapping and type information
        pcds_columns = pcds_meta['column']
        column_mapping = {}
        column_types = {}
        
        for _, row in pcds_columns.iterrows():
            pcds_col = row['column_name']
            aws_col = row.get('aws_colname', pcds_col.lower())
            pcds_type = row['data_type']
            
            if pd.notna(aws_col):
                column_mapping[pcds_col] = aws_col
                column_types[pcds_col] = pcds_type
        
        return {
            'pcds_columns': list(column_mapping.keys()),
            'aws_columns': list(column_mapping.values()),
            'column_mapping': column_mapping,
            'column_types': column_types,
            'pcds_row_count': pcds_meta.get('row', pd.DataFrame()).iloc[0].item() if not pcds_meta.get('row', pd.DataFrame()).empty else 0,
            'aws_row_count': aws_meta.get('row', pd.DataFrame()).iloc[0].item() if not aws_meta.get('row', pd.DataFrame()).empty else 0
        }
    
    def migrate_table_data(self, pcds_table: str, aws_table: str,
                          migration_info: Dict[str, Any],
                          date_column: Optional[str] = None,
                          partition_columns: Optional[List[str]] = None) -> None:
        """
        Migrate data for a single table from PCDS to AWS.
        
        Args:
            pcds_table: PCDS table identifier (service.table)
            aws_table: AWS table identifier (database.table)
            migration_info: Table migration metadata
            date_column: Date column for partitioning (optional)
            partition_columns: Additional partition columns
        """
        logger.info(f"Starting migration for {pcds_table} -> {aws_table}")
        
        # Parse table identifiers
        pcds_service, pcds_table_name = pcds_table.split('.', 1)
        aws_database, aws_table_name = aws_table.split('.', 1)
        
        # Prepare S3 path
        s3_path = f"data/{aws_database}/{aws_table_name}"
        
        # Extract and upload data
        total_rows_processed = 0
        
        try:
            # Choose extraction method based on date column availability
            if date_column and date_column in migration_info['pcds_columns']:
                data_iterator = self.extractor.extract_table_data_paginated(
                    table_name=pcds_table_name,
                    date_column=date_column,
                    columns=migration_info['pcds_columns'],
                    service=pcds_service
                )
            else:
                data_iterator = self.extractor.extract_table_data_chunked(
                    table_name=pcds_table_name,
                    columns=migration_info['pcds_columns'],
                    service=pcds_service
                )
            
            # Process data in chunks
            for chunk_idx, chunk_df in enumerate(data_iterator):
                if chunk_df.empty:
                    continue
                
                # Convert data types
                chunk_df = self.converter.convert_pcds_column_types(
                    chunk_df, migration_info['column_types']
                )
                
                # Rename columns to AWS format
                chunk_df = chunk_df.rename(columns=migration_info['column_mapping'])
                
                # Optimize for Parquet
                chunk_df = self.converter.optimize_dtypes_for_parquet(chunk_df)
                
                # Add partition columns if specified
                if partition_columns:
                    for partition_col in partition_columns:
                        if partition_col == 'migration_date':
                            chunk_df[partition_col] = dt.now().strftime('%Y-%m-%d')
                        elif partition_col == 'year' and date_column:
                            chunk_df[partition_col] = chunk_df[date_column].dt.year.astype(str)
                        elif partition_col == 'month' and date_column:
                            chunk_df[partition_col] = chunk_df[date_column].dt.month.astype(str).str.zfill(2)
                
                # Upload chunk to S3
                mode = 'overwrite' if chunk_idx == 0 else 'append'
                s3_uri, upload_info = self.uploader.upload_dataframe_to_s3(
                    df=chunk_df,
                    s3_path=s3_path,
                    table_name=aws_table_name,
                    partition_cols=partition_columns,
                    mode=mode
                )
                
                total_rows_processed += len(chunk_df)
                logger.info(f"Processed {total_rows_processed} rows for {pcds_table}")
            
            # Create Athena table
            if total_rows_processed > 0:
                # Use the last chunk for schema inference
                self.uploader.create_athena_table(
                    table_name=aws_table_name,
                    s3_path=s3_path,
                    database=aws_database,
                    df_sample=chunk_df,
                    partition_cols=partition_columns
                )
            
            logger.info(f"Migration completed for {pcds_table}: {total_rows_processed} rows")
            
        except Exception as e:
            logger.error(f"Migration failed for {pcds_table}: {e}")
            raise
    
    def run_migration(self, analysis_output_path: str,
                     table_filter: Optional[List[str]] = None,
                     partition_strategy: str = 'date',
                     metadata_output_path: Optional[str] = None) -> str:
        """
        Run the complete migration process.
        
        Args:
            analysis_output_path: Path to MigrationAnalyzer output
            table_filter: List of table names to migrate (None for all)
            partition_strategy: Partitioning strategy ('date', 'none')
            metadata_output_path: Path to save upload metadata (auto-generated if None)
            
        Returns:
            Path to the upload metadata file
        """
        logger.info("Starting PCDS to AWS data migration")
        
        # Start upload session
        self.upload_metadata.start_upload_session()
        
        # Generate metadata output path if not provided
        if not metadata_output_path:
            timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
            metadata_output_path = f"upload_metadata_{timestamp}.json"
        
        try:
            # Load analysis results
            analysis_data = self.load_migration_analysis_results(analysis_output_path)
            
            # Filter tables if specified
            tables_to_migrate = table_filter or list(analysis_data.keys())
            
            # Process each table
            for table_name in tqdm(tables_to_migrate, desc="Migrating tables"):
                try:
                    # Get migration info
                    migration_info = self.get_table_migration_info(table_name, analysis_data)
                    if not migration_info:
                        continue
                    
                    # Determine partitioning
                    partition_columns = []
                    date_column = None
                    
                    if partition_strategy == 'date':
                        # Try to find a date column
                        for col, col_type in migration_info['column_types'].items():
                            if 'DATE' in col_type.upper() or 'TIMESTAMP' in col_type.upper():
                                date_column = col
                                partition_columns = ['year', 'month']
                                break
                    
                    # Add migration date partition
                    partition_columns.append('migration_date')
                    
                    # Extract table identifiers from analysis
                    # This would need to be adapted based on your actual data structure
                    pcds_table = f"service.{table_name}"  # Placeholder
                    aws_table = f"database.{table_name}"   # Placeholder
                    
                    # Migrate the table
                    self.migrate_table_data(
                        pcds_table=pcds_table,
                        aws_table=aws_table,
                        migration_info=migration_info,
                        date_column=date_column,
                        partition_columns=partition_columns
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to migrate table {table_name}: {e}")
                    continue
            
            # End upload session
            self.upload_metadata.end_upload_session()
            
            # Save upload metadata
            self.upload_metadata.save_metadata_to_file(metadata_output_path)
            
            logger.info("Migration process completed")
            logger.info(f"Upload metadata saved to: {metadata_output_path}")
            
            # Print summary
            self._print_migration_summary()
            
            return metadata_output_path
            
        except Exception as e:
            logger.error(f"Migration process failed: {e}")
            # Still save partial metadata
            self.upload_metadata.save_metadata_to_file(metadata_output_path)
            raise
    
    def _print_migration_summary(self):
        """Print a summary of the migration process."""
        total_tables = len(self.upload_metadata.uploads)
        total_rows = sum(info['row_count'] for info in self.upload_metadata.uploads.values())
        
        session_duration = None
        if self.upload_metadata.upload_start_time and self.upload_metadata.upload_end_time:
            session_duration = (self.upload_metadata.upload_end_time - 
                              self.upload_metadata.upload_start_time).total_seconds()
        
        logger.info("=== Migration Summary ===")
        logger.info(f"Tables migrated: {total_tables}")
        logger.info(f"Total rows uploaded: {total_rows:,}")
        if session_duration:
            logger.info(f"Total duration: {session_duration:.2f} seconds")
        logger.info(f"Upload session start: {self.upload_metadata.upload_start_time}")
        logger.info(f"Upload session end: {self.upload_metadata.upload_end_time}")
        
        # Print per-table summary
        for table_name, info in self.upload_metadata.uploads.items():
            logger.info(f"  {table_name}: {info['row_count']:,} rows at {info['upload_datetime_str']}")
    
    def get_upload_cutoff_for_queries(self, table_name: str = None) -> Dict[str, str]:
        """
        Get upload cutoff information for constructing Athena queries.
        
        Args:
            table_name: Specific table name, or None for session-wide cutoff
            
        Returns:
            Dictionary with cutoff date and datetime strings for SQL queries
        """
        cutoff_time = self.upload_metadata.get_upload_cutoff_time(table_name)
        
        if not cutoff_time:
            cutoff_time = dt.now()
        
        return {
            'cutoff_date': cutoff_time.strftime('%Y-%m-%d'),
            'cutoff_datetime': cutoff_time.strftime('%Y-%m-%d %H:%M:%S'),
            'cutoff_timestamp': cutoff_time.isoformat(),
            'where_clause_date': f"_upload_date <= '{cutoff_time.strftime('%Y-%m-%d')}'",
            'where_clause_timestamp': f"_upload_timestamp <= TIMESTAMP '{cutoff_time.strftime('%Y-%m-%d %H:%M:%S')}'"
        }


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Upload PCDS data to AWS using MigrationAnalyzer output'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Configuration file path',
        required=True
    )
    parser.add_argument(
        '--analysis-output',
        type=str,
        help='Path to MigrationAnalyzer output pickle file',
        required=True
    )
    parser.add_argument(
        '--tables',
        type=str,
        nargs='*',
        help='Specific tables to migrate (default: all)',
        default=None
    )
    parser.add_argument(
        '--partition-strategy',
        type=str,
        choices=['date', 'none'],
        help='Partitioning strategy',
        default='date'
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    try:
        args = parse_arguments()
        
        # Initialize and run migration
        uploader = MigrationUploader(args.config)
        metadata_file = uploader.run_migration(
            analysis_output_path=args.analysis_output,
            table_filter=args.tables,
            partition_strategy=args.partition_strategy
        )
        
        # Print cutoff information for next steps
        cutoff_info = uploader.get_upload_cutoff_for_queries()
        print("\n=== Upload Cutoff Information for Athena Queries ===")
        print(f"Session cutoff date: {cutoff_info['cutoff_date']}")
        print(f"Session cutoff datetime: {cutoff_info['cutoff_datetime']}")
        print(f"Suggested WHERE clause: {cutoff_info['where_clause_timestamp']}")
        print(f"Upload metadata file: {metadata_file}")
        
        logger.info("Data migration completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Migration interrupted by user")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


# Utility class for reading upload metadata in subsequent processes
class UploadMetadataReader:
    """Utility class for reading upload metadata from previous migrations."""
    
    @staticmethod
    def load_upload_metadata(metadata_file_path: str) -> UploadMetadata:
        """
        Load upload metadata from a JSON file.
        
        Args:
            metadata_file_path: Path to the metadata JSON file
            
        Returns:
            UploadMetadata object with loaded data
        """
        metadata = UploadMetadata()
        metadata.load_metadata_from_file(metadata_file_path)
        return metadata
    
    @staticmethod
    def get_athena_query_filter(metadata_file_path: str, 
                               table_name: str = None) -> str:
        """
        Get WHERE clause for Athena queries to ensure data consistency.
        
        Args:
            metadata_file_path: Path to the metadata JSON file
            table_name: Specific table name, or None for session-wide filter
            
        Returns:
            WHERE clause string for Athena queries
        """
        metadata = UploadMetadataReader.load_upload_metadata(metadata_file_path)
        cutoff_time = metadata.get_upload_cutoff_time(table_name)
        
        if cutoff_time:
            return f"_upload_timestamp <= TIMESTAMP '{cutoff_time.strftime('%Y-%m-%d %H:%M:%S')}'"
        else:
            return "1=1"  # No filter if no metadata available
    
    @staticmethod
    def get_upload_summary(metadata_file_path: str) -> Dict[str, Any]:
        """
        Get a summary of the upload process.
        
        Args:
            metadata_file_path: Path to the metadata JSON file
            
        Returns:
            Dictionary containing upload summary information
        """
        metadata = UploadMetadataReader.load_upload_metadata(metadata_file_path)
        
        total_tables = len(metadata.uploads)
        total_rows = sum(info['row_count'] for info in metadata.uploads.values())
        
        session_duration = None
        if metadata.upload_start_time and metadata.upload_end_time:
            session_duration = (metadata.upload_end_time - metadata.upload_start_time).total_seconds()
        
        return {
            'total_tables': total_tables,
            'total_rows': total_rows,
            'session_start': metadata.upload_start_time,
            'session_end': metadata.upload_end_time,
            'session_duration_seconds': session_duration,
            'uploads_by_table': metadata.uploads
        }


if __name__ == '__main__':
    main()