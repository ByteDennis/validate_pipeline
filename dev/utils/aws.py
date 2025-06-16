#!/usr/bin/env python3
"""
AWS utilities for S3 and Athena operations.
Simplified from multiple AWS utility files.
"""

import os
import io
import json
import time
import uuid
import threading
from datetime import datetime as dt
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

import pandas as pd
import awswrangler as wr
import boto3
import botocore
import requests
import urllib3
from loguru import logger


class AWSManager:
    """Manages AWS operations with credential renewal and S3/Athena utilities."""
    
    def __init__(self, boto3_session: Optional[boto3.Session] = None):
        self.session = boto3_session or boto3.Session()
        self._lock = threading.RLock()
        
    def renew_credentials(self, seconds: int = 0, 
                         msg: str = 'AWS Credential Has Been Updated!') -> None:
        """Renew AWS credentials using corporate portal."""
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        usr, pwd = os.environ['EDP_USR'], os.environ['EDP_PWD']
        token_url = 'https://awsportal.barcapint.com/v1/jwttoken/'
        arn_url = ('https://awsportal.barcapint.com/v1/creds-provider/provide-credentials/'
                  'arn:aws:iam::355538383407:role/app-uscb-analytics=fglbuscloudanalytic@I+000')
        
        # Get JWT token
        resp = requests.post(
            token_url, 
            headers={'Accept': '*/*'}, 
            verify=False,
            json={"username": usr, "password": pwd}
        ).json()
        
        # Get temporary credentials
        headers = {"Accept": "*/*", "Authorization": "Bearer " + resp['token']}
        creds = requests.get(arn_url, headers=headers, verify=False).json()['Credentials']
        
        # Set environment variables
        os.environ['AWS_ACCESS_KEY_ID'] = creds['AccessKeyId']
        os.environ['AWS_SECRET_ACCESS_KEY'] = creds['SecretAccessKey']
        os.environ['AWS_SESSION_TOKEN'] = creds['SessionToken']
        os.environ['HTTPS_PROXY'] = f'http://{usr}:{pwd}@35.165.20.1:8080'
        
        if msg:
            logger.info(msg)
        
        # Update session
        self.session = boto3.Session(region_name='us-east-1')
        
        # Schedule next renewal
        if seconds > 0:
            threading.Timer(seconds, self.renew_credentials, args=(seconds,)).start()


class S3Utils:
    """S3 utility functions."""
    
    def __init__(self, session: Optional[boto3.Session] = None):
        self.session = session or boto3.Session()
    
    def upload_file(self, local_path: Union[str, Path], s3_url: str, **kwargs) -> None:
        """Upload file to S3."""
        try:
            wr.s3.upload(str(local_path), s3_url, boto3_session=self.session, **kwargs)
            logger.info(f"Successfully uploaded {local_path} to {s3_url}")
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to {s3_url}: {e}")
            raise
    
    def download_file(self, s3_url: str, local_path: Union[str, Path], **kwargs) -> None:
        """Download file from S3."""
        try:
            wr.s3.download(s3_url, str(local_path), boto3_session=self.session, **kwargs)
            logger.info(f"Successfully downloaded {s3_url} to {local_path}")
        except Exception as e:
            logger.error(f"Failed to download {s3_url} to {local_path}: {e}")
            raise
    
    def upload_dataframe(self, df: pd.DataFrame, s3_path: str, 
                        file_format: str = 'parquet',
                        partition_cols: Optional[List[str]] = None,
                        **kwargs) -> Dict[str, Any]:
        """Upload DataFrame to S3."""
        try:
            # Add metadata columns
            upload_time = dt.now()
            df_with_meta = df.copy()
            df_with_meta['_upload_timestamp'] = upload_time
            df_with_meta['_upload_date'] = upload_time.strftime('%Y-%m-%d')
            
            if file_format.lower() == 'parquet':
                result = wr.s3.to_parquet(
                    df=df_with_meta,
                    path=s3_path,
                    partition_cols=partition_cols,
                    boto3_session=self.session,
                    sanitize_columns=True,
                    **kwargs
                )
            elif file_format.lower() == 'csv':
                result = wr.s3.to_csv(
                    df=df_with_meta,
                    path=s3_path,
                    boto3_session=self.session,
                    **kwargs
                )
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            # Calculate metadata
            memory_usage = df_with_meta.memory_usage(deep=True).sum()
            estimated_size_mb = memory_usage / (1024 * 1024)
            
            upload_info = {
                'row_count': len(df),
                'estimated_size_mb': estimated_size_mb,
                's3_path': s3_path,
                'upload_timestamp': upload_time,
                'files_written': len(result.get('paths', [])) if isinstance(result, dict) else 1
            }
            
            logger.info(f"Uploaded {len(df)} rows to {s3_path} "
                       f"({estimated_size_mb:.2f}MB)")
            
            return upload_info
            
        except Exception as e:
            logger.error(f"Failed to upload DataFrame to {s3_path}: {e}")
            raise
    
    def read_dataframe(self, s3_path: str, file_format: str = 'parquet',
                      **kwargs) -> pd.DataFrame:
        """Read DataFrame from S3."""
        try:
            if file_format.lower() == 'parquet':
                df = wr.s3.read_parquet(s3_path, boto3_session=self.session, **kwargs)
            elif file_format.lower() == 'csv':
                df = wr.s3.read_csv(s3_path, boto3_session=self.session, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            logger.info(f"Read {len(df)} rows from {s3_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to read from {s3_path}: {e}")
            raise
    
    def exists(self, s3_url: str) -> bool:
        """Check if S3 object exists."""
        try:
            objects = wr.s3.list_objects(s3_url, boto3_session=self.session)
            return len(objects) > 0
        except Exception:
            return False
    
    def delete(self, s3_url: str) -> None:
        """Delete S3 object(s)."""
        try:
            wr.s3.delete_objects(s3_url, boto3_session=self.session)
            logger.info(f"Deleted {s3_url}")
        except Exception as e:
            logger.error(f"Failed to delete {s3_url}: {e}")
            raise
    
    def list_objects(self, s3_path: str, prefix: str = None) -> List[str]:
        """List S3 objects with optional prefix filter."""
        try:
            objects = wr.s3.list_objects(s3_path, boto3_session=self.session)
            if prefix:
                objects = [obj for obj in objects if prefix in obj]
            return objects
        except Exception as e:
            logger.error(f"Failed to list objects in {s3_path}: {e}")
            raise
    
    def get_metadata(self, s3_url: str) -> Dict[str, Any]:
        """Get S3 object metadata."""
        try:
            return wr.s3.describe_objects(s3_url, boto3_session=self.session)
        except Exception as e:
            logger.error(f"Failed to get metadata for {s3_url}: {e}")
            raise
    
    def save_json(self, data: Dict[str, Any], s3_url: str) -> None:
        """Save dictionary as JSON to S3."""
        try:
            df = pd.DataFrame([data])
            wr.s3.to_json(df, s3_url, boto3_session=self.session)
            logger.info(f"Saved JSON to {s3_url}")
        except Exception as e:
            logger.error(f"Failed to save JSON to {s3_url}: {e}")
            raise
    
    def load_json(self, s3_url: str) -> Dict[str, Any]:
        """Load JSON from S3."""
        try:
            df = wr.s3.read_json(s3_url, boto3_session=self.session)
            return df.to_dict()
        except Exception as e:
            logger.error(f"Failed to load JSON from {s3_url}: {e}")
            raise


class AthenaUtils:
    """Athena utility functions."""
    
    def __init__(self, session: Optional[boto3.Session] = None):
        self.session = session or boto3.Session()
        self.default_database = "default"
        self.default_workgroup = "uscb-analytics"
    
    def execute_query(self, sql: str, database: str = None, **kwargs) -> pd.DataFrame:
        """Execute Athena query and return DataFrame."""
        try:
            database = database or self.default_database
            result = wr.athena.read_sql_query(
                sql=sql,
                database=database,
                boto3_session=self.session,
                **kwargs
            )
            logger.info(f"Executed Athena query, returned {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Athena query failed: {e}")
            raise
    
    def create_table_from_s3(self, database: str, table: str, s3_path: str, 
                           df_sample: pd.DataFrame, 
                           partition_cols: Optional[List[str]] = None) -> None:
        """Create Athena table pointing to S3 data."""
        try:
            # Extract column types from sample DataFrame
            columns_types = wr.catalog.extract_athena_types(df_sample)
            partitions_types = {col: 'string' for col in (partition_cols or [])}
            
            wr.catalog.create_parquet_table(
                database=database,
                table=table,
                path=s3_path,
                columns_types=columns_types,
                partitions_types=partitions_types,
                boto3_session=self.session,
                mode='overwrite'
            )
            
            logger.info(f"Created Athena table {database}.{table}")
            
        except Exception as e:
            logger.error(f"Failed to create Athena table {table}: {e}")
            raise
    
    def get_table_types(self, database: str, table: str) -> Dict[str, str]:
        """Get column types for Athena table."""
        try:
            return wr.catalog.get_table_types(
                database=database, 
                table=table, 
                boto3_session=self.session
            )
        except Exception as e:
            logger.error(f"Failed to get table types for {database}.{table}: {e}")
            raise
    
    def query_with_s3_output(self, sql: str, s3_output_path: str, 
                           database: str = None, **kwargs) -> str:
        """Execute query and save results to S3."""
        try:
            database = database or self.default_database
            
            # Generate unique S3 path for results
            unique_path = f"{s3_output_path}/{uuid.uuid4()}"
            
            # Execute query with S3 output
            query_id = wr.athena.start_query_execution(
                sql=sql,
                database=database,
                s3_output=unique_path,
                boto3_session=self.session,
                **kwargs
            )
            
            # Wait for completion
            wr.athena.wait_query(query_id, boto3_session=self.session)
            
            logger.info(f"Query results saved to {unique_path}")
            return unique_path
            
        except Exception as e:
            logger.error(f"Failed to execute query with S3 output: {e}")
            raise


# Convenience functions
def get_aws_manager(renew_credentials: bool = True) -> AWSManager:
    """Get configured AWS manager."""
    manager = AWSManager()
    if renew_credentials:
        manager.renew_credentials()
    return manager


def get_s3_utils(session: Optional[boto3.Session] = None) -> S3Utils:
    """Get S3 utilities."""
    return S3Utils(session)


def get_athena_utils(session: Optional[boto3.Session] = None) -> AthenaUtils:
    """Get Athena utilities."""
    return AthenaUtils(session)


# Migration-specific functions
def upload_migration_data(df: pd.DataFrame, table_name: str, s3_bucket: str,
                         partition_strategy: str = 'date') -> Dict[str, Any]:
    """Upload migration data with appropriate partitioning."""
    s3_utils = get_s3_utils()
    
    # Determine partitioning
    partition_cols = []
    if partition_strategy == 'date':
        # Add date-based partitions
        upload_date = dt.now()
        df['migration_year'] = str(upload_date.year)
        df['migration_month'] = f"{upload_date.month:02d}"
        df['migration_date'] = upload_date.strftime('%Y-%m-%d')
        partition_cols = ['migration_year', 'migration_month']
    
    # Upload to S3
    s3_path = f"s3://{s3_bucket}/migration_data/{table_name}/"
    
    return s3_utils.upload_dataframe(
        df=df,
        s3_path=s3_path,
        partition_cols=partition_cols
    )


def create_athena_table_for_migration(table_name: str, s3_path: str, 
                                    database: str, df_sample: pd.DataFrame) -> None:
    """Create Athena table for migrated data."""
    athena_utils = get_athena_utils()
    
    athena_utils.create_table_from_s3(
        database=database,
        table=table_name,
        s3_path=s3_path,
        df_sample=df_sample,
        partition_cols=['migration_year', 'migration_month']
    )


def execute_comparison_query(table_name: str, database: str, 
                           upload_cutoff: str) -> pd.DataFrame:
    """Execute standardized comparison query."""
    athena_utils = get_athena_utils()
    
    query = f"""
        SELECT * FROM {database}.{table_name}
        WHERE _upload_timestamp <= TIMESTAMP '{upload_cutoff}'
        ORDER BY _upload_timestamp
    """
    
    return athena_utils.execute_query(query, database=database)