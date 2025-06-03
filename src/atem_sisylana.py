#!/usr/bin/env python3
"""
Database Migration Analysis Tool

This tool performs comprehensive analysis of database table migrations between
PCDS (Oracle) and AWS (PostgreSQL/Athena) systems. It compares table schemas,
data types, row counts, and temporal data distributions to identify discrepancies
and ensure data integrity during migration processes.

Features:
- Schema comparison between PCDS and AWS tables
- Data type mapping validation
- Row count verification
- Temporal data analysis
- Column mapping verification
- Comprehensive reporting with CSV output

Author: Xiang Liu
Date: 20250603
"""


import re
import os
import pickle
import json
import csv
import itertools as it
import shutil
import argparse
from pathlib import Path
from datetime import datetime as dt
from enum import Enum
from typing import Literal, Dict, Any, Tuple, Optional, List
from warnings import filterwarnings
from collections import defaultdict

import utils
import pandas as pd
import pyathena as pa
from loguru import logger
from confection import Config
from tqdm import tqdm
from dotenv import load_dotenv

# Suppress pandas SQLAlchemy warnings
filterwarnings("ignore", category=UserWarning, 
               message='.*pandas only supports SQLAlchemy connectable.*')


class PullStatus(Enum):
    """Enumeration of possible data pull statuses during migration analysis."""
    NONEXIST_PCDS = 'Nonexisting PCDS Table'
    NONEXIST_AWS = 'Nonexisting AWS Table'
    NONDATE_PCDS = 'Nonexisting Date Variable in PCDS'
    NONDATE_AWS = 'Nonexisting Date Variable in AWS'
    EMPTY_PCDS = 'Empty PCDS Table'
    EMPTY_AWS = 'Empty AWS Table'
    NO_MAPPING = 'Column Mapping Not Provided'
    SUCCESS = 'Successful Data Access'


class DatabaseQueryBuilder:
    """Builds SQL queries for different database platforms."""
    
    # PCDS (Oracle) SQL Templates
    PCDS_SQL_META = """
    select
        column_name,
        data_type || case
        when data_type = 'NUMBER' then 
            case when data_precision is NULL AND data_scale is NULL
                then NULL
            else
                '(' || TO_CHAR(data_precision) || ',' || TO_CHAR(data_scale) || ')'
            end
        when data_type LIKE '%CHAR%'
            then
                '(' || TO_CHAR(data_length) || ')'
            else NULL
        end AS data_type
    from all_tab_cols
    where table_name = UPPER('{table}')
    order by column_id
    """.strip()

    PCDS_SQL_NROW = """
    SELECT COUNT(*) AS nrow FROM {table}
    """.strip()

    PCDS_SQL_DATE = """
    SELECT {date}, count(*) AS nrows
    FROM {table} GROUP BY {date}
    """.strip()

    # AWS (PostgreSQL/Athena) SQL Templates
    AWS_SQL_META = """
    select column_name, data_type from information_schema.columns
    where table_schema = LOWER('{db}') and table_name = LOWER('{table}')
    """.strip()

    AWS_SQL_NROW = """
    SELECT COUNT(*) AS nrow FROM {db}.{table}
    """.strip()

    AWS_SQL_DATE = """
    SELECT {date}, count(*) AS nrows
    FROM {db}.{table} GROUP BY {date}
    """.strip()

    @classmethod
    def get_query(cls, platform: str, category: Literal['date', 'nrow', 'meta'], 
                  table: str, db: Optional[str] = None, date_col: Optional[str] = None) -> str:
        """
        Generate SQL query based on platform and category.
        
        Args:
            platform: Database platform ('PCDS' or 'AWS')
            category: Query type ('date', 'nrow', 'meta')
            table: Table name
            db: Database name (required for AWS)
            date_col: Date column name (required for date queries)
            
        Returns:
            Formatted SQL query string
        """
        query_map = {
            'PCDS': {
                'date': cls.PCDS_SQL_DATE,
                'meta': cls.PCDS_SQL_META,
                'nrow': cls.PCDS_SQL_NROW
            },
            'AWS': {
                'date': cls.AWS_SQL_DATE,
                'meta': cls.AWS_SQL_META,
                'nrow': cls.AWS_SQL_NROW
            }
        }
        
        template = query_map[platform][category]
        
        if category == 'date':
            return template.format(table=table, db=db, date=date_col)
        elif platform == 'AWS':
            return template.format(table=table, db=db)
        else:
            return template.format(table=table)


class DataTypeMapper:
    """Handles data type mapping between PCDS (Oracle) and AWS (PostgreSQL) systems."""
    
    @staticmethod
    def map_pcds_to_aws(row: pd.Series) -> bool:
        """
        Compare PCDS and AWS data types to determine compatibility.
        
        Args:
            row: Pandas Series containing data_type_pcds and data_type_aws columns
            
        Returns:
            Boolean indicating if data types are compatible
        """
        aws_dtype = row.data_type_aws
        pcds_dtype = row.data_type_pcds
        
        # Handle Oracle NUMBER type
        if pcds_dtype == 'NUMBER':
            return aws_dtype == 'double'
        
        # Handle Oracle NUMBER with precision/scale
        elif pcds_dtype.startswith('NUMBER'):
            match = re.match(r'NUMBER\(\d*,(\d+)\)', pcds_dtype)
            if match:
                scale = match.group(1)
                aws_match = re.match(r'decimal\(\d*,(\d+)\)', aws_dtype)
                return bool(aws_match and aws_match.group(1) == scale)
        
        # Handle VARCHAR2 types
        elif pcds_dtype.startswith('VARCHAR2'):
            return pcds_dtype.replace('VARCHAR2', 'varchar') == aws_dtype
        
        # Handle CHAR types
        elif pcds_dtype.startswith('CHAR'):
            n_match = re.match(r'CHAR\((\d+)\)', pcds_dtype)
            if n_match:
                n = n_match.group(1)
                return not (aws_dtype.startswith('VARCHAR') and n != '1')
        
        # Handle DATE types
        elif pcds_dtype == 'DATE':
            return aws_dtype == 'date' or aws_dtype.startswith('timestamp')
        
        # Handle TIMESTAMP types
        elif pcds_dtype.startswith('TIMESTAMP'):
            return aws_dtype.startswith('timestamp')
        
        else:
            logger.info(f">>> Mismatched type on {row.get('column_name_aws', 'unknown')}\n"
                       f"\tPCDS ({pcds_dtype}) ==> AWS ({aws_dtype})")
            return False
        
        return False


class StringMatcher:
    """Utilities for matching strings with different strategies."""
    
    @staticmethod
    def has_prefix_match(a: str, b: str) -> bool:
        """Check if two strings have a prefix relationship."""
        return a.startswith(b) or b.startswith(a)
    
    @staticmethod
    def find_differences(a: List[str], b: List[str], mode: str = 'prefix', 
                        drop: Optional[List[str]] = None) -> set:
        """
        Find items in list 'a' that don't match items in list 'b'.
        
        Args:
            a: First list of strings
            b: Second list of strings  
            mode: Matching mode ('exact' or 'prefix')
            drop: Patterns to exclude from comparison
            
        Returns:
            Set of unmatched items from list 'a'
        """
        if drop is None:
            drop = []
            
        def should_compare(x: str) -> bool:
            if drop and re.match(r'|'.join(drop), x):
                return False
            if mode == 'exact':
                return x not in b
            elif mode == 'prefix':
                return not any(StringMatcher.has_prefix_match(x, y) for y in b)
            return True
            
        return set(filter(should_compare, a))
    
    @staticmethod
    def find_common_mappings(list_a: List[str], list_b: List[str]) -> Dict[str, str]:
        """
        Find common mappings between two lists using exact and prefix matching.
        
        Args:
            list_a: Source list
            list_b: Target list
            
        Returns:
            Dictionary mapping items from list_a to list_b
        """
        result = {}
        visited = set()
        prefix_mappings = defaultdict(list)
        
        # Build prefix mapping candidates
        for x, y in it.product(list_a, list_b):
            if StringMatcher.has_prefix_match(x, y):
                prefix_mappings[x].append(y)
        
        # Prioritize exact matches
        for x in list_a:
            if x in list_b and x not in visited:
                result[x] = x
                visited.add(x)
        
        # Handle prefix matches for remaining items
        for x in list_a:
            if x in result:
                continue
            for y in prefix_mappings[x]:
                if y not in visited:
                    result[x] = y
                    visited.add(y)
                    break
                    
        return result


class MigrationAnalyzer:
    """Main class for conducting database migration analysis."""
    
    def __init__(self, config_path: Path):
        """
        Initialize the migration analyzer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config().from_disk(config_path)
        self.config = utils.MetaConfig(**self.config)
        self.col_maps = utils.read_column_mapping(self.config.column_maps)
        self.tbl_list = utils.read_input_excel(self.config.input.table)
        
        # Constants
        self.SEP = '; '
        self.mismatch_data = {}
        self.next_data = {}
        
        # Setup logging and output directories
        self._setup_environment()
    
    def _setup_environment(self) -> None:
        """Setup logging, environment variables, and output directories."""
        # Clean up previous runs if starting from beginning
        try:
            start_row = self.config.input.range[0] if self.config.input.range else 1
            if start_row <= 1:
                os.remove(self.config.output.csv.path)
        except (TypeError, AttributeError, FileNotFoundError):
            pass
        
        # Create output directory
        os.makedirs(self.config.output.folder, exist_ok=True)
        
        # Setup logging
        logger.add(**self.config.output.log.todict())
        logger.info('Configuration:\n' + self.config.tolog())
        
        # Load environment variables
        load_dotenv(self.config.input.env)
    
    @utils.cache(expire_hours=None)
    def query_database(self, table: str, category: Literal['date', 'nrow', 'meta'], 
                      db: Optional[str] = None, date_col: Optional[str] = None, 
                      **kwargs) -> pd.DataFrame:
        """
        Query database with caching support.
        
        Args:
            table: Table name to query
            category: Type of query ('date', 'nrow', 'meta')
            db: Database name (None for PCDS, required for AWS)
            date_col: Date column name for date queries
            **kwargs: Additional arguments for database connector
            
        Returns:
            Query results as DataFrame
        """
        platform = 'PCDS' if db is None else 'AWS'
        query_str = DatabaseQueryBuilder.get_query(platform, category, table, db, date_col)
        
        if platform == 'AWS':
            return utils.DatabaseConnector('AWS').query(query_str, **kwargs)
        else:
            return utils.DatabaseConnector('PCDS').query(query_str, **kwargs)
    
    def process_pcds_metadata(self, table_info: str, col_map: str) -> Tuple[Dict[str, Any], bool]:
        """
        Process PCDS table metadata including schema and row count.
        
        Args:
            table_info: Table identifier in format 'service.table'
            col_map: Column mapping key
            
        Returns:
            Tuple of (metadata dict, has_mapping boolean)
            
        Raises:
            utils.NONEXIST_TABLE: If table doesn't exist
        """
        service, table = table_info.split('.', maxsplit=1)
        logger.info(f"\tStart processing {table_info}")
        
        try:
            df_schema, service = self.query_database(table, 'meta', svc=service, return_svc=True)
            df_rowcount = self.query_database(table, 'nrow', svc=service)
        except (utils.NONEXIST_TABLE, pd.errors.DatabaseError):
            logger.warning(f"Couldn't find {table.upper()} in {service.upper()}")
            raise utils.NONEXIST_TABLE("PCDS View Not Existing")
        
        # Normalize column names and apply mappings
        df_schema.columns = [x.lower() for x in df_schema.columns]
        column_mappings = self.col_maps.get(col_map, {}).get('pcds2aws', {})
        df_schema['aws_colname'] = df_schema['column_name'].map(column_mappings)
        
        metadata = {
            'column': df_schema,
            'row': df_rowcount,
            'svc': service
        }
        
        return metadata, len(column_mappings) > 0
    
    def process_pcds_date_data(self, table_info: str, date_var: str) -> pd.DataFrame:
        """
        Process PCDS temporal data distribution.
        
        Args:
            table_info: Table identifier in format 'service.table'
            date_var: Date column name
            
        Returns:
            DataFrame with date distribution
            
        Raises:
            utils.NONEXIST_DATEVAR: If date variable doesn't exist
        """
        service, table = table_info.split('.', maxsplit=1)
        
        try:
            df_date = self.query_database(table, 'date', svc=service, date_col=date_var)
            logger.info(f"\tFinish Processing {table_info}")
            return df_date
        except (utils.NONEXIST_TABLE, pd.errors.DatabaseError):
            logger.warning(f"Column {date_var.upper()} not found in {table.upper()}")
            raise utils.NONEXIST_DATEVAR("Date-like Variable Not In PCDS")
    
    def process_aws_metadata(self, table_info: str) -> Dict[str, pd.DataFrame]:
        """
        Process AWS table metadata including schema and row count.
        
        Args:
            table_info: Table identifier in format 'database.table'
            
        Returns:
            Dictionary containing column and row metadata
            
        Raises:
            utils.NONEXIST_TABLE: If table doesn't exist
        """
        database, table = table_info.split('.', maxsplit=1)
        logger.info(f"\tStart processing {table_info}")
        
        try:
            df_schema = self.query_database(table, 'meta', db=database)
            df_rowcount = self.query_database(table, 'nrow', db=database)
        except (utils.NONEXIST_TABLE, pd.errors.DatabaseError):
            logger.warning(f"Couldn't find {table.lower()} in {database.lower()}")
            raise utils.NONEXIST_TABLE("AWS View Not Existing")
        
        return {'column': df_schema, 'row': df_rowcount}
    
    def process_aws_date_data(self, table_info: str, date_var: str) -> pd.DataFrame:
        """
        Process AWS temporal data distribution.
        
        Args:
            table_info: Table identifier in format 'database.table'
            date_var: Date column name
            
        Returns:
            DataFrame with date distribution
            
        Raises:
            utils.NONEXIST_DATEVAR: If date variable doesn't exist
        """
        database, table = table_info.split('.', maxsplit=1)
        
        try:
            df_date = self.query_database(table, 'date', db=database, date_col=date_var)
            logger.info(f"\tFinish Processing {table_info}")
            return df_date
        except (utils.NONEXIST_TABLE, pd.errors.DatabaseError):
            logger.warning(f"Column {date_var.upper()} not found in {table.upper()}")
            raise utils.NONEXIST_DATEVAR("Date-like Variable Not In AWS")
    
    def merge_and_compare_schemas(self, pcds_df: pd.DataFrame, aws_df: pd.DataFrame, 
                                 pull_status: PullStatus) -> utils.MetaMerge:
        """
        Merge and compare PCDS and AWS table schemas.
        
        Args:
            pcds_df: PCDS schema DataFrame
            aws_df: AWS schema DataFrame
            pull_status: Current processing status
            
        Returns:
            MetaMerge object containing comparison results
        """
        # Find unmapped columns
        unmapped_pcds = (
            pcds_df.query('aws_colname != aws_colname')['column_name']
            .str.lower().to_list()
        )
        unmapped_aws = (
            aws_df.query('~column_name.isin(@pcds_df.aws_colname)')['column_name']
            .to_list()
        )
        
        # Find automatic mappings for uncaptured columns
        auto_mappings = StringMatcher.find_common_mappings(unmapped_pcds, unmapped_aws)
        auto_mappings = {k.upper(): v for k, v in auto_mappings.items()}
        uncaptured_str = self.SEP.join(f'{k}->{v}' for k, v in auto_mappings.items())
        
        # Apply automatic mappings
        pcds_df['aws_colname'] = (
            pcds_df['aws_colname']
            .combine_first(pcds_df['column_name'].map(auto_mappings))
        )
        
        # Perform outer merge to find matches and unique columns
        merged_df = pd.merge(
            pcds_df, aws_df,
            left_on='aws_colname', right_on='column_name',
            suffixes=['_pcds', '_aws'],
            how='outer', indicator=True
        )
        
        # Extract unique columns
        pcds_unique = merged_df.query('_merge == "left_only"')[
            ['column_name_pcds', 'data_type_pcds']
        ]
        aws_unique = merged_df.query('_merge == "right_only"')[
            ['column_name_aws', 'data_type_aws']
        ]
        
        # Handle type comparison based on status
        if pull_status == PullStatus.NO_MAPPING:
            mismatched_str = ''
            type_comparison = None
        else:
            type_comparison = (
                merged_df.query('_merge == "both"')
                .drop(columns=['aws_colname', '_merge'])
            )
            type_comparison['type_match'] = type_comparison.apply(
                DataTypeMapper.map_pcds_to_aws, axis=1
            )
            
            # Store mismatched types globally for later use
            self.mismatch_data = type_comparison.query('~type_match')
            mismatch_summary = (
                self.mismatch_data[['data_type_pcds', 'data_type_aws']]
                .drop_duplicates()
            )
            mismatched_str = self.SEP.join(
                f'{row.data_type_pcds}->{row.data_type_aws}' 
                for row in mismatch_summary.itertuples()
            )
        
        return utils.MetaMerge(
            unique_pcds=pcds_unique['column_name_pcds'].str.upper().to_list(),
            unique_aws=aws_unique['column_name_aws'].str.lower().to_list(),
            col_mapping=type_comparison,
            mismatches=mismatched_str,
            uncaptured=uncaptured_str
        )
    
    def analyze_metadata_differences(self, pcds_meta: Dict[str, Any], 
                                   aws_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze metadata differences between PCDS and AWS tables.
        
        Args:
            pcds_meta: PCDS metadata dictionary
            aws_meta: AWS metadata dictionary
            
        Returns:
            Dictionary containing analysis results
        """
        pcds_columns, aws_columns = pcds_meta['column'], aws_meta['column']
        
        # Handle case where no column mapping is provided
        if pcds_columns['aws_colname'].isna().all():
            pcds_columns['aws_colname'] = pcds_columns['column_name'].str.lower()
            uncaptured_msg = "Column Mapping Not Provided"
        else:
            uncaptured_msg = ""
        
        # Perform schema comparison
        comparison_result = self.merge_and_compare_schemas(
            pcds_columns, aws_columns, 
            PullStatus.NO_MAPPING if uncaptured_msg else PullStatus.SUCCESS
        )
        logger.info(">>> Finish Merging Type Data")
        
        # Prepare data for next processing step
        if comparison_result.col_mapping is not None:
            column_data = (
                comparison_result.col_mapping
                .drop(columns='type_match')
                .apply(lambda x: self.SEP.join(x.astype(str).tolist()), axis=0)
                .to_dict()
            )
            
            self.next_data.update(
                pcds_cols=column_data.get('column_name_pcds', ''),
                pcds_types=column_data.get('data_type_pcds', ''),
                pcds_nrows=int(pcds_meta['row'].iloc[0].item()),
                aws_cols=column_data.get('column_name_aws', ''),
                aws_types=column_data.get('data_type_aws', ''),
                aws_nrows=int(aws_meta['row'].iloc[0].item()),
            )
        
        # Return analysis summary
        pcds_row_count = int(pcds_meta['row'].iloc[0].item())
        aws_row_count = int(aws_meta['row'].iloc[0].item())
        
        return {
            'Row UnMatch': pcds_row_count != aws_row_count,
            'Row UnMatch Details': f"PCDS({pcds_row_count}) : AWS({aws_row_count})",
            'Col Count Details': f'PCDS({len(pcds_columns)}) : AWS({len(aws_columns)})',
            'Type UnMatch Details': comparison_result.mismatches,
            'Column Type UnMatch': len(comparison_result.mismatches) > 0,
            'PCDS Unique Columns': self.SEP.join(comparison_result.unique_pcds),
            'AWS Unique Columns': self.SEP.join(comparison_result.unique_aws),
            'Uncaptured Column Mappings': uncaptured_msg or comparison_result.uncaptured,
        }
    
    def convert_date_format(self, df: pd.DataFrame, column: str, data_type: str) -> None:
        """
        Convert date column to consistent string format.
        
        Args:
            df: DataFrame containing the date column
            column: Name of the date column
            data_type: Data type of the column
        """
        if data_type.lower() == 'date' or data_type.startswith('timestamp'):
            df[column] = df[column].dt.strftime('%Y-%m-%d')
    
    def analyze_temporal_differences(self, pcds_date_df: pd.DataFrame, pcds_date_col: str,
                                   aws_date_df: pd.DataFrame, aws_date_col: str) -> Dict[str, Any]:
        """
        Analyze temporal data distribution differences.
        
        Args:
            pcds_date_df: PCDS date distribution DataFrame
            pcds_date_col: PCDS date column name
            aws_date_df: AWS date distribution DataFrame  
            aws_date_col: AWS date column name
            
        Returns:
            Dictionary containing temporal analysis results
        """
        # Handle data type mismatches for date columns
        try:
            if hasattr(self, 'mismatch_data') and len(self.mismatch_data) > 0:
                mismatch_row = self.mismatch_data.query(
                    f'column_name_aws == "{aws_date_col}"'
                )
                if len(mismatch_row) > 0:
                    mismatch_info = mismatch_row.squeeze()
                    self.convert_date_format(pcds_date_df, pcds_date_col, 
                                           mismatch_info.data_type_pcds)
                    self.convert_date_format(aws_date_df, aws_date_col, 
                                           mismatch_info.data_type_aws)
        except (AttributeError, KeyError):
            pass
        
        # Ensure consistent string format for comparison
        pcds_date_df[pcds_date_col] = pcds_date_df[pcds_date_col].astype(str)
        aws_date_df[aws_date_col] = aws_date_df[aws_date_col].astype(str)
        
        # Merge date distributions
        merged_dates = pd.merge(
            pcds_date_df, aws_date_df,
            left_on=pcds_date_col, right_on=aws_date_col,
            suffixes=['_pcds', '_aws'], how='outer'
        )
        
        # Find time periods with mismatched counts
        time_mismatches = merged_dates.query('NROWS != nrows')
        mismatch_dates = time_mismatches[pcds_date_col].fillna(
            time_mismatches[aws_date_col]
        )
        
        # Store results for next processing step
        self.next_data.update(
            time_excludes=self.SEP.join(utils.get_datesort(mismatch_dates))
        )
        
        return {
            'Time Span UnMatch': len(time_mismatches) > 0,
            'Time Span Variable': f'{pcds_date_col} : {aws_date_col}',
            'Time UnMatch Details': self.next_data['time_excludes']
        }
    
    def clean_column_lists(self, pcds_unique: str, aws_unique: str) -> Tuple[str, str]:
        """
        Clean unique column lists by removing specified patterns.
        
        Args:
            pcds_unique: PCDS unique columns string
            aws_unique: AWS unique columns string
            
        Returns:
            Tuple of cleaned column strings
        """
        def remove_patterns(input_str: str, patterns: List[str]) -> str:
            if not patterns:
                return input_str
            pattern = '|'.join(rf'\b{re.escape(x)}\b;?\s?' for x in patterns)
            return re.sub(pattern, '', input_str).rstrip('; ')
        
        cleaned_pcds = remove_patterns(pcds_unique, 
                                     getattr(self.config.match, 'drop_cols', []))
        cleaned_aws = remove_patterns(aws_unique, 
                                    getattr(self.config.match, 'add_cols', []))
        
        return cleaned_pcds, cleaned_aws
    
    def upload_results_to_s3(self) -> None:
        """Upload analysis results to S3 bucket."""
        try:
            s3_root = utils.urljoin(f'{self.config.output.to_s3.run}/', 
                                  self.config.input.name)
            
            for root, _, files in os.walk(self.config.output.folder):
                for file in files:
                    if file.startswith(self.config.input.step):
                        s3_url = utils.urljoin(f'{s3_root}/', file)
                        local_path = os.path.join(root, file)
                        utils.s3_upload(local_path, s3_url)
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
    
    def run_analysis(self) -> None:
        """
        Execute the complete migration analysis workflow.
        
        This method orchestrates the entire analysis process including:
        1. Database connection setup
        2. Table processing and comparison
        3. Result compilation and output
        4. Cleanup and reporting
        """
        df_dict, df_next = {}, {}
        start_row, end_row = getattr(self.config.input, 'range', (1, float('inf')))
        has_header = False
        
        try:
            # Initialize analysis environment
            utils.start_run()
            utils.aws_creds_renew(15 * 60)  # Renew AWS credentials for 15 minutes
            
            # Process unique tables only
            unique_tables = self.tbl_list.groupby('pcds_tbl').first().reset_index()
            total_tables = len(unique_tables)
            
            logger.info(f"Starting analysis of {total_tables} tables")
            
            # Process each table
            for i, row in enumerate(tqdm(
                unique_tables.itertuples(), 
                desc='Processing tables...', 
                total=total_tables
            ), start=1):
                
                # Skip rows outside specified range
                if i < start_row or i > end_row:
                    has_header = False
                    continue
                
                # Initialize processing variables
                pcds_meta, pcds_date = {}, None
                aws_meta, aws_date = {}, None
                self.mismatch_data, self.next_data = {}, self.config.output.next.fields.copy()
                
                table_name = row.pcds_tbl.split('.')[1].lower()
                logger.info(f">>> Start processing {table_name} ({i}/{total_tables})")
                
                # Update next data with basic info
                self.next_data.update(
                    pcds_tbl=row.pcds_tbl,
                    aws_tbl=row.aws_tbl,
                    pcds_id=row.pcds_id,
                    aws_id=row.aws_id,
                    last_modified=dt.now().strftime('%Y-%m-%d'),
                )
                
                # Initialize row result data
                row_result = {
                    'Consumer Loans Data Product': row.group,
                    'PCDS Table Details with DB Name': table_name,
                    'Tables delivered in AWS with DB Name': row.aws_tbl,
                    'Hydrated Table only in AWS': row.hydrate_only
                }
                
                pull_status = PullStatus.SUCCESS
                
                # Step 1: Process PCDS metadata
                try:
                    pcds_meta, has_mapping = self.process_pcds_metadata(
                        row.pcds_tbl, row.col_map.lower()
                    )
                except utils.NONEXIST_TABLE:
                    pull_status = PullStatus.NONEXIST_PCDS
                
                # Handle dynamic service discovery
                if (row.pcds_tbl.startswith('no_server_provided') and 
                    'svc' in pcds_meta):
                    new_table = row.pcds_tbl.replace('no_server_provided', 
                                                   pcds_meta.pop('svc'))
                    row = row._replace(pcds_tbl=new_table)
                
                # Check for missing column mapping
                if pull_status == PullStatus.SUCCESS and not has_mapping:
                    pull_status = PullStatus.NO_MAPPING
                
                # Step 2: Process PCDS date data
                try:
                    pcds_date = self.process_pcds_date_data(row.pcds_tbl, row.pcds_id)
                    
                    # Check for empty PCDS table
                    if (pull_status == PullStatus.SUCCESS and 
                        pcds_meta and len(pcds_meta.get('column', [])) == 0):
                        pull_status = PullStatus.EMPTY_PCDS
                        
                except utils.NONEXIST_DATEVAR:
                    if pull_status == PullStatus.SUCCESS:
                        pull_status = PullStatus.NONDATE_PCDS
                
                # Step 3: Process AWS metadata
                try:
                    aws_meta = self.process_aws_metadata(row.aws_tbl)
                except utils.NONEXIST_TABLE:
                    if pull_status == PullStatus.SUCCESS:
                        pull_status = PullStatus.NONEXIST_AWS
                
                # Step 4: Process AWS date data
                try:
                    aws_date = self.process_aws_date_data(row.aws_tbl, row.aws_id)
                    
                    # Check for empty AWS table
                    if (aws_meta and len(aws_meta.get('column', [])) == 0):
                        pull_status = PullStatus.EMPTY_AWS
                        
                except utils.NONEXIST_DATEVAR:
                    if pull_status == PullStatus.SUCCESS:
                        pull_status = PullStatus.NONDATE_AWS
                
                # Initialize analysis results
                analysis_results = {
                    'PCDS Table Service Name': row.pcds_tbl.split('.')[0],
                    'Status': pull_status.value,
                    'Row UnMatch': False,
                    'Row UnMatch Details': '',
                    'Col Count Details': '',
                    'Time Span UnMatch': False,
                    'Time Span Variable': f'{row.pcds_id} : {row.aws_id}',
                    'Time UnMatch Details': '',
                    'Column Type UnMatch': False,
                    'Type UnMatch Details': '',
                    'PCDS Unique Columns': '',
                    'AWS Unique Columns': '',
                    'Uncaptured Column Mappings': '',
                }
                
                # Step 5: Analyze metadata based on pull status
                if pull_status in (PullStatus.NONEXIST_PCDS, PullStatus.NONDATE_AWS):
                    # No data available for analysis
                    pass
                    
                elif pull_status == PullStatus.NO_MAPPING:
                    # Handle case with no column mapping
                    analysis_results.update(
                        self._analyze_no_mapping_case(pcds_meta, aws_meta)
                    )
                    
                else:
                    # Perform full metadata analysis
                    metadata_analysis = self.analyze_metadata_differences(
                        pcds_meta, aws_meta
                    )
                    analysis_results.update(metadata_analysis)
                
                # Step 6: Analyze temporal differences if applicable
                if (analysis_results['Row UnMatch'] and 
                    pull_status not in (PullStatus.NONDATE_AWS, PullStatus.NONDATE_PCDS)):
                    try:
                        temporal_analysis = self.analyze_temporal_differences(
                            pcds_date, row.pcds_id, aws_date, row.aws_id
                        )
                        analysis_results.update(temporal_analysis)
                    except TypeError:
                        logger.warning("Failed to analyze temporal differences")
                        continue
                
                # Step 7: Clean up column lists
                cleaned_pcds, cleaned_aws = self.clean_column_lists(
                    analysis_results['PCDS Unique Columns'],
                    analysis_results['AWS Unique Columns']
                )
                analysis_results.update({
                    'PCDS Unique Columns': cleaned_pcds,
                    'AWS Unique Columns': cleaned_aws
                })
                
                # Step 8: Write results to CSV
                self._write_results_to_csv(row_result, analysis_results, has_header)
                has_header = True
                
                # Step 9: Store detailed data for further processing
                if True:  # Flag for detailed data storage
                    df_dict[table_name] = {
                        'pcds_meta': pcds_meta,
                        'pcds_date': pcds_date,
                        'aws_meta': aws_meta,
                        'aws_date': aws_date,
                        'mismatch': self.mismatch_data.copy() if hasattr(self, 'mismatch_data') else {}
                    }
                    df_next[table_name] = self.next_data.copy()
                    
                    # Save intermediate results
                    with open(self.config.output.next.file, 'w') as fp:
                        json.dump(df_next, fp)
                
                utils.seperator()
                logger.info(f">>> Completed {table_name}")
        
        except Exception as e:
            logger.error("Error in processing ... Stopping")
            logger.exception(e)
            raise
            
        finally:
            # Cleanup and finalization
            self._finalize_analysis(df_dict)
    
    def _analyze_no_mapping_case(self, pcds_meta: Dict[str, Any], 
                                aws_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze case where no column mapping is provided.
        
        Args:
            pcds_meta: PCDS metadata dictionary
            aws_meta: AWS metadata dictionary
            
        Returns:
            Dictionary containing analysis results for no mapping case
        """
        # Extract basic counts
        pcds_row_count = int(pcds_meta['row'].iloc[0].item())
        aws_row_count = int(aws_meta['row'].iloc[0].item())
        pcds_col_count = len(pcds_meta['column'])
        aws_col_count = len(aws_meta['column'])
        
        # Get column lists
        pcds_columns = [x.lower() for x in pcds_meta['column'].column_name]
        aws_columns = aws_meta['column'].column_name.to_list()
        
        # Find automatic mappings
        auto_mappings = StringMatcher.find_common_mappings(pcds_columns, aws_columns)
        pcds_to_aws = {k.upper(): v for k, v in auto_mappings.items()}
        aws_to_pcds = {v: k for k, v in pcds_to_aws.items()}
        
        # Create type mappings
        pcds_types = pcds_meta['column'].set_index('column_name')['data_type'].to_dict()
        aws_types = aws_meta['column'].set_index('column_name')['data_type'].to_dict()
        
        # Find unique columns
        pcds_unique = [x.upper() for x in pcds_columns if x.upper() not in pcds_to_aws]
        aws_unique = [x.lower() for x in aws_columns if x.lower() not in aws_to_pcds]
        
        # Update next data for downstream processing
        self.next_data.update(
            pcds_cols=self.SEP.join(pcds_to_aws.keys()),
            pcds_types=self.SEP.join([pcds_types[x] for x in pcds_to_aws.keys()]),
            pcds_nrows=pcds_row_count,
            aws_cols=self.SEP.join(aws_to_pcds.keys()),
            aws_types=self.SEP.join([aws_types[x] for x in aws_to_pcds.keys()]),
            aws_nrows=aws_row_count
        )
        
        return {
            'Row UnMatch': pcds_row_count != aws_row_count,
            'Row UnMatch Details': f"PCDS({pcds_row_count}) : AWS({aws_row_count})",
            'Col Count Details': f"PCDS({pcds_col_count}) : AWS({aws_col_count})",
            'PCDS Unique Columns': self.SEP.join(pcds_unique),
            'AWS Unique Columns': self.SEP.join(aws_unique),
            'Uncaptured Column Mappings': self.SEP.join(
                f'{k}->{v}' for k, v in pcds_to_aws.items()
            ),
        }
    
    def _write_results_to_csv(self, row_result: Dict[str, Any], 
                             analysis_results: Dict[str, Any], 
                             has_header: bool) -> None:
        """
        Write analysis results to CSV file.
        
        Args:
            row_result: Basic row information
            analysis_results: Analysis results dictionary
            has_header: Whether CSV header has been written
        """
        with open(self.config.output.csv.path, 'a+', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=self.config.output.csv.columns)
            
            if not has_header:
                writer.writeheader()
            
            writer.writerow({**row_result, **analysis_results})
    
    def _finalize_analysis(self, df_dict: Dict[str, Any]) -> None:
        """
        Finalize analysis by saving data and uploading results.
        
        Args:
            df_dict: Dictionary containing all processed data
        """
        try:
            # Save processed data to pickle file
            with open(self.config.output.to_pkl, 'wb') as fp:
                pickle.dump(df_dict, fp)
            
            # Copy configuration file to output directory
            config_filename = f'{self.config.input.step}.cfg'
            shutil.copy(
                self.config.config_path, 
                os.path.join(self.config.output.folder, config_filename)
            )
            
            # Upload results to S3
            self.upload_results_to_s3()
            
            logger.info("Analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in finalization: {e}")
            
        finally:
            utils.end_run()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Conduct comprehensive database migration analysis between PCDS and AWS systems'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Configuration file path for conducting meta analysis',
        default=r'FILE\Input\config_meta.cfg'
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the migration analysis tool.
    
    This function initializes the analyzer and runs the complete analysis workflow.
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Initialize and run analyzer
        analyzer = MigrationAnalyzer(args.config)
        analyzer.run_analysis()
        
        logger.info("Migration analysis completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.exception("Full traceback:")
        raise


if __name__ == '__main__':
    main()