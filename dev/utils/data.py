#!/usr/bin/env python3
"""
Data processing utilities for the migration pipeline.
Handles data type conversions, transformations, and validation.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from loguru import logger

from .types import PLATFORM


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
        """Convert PCDS data types to pandas-compatible types."""
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
                
                # Handle boolean types
                elif pcds_type.upper() in ('BOOLEAN', 'BOOL'):
                    df_converted[column] = df_converted[column].astype('boolean')
                    
            except Exception as e:
                logger.warning(f"Failed to convert column {column} of type {pcds_type}: {e}")
                
        return df_converted
    
    @classmethod
    def optimize_dtypes_for_parquet(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for efficient Parquet storage."""
        df_optimized = df.copy()
        
        for column in df_optimized.columns:
            col_data = df_optimized[column]
            
            # Optimize integer columns
            if pd.api.types.is_integer_dtype(col_data):
                df_optimized[column] = pd.to_numeric(col_data, downcast='integer')
            
            # Optimize float columns
            elif pd.api.types.is_float_dtype(col_data):
                df_optimized[column] = pd.to_numeric(col_data, downcast='float')
            
            # Convert object columns to category if beneficial
            elif col_data.dtype == 'object':
                unique_ratio = col_data.nunique() / len(col_data)
                if unique_ratio < 0.5 and col_data.nunique() < 1000:
                    df_optimized[column] = col_data.astype('category')
        
        return df_optimized


class StatisticsCalculator:
    """Calculate comprehensive column statistics for comparison."""
    
    @staticmethod
    def calculate_column_stats(df: pd.DataFrame, platform: PLATFORM) -> pd.DataFrame:
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
                'platform': platform,
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
    
    @staticmethod
    def compare_statistics(pcds_stats: pd.DataFrame, aws_stats: pd.DataFrame) -> pd.DataFrame:
        """Compare statistics between PCDS and AWS data."""
        # Merge on column name
        comparison = pd.merge(
            pcds_stats, aws_stats,
            on='column_name', suffixes=('_pcds', '_aws'),
            how='outer', indicator=True
        )
        
        # Calculate differences for numeric columns
        numeric_cols = ['row_count', 'null_count', 'unique_count', 'min_value', 'max_value', 'mean_value']
        
        for col in numeric_cols:
            pcds_col = f'{col}_pcds'
            aws_col = f'{col}_aws'
            
            if pcds_col in comparison.columns and aws_col in comparison.columns:
                comparison[f'{col}_diff'] = comparison[aws_col] - comparison[pcds_col]
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


class DataValidator:
    """Data validation utilities."""
    
    @staticmethod
    def validate_schema_compatibility(pcds_df: pd.DataFrame, aws_df: pd.DataFrame,
                                    column_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Validate schema compatibility between PCDS and AWS data."""
        results = {
            'compatible': True,
            'issues': [],
            'warnings': []
        }
        
        # Check mapped columns exist
        for pcds_col, aws_col in column_mapping.items():
            if pcds_col not in pcds_df.columns:
                results['issues'].append(f"PCDS column '{pcds_col}' not found")
                results['compatible'] = False
            
            if aws_col not in aws_df.columns:
                results['issues'].append(f"AWS column '{aws_col}' not found")
                results['compatible'] = False
        
        # Check data types compatibility
        for pcds_col, aws_col in column_mapping.items():
            if pcds_col in pcds_df.columns and aws_col in aws_df.columns:
                pcds_dtype = pcds_df[pcds_col].dtype
                aws_dtype = aws_df[aws_col].dtype
                
                if not DataValidator._types_compatible(pcds_dtype, aws_dtype):
                    results['warnings'].append(
                        f"Type mismatch: {pcds_col}({pcds_dtype}) -> {aws_col}({aws_dtype})"
                    )
        
        return results
    
    @staticmethod
    def _types_compatible(pcds_dtype, aws_dtype) -> bool:
        """Check if data types are compatible."""
        # Convert to string for comparison
        pcds_str = str(pcds_dtype).lower()
        aws_str = str(aws_dtype).lower()
        
        # Define compatibility rules
        compatible_pairs = [
            ('int', 'int'), ('float', 'float'), ('object', 'string'),
            ('datetime', 'datetime'), ('bool', 'bool')
        ]
        
        for pcds_type, aws_type in compatible_pairs:
            if pcds_type in pcds_str and aws_type in aws_str:
                return True
        
        return pcds_str == aws_str
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame, 
                            required_columns: List[str] = None,
                            max_null_percentage: float = 50.0) -> Dict[str, Any]:
        """Validate data quality metrics."""
        results = {
            'passed': True,
            'issues': [],
            'metrics': {}
        }
        
        # Check required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                results['issues'].append(f"Missing required columns: {missing_cols}")
                results['passed'] = False
        
        # Check null percentages
        for col in df.columns:
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            results['metrics'][f'{col}_null_pct'] = null_pct
            
            if null_pct > max_null_percentage:
                results['issues'].append(f"Column '{col}' has {null_pct:.1f}% nulls")
                results['passed'] = False
        
        # Check for completely empty DataFrame
        if df.empty:
            results['issues'].append("DataFrame is empty")
            results['passed'] = False
        
        # Check for duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            results['issues'].append(f"Duplicate columns: {duplicate_cols}")
            results['passed'] = False
        
        return results


class DataProcessor:
    """Main data processing orchestrator."""
    
    def __init__(self):
        self.converter = DataTypeConverter()
        self.validator = DataValidator()
        self.stats_calc = StatisticsCalculator()
    
    def process_pcds_data(self, df: pd.DataFrame, column_types: Dict[str, str],
                         column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Process PCDS data for migration."""
        # Convert data types
        df_processed = self.converter.convert_pcds_column_types(df, column_types)
        
        # Rename columns according to mapping
        df_processed = df_processed.rename(columns=column_mapping)
        
        # Optimize for Parquet
        df_processed = self.converter.optimize_dtypes_for_parquet(df_processed)
        
        logger.info(f"Processed PCDS data: {len(df_processed)} rows, {len(df_processed.columns)} columns")
        return df_processed
    
    def compare_datasets(self, pcds_df: pd.DataFrame, aws_df: pd.DataFrame,
                        column_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Compare PCDS and AWS datasets."""
        # Validate compatibility
        compatibility = self.validator.validate_schema_compatibility(
            pcds_df, aws_df, column_mapping
        )
        
        if not compatibility['compatible']:
            return {
                'status': 'incompatible',
                'compatibility': compatibility,
                'statistics': None
            }
        
        # Calculate statistics
        pcds_stats = self.stats_calc.calculate_column_stats(pcds_df, 'PCDS')
        aws_stats = self.stats_calc.calculate_column_stats(aws_df, 'AWS')
        
        # Compare statistics
        stats_comparison = self.stats_calc.compare_statistics(pcds_stats, aws_stats)
        
        return {
            'status': 'complete',
            'compatibility': compatibility,
            'statistics': {
                'pcds': pcds_stats,
                'aws': aws_stats,
                'comparison': stats_comparison
            }
        }


# Utility functions
def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 50000) -> List[pd.DataFrame]:
    """Split DataFrame into chunks."""
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()
        chunks.append(chunk)
    return chunks


def merge_chunks(chunks: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge DataFrame chunks back together."""
    if not chunks:
        return pd.DataFrame()
    
    return pd.concat(chunks, ignore_index=True)


def add_migration_metadata(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Add migration tracking metadata to DataFrame."""
    df_with_meta = df.copy()
    timestamp = pd.Timestamp.now()
    
    df_with_meta['_migration_table'] = table_name
    df_with_meta['_migration_timestamp'] = timestamp
    df_with_meta['_migration_date'] = timestamp.strftime('%Y-%m-%d')
    df_with_meta['_migration_batch'] = timestamp.strftime('%Y%m%d_%H%M%S')
    
    return df_with_meta