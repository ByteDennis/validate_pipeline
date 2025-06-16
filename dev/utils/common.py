#!/usr/bin/env python3
"""
Common utility functions used across the migration pipeline.
Consolidated from various utility files.
"""

import os
import re
import time
import json
import csv
from datetime import datetime as dt
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import itertools as it
from collections import defaultdict

import pandas as pd
from loguru import logger

from .types import UDict


# Timing and logging utilities
class Timer:
    """Simple timer context manager."""
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        pass
    
    @property
    def elapsed(self):
        return time.perf_counter() - self.start
    
    def pause(self):
        elapsed = self.elapsed
        self.start = time.perf_counter()
        return elapsed
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human readable format."""
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f'{int(hours)}h {int(minutes)}m {seconds:.1f}s'


def start_run():
    """Log start of pipeline run."""
    logger.info('\n\n' + '=' * 80)


def end_run():
    """Log end of pipeline run."""
    logger.info('\n\n' + '=' * 80)


def separator():
    """Log separator line."""
    logger.info('-' * 80)


# File and data utilities
def get_memory_usage(df: pd.DataFrame, human_readable: bool = True) -> Union[str, int]:
    """Get DataFrame memory usage."""
    bytes_used = df.memory_usage(deep=True).sum()
    
    if not human_readable:
        return bytes_used
    
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_used < 1024:
            return f'{bytes_used:.2f} {unit}'
        bytes_used /= 1024
    return f'{bytes_used:.2f} PB'


def read_excel_input(file_path: str, sheet_name: str = None, 
                    skip_rows: int = 0, **kwargs) -> pd.DataFrame:
    """Read Excel file with error handling."""
    try:
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            skiprows=skip_rows,
            **kwargs
        )
        logger.info(f"Read {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to read Excel file {file_path}: {e}")
        raise


def read_column_mapping(file_path: str, sheet_excludes: List[str] = None) -> Dict[str, Dict[str, str]]:
    """Read column mapping from Excel file."""
    sheet_excludes = sheet_excludes or []
    
    try:
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        mappings = {}
        
        for sheet_name, df in all_sheets.items():
            if sheet_name in sheet_excludes:
                continue
            
            # Process sheet to extract column mappings
            mapping = {}
            for _, row in df.iterrows():
                pcds_col = row.get('PCDS_Column', row.get('pcds_col'))
                aws_col = row.get('AWS_Column', row.get('aws_col'))
                
                if pd.notna(pcds_col) and pd.notna(aws_col):
                    mapping[str(pcds_col).upper()] = str(aws_col).lower()
            
            if mapping:
                mappings[sheet_name.lower()] = mapping
                logger.info(f"Loaded {len(mapping)} column mappings from {sheet_name}")
        
        return UDict(mappings)
        
    except Exception as e:
        logger.error(f"Failed to read column mappings from {file_path}: {e}")
        raise


def write_csv_report(data: List[Dict[str, Any]], output_path: str, 
                    fieldnames: List[str] = None) -> None:
    """Write data to CSV report."""
    try:
        if not data:
            logger.warning("No data to write to CSV")
            return
        
        fieldnames = fieldnames or list(data[0].keys())
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        logger.info(f"Wrote {len(data)} rows to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to write CSV report to {output_path}: {e}")
        raise


# String and data processing utilities
def has_prefix_match(a: str, b: str) -> bool:
    """Check if two strings have prefix relationship."""
    return a.startswith(b) or b.startswith(a)


def find_common_mappings(list_a: List[str], list_b: List[str]) -> Dict[str, str]:
    """Find common mappings between two lists using exact and prefix matching."""
    result = {}
    visited = set()
    prefix_mappings = defaultdict(list)
    
    # Build prefix mapping candidates
    for x, y in it.product(list_a, list_b):
        if has_prefix_match(x, y):
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


def clean_column_name(name: str) -> str:
    """Clean and standardize column name."""
    if pd.isna(name):
        return 'unknown_column'
    
    # Remove extra whitespace and special characters
    cleaned = re.sub(r'\s+', '_', str(name).strip())
    cleaned = re.sub(r'[^\w_]', '', cleaned)
    
    return cleaned.lower()


def remove_items_from_string(input_str: str, items_to_remove: List[str], 
                           separator: str = '; ') -> str:
    """Remove specified items from delimited string."""
    if not items_to_remove:
        return input_str
    
    pattern = '|'.join(rf'\b{re.escape(item)}\b;?\s?' for item in items_to_remove)
    result = re.sub(pattern, '', input_str)
    return result.rstrip(f'{separator} ')


# Date and time utilities
def get_date_sorted_list(date_series: pd.Series, format_str: str = '%Y-%m-%d') -> List[str]:
    """Get sorted list of dates in specified format."""
    if pd.api.types.is_string_dtype(date_series):
        date_series = pd.to_datetime(date_series, errors='coerce')
    
    return date_series.sort_values().dt.strftime(format_str).tolist()


def parse_time_range(time_range_str: str) -> Dict[str, Optional[str]]:
    """Parse time range string into start and end dates."""
    if not time_range_str:
        return {'start_date': None, 'end_date': None}
    
    # Handle different time range formats
    if ' to ' in time_range_str:
        start, end = time_range_str.split(' to ', 1)
        return {'start_date': start.strip(), 'end_date': end.strip()}
    elif ' - ' in time_range_str:
        start, end = time_range_str.split(' - ', 1)
        return {'start_date': start.strip(), 'end_date': end.strip()}
    else:
        # Single date
        return {'start_date': time_range_str.strip(), 'end_date': None}


# Configuration utilities
def load_json_config(file_path: str) -> Dict[str, Any]:
    """Load JSON configuration file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {file_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load JSON config from {file_path}: {e}")
        raise


def save_json_config(data: Dict[str, Any], file_path: str) -> None:
    """Save configuration to JSON file."""
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved configuration to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON config to {file_path}: {e}")
        raise


def merge_configurations(base_config: Dict[str, Any], 
                        override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries with override support."""
    result = base_config.copy()
    
    for key, value in override_config.items():
        if (key in result and 
            isinstance(result[key], dict) and 
            isinstance(value, dict)):
            result[key] = merge_configurations(result[key], value)
        else:
            result[key] = value
    
    return result


# Validation utilities
def validate_file_exists(file_path: Union[str, Path]) -> bool:
    """Validate that file exists."""
    path = Path(file_path)
    exists = path.exists() and path.is_file()
    if not exists:
        logger.warning(f"File does not exist: {file_path}")
    return exists


def validate_directory_exists(dir_path: Union[str, Path], create: bool = False) -> bool:
    """Validate that directory exists, optionally create it."""
    path = Path(dir_path)
    
    if path.exists() and path.is_dir():
        return True
    
    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")
            return False
    
    logger.warning(f"Directory does not exist: {dir_path}")
    return False


def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame has required columns."""
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    return True


# Progress tracking
def create_progress_tracker(total_items: int, description: str = "Processing") -> object:
    """Create progress tracker (tqdm wrapper)."""
    try:
        from tqdm import tqdm
        return tqdm(total=total_items, desc=description)
    except ImportError:
        # Fallback simple progress tracker
        class SimpleProgress:
            def __init__(self, total, desc):
                self.total = total
                self.current = 0
                self.desc = desc
            
            def update(self, n=1):
                self.current += n
                if self.current % max(1, self.total // 10) == 0:
                    logger.info(f"{self.desc}: {self.current}/{self.total}")
            
            def close(self):
                logger.info(f"{self.desc}: Complete ({self.total}/{self.total})")
        
        return SimpleProgress(total_items, description)


# Error handling utilities
def safe_execute(func, *args, default=None, log_error=True, **kwargs):
    """Safely execute function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_error:
            logger.error(f"Error executing {func.__name__}: {e}")
        return default


def retry_on_failure(func, max_retries: int = 3, delay: float = 1.0, 
                    backoff_factor: float = 2.0):
    """Retry function on failure with exponential backoff."""
    def wrapper(*args, **kwargs):
        current_delay = delay
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(f"Function {func.__name__} failed after {max_retries} attempts")
                    raise
                
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                time.sleep(current_delay)
                current_delay *= backoff_factor
    
    return wrapper