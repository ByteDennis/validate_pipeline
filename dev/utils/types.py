#!/usr/bin/env python3
"""
Data types and configuration classes for the migration pipeline.
Consolidated from utils_type.py and paste files.
"""

import pprint
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Literal
from pathlib import Path
import pandas as pd


# Enums
class PullStatus(Enum):
    NONEXIST_PCDS = 'Nonexisting PCDS Table'
    NONEXIST_AWS = 'Nonexisting AWS Table'
    NONDATE_PCDS = 'Nonexisting Date Variable in PCDS'
    NONDATE_AWS = 'Nonexisting Date Variable in AWS'
    EMPTY_PCDS = 'Empty PCDS Table'
    EMPTY_AWS = 'Empty AWS Table'
    NO_MAPPING = 'Column Mapping Not Provided'
    SUCCESS = 'Successful Data Access'


# Type aliases
PLATFORM = Literal['PCDS', 'AWS']
TPartition = Literal['none', 'year', 'year_month']


# Exceptions
class NONEXIST_TABLE(Exception):
    """Table does not exist"""
    pass


class NONEXIST_DATEVAR(Exception):
    """Date variable does not exist"""
    pass


# Helper functions
def read_str_lst(lst_str: str, sep: str = '\n') -> List[str]:
    """Parse delimited string into list."""
    return [x.strip() for x in lst_str.strip().split(sep) if x.strip()]


def read_dstr_lst(dct_str: str, sep: str = '=') -> Dict[str, str]:
    """Parse string with key=value pairs into dictionary."""
    result = {}
    for line in read_str_lst(dct_str):
        if sep in line:
            key, value = line.split(sep, 1)
            result[key.strip()] = value.strip()
    return result


# Base configuration class
@dataclass
class BaseType:
    """Base class for configuration objects."""
    
    def __post_init__(self):
        for field in fields(self):
            if is_dataclass(field.type):
                field_val = field.type(**getattr(self, field.name))
                setattr(self, field.name, field_val)

    def tolog(self, indent: int = 1, padding: str = '') -> str:
        """Generate formatted string for logging."""
        def get_val(x, pad):
            if isinstance(x, BaseType):
                return x.tolog(indent, pad)
            elif isinstance(x, Dict):
                return pprint.pformat(x, indent)
            else:
                return repr(x)
        
        cls_name = self.__class__.__name__
        padding = padding + '\t' * indent
        field_strings = [
            f'{padding}{k}={get_val(v, padding)}' 
            for k, v in vars(self).items()
        ]
        return f'{cls_name}(\n' + ',\n'.join(field_strings) + '\n)'


# Configuration structures
@dataclass
class MetaRange:
    """Row range configuration."""
    start_rows: Optional[int]
    end_rows: Optional[int]

    def __iter__(self):
        yield from [self.start_rows or 1, self.end_rows or float('inf')]


@dataclass
class MetaTable(BaseType):
    """Table configuration."""
    file: Path
    sheet: str
    skip_rows: int
    select_cols: str
    select_rows: str

    def __post_init__(self):
        self.select_cols = read_dstr_lst(self.select_cols)
        self.select_rows = read_str_lst(str(self.select_rows))


@dataclass
class LogConfig:
    """Logging configuration."""
    level: str
    format: str
    file: str
    overwrite: bool

    def todict(self) -> Dict[str, Any]:
        return {
            'level': self.level.upper(), 
            'format': self.format,
            'sink': self.file,
            'mode': 'w' if self.overwrite else 'a'
        }


@dataclass
class S3Config:
    """S3 configuration."""
    run: Path
    data: Path


@dataclass
class CSVConfig:
    """CSV output configuration."""
    file: Path
    columns: str
    
    def __post_init__(self):
        self.columns = read_str_lst(self.columns)


@dataclass
class MetaInput(BaseType):
    """Input configuration for analysis."""
    name: str
    step: str
    env: str
    range: MetaRange
    table: MetaTable


@dataclass
class MetaOutput(BaseType):
    """Output configuration for analysis."""
    folder: Path
    to_pkl: Path
    csv: CSVConfig
    to_s3: S3Config
    log: LogConfig


@dataclass
class ColumnMap:
    """Column mapping configuration."""
    to_json: Path
    file: Path
    na_str: str
    overwrite: bool
    excludes: List[str]
    pcds_col: str
    aws_col: str

    def __post_init__(self):
        self.excludes = read_str_lst(self.excludes) if isinstance(self.excludes, str) else self.excludes
        self.pcds_col = read_str_lst(self.pcds_col)
        self.aws_col = read_str_lst(self.aws_col)


@dataclass
class MetaConfig(BaseType):
    """Main configuration for migration analysis."""
    input: MetaInput
    output: MetaOutput
    column_maps: ColumnMap


# Data structures
@dataclass
class MetaMerge:
    """Schema merge results."""
    unique_pcds: List[str]
    unique_aws: List[str]
    col_mapping: pd.DataFrame
    mismatches: str
    uncaptured: str


@dataclass
class MetaOut:
    """Metadata output structure."""
    col2COL: Dict[str, str]
    col2type: Dict[str, str]
    infostr: str
    rowvar: str
    rowexclude: List[str]
    nrows: int


@dataclass
class TimeRange:
    """Time range for data filtering."""
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    def __str__(self) -> str:
        if self.start_date and self.end_date:
            return f"{self.start_date}_{self.end_date}"
        elif self.start_date:
            return f"from_{self.start_date}"
        elif self.end_date:
            return f"until_{self.end_date}"
        return "all_time"


@dataclass
class UploadMetadata:
    """Upload operation metadata."""
    table_name: str
    upload_timestamp: str
    upload_date: str
    s3_path: str
    row_count: int
    file_size_mb: Optional[float] = None


# Utility dictionary class
class UDict(dict):
    """Case-insensitive dictionary."""
    
    def __getitem__(self, key):
        return super().__getitem__(self._match(key))
    
    def __contains__(self, key):
        try:
            self._match(key)
            return True
        except KeyError:
            return False
    
    def _match(self, key):
        for k in self:
            if k.lower() == key.lower():
                return k
        raise KeyError(key)


# Migration-specific data types
@dataclass
class TableComparison:
    """Table comparison results."""
    table_name: str
    pcds_row_count: int
    aws_row_count: int
    row_match: bool
    column_matches: int
    column_mismatches: int
    type_mismatches: List[str]
    unique_pcds_columns: List[str]
    unique_aws_columns: List[str]


@dataclass
class StatisticsComparison:
    """Statistics comparison results."""
    table_name: str
    total_columns: int
    matching_columns: int
    mismatched_columns: List[str]
    mismatch_percentage: float
    detailed_results: Dict[str, Any]


@dataclass
class AlignmentResult:
    """Data alignment results."""
    table_name: str
    status: str
    aligned_columns: List[str]
    s3_row_count: int
    athena_row_count: int
    match_percentage: float
    comparison_details: Dict[str, Any]