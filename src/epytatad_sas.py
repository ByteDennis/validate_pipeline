import numpy as np
import pandas as pd
from datetime import datetime, date, time
from typing import Any, List, Dict, Union, Optional
import re

class SASArray:
    """Equivalent to SAS ARRAY statement - groups variables for processing"""
    
    def __init__(self, name: str, variables: List[str], data: Optional[Dict] = None):
        self.name = name
        self.variables = variables
        self.data = data or {}
        self.size = len(variables)
    
    def __getitem__(self, index: int):
        if 1 <= index <= self.size:  # SAS uses 1-based indexing
            var_name = self.variables[index - 1]
            return self.data.get(var_name)
        raise IndexError(f"Array index {index} out of bounds (1-{self.size})")
    
    def __setitem__(self, index: int, value):
        if 1 <= index <= self.size:
            var_name = self.variables[index - 1]
            self.data[var_name] = value
        else:
            raise IndexError(f"Array index {index} out of bounds (1-{self.size})")
    
    def dim(self):
        return self.size

class SASNumeric:
    """SAS numeric data type - handles missing values and formatting"""
    
    def __init__(self, value: Union[float, int, None] = None, format_str: str = "BEST12."):
        self.value = value
        self.format = format_str
        self.missing = value is None or (isinstance(value, float) and np.isnan(value))
    
    def __str__(self):
        if self.missing:
            return "."
        return str(self.value)
    
    def __float__(self):
        return float('nan') if self.missing else float(self.value)
    
    def __int__(self):
        if self.missing:
            raise ValueError("Cannot convert missing value to int")
        return int(self.value)
    
    def is_missing(self):
        return self.missing

class SASCharacter:
    """SAS character data type with fixed length and formatting"""
    
    def __init__(self, value: str = "", length: int = 200, format_str: str = None):
        self.length = length
        self.format = format_str or f"${length}."
        self.value = self._pad_or_truncate(value)
    
    def _pad_or_truncate(self, value: str) -> str:
        if len(value) > self.length:
            return value[:self.length]
        return value.ljust(self.length)
    
    def __str__(self):
        return self.value.rstrip()  # Remove trailing spaces for display
    
    def __len__(self):
        return self.length
    
    def strip(self):
        return self.value.strip()

class SASDate:
    """SAS date - stored as days since January 1, 1960"""
    
    SAS_EPOCH = datetime(1960, 1, 1)
    
    def __init__(self, value: Union[int, datetime, date, None] = None):
        if value is None:
            self.sas_value = None
        elif isinstance(value, (datetime, date)):
            delta = value - self.SAS_EPOCH.date() if isinstance(value, date) else value - self.SAS_EPOCH
            self.sas_value = delta.days
        else:
            self.sas_value = int(value)
    
    def to_python_date(self) -> Optional[date]:
        if self.sas_value is None:
            return None
        return (self.SAS_EPOCH + pd.Timedelta(days=self.sas_value)).date()
    
    def __str__(self):
        if self.sas_value is None:
            return "."
        return self.to_python_date().strftime("%d%b%Y").upper()

class SASDateTime:
    """SAS datetime - stored as seconds since January 1, 1960 00:00:00"""
    
    SAS_EPOCH = datetime(1960, 1, 1)
    
    def __init__(self, value: Union[float, datetime, None] = None):
        if value is None:
            self.sas_value = None
        elif isinstance(value, datetime):
            delta = value - self.SAS_EPOCH
            self.sas_value = delta.total_seconds()
        else:
            self.sas_value = float(value)
    
    def to_python_datetime(self) -> Optional[datetime]:
        if self.sas_value is None:
            return None
        return self.SAS_EPOCH + pd.Timedelta(seconds=self.sas_value)
    
    def __str__(self):
        if self.sas_value is None:
            return "."
        return self.to_python_datetime().strftime("%d%b%Y:%H:%M:%S")

class SASTime:
    """SAS time - stored as seconds since midnight"""
    
    def __init__(self, value: Union[float, time, None] = None):
        if value is None:
            self.sas_value = None
        elif isinstance(value, time):
            self.sas_value = value.hour * 3600 + value.minute * 60 + value.second
        else:
            self.sas_value = float(value)
    
    def to_python_time(self) -> Optional[time]:
        if self.sas_value is None:
            return None
        hours = int(self.sas_value // 3600)
        minutes = int((self.sas_value % 3600) // 60)
        seconds = int(self.sas_value % 60)
        return time(hours, minutes, seconds)
    
    def __str__(self):
        if self.sas_value is None:
            return "."
        return self.to_python_time().strftime("%H:%M:%S")

class SASFormat:
    """SAS format specification"""
    
    def __init__(self, name: str, width: int = 8, decimals: int = 0):
        self.name = name.upper()
        self.width = width
        self.decimals = decimals
    
    def apply(self, value):
        """Apply format to a value"""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "."
        
        if self.name.startswith('$'):
            # Character format
            return str(value)[:self.width].ljust(self.width)
        elif self.name in ['BEST', 'F', 'COMMA', 'DOLLAR']:
            # Numeric formats
            if self.decimals > 0:
                return f"{float(value):.{self.decimals}f}"
            return str(int(value))
        elif self.name == 'DATE':
            if isinstance(value, (int, float)):
                sas_date = SASDate(value)
                return str(sas_date)
        elif self.name == 'DATETIME':
            if isinstance(value, (int, float)):
                sas_datetime = SASDateTime(value)
                return str(sas_datetime)
        elif self.name == 'TIME':
            if isinstance(value, (int, float)):
                sas_time = SASTime(value)
                return str(sas_time)
        
        return str(value)

class SASInformat:
    """SAS informat for reading data"""
    
    def __init__(self, name: str, width: int = 8, decimals: int = 0):
        self.name = name.upper()
        self.width = width
        self.decimals = decimals
    
    def read(self, input_string: str):
        """Parse input string according to informat"""
        if not input_string or input_string.strip() == '.':
            return None
        
        input_string = input_string.strip()
        
        if self.name.startswith('$'):
            # Character informat
            return input_string[:self.width]
        elif self.name in ['F', 'BEST', 'COMMA', 'DOLLAR']:
            # Numeric informats
            try:
                # Remove common formatting characters
                clean_str = re.sub(r'[,$%]', '', input_string)
                return float(clean_str)
            except ValueError:
                return None
        elif self.name == 'DATE':
            # Date informat parsing would be more complex in practice
            try:
                return datetime.strptime(input_string, '%d%b%Y').date()
            except ValueError:
                return None
        
        return input_string

class SASMissing:
    """SAS missing value representation"""
    
    NUMERIC_MISSING = float('nan')
    CHARACTER_MISSING = ""
    
    @staticmethod
    def is_missing_numeric(value):
        return value is None or (isinstance(value, float) and np.isnan(value))
    
    @staticmethod
    def is_missing_character(value):
        return value is None or value == ""
    
    @staticmethod
    def coalesce(*values):
        """Return first non-missing value (like SAS COALESCE function)"""
        for value in values:
            if not SASMissing.is_missing_numeric(value) and not SASMissing.is_missing_character(value):
                return value
        return None

class SASLabel:
    """SAS variable label"""
    
    def __init__(self, text: str = ""):
        self.text = text[:256]  # SAS labels max 256 characters
    
    def __str__(self):
        return self.text

class SASLength:
    """SAS variable length specification"""
    
    def __init__(self, length: int, is_character: bool = False):
        self.length = length
        self.is_character = is_character
        
        # SAS length limits
        if is_character:
            self.length = min(length, 32767)  # Max character length
        else:
            self.length = min(max(length, 3), 8)  # Numeric length 3-8

# Example usage and demonstration
if __name__ == "__main__":
    # Array example
    sales_array = SASArray("sales", ["jan_sales", "feb_sales", "mar_sales"])
    sales_array[1] = 1000
    sales_array[2] = 1200
    sales_array[3] = 1150
    print(f"Q1 Sales: {sales_array[1]}, {sales_array[2]}, {sales_array[3]}")
    
    # Numeric with missing
    price = SASNumeric(19.95)
    missing_price = SASNumeric(None)
    print(f"Price: {price}, Missing: {missing_price}")
    
    # Character with fixed length
    name = SASCharacter("John Doe", length=20)
    print(f"Name: '{name}' (length: {len(name)})")
    
    # Date handling
    today = SASDate(datetime.now())
    print(f"SAS Date: {today}")
    
    # Format application
    dollar_fmt = SASFormat("DOLLAR", width=10, decimals=2)
    formatted_value = dollar_fmt.apply(1234.56)
    print(f"Formatted: {formatted_value}")