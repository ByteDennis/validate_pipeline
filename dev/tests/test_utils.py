import pytest
import pandas as pd
from utils.data import DataTypeConverter, DataValidator

def test_data_type_converter():
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
    column_types = {'col1': 'NUMBER', 'col2': 'VARCHAR2(10)'}
    
    result = DataTypeConverter.convert_pcds_column_types(df, column_types)
    assert result['col1'].dtype == 'float64'
    assert result['col2'].dtype == 'string'

def test_data_validator(sample_pcds_data, sample_aws_data, column_mapping):
    result = DataValidator.validate_schema_compatibility(
        sample_pcds_data, sample_aws_data, column_mapping
    )
    assert result['compatible'] == True