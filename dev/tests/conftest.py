import pytest
import pandas as pd
from pathlib import Path

@pytest.fixture
def sample_pcds_data():
    return pd.DataFrame({
        'ID': [1, 2, 3],
        'NAME': ['Test1', 'Test2', 'Test3'],
        'VALUE': [100.0, 200.0, 300.0]
    })

@pytest.fixture  
def sample_aws_data():
    return pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Test1', 'Test2', 'Test3'], 
        'value': [100.0, 200.0, 300.0]
    })

@pytest.fixture
def column_mapping():
    return {'ID': 'id', 'NAME': 'name', 'VALUE': 'value'}