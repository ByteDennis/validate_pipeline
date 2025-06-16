#!/usr/bin/env python3
"""
Database utilities for PCDS and AWS connections.
Simplified from multiple utility files.
"""

import os
import re
from typing import Dict, Any, Optional
import pandas as pd
import pandas.io.sql as psql
from loguru import logger

from .types import PLATFORM, NONEXIST_TABLE, NONEXIST_DATEVAR


class DatabaseConnector:
    """Unified database connector for PCDS and AWS."""
    
    def __init__(self, platform: PLATFORM):
        self.platform = platform
        self.svc2server = {
            'p_uscb_cnsmrlnd_svc': '21P',
            'p_uscb_rft_svc': '30P',
            'pcds_svc': '00P'
        }
    
    def query(self, sql: str, svc: Optional[str] = None, 
              return_svc: bool = False, **kwargs) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        if self.platform == 'PCDS':
            return self._query_pcds(sql, svc, return_svc, **kwargs)
        else:
            return self._query_aws(sql, **kwargs)
    
    def _query_pcds(self, sql: str, svc: Optional[str] = None, 
                   return_svc: bool = False, **kwargs) -> pd.DataFrame:
        """Execute PCDS query."""
        try:
            with self._pcds_connect(svc) as conn:
                df = psql.read_sql_query(sql, conn)
                if return_svc:
                    return df, svc
                return df
        except Exception as e:
            logger.error(f"PCDS query failed: {e}")
            raise NONEXIST_TABLE(f"PCDS query error: {e}")
    
    def _query_aws(self, sql: str, **kwargs) -> pd.DataFrame:
        """Execute AWS Athena query."""
        try:
            import pyathena as pa
            conn = pa.connect(
                s3_staging_dir="s3://355538383407-us-east-1-athena-output/uscb-analytics/",
                region_name="us-east-1",
            )
            return psql.read_sql_query(sql, conn)
        except Exception as e:
            logger.error(f"AWS query failed: {e}")
            raise NONEXIST_TABLE(f"AWS query error: {e}")
    
    def _pcds_connect(self, service_name: str):
        """Create PCDS Oracle connection."""
        import oracledb
        
        if service_name not in self.svc2server:
            raise pd.errors.DatabaseError("Service Name Is Not Provided")
        
        # Resolve LDAP
        ldap_service = 'ldap://oid.barcapint.com:4050'
        dns_tns = self._solve_ldap(
            f'{ldap_service}/{service_name},cn=OracleContext,dc=barcapint,dc=com'
        )
        
        # Get credentials
        PCDS_PWD = f'PCDS_{self.svc2server[service_name]}'
        usr, pwd = os.environ['PCDS_USR'], os.environ[PCDS_PWD]
        
        return oracledb.connect(user=usr, password=pwd, dsn=dns_tns)
    
    def _solve_ldap(self, ldap_dsn: str) -> str:
        """Resolve LDAP DSN to TNS connect string."""
        from ldap3 import Server, Connection
        
        pattern = r"^ldap:\/\/(.+)\/(.+)\,(cn=OracleContext.*)$"
        match = re.match(pattern, ldap_dsn)
        if not match:
            return None
        
        ldap_server, db, ora_context = match.groups()
        server = Server(ldap_server)
        conn = Connection(server)
        conn.bind()
        conn.search(ora_context, f"(cn={db})", attributes=['orclNetDescString'])
        
        return conn.entries[0].orclNetDescString.value


class QueryBuilder:
    """SQL query builder for different platforms."""
    
    # PCDS (Oracle) queries
    PCDS_QUERIES = {
        'meta': """
            SELECT column_name,
                   data_type || CASE
                       WHEN data_type = 'NUMBER' THEN 
                           CASE WHEN data_precision IS NULL AND data_scale IS NULL
                                THEN NULL
                           ELSE '(' || TO_CHAR(data_precision) || ',' || TO_CHAR(data_scale) || ')'
                           END
                       WHEN data_type LIKE '%CHAR%' THEN '(' || TO_CHAR(data_length) || ')'
                       ELSE NULL
                   END AS data_type
            FROM all_tab_cols
            WHERE table_name = UPPER('{table}')
            ORDER BY column_id
        """,
        'nrow': "SELECT COUNT(*) AS nrow FROM {table}",
        'date': "SELECT {date}, count(*) AS nrows FROM {table} GROUP BY {date}"
    }
    
    # AWS (Athena) queries  
    AWS_QUERIES = {
        'meta': """
            SELECT column_name, data_type 
            FROM information_schema.columns
            WHERE table_schema = LOWER('{db}') AND table_name = LOWER('{table}')
        """,
        'nrow': "SELECT COUNT(*) AS nrow FROM {db}.{table}",
        'date': "SELECT {date}, count(*) AS nrows FROM {db}.{table} GROUP BY {date}"
    }
    
    @classmethod
    def get_query(cls, platform: PLATFORM, query_type: str, **params) -> str:
        """Get formatted SQL query."""
        queries = cls.PCDS_QUERIES if platform == 'PCDS' else cls.AWS_QUERIES
        template = queries[query_type]
        return template.format(**params).strip()


def costly_query(sql: str, connection) -> pd.DataFrame:
    """Execute query with caching (placeholder for actual cache)."""
    return psql.read_sql_query(sql, connection)


# Migration-specific database operations
def get_table_metadata(platform: PLATFORM, table: str, **kwargs) -> pd.DataFrame:
    """Get table column metadata."""
    connector = DatabaseConnector(platform)
    query = QueryBuilder.get_query(platform, 'meta', table=table, **kwargs)
    return connector.query(query, **kwargs)


def get_table_row_count(platform: PLATFORM, table: str, **kwargs) -> pd.DataFrame:
    """Get table row count."""
    connector = DatabaseConnector(platform)
    query = QueryBuilder.get_query(platform, 'nrow', table=table, **kwargs)
    return connector.query(query, **kwargs)


def get_date_distribution(platform: PLATFORM, table: str, date_col: str, 
                         **kwargs) -> pd.DataFrame:
    """Get date column distribution."""
    try:
        connector = DatabaseConnector(platform)
        query = QueryBuilder.get_query(platform, 'date', table=table, 
                                      date=date_col, **kwargs)
        return connector.query(query, **kwargs)
    except Exception:
        raise NONEXIST_DATEVAR(f"Date column {date_col} not found in {table}")


# Data type mapping utilities
def map_pcds_to_aws_type(pcds_type: str, aws_type: str) -> bool:
    """Check if PCDS and AWS data types are compatible."""
    # Handle Oracle NUMBER type
    if pcds_type == 'NUMBER':
        return aws_type == 'double'
    
    # Handle Oracle NUMBER with precision/scale
    elif pcds_type.startswith('NUMBER'):
        match = re.match(r'NUMBER\(\d*,(\d+)\)', pcds_type)
        if match:
            scale = match.group(1)
            aws_match = re.match(r'decimal\(\d*,(\d+)\)', aws_type)
            return bool(aws_match and aws_match.group(1) == scale)
    
    # Handle VARCHAR2 types
    elif pcds_type.startswith('VARCHAR2'):
        return pcds_type.replace('VARCHAR2', 'varchar') == aws_type
    
    # Handle CHAR types
    elif pcds_type.startswith('CHAR'):
        n_match = re.match(r'CHAR\((\d+)\)', pcds_type)
        if n_match:
            n = n_match.group(1)
            return not (aws_type.startswith('VARCHAR') and n != '1')
    
    # Handle DATE types
    elif pcds_type == 'DATE':
        return aws_type == 'date' or aws_type.startswith('timestamp')
    
    # Handle TIMESTAMP types
    elif pcds_type.startswith('TIMESTAMP'):
        return aws_type.startswith('timestamp')
    
    return False


# Test connectivity
def test_connections() -> Dict[str, bool]:
    """Test database connections for both platforms."""
    results = {}
    
    # Test PCDS
    try:
        pcds_connector = DatabaseConnector('PCDS')
        pcds_connector.query("SELECT 1 FROM DUAL", svc='pcds_svc')
        results['PCDS'] = True
    except Exception as e:
        logger.error(f"PCDS connection failed: {e}")
        results['PCDS'] = False
    
    # Test AWS
    try:
        aws_connector = DatabaseConnector('AWS')
        aws_connector.query("SELECT 1")
        results['AWS'] = True
    except Exception as e:
        logger.error(f"AWS connection failed: {e}")
        results['AWS'] = False
    
    return results