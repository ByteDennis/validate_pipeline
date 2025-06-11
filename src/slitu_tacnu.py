import pandas as pd
import pickle
from typing import Dict, List, Tuple, Any


class DataMismatchProcessor:
    """Process CSV and pickle data to generate a mismatch matrix."""
    
    def __init__(self, csv_path: str, pkl_path: str):
        """
        Initialize the processor with file paths.
        
        Args:
            csv_path: Path to CSV file containing mismatched dates
            pkl_path: Path to pickle file containing dataframes
        """
        self.csv_path = csv_path
        self.pkl_path = pkl_path
        self.mismatched_dates_df = None
        self.data_dict = None
        
    def load_data(self) -> None:
        """Load CSV and pickle data."""
        # Load CSV with mismatched dates
        self.mismatched_dates_df = pd.read_csv(self.csv_path, skipinitialspace=True)
        
        # Load pickle data
        with open(self.pkl_path, 'rb') as f:
            self.data_dict = pickle.load(f)
    
    def _parse_dates(self, date_string: str) -> List[str]:
        """Parse semicolon-separated dates from string."""
        if pd.isna(date_string) or not date_string.strip():
            return []
        return [date.strip() for date in date_string.split(';')]
    
    def _get_record_count(self, df: pd.DataFrame, date: str) -> int:
        """Get record count for a specific date from dataframe."""
        df['date'] = pd.to_datetime(df['date'])
        target_date = pd.to_datetime(date)
        
        matching_rows = df[df['date'] == target_date]
        return matching_rows['number of record'].iloc[0] if not matching_rows.empty else 0
    
    def _check_mismatch(self, table: str, date: str) -> Tuple[bool, str]:
        """
        Check if there's a mismatch for a table on a specific date.
        
        Returns:
            Tuple of (has_mismatch, display_string)
        """
        if table not in self.data_dict:
            return False, "No"
        
        table_data = self.data_dict[table]
        pcds_df = table_data.get('pcds')
        aws_df = table_data.get('aws')
        
        if pcds_df is None or aws_df is None:
            return False, "No"
        
        pcds_count = self._get_record_count(pcds_df, date)
        aws_count = self._get_record_count(aws_df, date)
        
        if pcds_count != aws_count:
            return True, f"Yes ({pcds_count} : {aws_count})"
        else:
            return False, "No"
    
    def generate_matrix(self) -> pd.DataFrame:
        """
        Generate the final mismatch matrix.
        
        Returns:
            DataFrame with mismatch results
        """
        if self.mismatched_dates_df is None or self.data_dict is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get all unique dates from all tables
        all_dates = set()
        for _, row in self.mismatched_dates_df.iterrows():
            dates = self._parse_dates(row['mismatched date'])
            all_dates.update(dates)
        
        all_dates = sorted(all_dates)
        tables = self.mismatched_dates_df['tbl'].tolist()
        
        # Create result matrix
        result_data = []
        for table in tables:
            row_data = {'tbl': table}
            for date in all_dates:
                _, display_value = self._check_mismatch(table, date)
                row_data[date] = display_value
            result_data.append(row_data)
        
        return pd.DataFrame(result_data)
    
    def process(self) -> pd.DataFrame:
        """
        Complete processing pipeline.
        
        Returns:
            Final mismatch matrix DataFrame
        """
        self.load_data()
        return self.generate_matrix()