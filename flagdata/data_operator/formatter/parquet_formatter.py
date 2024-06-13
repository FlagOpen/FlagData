from .base_formatter import DataFormatter
import pandas as pd

class ParquetFormatter(DataFormatter):
    def read_data(self, file_path):
        """Reads a Parquet file and returns a DataFrame."""
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            raise

    def write_data(self, data, file_path):
        """Writes a DataFrame to a Parquet file."""
        try:
            data.to_parquet(file_path)
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            raise
