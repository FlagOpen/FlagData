from .base_formatter import DataFormatter
import pandas as pd

class CSVFormatter(DataFormatter):
    """Class to read and write CSV files."""

    def read_data(self, file_path, encoding='utf-8'):
        """Reads a CSV file and returns a DataFrame. Encoding parameter added."""
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            raise

    def write_data(self, data, file_path, index=False):
        """Writes a DataFrame to a CSV file. Added a parameter to include or exclude the index."""
        try:
            data.to_csv(file_path, index=index)
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            raise
