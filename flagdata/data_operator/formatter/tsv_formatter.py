from .base_formatter import DataFormatter
import pandas as pd

class TSVFormatter(DataFormatter):
    def read_data(self, file_path, encoding='utf-8'):
        """Reads a TSV file and returns a DataFrame. An encoding parameter has been added."""
        try:
            return pd.read_csv(file_path, sep='\t', encoding=encoding)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            raise

    def write_data(self, data, file_path, index=False):
        """Writes a DataFrame to a TSV file, using a tab as the separator. An option to include the index has been added."""
        try:
            data.to_csv(file_path, sep='\t', index=index)
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            raise
