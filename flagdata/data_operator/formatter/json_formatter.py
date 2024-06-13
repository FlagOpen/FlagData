from .base_formatter import DataFormatter
import pandas as pd
import json

class JSONFormatter(DataFormatter):
    def read_data(self, file_path, auto_detect_lines=True):
        """Reads a JSON file and returns a DataFrame. Can automatically detect whether to use lines=True, default is True."""
        if auto_detect_lines:
            lines = self._detect_lines(file_path)
        else:
            lines = True  # Default to handling JSON Lines format

        try:
            return pd.read_json(file_path, lines=lines)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            raise

    def write_data(self, data, file_path, orient='records', lines=True):
        """Writes a DataFrame to a JSON file. Added parameters to control the output format."""
        try:
            data.to_json(file_path, orient=orient, lines=lines)
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            raise

    def _detect_lines(self, file_path):
        """Attempts to detect if the JSON file format is one JSON object per line."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    # Try to parse the first line; if successful, assume it is JSON Lines format
                    json.loads(line)
                    return True
        except json.JSONDecodeError:
            # If the first line parsing fails, assume it is not JSON Lines format
            return False
        return False  # Default to not using lines=True if there's no data or in other cases
