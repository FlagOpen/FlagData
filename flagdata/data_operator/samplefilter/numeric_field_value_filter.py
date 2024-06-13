import sys
from ..base_operator import BaseOperator

class NumericFieldValueFilter(BaseOperator):
    """Class to filter samples based on numeric values of a specified field
    within the text, ensuring these values fall within a specified range."""

    def __init__(self,
                 key_path: str = '',
                 min_value: float = -sys.maxsize,
                 max_value: float = sys.maxsize,
             ):
        """
        Constructor to initialize the filter settings.

        :param key_path: Path to the field in the sample dictionary, with levels separated by '.'.
        :param min_value: The minimum acceptable value for the numeric field to keep a sample.
        :param max_value: The maximum acceptable value for the numeric field to keep a sample.
        """
        super().__init__()
        self.key_path = key_path
        self.min_value = min_value
        self.max_value = max_value

    def validate_numeric(self, value):
        """Check if the provided value is numeric and convert if necessary."""
        try:
            numeric_value = float(value)
            return True, numeric_value
        except ValueError:
            return False, None

    def process(self, sample):
        """
        Processes the given sample to determine if the numeric field value meets the specified range.

        :param sample: Dictionary containing data to process.
        :return: Boolean indicating if the sample should be kept or not.
        """
        field_value = sample
        for key in self.key_path.split('.'):
            if key not in field_value:
                raise ValueError(f"'{key}' not found in the keys of the sample.")
            field_value = field_value[key]

        is_numeric, numeric_value = self.validate_numeric(field_value)
        if is_numeric:
            return self.min_value <= numeric_value <= self.max_value
        else:
            return False
