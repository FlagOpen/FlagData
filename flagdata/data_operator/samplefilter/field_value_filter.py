from ..base_operator import BaseOperator
from typing import Union, List, Tuple

class FieldValueFilter(BaseOperator):
    """Class to filter samples based on the value of a specified field within
    the data, ensuring it matches the target values."""

    def __init__(self,
                 key_path: str = '',
                 valid_values: Union[List, Tuple] = [],
        ):
        """
        Constructor to initialize the filter settings.

        :param key_path: Path to the field in the sample dictionary, with levels separated by '.'.
        :param valid_values: List or tuple of values that are considered valid for the specified field.
        """
        super().__init__()
        self.key_path = key_path
        self.valid_values = set(valid_values)  # Using a set for faster membership checking

    def process(self, sample):
        """
        Processes the given sample to determine if the field value matches the specified valid values.

        :param sample: Dictionary containing data to process.
        :return: Boolean indicating if the sample should be kept or not.
        """
        field_value = sample
        for key in self.key_path.split('.'):
            if key not in field_value:
                raise ValueError(f"'{key}' not found in the keys of the sample.")
            field_value = field_value[key]

        if not isinstance(field_value, (list, tuple)):
            field_value = [field_value]

        return any(value in self.valid_values for value in field_value)

