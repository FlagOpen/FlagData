from ..base_operator import BaseOperator
from typing import Union, List, Tuple

class SuffixFilter(BaseOperator):
    """Class to filter samples based on the suffix of a specified field within
    the data, ensuring the field ends with specified suffixes."""

    def __init__(self,
                 field_key: str = '',
                 allowed_suffixes: Union[str, List[str], Tuple[str, ...]] = [],
          ):
        """
        Constructor to initialize the filter settings.

        :param field_key: Key in the sample dictionary containing the field to evaluate for suffixes.
        :param allowed_suffixes: Suffix or list of suffixes that are acceptable to retain a sample.
        """
        super().__init__()
        self.field_key = field_key
        self.allowed_suffixes = set(allowed_suffixes if isinstance(allowed_suffixes, (list, tuple)) else [allowed_suffixes])

    def process(self, sample):
        """
        Processes the given sample to determine if the field ends with an allowed suffix.

        :param sample: Dictionary containing data to process.
        :return: Boolean indicating if the sample should be kept or not.
        """
        field_value = sample.get(self.field_key, '')
        return any(field_value.endswith(suffix) for suffix in self.allowed_suffixes)
