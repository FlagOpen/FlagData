import sys
from ..base_operator import BaseOperator

class AlphanumericRatioFilter(BaseOperator):
    """Class to filter samples based on the ratio of alphanumeric characters
    within the text, ensuring it falls within a specified range."""

    def __init__(self,
                 text_key='content',
                 min_ratio: float = 0.0,
                 max_ratio: float = 0.8,):
        """
        Constructor to initialize the filter settings.

        :param text_key: Key in the sample dictionary containing the text to evaluate.
        :param min_ratio: The minimum ratio of alphanumeric characters required to keep a sample.
        :param max_ratio: The maximum ratio of alphanumeric characters allowed to keep a sample.
        """
        super().__init__()

        self.text_key = text_key
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def evaluate_content(self, sample):
        """Calculate the alphanumeric ratio and update the sample's evaluation."""
        content = sample[self.text_key]
        alnum_count = sum(char.isalnum() for char in content)
        total_count = len(content)
        alnum_ratio = alnum_count / total_count if total_count != 0 else 0.0

        return alnum_ratio

    def process(self, sample):
        """
        Process the given sample to determine if it meets the criteria based on the alphanumeric ratio.

        :param sample: Dictionary containing data to process.
        :return: Boolean indicating if the sample should be kept or not.
        """
        alnum_ratio = self.evaluate_content(sample)
        return self.min_ratio <= alnum_ratio <= self.max_ratio
