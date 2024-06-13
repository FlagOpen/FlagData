import sys
from ..base_operator import BaseOperator

class AvgLineLengthFilter(BaseOperator):
    """Class to evaluate samples based on the average line length within the text,
    ensuring it falls within a specified range."""

    def __init__(self,
                 text_key='content',
                 min_length: int = 10,
                 max_length: int = 10000,
                 ):
        """
        Constructor to initialize the evaluator settings.

        :param text_key: Key in the sample dictionary containing the text to evaluate.
        :param min_length: The minimum average line length required to keep a sample.
        :param max_length: The maximum average line length allowed to keep a sample.
        """
        super().__init__()

        self.text_key = text_key
        self.min_length = min_length
        self.max_length = max_length

    def evaluate_avg_line_length(self, sample):
        """Calculates the average line length of the text."""
        lines = sample[self.text_key].splitlines()
        total_length = sum(len(line) for line in lines)
        avg_length = total_length / len(lines) if lines else 0
        return avg_length

    def process(self, sample):
        """
        Process the given sample to determine if it meets the criteria based on the average line length.

        :param sample: Dictionary containing data to process.
        :return: Boolean indicating if the sample should be kept or not.
        """
        average_length = self.evaluate_avg_line_length(sample)
        return self.min_length <= average_length <= self.max_length
