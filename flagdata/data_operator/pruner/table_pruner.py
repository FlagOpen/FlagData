import regex as re
from ..base_operator import BaseOperator

class TablePruner(BaseOperator):
    """
    Pruner to remove table texts from text samples.

    Uses a regular expression to identify and remove text that appears in a table format,
    within the specified range of column numbers.
    """

    def __init__(self, text_key='content', min_col = 3, max_col = 30):
        self.text_key = text_key
        self.min_col = min_col
        self.max_col = max_col
        # Regular expression pattern to match text formatted as a table with a specified range of columns
        self.pattern = r'(?<=\n)((\S+)(\t|\s{2,})(\S+)){%d,}\n+'

    def process(self, sample):
        """Processes the given sample to remove text that forms table-like structures.

        Args:
            sample (dict): A dictionary containing text data.

        Returns:
            dict: The modified sample with table texts removed based on the specified column range.
        """
        if self.text_key not in sample:
            raise ValueError(f"Expected key '{self.text_key}' not found in the provided sample.")

        text = sample[self.text_key]
        # Apply the regex to remove table text from the specified column count
        for i in range(self.min_col - 1, self.max_col):
            pattern = re.compile(self.pattern % i)
            text = pattern.sub('', text)

        sample[self.text_key] = text
        return sample
