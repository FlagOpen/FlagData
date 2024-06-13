from ..base_operator import BaseOperator

class MaxLineLengthFilter(BaseOperator):
    """Class to filter samples based on the maximum line length within the text,
    ensuring it falls within a specified range."""

    def __init__(self,
                 text_key='content',
                 min_length: int = 10,
                 max_length: int = 10000,
               ):
        """
        Constructor to initialize the pruner settings.

        :param text_key: Key in the sample dictionary containing the text to evaluate.
        :param min_length: The minimum acceptable maximum line length to keep a sample.
        :param max_length: The maximum acceptable maximum line length to keep a sample.
        """
        super().__init__()
        self.text_key = text_key
        self.min_length = min_length
        self.max_length = max_length

    def evaluate_max_line_length(self, text):
        """Calculates the maximum line length of the text."""
        lines = text.splitlines()
        max_length = max((len(line) for line in lines), default=0)
        return max_length

    def process(self, sample):
        """
        Processes the given sample to determine if it meets the criteria based on the maximum line length.

        :param sample: Dictionary containing data to process.
        :return: Boolean indicating if the sample should be kept or not.
        """
        text = sample[self.text_key]
        max_line_length = self.evaluate_max_line_length(text)
        return self.min_length <= max_line_length <= self.max_length
