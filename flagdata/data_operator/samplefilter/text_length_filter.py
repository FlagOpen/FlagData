from ..base_operator import BaseOperator

class TextLengthFilter(BaseOperator):
    """Class to filter samples based on the length of text within them,
    ensuring the length is within a specified range."""

    def __init__(self,
                 text_key='content',
                 min_length: int = 10,
                 max_length: int = 100000000,
                 ):
        """
        Constructor to initialize the filter settings.

        :param text_key: Key in the sample dictionary containing the text to evaluate.
        :param min_length: The minimum length of text required to keep a sample.
        :param max_length: The maximum length of text allowed to keep a sample.
        """
        super().__init__()
        self.text_key = text_key
        self.min_length = min_length
        self.max_length = max_length

    def process(self, sample):
        """
        Processes the given sample to determine if the text length meets the specified bounds.

        :param sample: Dictionary containing data to process.
        :return: Boolean indicating if the sample should be kept or not.
        """
        text_length = len(sample.get(self.text_key, ""))
        return self.min_length <= text_length <= self.max_length
