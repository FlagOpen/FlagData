import unicodedata
from ..base_operator import BaseOperator

class ControlCharPruner(BaseOperator):
    """Class to remove control characters from text, preserving newlines."""

    def __init__(self, text_key='content'):
        super().__init__()
        self.text_key = text_key

    def process(self, sample):
        """Process the sample to remove control characters except for newlines.

        Args:
            sample (dict): A dictionary containing text data.

        Raises:
            ValueError: If the expected text key is not found in the sample.

        Returns:
            dict: The modified sample with control characters removed, except newlines.
        """
        if self.text_key not in sample:
            raise ValueError(f"Expected key '{self.text_key}' not found in the provided sample.")

        text = sample[self.text_key]
        cleaned_text = self.remove_control_chars(text)
        sample[self.text_key] = cleaned_text

        return sample

    def remove_control_chars(self, text):
        """Remove control characters from text while preserving newlines.

        Args:
            text (str): The text to clean.

        Returns:
            str: The text with control characters removed, except for newlines.
        """
        return ''.join(ch for ch in text if (unicodedata.category(ch) != 'Cc' or ch == '\n'))
