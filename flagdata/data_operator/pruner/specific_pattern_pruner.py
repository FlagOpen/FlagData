import re
from ..base_operator import BaseOperator

class SpecificPatternPruner(BaseOperator):
    """Class to remove specific patterns from text based on a provided list of patterns."""

    def __init__(self, text_key='content', patterns=None):
        super().__init__()
        self.text_key = text_key
        self.patterns = patterns if patterns is not None else []

    def process(self, sample):
        """Process the sample to remove specific patterns.

        Args:
            sample (dict): A dictionary containing text data.

        Raises:
            ValueError: If the expected text key is not found in the sample.

        Returns:
            dict: The modified sample with specific patterns removed.
        """
        if self.text_key not in sample:
            raise ValueError(f"Expected key '{self.text_key}' not found in the provided sample.")

        text = sample[self.text_key]
        text = self.remove_specific_patterns(text)
        sample[self.text_key] = text.strip()

        return sample

    def remove_specific_patterns(self, text):
        """Remove specific patterns from the text.

        Args:
            text (str): The text to clean.

        Returns:
            str: The text with specific patterns removed.
        """
        for pattern in self.patterns:
            text = re.sub(r'^.*' + re.escape(pattern) + r'.*$\n?', '', text, flags=re.MULTILINE)
        return text
