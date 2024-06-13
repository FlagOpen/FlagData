import re
from ..base_operator import BaseOperator

class ConsecutiveNewlinesPruner(BaseOperator):
    """Class to remove excessive consecutive newlines from text, allowing a maximum of two consecutive newlines."""

    def __init__(self, text_key='content'):
        super().__init__()
        self.text_key = text_key
        # Regular expression to match more than two consecutive newlines
        self.consecutive_newlines_pat = re.compile(r'\n{3,}')

    def process(self, sample):
        """Process the sample to reduce the number of consecutive newlines.

        Args:
            sample (dict): A dictionary containing text data.

        Raises:
            ValueError: If the expected text key is not found in the sample.

        Returns:
            dict: The modified sample with excessive newlines removed.
        """
        # Ensure the correct key is present in the sample
        if self.text_key not in sample:
            raise ValueError(f"Expected key '{self.text_key}' not found in the provided sample.")

        text = sample[self.text_key]

        # Replace excessive newlines with exactly two newlines
        text = self.consecutive_newlines_pat.sub('\n\n', text)

        # Update the text in the sample dictionary
        sample[self.text_key] = text

        return sample
