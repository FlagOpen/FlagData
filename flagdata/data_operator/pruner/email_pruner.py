import re
from ..base_operator import BaseOperator

class EmailPruner(BaseOperator):
    """Class to clean email addresses from text samples."""

    def __init__(self, text_key='content'):
        super().__init__()
        self.text_key = text_key
        # Regex pattern to match common email formats
        self.email_pat = re.compile(r'[A-Za-z0-9.\-+_]+@[a-z0-9.\-+_]+\.[a-z]+', re.IGNORECASE)

    def process(self, sample):
        """Process the sample to remove email addresses.

        Args:
            sample (dict): A dictionary containing text data.

        Raises:
            ValueError: If the expected text key is not found in the sample.

        Returns:
            dict: The modified sample with email addresses removed.
        """
        # Check if the expected key that contains text is present in the sample
        if self.text_key not in sample:
            raise ValueError(f"Expected key '{self.text_key}' not found in the provided sample.")
        
        text = sample[self.text_key]

        # Remove all matching email addresses using the regex pattern
        text = re.sub(self.email_pat, '', text)

        # Update the text in the sample dictionary
        sample[self.text_key] = text

        return sample
