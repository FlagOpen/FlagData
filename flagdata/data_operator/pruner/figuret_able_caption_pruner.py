import re
from ..base_operator import BaseOperator

class FigureTableCaptionPruner(BaseOperator):
    """Class to remove figure and table caption line from lines"""

    def __init__(self, text_key='content'):
        super().__init__()
        self.text_key = text_key
        # Pattern to match figure and table annotations
        self.figure_pattern = re.compile(
            r"^(图|表|Figure|Table)\s*\d+(-\d+)?(\.\d+)?\s*[\u4e00-\u9fa5A-Za-z]*$",
            re.IGNORECASE
        )

    def process(self, sample):
        """Process the sample to remove figure and table captions.

        Args:
            sample (dict): A dictionary containing text data.

        Raises:
            ValueError: If the expected text key is not found in the sample.

        Returns:
            dict: The modified sample with figure and table captions removed.
        """
        # Ensure the correct key is present in the sample
        if self.text_key not in sample:
            raise ValueError(f"Expected key '{self.text_key}' not found in the provided sample.")

        text = sample[self.text_key]

        # Process lines to remove figure and table annotations
        lines = text.split('\n')
        lines = [line.strip() for line in lines if not self.figure_pattern.fullmatch(line.strip())]
        text = '\n'.join(lines)

        # Update the text in the sample dictionary
        sample[self.text_key] = text

        return sample
