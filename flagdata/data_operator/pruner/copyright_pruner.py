import re
from ..base_operator import BaseOperator

class CopyrightPruner(BaseOperator):
    """Class to clean copyright notices from code samples."""

    def __init__(self, text_key='content'):
        super().__init__()
        self.text_key = text_key
        # Pattern to match multi-line comments
        self.multi_line_pat = re.compile(r'/\*[^*]*\*+(?:[^/*][^*]*\*+)*/', re.DOTALL)
        # Pattern to search for "copyright"
        self.copyright_pat = re.compile('copyright', re.IGNORECASE)
        # Pattern to match single-line comments
        self.single_line_pat = re.compile(r'^\s*(//|#|--).*(copyright).*$', re.IGNORECASE)

    def process(self, sample):
        """Process the sample to remove copyright notices.

        Args:
            sample (dict): A dictionary containing text data.

        Raises:
            ValueError: If the expected text key is not found in the sample.

        Returns:
            dict: The modified sample with copyright notices removed.
        """
        # Ensure the correct key is present in the sample
        if self.text_key not in sample:
            raise ValueError(f"Expected key '{self.text_key}' not found in the provided sample.")
        
        text = sample[self.text_key]

        # Process multi-line comments
        for match in self.multi_line_pat.finditer(text):
            sub = match.group(0)
            if self.copyright_pat.search(sub):
                text = text.replace(sub, '')  # Remove the matching multi-line comments

        # Process single-line comments
        lines = text.split('\n')
        new_lines = []
        for line in lines:
            if self.single_line_pat.search(line):
                continue  # Skip lines containing copyright information
            new_lines.append(line)
        text = '\n'.join(new_lines)

        # Update the text in the sample dictionary
        sample[self.text_key] = text

        return sample
