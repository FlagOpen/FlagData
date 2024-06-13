import re
from ..base_operator import BaseOperator

class LinkPruner(BaseOperator):
    """Class to clean HTTP, HTTPS, and FTP links from text samples."""

    def __init__(self, text_key='content'):
        super().__init__()
        self.text_key = text_key
        # Regex pattern to match HTTP, HTTPS, and FTP URLs
        self.link_pat = r'(?i)\b(?:https?|ftp):\/\/'  # Start with HTTP, HTTPS, or FTP
        self.link_pat += r'[\w-]+(?:\.[\w-]+)+'  # Domain name
        self.link_pat += r'(?:\:[0-9]+)?'  # Optional port
        self.link_pat += r'(?:\/\S*)?'  # Path
        self.repl = ''  # Replacement string, default is empty to remove URLs

    def process(self, sample):
        """Processes the given sample to remove URLs.

        Args:
            sample (dict): A dictionary containing text data.

        Raises:
            ValueError: If the expected key is not found in the sample.

        Returns:
            dict: The modified sample with URLs removed.
        """
        if self.text_key not in sample:
            raise ValueError(f"Expected key '{self.text_key}' not found in the provided sample.")

        text = sample[self.text_key]
        # Replace all found URLs with the empty string
        modified_text = re.sub(self.link_pat, self.repl, text, flags=re.DOTALL)
        sample[self.text_key] = modified_text

        return sample
