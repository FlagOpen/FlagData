import re
from ..base_operator import BaseOperator

class IpPruner(BaseOperator):
    """Class to clean IPv4 and IPv6 addresses from text samples."""

    def __init__(self, text_key='content'):
        super().__init__()
        self.text_key = text_key
        # Regular expression pattern to match IPv4 and IPv6 addresses
        self.ip_pat = r'(?:(?:1[0-9][0-9]\.)|(?:2[0-4][0-9]\.)|'\
                      r'(?:25[0-5]\.)|(?:[1-9][0-9]\.)|(?:[0-9]\.))'\
                      r'{3}(?:(?:1[0-9][0-9])|(?:2[0-4][0-9])|'\
                      r'(?:25[0-5])|(?:[1-9][0-9])|(?:[0-9]))|'\
                      r'([\da-fA-F]{1,4}:){7}[\da-fA-F]{1,4}'  # ipv6
        self.repl = ''  # Replacement string, default is empty to remove IPs

    def process(self, sample):
        """Processes the given sample to remove IP addresses.

        Args:
            sample (dict): A dictionary containing text data.

        Raises:
            ValueError: If the expected text key is not found in the sample.

        Returns:
            dict: The modified sample with IP addresses removed.
        """
        # Ensure the text key is present in the sample
        if self.text_key not in sample:
            raise ValueError(f"Expected key '{self.text_key}' not found in the provided sample.")

        text = sample[self.text_key]
        # Replace all found IP addresses using the compiled regex pattern
        modified_text = re.sub(self.ip_pat, self.repl, text, flags=re.DOTALL)
        sample[self.text_key] = modified_text

        return sample
