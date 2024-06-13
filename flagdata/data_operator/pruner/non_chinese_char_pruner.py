import re
from ..base_operator import BaseOperator

class NonChineseCharPruner(BaseOperator):
    """Class to remove non-Chinese characters from text samples, with options to keep alphabets, numbers, and punctuation."""

    def __init__(self, text_key='content', keep_alphabet=True, keep_number=True, keep_punc=True,):
        super().__init__()
        self.text_key = text_key
        # Build regex pattern based on constructor parameters
        self.pattern = '[^\u4e00-\u9fa5'  # Start pattern with matching non-Chinese characters
        if keep_alphabet:
            self.pattern += 'A-Za-z'  # Add alphabet characters if keeping
        if keep_number:
            self.pattern += '0-9'  # Add numerical characters if keeping
        if keep_punc:
            self.pattern += r'.， ,\-。%《*》/•、&＆(—)（+）：？!！“”·]'  # Add specified punctuation if keeping
        else:
            self.pattern += ']'  # Close character class if not keeping punctuation

    def process(self, sample):
        """Processes the given sample to remove unwanted characters, retaining only specified types.

        Args:
            sample (dict): A dictionary containing text data.

        Raises:
            ValueError: If the expected text key is not found in the sample.

        Returns:
            dict: The modified sample with non-Chinese characters removed based on the settings.
        """
        if self.text_key not in sample:
            raise ValueError(f"Expected key '{self.text_key}' not found in the provided sample.")

        # Replace non-desired characters with an empty string
        text = sample[self.text_key]
        modified_text = re.sub(self.pattern, '', text, flags=re.DOTALL)
        sample[self.text_key] = modified_text

        return sample
