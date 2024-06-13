import re
from ..base_operator import BaseOperator

class EndAtLastPunctuationPruner(BaseOperator):
    """Class to clip text to the last occurrence of designated punctuation marks."""

    def __init__(self, text_key='content'):
        super().__init__()
        self.text_key = text_key
        self.chinese_punctuation = "。！？"
        self.english_punctuation = ".?!"
        self.all_punctuation = self.chinese_punctuation + self.english_punctuation

    def process(self, sample):
        """Process the sample to clip text to the last punctuation mark.

        Args:
            sample (dict): A dictionary containing text data.

        Raises:
            ValueError: If the expected text key is not found in the sample.

        Returns:
            dict: The modified sample with text clipped to the last punctuation mark.
        """
        if self.text_key not in sample:
            raise ValueError(f"Expected key '{self.text_key}' not found in the provided sample.")

        text = sample[self.text_key]
        clipped_text = self.end_clip(text)
        sample[self.text_key] = clipped_text

        return sample

    def end_clip(self, text):
        """Clip text to the last punctuation mark if present.

        Args:
            text (str): The text to be clipped.

        Returns:
            str: Text clipped up to the last punctuation mark.
        """
        if text[-1] in self.all_punctuation:
            return text
        else:
            match = re.search(r'[{}]'.format(re.escape(self.all_punctuation)), text[::-1])
            if match:
                return text[:len(text) - match.start()]
            else:
                return ""
