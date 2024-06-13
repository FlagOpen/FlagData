from ..base_operator import BaseOperator

class ChineseConversionPruner(BaseOperator):
    """Class to convert Chinese between Traditional Chinese, Simplified Chinese, and Japanese Kanji using OpenCC."""

    def __init__(self, text_key='content', mode='s2t'):
        super().__init__()
        self.text_key = text_key
        self.mode = mode
        self.converter = None
        self.mode_list = [
            's2t', 't2s', 's2tw', 'tw2s', 's2hk', 'hk2s', 's2twp', 'tw2sp',
            't2tw', 'tw2t', 'hk2t', 't2hk', 't2jp', 'jp2t'
        ]
        assert self.mode in self.mode_list, f'Please make sure mode is one of {self.mode_list}'
        self.prepare_converter()

    def prepare_converter(self):
        """Prepare the OpenCC converter with the specified mode."""
        import opencc
        mode_path = f'{self.mode}.json'
        if self.converter is None or not self.converter.config.endswith(mode_path):
            self.converter = opencc.OpenCC(mode_path)

    def process(self, sample):
        """Convert text in the sample from one Chinese script to another.

        Args:
            sample (dict): A dictionary containing text data.

        Raises:
            ValueError: If the expected text key is not found in the sample.

        Returns:
            dict: The modified sample with text converted as specified.
        """
        if self.text_key not in sample:
            raise ValueError(f"Expected key '{self.text_key}' not found in the provided sample.")

        sample[self.text_key] = self.converter.convert(sample[self.text_key])
        return sample
