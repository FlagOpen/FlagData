import regex as re
from ..base_operator import BaseOperator

class UnicodePruner(BaseOperator):
    """Pruner to normalize both various kinds of whitespaces and unicode punctuations
    to their standard forms in text samples."""

    def __init__(self, text_key='content'):
        self.text_key = text_key
        self.whitespace_chars = ' \t\n\r\x0b\x0c'
        self.punctuation_map = {
            '，': ',',
            '。': '.',
            '、': ',',
            '„': '"',
            '”': '"',
            '“': '"',
            '«': '"',
            '»': '"',
            '１': '"',
            '」': '"',
            '「': '"',
            '《': '"',
            '》': '"',
            '´': "'",
            '∶': ':',
            '：': ':',
            '？': '?',
            '！': '!',
            '（': '(',
            '）': ')',
            '；': ';',
            '–': '-',
            '—': ' - ',
            '．': '. ',
            '～': '~',
            '’': "'",
            '…': '...',
            '━': '-',
            '〈': '<',
            '〉': '>',
            '【': '[',
            '】': ']',
            '％': '%',
            '►': '-',
        }

    def process(self, sample):
        if self.text_key not in sample:
            raise ValueError(f"Expected key '{self.text_key}' not found in the provided sample.")

        text = sample[self.text_key]

        # Normalize whitespaces
        text = re.sub(r'\s', ' ', text.strip())

        # Normalize punctuations
        text = ''.join([self.punctuation_map.get(char, char) for char in text])

        sample[self.text_key] = text
        return sample
