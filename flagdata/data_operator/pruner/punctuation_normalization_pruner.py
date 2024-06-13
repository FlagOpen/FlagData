from ..base_operator import BaseOperator

class PunctuationNormalizationPruner(BaseOperator):
    """Pruner to normalize unicode punctuations to English punctuations in text samples."""

    def __init__(self, text_key='content', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_key = text_key
        self.punctuation_unicode = {
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

        sample[self.text_key] = ''.join([
            self.punctuation_unicode.get(c, c) for c in sample[self.text_key]
        ])
        return sample
