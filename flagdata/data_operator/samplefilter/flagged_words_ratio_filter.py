from ..base_operator import BaseOperator
import spacy

class FlaggedWordsRatioFilter(BaseOperator):
    """Class to screen samples based on the presence of flagged words,
    ensuring the ratio of such words in a specified threshold."""

    def __init__(self,
                 text_key='content',
                 min_ratio: float = 0.000,
                 max_ratio: float = 0.045,
                 flagged_words_list: list = [],
                 flagged_words_file: str = None,
                 language: str = None):
        """
        Constructor to initialize the pruner settings.

        :param text_key: Key in the sample dictionary containing the text to evaluate.
        :param max_ratio: The maximum allowed ratio of flagged words in the text.
        :param flagged_words_file: Path to the JSON file containing flagged words.
        :param language: Language model to use ('en' for English, 'zh' for Chinese, 'mix' for both).
                         If not provided, it will use the BaseOperator's language.
        """
        super().__init__()
        self.text_key = text_key
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.flagged_words_list = flagged_words_list
        if flagged_words_file:
            self.flagged_words = self.load_flagged_words(flagged_words_file)
        else:
            self.flagged_words  = []
        self.flagged_words.extend(flagged_words_list)

        if language is not None:
            self.language = language
        self.nlp = self.load_language_model(self.language)

    def load_flagged_words(self, file_path):
        """Loads flagged words from a JSON file."""
        import json
        with open(file_path, 'r') as file:
            return json.load(file)

    def compute_flagged_ratio(self, text):
        """Calculates the ratio of flagged words in the text."""
        if self.language == 'mix':
            doc_en, doc_zh = self.nlp
            words_en = [token.text.lower() for token in doc_en(text)]
            words_zh = [token.text.lower() for token in doc_zh(text)]
            words = words_en + words_zh
        else:
            doc = self.nlp(text)
            words = [token.text.lower() for token in doc]

        flagged_count = sum(1 for word in words if word in self.flagged_words)
        total_words = len(words)
        return flagged_count / total_words if total_words > 0 else 0
    
    def process(self, sample):
        """
        Processes the given sample to determine if it meets the criteria based on the flagged word ratio.

        :param sample: Dictionary containing data to process.
        :return: Boolean indicating if the sample should be kept or not.
        """
        text = sample[self.text_key]
        flagged_ratio = self.compute_flagged_ratio(text)
        return  self.min_ratio <= flagged_ratio <= self.max_ratio
