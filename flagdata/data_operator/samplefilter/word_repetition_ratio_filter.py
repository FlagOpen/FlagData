from ..base_operator import BaseOperator

class WordRepetitionRationFilter(BaseOperator):
    """Class to filter samples based on word-level n-gram repetition ratio,
    ensuring the repetition does not exceed a specific threshold."""

    def __init__(self,
                 text_key='content',
                 ngram_length: int = 10,
                 min_ratio: float = 0.0,
                 max_ratio: float = 0.5,
                 language: str = None):
        """
        Constructor to initialize the filter settings.

        :param text_key: Key in the sample dictionary containing the text to evaluate.
        :param ngram_length: The length of the n-grams to analyze for repetition.
        :param min_ratio: The minimum acceptable ratio of repeated n-grams.
        :param max_ratio: The maximum acceptable ratio of repeated n-grams.
        :param language: Language model to use ('en' for English, 'zh' for Chinese, 'mix' for both).
                    If not provided, it will use the BaseOperator's language.
        """
        super().__init__()
        self.text_key = text_key
        self.ngram_length = ngram_length
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        if language is not None:
            self.language = language  
        self.nlp = self.load_language_model(self.language)
        # if language is 'mix' use zh_core_web_sm
        if self.language == 'mix':
            self.nlp  = self.nlp[-1]
     
    def calculate_ngram_repetition(self, text):
        """Calculate the repetition ratio of n-grams in the text."""
        doc = self.nlp(text)
        words = [token.text for token in doc]

        ngrams = [' '.join(words[i:i + self.ngram_length]) for i in range(len(words) - self.ngram_length + 1)]
        ngram_counts = {}
        for ngram in ngrams:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        total_ngrams = len(ngrams)
        repeated_ngrams = sum(1 for count in ngram_counts.values() if count > 1)

        return repeated_ngrams / total_ngrams if total_ngrams > 0 else 0

    def process(self, sample):
        """
        Processes the given sample to determine if the n-gram repetition ratio meets the specified criteria.

        :param sample: Dictionary containing data to process.
        :return: Boolean indicating if the sample should be kept or not.
        """
        text = sample.get(self.text_key, "")
        repetition_ratio = self.calculate_ngram_repetition(text)
        return self.min_ratio <= repetition_ratio <= self.max_ratio
