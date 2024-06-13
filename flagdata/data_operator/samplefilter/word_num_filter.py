from ..base_operator import BaseOperator
import spacy

class WordNumFilter(BaseOperator):
    """Class to filter samples based on the number of words in the text,
    ensuring the word count is within a specified range."""

    def __init__(self,
                 text_key='content',
                 min_words: int = 10,
                 max_words: int = 100000000,
                 language: str = None
                 ):
        """
        Constructor to initialize the filter settings.

        :param text_key: Key in the sample dictionary containing the text to evaluate.
        :param min_words: The minimum number of words required to keep a sample.
        :param max_words: The maximum number of words allowed to keep a sample.
        :param language: Language model to use ('en' for English, 'zh' for Chinese, 'mix' for both).
                    If not provided, it will use the BaseOperator's language.
        """
        super().__init__()
        self.text_key = text_key
        self.min_words = min_words
        self.max_words = max_words
        if language is not None:
            self.language = language  
        self.nlp = self.load_language_model(self.language)
        # if language is 'mix' use zh_core_web_sm
        if self.language == 'mix':
            self.nlp  = self.nlp[-1]
            
    def count_words(self, text):
        """Counts the words in the provided text using SpaCy for different languages."""

        doc = self.nlp(text)
        words = [token.text for token in doc]

        return len(words)

    def process(self, sample):
        """
        Processes the given sample to determine if the word count meets the specified bounds.

        :param sample: Dictionary containing data to process.
        :return: Boolean indicating if the sample should be kept or not.
        """
        text = sample.get(self.text_key, "")
        word_count = self.count_words(text)
        return self.min_words <= word_count <= self.max_words
