from ..base_operator import BaseOperator
import spacy

class StopwordRatioFilter(BaseOperator):
    """Class to filter samples based on the stopword ratio within the text,
    ensuring the ratio exceeds a specified minimum value."""

    def __init__(self,
                 text_key='content',
                 min_ratio=0.3,
                 stopwords_list=None,
                 language: str = None):
        """
        Constructor to initialize the filter settings.

        :param text_key: Key in the sample dictionary containing the text to evaluate.
        :param language: Language model to use ('en' for English, 'zh' for Chinese, 'mix' for both).
                         If not provided, it will use the BaseOperator's language.
        :param min_ratio: The minimum ratio of stopwords required to keep a sample.
        :param stopwords_list: A list of stopwords. If None, a default list based on the language will be used.
        """
        super().__init__()
        self.text_key = text_key
        if language is not None:
            self.language = language        
        self.min_ratio = min_ratio
        self.stopwords = stopwords_list or self.load_default_stopwords(self.language)
        self.nlp = self.load_language_model(language)

    def load_default_stopwords(self, language):
        """Load default stopwords based on the language setting using SpaCy."""
        try:
            if language == 'mix':
                stopwords_en = spacy.load("en_core_web_sm").Defaults.stop_words
                stopwords_zh = spacy.load("zh_core_web_sm").Defaults.stop_words
                return set(stopwords_en).union(set(stopwords_zh))
            else:
                nlp = spacy.load(f"{language}_core_web_sm")  # 加载对应语言的模型
                return set(nlp.Defaults.stop_words)
        except OSError:
            print(f"SpaCy model for {language} is not installed. Please install it using 'python -m spacy download {language}_core_web_sm'.")
            return set()

    def calculate_stopword_ratio(self, text):
        """Calculate the ratio of stopwords in the text."""
        if self.language == 'mix':
            doc_en, doc_zh = self.nlp
            words_en = [token.text.lower() for token in doc_en(text)]
            words_zh = [token.text.lower() for token in doc_zh(text)]
            words = words_en + words_zh
        else:
            doc = self.nlp(text)
            words = [token.text.lower() for token in doc]
        
        total_words = len(words)
        stopword_count = sum(1 for word in words if word in self.stopwords)
        return stopword_count / total_words if total_words > 0 else 0
    
    def process(self, sample):
        """
        Process the given sample to determine if the stopword ratio meets the specified minimum.

        :param sample: Dictionary containing data to process.
        :return: Boolean indicating if the sample should be kept or not.
        """
        text = sample[self.text_key]
        stopword_ratio = self.calculate_stopword_ratio(text)
        return stopword_ratio >= self.min_ratio
