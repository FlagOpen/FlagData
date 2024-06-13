from ..base_operator import BaseOperator
import spacy, re

class ActionableVerbNumFilter(BaseOperator):
    """Class to filter samples based on the presence of actionable verbs within
    the text, ensuring there are enough such verbs according to a specified minimum count."""

    def __init__(self,
                 text_key='content',
                 min_verbs: int = 1,
                 language: str = None):
        """
        Constructor to initialize the filter settings.

        :param text_key: Key in the sample dictionary containing the text to evaluate for verbs.
        :param min_verbs: The minimum number of verbs required to keep a sample.
        :param language: Language model to use ('en' for English, 'zh' for Chinese, 'mix' for both).
                         If not provided, it will use the BaseOperator's language.
        """
        super().__init__()
        self.text_key = text_key
        self.min_verbs = min_verbs
        if language is not None:
            self.language = language
        
        self.nlp = self.load_language_model(self.language)


    def evaluate_text(self, text):
        """Utilizes NLP to count the number of verbs in the text."""
        if self.language in ['en', 'zh']:
            doc = self.nlp(text)
            if self.language == 'en':
                verb_tags = {'VERB', 'VB', 'VBP', 'VBZ', 'VBD', 'VBG', 'VBN'}
                verb_count = sum(1 for token in doc if token.tag_ in verb_tags)
            elif self.language == 'zh':
                verb_count = sum(1 for token in doc if token.pos_ == 'VERB')
        elif self.language == 'mix':
            doc_en, doc_zh = self.nlp
            # doc_en = doc_en(text)
            # doc_zh = doc_zh(text)
            # verb_tags = {'VERB', 'VB', 'VBP', 'VBZ', 'VBD', 'VBG', 'VBN'}
            # verb_count = (sum(1 for token in doc_en if token.tag_ in verb_tags) +
            #               sum(1 for token in doc_zh if token.pos_ == 'VERB'))
            english_text = re.findall(r'[a-zA-Z0-9,.!?;:]+', text)
            chinese_text = re.findall(r'[\u4e00-\u9fff，。！？；：]+', text)
            
            english_text = " ".join(english_text)
            chinese_text = "".join(chinese_text)

            # 处理英文部分
            doc_en = doc_en(english_text)
            doc_zh = doc_zh(chinese_text)

            verb_tags = {'VERB', 'VB', 'VBP', 'VBZ', 'VBD', 'VBG', 'VBN'}
            verb_count_en = sum(1 for token in doc_en if token.tag_ in verb_tags)
            
            # 处理中文部分
            verb_count_zh = sum(1 for token in doc_zh if token.pos_ == 'VERB')
            verb_count = verb_count_en + verb_count_zh
        else:
            raise ValueError(f"Unsupported language: {self.language}")
        return verb_count


    def process(self, sample):
        """
        Processes the given sample to determine if the count of verbs meets the specified minimum.

        :param sample: Dictionary containing data to process.
        :return: Boolean indicating if the sample should be kept or not.
        """
        text = sample[self.text_key]
        verb_count = self.evaluate_text(text)
        return verb_count >= self.min_verbs
