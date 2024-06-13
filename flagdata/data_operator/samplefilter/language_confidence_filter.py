from ..base_operator import BaseOperator
from typing import Union, List, Tuple
from langdetect import detect_langs

class LanguageConfidenceFilter(BaseOperator):
    """Class to filter samples based on language identification confidence
    scores, ensuring the language matches the specified criteria and the
    confidence is above a minimum threshold."""

    def __init__(self,
                 text_key='content',
                 languages: Union[str, List[str], Tuple[str, ...]] = '',
                 min_confidence: float = 0.8,
            ):
        """
        Constructor to initialize the language confidence settings.

        :param text_key: Key in the sample dictionary containing the text to evaluate.
        :param languages: Samples in which languages to keep. Can be a string or a list of strings.
        :param min_confidence: The minimum language identification confidence score to keep a sample.
        """
        super().__init__()
        self.text_key = text_key
        self.languages = [languages] if isinstance(languages, str) else languages
        self.min_confidence = min_confidence


    def identify_language(self, text):
        try:
            # 使用 langdetect 来检测语言和置信度
            results = detect_langs(text)
            # results 是一个列表，包含语言和对应置信度，按置信度排序
            if results:
                # 选择置信度最高的语言
                highest_confidence_result = results[0]
                return (highest_confidence_result.lang, highest_confidence_result.prob)
            else:
                return (None, -1)
        except Exception as e:
            # 处理任何异常，可能是无法识别语言
            print(f"Error in language detection: {str(e)}")
            return (None, -1)

    def process(self, sample):
        """
        Processes the given sample to determine if it meets the language confidence criteria.

        :param sample: Dictionary containing data to process.
        :return: Boolean indicating if the sample should be kept or not.
        """
        text = sample[self.text_key]
        lang_code, confidence = self.identify_language(text)
        if lang_code not in self.languages:
            return False
        return confidence >= self.min_confidence
