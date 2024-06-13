
import spacy
import subprocess
import sys

def download_model(model_name):
    try:
        spacy.load(model_name)
    except OSError:
        print(f"Model '{model_name}' not found. Downloading...")
        subprocess.run(["python", "-m", "spacy", "download", model_name])


class BaseOperator:
    """Base text cleaner class that provides common functionality for specific text cleaning tasks."""

    def __init__(self, text_key='content', language='mix'):
        """
        Args:
            text_key (str, optional): The key in the sample dictionary containing the text data to be processed. Default is 'content'.
            language (str, optional): The language of the text to be processed, default is 'mix'. Some operators require this parameter. Currently, only Chinese ('zh'), English ('en'), and mixed text ('mix') are supported. For more accurate results, use 'zh' or 'en' if the language is known.
        """
        self.text_key = text_key
        self.language = language

    def process(self, sample):
        raise NotImplementedError("Each pruner must implement its own `process` method.")

    def load_language_model(self, language):
        if language == 'en':
            download_model("en_core_web_sm")
            return spacy.load("en_core_web_sm")
        elif language == 'zh':
            download_model("zh_core_web_sm")
            return spacy.load("zh_core_web_sm")
        elif language == 'mix':
            download_model("en_core_web_sm")
            download_model("zh_core_web_sm")
            return (spacy.load("en_core_web_sm"), spacy.load("zh_core_web_sm"))

        else:
            raise ValueError(f"Unsupported language: {language}")