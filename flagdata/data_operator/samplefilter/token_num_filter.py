from ..base_operator import BaseOperator
from transformers import AutoTokenizer

class TokenNumFilter(BaseOperator):
    """Class to filter samples based on the number of tokens in the text,
    ensuring the token count is within a specified range."""

    def __init__(self,
                 text_key='content',
                 tokenizer_name='bert-base-uncased',
                 min_tokens: int = 10,
                 max_tokens: int = 100000000,
               ):
        """
        Constructor to initialize the filter settings.

        :param text_key: Key in the sample dictionary containing the text to tokenize.
        :param tokenizer_name: Identifier for a pre-trained tokenizer from Hugging Face.
        :param min_tokens: The minimum number of tokens required to keep a sample.
        :param max_tokens: The maximum number of tokens allowed to keep a sample.
        """
        super().__init__()
        self.text_key = text_key
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def count_tokens(self, text):
        """Tokenize the text and count the number of tokens."""
        tokens = self.tokenizer.tokenize(text)
        return len(tokens)

    def process(self, sample):
        """
        Processes the given sample to determine if the token count meets the specified bounds.

        :param sample: Dictionary containing data to process.
        :return: Boolean indicating if the sample should be kept or not.
        """
        text = sample[self.text_key]
        token_count = self.count_tokens(text)
        return self.min_tokens <= token_count <= self.max_tokens
