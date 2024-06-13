import regex as re
from ..base_operator import BaseOperator

def split_sentence(text):
    """Utility function to split text into sentences based on punctuation marks."""
    text = re.sub('([.。！!？\?])([^’”])', r'\1\n\2', text)
    text = re.sub('(\.{6})([^’”])', r'\1\n\2', text)
    text = re.sub('(\…{2})([^’”])', r'\1\n\2', text)
    text = re.sub('([.。!！？\?\.{6}\…{2}][’”])([^’”])', r'\1\n\2', text)
    return text.split('\n')

class RepeatSentencePruner(BaseOperator):
    """Pruner to remove repeated sentences in text samples."""

    def __init__(self, text_key='content', lowercase=False, ignore_special_character=True, min_repeat_sentence_length=2):
        self.text_key = text_key
        self.lowercase = lowercase
        self.min_repeat_sentence_length = min_repeat_sentence_length
        self.remove_regex = re.compile(r'[^a-zA-Z0-9\u4e00-\u9fa5\n\t ]') if ignore_special_character else None

    def process(self, sample):
        """Processes the given sample to remove repeated sentences based on the specified configurations.

        Args:
            sample (dict): A dictionary containing text data.

        Returns:
            dict: The modified sample with repeated sentences removed.
        """
        if self.text_key not in sample:
            raise ValueError(f"Expected key '{self.text_key}' not found in the provided sample.")
        
        text = sample[self.text_key]
        new_lines = []
        hash_set = set()
        for line in text.split('\n'):
            new_sent = ''
            sentences = split_sentence(line)
            for sentence in sentences:
                processed_sentence = sentence.strip()
                if self.lowercase:
                    processed_sentence = processed_sentence.lower()
                if self.remove_regex:
                    processed_sentence = self.remove_regex.sub('', processed_sentence)

                if len(processed_sentence) >= self.min_repeat_sentence_length and processed_sentence not in hash_set:
                    new_sent += sentence
                    hash_set.add(processed_sentence)
            if new_sent:  # only add non-empty new sentences
                new_lines.append(new_sent)

        sample[self.text_key] = '\n'.join(new_lines)
        return sample
