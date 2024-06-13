import regex as re
from ..base_operator import BaseOperator
from typing import Union, List

class ReplacePruner(BaseOperator):
    """Pruner to replace all content in the text that matches
    a specific regular expression pattern with a designated
    replacement string."""

    def __init__(self,
                 text_key='content',
                 pattern: Union[str, List[str]] = None,
                 repl: Union[str, List[str]] = '',
):
        """
        Initialization method.

        :param text_key: The key in the sample dict containing the text to be processed.
        :param pattern: Regular expression pattern(s) to search for within text.
        :param repl: Replacement string(s), default is an empty string.
        """
        self.text_key = text_key
        self.pattern = pattern
        self.repl = repl
        self.compiled_patterns = []
        if isinstance(pattern, str):
            self.compiled_patterns.append(self.compile_pattern(pattern))
        elif isinstance(pattern, list):
            for p in pattern:
                self.compiled_patterns.append(self.compile_pattern(p))

    def compile_pattern(self, pattern: str) -> re.Pattern:
        """Prepare the regular expression pattern."""
        return re.compile(pattern, flags=re.DOTALL)

    def process(self, sample):
        """
        Processes the given sample to replace text based on the configured patterns and replacements.

        Args:
            sample (dict): A dictionary containing text data.

        Returns:
            dict: The modified sample with text replaced as per the specified patterns.
        """
        if self.text_key not in sample:
            raise ValueError(f"Expected key '{self.text_key}' not found in the provided sample.")

        text = sample[self.text_key]
        for i, pattern in enumerate(self.compiled_patterns):
            replacement = self.repl[i] if isinstance(self.repl, list) and i < len(self.repl) else self.repl
            text = pattern.sub(replacement, text)

        sample[self.text_key] = text
        return sample
