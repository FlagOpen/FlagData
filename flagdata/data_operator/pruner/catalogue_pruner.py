import re
from ..base_operator import BaseOperator

class CataloguePruner(BaseOperator):
    """Class to remove catalogue entries from text based on specific patterns and conditions."""

    def __init__(self, text_key='content', length_threshold=100, consecutive_threshold=3):
        super().__init__()
        self.text_key = text_key
        self.length_threshold = length_threshold
        self.consecutive_threshold = consecutive_threshold

    def process(self, sample):
        """Process the sample to remove catalogue entries.

        Args:
            sample (dict): A dictionary containing text data.

        Raises:
            ValueError: If the expected text key is not found in the sample.

        Returns:
            dict: The modified sample with catalogue entries removed.
        """
        if self.text_key not in sample:
            raise ValueError(f"Expected key '{self.text_key}' not found in the provided sample.")

        lines = sample[self.text_key].split('\n')
        lines_length = [len(line) for line in lines]
        consecutive_count = 0
        temp = set()
        lines_to_delete = set()

        for i, line in enumerate(lines):
            if lines_length[i] < self.length_threshold and (
                ('第' in line and '章' in line) or 
                ('第' in line and '幕' in line) or 
                ('chapter' in line.lower()) or 
                re.match(r'^\d+\.', line.strip()) or 
                ('............' in line)
            ):
                consecutive_count += 1
                temp.add(i)
                if consecutive_count >= self.consecutive_threshold:
                    lines_to_delete.update(temp)
            elif line in ['', '\n']:
                consecutive_count += 0
                temp.add(i)
            else:
                consecutive_count = 0
                temp.clear()

        # Generate a new list, excluding the rows to be deleted
        lines = [line for i, line in enumerate(lines) if i not in lines_to_delete]
        sample[self.text_key] = '\n'.join(lines)

        return sample
