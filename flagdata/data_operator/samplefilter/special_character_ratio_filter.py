# Code of SPECIAL_CHARACTERS has been modified from:
# https://huggingface.co/spaces/huggingface/text-data-filtering
# --------------------------------------------------------

from ..base_operator import BaseOperator

import string

import emoji

# special characters
MAIN_SPECIAL_CHARACTERS = string.punctuation + string.digits \
                          + string.whitespace
OTHER_SPECIAL_CHARACTERS = (
    " ￼’“”–ー一▬…✦�­£​•€«»°·═"
    "×士＾˘⇓↓↑←→（）§″′´¿−±∈﻿¢ø‚„½¼¾¹²³―⁃，ˌ¸‹›ʺˈʻ¦‰‑≤≥‖"
    "◆●■►▼▲▴∆▻¡★☆✱ːº。¯˜¥ɪ≈†上ン：∼⁄・♡✓⊕․．⋅÷１‟；،、¨াাी्े◦˚"
    "゜ʼ≖ʼ¤ッツシ℃√！【】‿∞➤～πه۩☛₨➩☻๑٪♥ıॽ《‘©﴿٬？▷Г♫∟™ª₪®「—❖"
    "」﴾》"
)
EMOJI = list(emoji.EMOJI_DATA.keys())
SPECIAL_CHARACTERS = set(OTHER_SPECIAL_CHARACTERS)
# SPECIAL_CHARACTERS.update(EMOJI)
# SPECIAL_CHARACTERS.update(MAIN_SPECIAL_CHARACTERS)

class SpecialCharacterRatioFilter(BaseOperator):
    """Class to filter samples based on the ratio of designated characters
    within the text, ensuring it falls within a specified range."""

    def __init__(self,
                 text_key='content',
                 min_ratio: float = 0.0,
                 max_ratio: float = 0.25,
                 characters: set = set() ,  # Define or import SPECIAL_CHARACTERS as needed
                 add_default = False,
                 add_emoji = False,
                 add_math = False
                 ):
        """
        Constructor to initialize the filter settings.

        :param text_key: Key in the sample dictionary containing the text to evaluate.
        :param min_ratio: The minimum acceptable ratio of designated characters to keep a sample.
        :param max_ratio: The maximum acceptable ratio of designated characters to keep a sample.
        :param characters: A set of characters considered for the ratio calculation.
        """
        super().__init__()
        self.text_key = text_key
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.characters = set(characters)

        if add_default:
            self.characters.update(SPECIAL_CHARACTERS)
        if add_emoji:
            self.characters.update(EMOJI)
        if add_math:
            self.characters.update(MAIN_SPECIAL_CHARACTERS)

    def compute_character_ratio(self, text):
        """Calculates the ratio of designated characters in the text."""
        char_count = sum(1 for char in text if char in self.characters)
        total_chars = len(text)
        return char_count / total_chars if total_chars > 0 else 0

    def process(self, sample):
        """
        Processes the given sample to determine if it meets the criteria based on the character ratio.

        :param sample: Dictionary containing data to process.
        :return: Boolean indicating if the sample should be kept or not.
        """
        text = sample[self.text_key]
        char_ratio = self.compute_character_ratio(text)
        return self.min_ratio <= char_ratio <= self.max_ratio
