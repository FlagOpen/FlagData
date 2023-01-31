# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from collections import OrderedDict
import re
import unicodedata
import json
import html
from urlextract import URLExtract
import string
from .ruleset import URLRuleSet, UserPrivacyRuleSet, TextIntegrityRuleSet
import opencc
import pathlib


class BasicCleaner:
    @classmethod
    def get_subclasses(self):
        class_dict = OrderedDict()
        classes = self.__subclasses__()
        for subclass in classes:
            class_dict[subclass.__name__] = subclass
        return class_dict

    def clean_article(self, article):
        raise NotImplementedError


class CleanerPipeline:
    def __init__(self, *args):
        self.cleaners = args

    def clean(self, article):
        for cleaner in self.cleaners:
            article = cleaner.clean_article(article)
        return article


class SimplifiedFilter(BasicCleaner):
    def __init__(self, config_file: str):
        try:
            self.converter = opencc.OpenCC(config_file)
        except:
            self.converter = opencc.OpenCC(config_file.replace(".json", ""))

    def clean_article(self, article):
        return self.converter.convert(article)


class SymbolFilter(BasicCleaner):
    """
    remove emojis and meaningless chars
    """

    def __init__(self, emoji_path=None, filter_emoji=True, filter_control=True):
        try:
            assert filter_emoji or filter_control
        except:
            raise AssertionError("At least one filter need to be specified!")
        if filter_emoji:
            self.symbol_regex_exp = self._build_regex()
            self.emoji_set = self._build_emoji_set(emoji_path)
            self.do_filter_emoji = filter_emoji
        self.do_filter_control = filter_control

    def _build_regex(self):
        regex_msg = re.compile(
            '[\U0001F600-\U0001F92F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F190-\U0001F1FF\U00002702-\U000027B0\U0001F926-\U0001FA9F\u200d\u2640-\u2642\u2600-\u2B55\u23cf\u23e9\u231a\ufe0f\u23ee\U0000200D' + ']+')
        return regex_msg

    def _build_emoji_set(self, path):
        if path is None:
            path = pathlib.Path(__file__).parent.parent / "resource" / "emojis.json"
        with open(path, "r", encoding="utf8") as fr:
            data = json.load(fr)
        emojis = [eval(repr(emoji_code).replace("\\\\", "\\"))
                  for emoji_code in data.keys()]
        emojis = [emoji_code for emoji_code in emojis if len(emoji_code) == 1]
        return set(emojis)

    def _is_control(self, char):
        """Checks whether `chars` is a control character."""
        # skip the following control chars
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    def remove_emoji(self, article: str):
        '''
            filter emojis
        '''
        cleaned_text = self.symbol_regex_exp.sub(u'', article)
        cleaned_chars = [
            char for char in cleaned_text if char not in self.emoji_set]
        return cleaned_chars

    def remove_control(self, char_list: list):
        """
        remove control chars
        """
        for idx, char in enumerate(char_list):
            if self._is_control(char):
                char_list[idx] = ""
        return char_list

    def clean_article(self, article: str):
        if not self.do_filter_control:
            cleaned_article = self.remove_emoji(article)
        elif not self.do_filter_emoji:
            article = list(article)
            cleaned_article = self.remove_control(article)
        else:
            cleaned_article = self.remove_control(self.remove_emoji(article))
        return "".join(cleaned_article).strip()


class TextCleaner(BasicCleaner):
    def __init__(self, filter_personal=True, filter_url=True, filter_extraspace=True):
        self.url_regex_set = URLRuleSet()
        self.privacy_regex_set = UserPrivacyRuleSet()
        if filter_url:
            self.url_extractor = URLExtract()

        self.clean_pipeline = self.build_pipeline(
            filter_personal, filter_url, filter_extraspace)
        assert sum(self.clean_pipeline.values()) >= 1

    def build_pipeline(self, filter_private, filter_url, filter_extraspace):
        check_pipeline = OrderedDict()
        pipeline = [self.remove_personal_msg,
                    self.remove_link, self.remove_extraspace]
        status = [filter_private, filter_url, filter_extraspace]
        for k, v in zip(pipeline, status):
            check_pipeline[k] = v
        return check_pipeline

    def _contain_url(self, article):
        url_hits_num = [
            True for hit in self.url_regex_set.URL_HITS if hit in article]
        if len(url_hits_num) > 0:
            return True
        else:
            return False

    def remove_personal_msg(self, article):
        '''
            remove private info: wechat, qq, email, phone number, ID card number, etc.
        '''
        cleaned_article = html.unescape(article)
        for regex_exp in self.privacy_regex_set.REGEX_PIPELINE:
            cleaned_article = re.sub(regex_exp, "", cleaned_article, 500)
        return cleaned_article

    def remove_link(self, article):
        '''
           remove urls and img links
        '''
        if not self._contain_url(article):
            return article
        article = html.unescape(article)
        urls = [re.escape(url)
                for url in self.url_extractor.find_urls(article)]
        patternsRegex = '(' + '|'.join(urls) + ')'
        article = re.sub(patternsRegex, '', article).strip()
        urls = re.findall(self.url_regex_set.URL_REGEX, article)
        all_regex_urls_temp = [y for y in list(set(sum([re.split(r'http:|https:|ftp:|HTTP:|HTTPS:|FTP:', ''.join(
            x for x in url if x in string.printable)) for url in urls], []))) if len(y) > 1]
        all_regex_urls = list(map(lambda x: 'http:' + x, all_regex_urls_temp)) + list(map(lambda x: 'https:' + x, all_regex_urls_temp)) + list(map(lambda x: 'ftp:' + x, all_regex_urls_temp)) + \
            list(map(lambda x: 'HTTP:' + x, all_regex_urls_temp)) + list(map(lambda x: 'HTTPS:' + x, all_regex_urls_temp)) + list(map(lambda x: 'FTP:' + x, all_regex_urls_temp))
        patterns_regex = '|'.join(all_regex_urls).replace(',', '').replace(':', '\:').replace('?', '\?').replace(
            '=', '\=').replace('&', '\&').replace('.', '\.').replace('#', '\#').replace('/', '\/')
        cleaned_article = re.sub(
            re.escape(patterns_regex), '', article, 1000, flags=re.I).strip()
        if 'http' in cleaned_article:
            urls_further = re.findall(
                self.url_regex_set.URL_REGEX_FURTHER, cleaned_article)
            all_regex_urls_temp_further = [y for y in list(set(sum([re.split(r'http:|https:|ftp:|HTTP:|HTTPS:|FTP:', ''.join(
                x for x in url if x in string.printable)) for url in urls_further], []))) if len(y) > 1]
            all_regex_urls_further = list(map(lambda x: 'http:' + x, all_regex_urls_temp_further)) + list(map(lambda x: 'https:' + x, all_regex_urls_temp_further)) + list(map(lambda x: 'ftp:' + x, all_regex_urls_temp_further)) + list(
                map(lambda x: 'HTTP:' + x, all_regex_urls_temp_further)) + list(map(lambda x: 'HTTPS:' + x, all_regex_urls_temp_further)) + list(map(lambda x: 'FTP:' + x, all_regex_urls_temp_further))
            patterns_regex_further = '|'.join(all_regex_urls_further).replace(',', '').replace(':', '\:').replace(
                '?', '\?').replace('=', '\=').replace('&', '\&').replace('.', '\.').replace('#', '\#').replace('/', '\/')
            cleaned_article = re.sub(
                patterns_regex_further, '', cleaned_article, 1000, flags=re.I).strip()
        return cleaned_article

    def remove_extraspace(self, article):
        article = re.sub("[\t\u3000\s]+", " ", article)
        article = re.sub("[\r\n]+", "\n", article).strip()
        return article

    def clean_article(self, article):
        for clean_func, enabled in self.clean_pipeline.items():
            if enabled:
                article = clean_func(article)
        return article


class TextIntegrityChecker(BasicCleaner):
    def __init__(self, min_length=16, length_check=True, do_end_clip=True, end_mark_check=True, double_mark_check=True):
        self.rules = TextIntegrityRuleSet()
        self.min_length = min_length
        self.check_pipeline = self.build_pipeline(
            length_check, do_end_clip, end_mark_check, double_mark_check)
        assert sum(self.check_pipeline.values()) >= 1

    def build_pipeline(self, length_check, do_end_clip, end_mark_check, double_mark_check):
        check_pipeline = OrderedDict()
        if do_end_clip:
            pipeline = [self._length_check, self._end_clip,
                        self._length_check]
            status = [length_check, True, True]
        else:
            pipeline = [self._length_check,
                        self._contain_end_mark, self._contain_double_mark]
            status = [length_check, end_mark_check, double_mark_check]

        for k, v in zip(pipeline, status):
            check_pipeline[k] = v
        return check_pipeline

    def _end_clip(self, article):
        clip_ret = re.search(self.rules.ALL_LANGUAGE_END_CLIP_REGEX, article)
        if clip_ret:
            return clip_ret.group()
        else:
            return ""

    def _contain_end_mark(self, article):
        '''
            make sure at least one sentence exists in this article, otherwise drop it
        '''
        new_article = [
            x for x in self.rules.ALL_LANGUAGE_END_PUNCTION if x in article]
        if len(new_article) >= 1:
            return article
        else:
            return ""

    def _contain_double_mark(self, article):
        '''
            double-mark validation
        '''
        new_article = [x for x in self.rules.ALL_LANGUAGE_WHOLE_SENTENCE_MARK if (
            (x[0] in article) and (x[1] in article))]
        if len(new_article) >= 1:
            return article
        else:
            return ""

    def _length_check(self, article):
        num_chars = len(re.findall("\S", article))
        if num_chars >= self.min_length:
            return article
        else:
            return ""

    def clean_article(self, article):
        for check_func, enabled in self.check_pipeline.items():
            if enabled:
                article = check_func(article)
                if not article:
                    break
        return article.strip()
