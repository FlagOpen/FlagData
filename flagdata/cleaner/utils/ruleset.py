# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import re
from dataclasses import dataclass
from collections import OrderedDict


@dataclass
class URLRuleSet:
    URL_HITS = ['http', 'HTTP', 'https', 'HTTPS', 'src', 'SRC', 'ftp', 'FTP', 'ww', 'WW', 'www',
                        'WWW', 'href', ':', '//', '\/\/', '.com', '.cn', '.ar', '.org', '.uk', '.ac', '.edu', '.net', '.php']
    URL_REGEX = r"""((?:(?:https|ftp|http|HTTPS|FTP|HTTP)?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|org|uk)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|uk|ac)\b/?(?!@)))"""
    URL_REGEX_FURTHER = '(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%.]+'


class UserPrivacyRuleSet:
    def __init__(self):
        self.REGEX_MAP = self.group_regex()
        self.REGEX_PIPELINE = self.build_pipeline(self.REGEX_MAP)

    def build_pipeline(self, regex_map):
        REGEX_PIPELINE = []
        for cat, regex_group in regex_map.items():
            for regex_exp in regex_group:
                compiled_regex = re.compile(regex_exp, flags=re.I)
                REGEX_PIPELINE.append(compiled_regex)
        return REGEX_PIPELINE

    def group_regex(self):
        regex_map = OrderedDict()
        regex_map["wechat"] = [r'vxin[：|:][a-zA-Z0-9{3,20}]+',
                               r'vx[：|:][a-zA-Z0-9{3,20}]+',
                               r'VX[：|:][a-zA-Z0-9{3,20}]+',
                               r'Vxin[：|:][a-zA-Z0-9{3,20}]+',
                               r'wx[：|:][a-zA-Z0-9{3,20}]+',
                               r'WX[：|:][a-zA-Z0-9{3,20}]+',
                               r'wei xin[：|:][a-zA-Z0-9{3,20}]+',
                               r'weixin[：|:][a-zA-Z0-9{3,20}]+',
                               r'微信[：|:][a-zA-Z0-9{3,20}]+',
                               r'薇信[：|:][a-zA-Z0-9{3,20}]+',
                               r'v信[：|:][a-zA-Z0-9{3,20}]+',
                               r'V信[：|:][a-zA-Z0-9{3,20}]+',
                               r'[1-9][0-9]{4,}']
        regex_map["qq"] = [
            r'qq：|qq:|QQ：|QQ:|qQ：|qQ:|Qq：|Qq:|pp：|pp:|企鹅号：|企鹅号:']
        regex_map["postal_code"] = [r'[1-9]\d{5}(?!\d)']
        regex_map["id_card"] = [r'^[1-9]\d{7}((0\d)|(1[0-2]))(([0|1|2]\d)|3[0-1])\d{3}$',
                                r'^[1-9]\d{5}[1-9]\d{3}((0\d)|(1[0-2]))(([0|1|2]\d)|3[0-1])\d{4}$',
                                r'^\d{8,18}|[0-9x]{8,18}|[0-9X]{8,18}?$']
        regex_map["phone_number"] = [r'^(13[0-9]|14[5|7]|15[0|1|2|3|5|6|7|8|9]|18[0|1|2|3|5|6|7|8|9])\d{8}$',
                                     r'(\(\d{3,4}\)|\d{3,4}-|\s)?\d{8}',
                                     r'(\(\d{3,4}\)|\d{3,4}-|\s)?\d{7,14}',
                                     r'\d{3}-\d{8}|\d{4}-\d{7}',
                                     r'\d{3}\s{1,3}\d{4}\s{1,3}\d{4}',
                                     r'(\+\d{2,3}-)|(\d{3,5}-)']
        regex_map["email"] = [r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)',
                              r'(@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)']
        regex_map["ip"] = [r'\d+\.\d+\.\d+\.\d+']
        return regex_map


class TextIntegrityRuleSet:
    def __init__(self):
        AR_END_PUNCTION = ['؛', '.', '؟', '!']
        EN_END_PUNCTION = [';', '.', '?', '!']
        CN_END_PUNCTION = ['；', '。', '？', '！']

        AR_WHOLE_SENTENCE_MARK = [('.', '،'), ('.', '؛'), ('.', ':'), ('.', '؟'), (
            '.', '!'), ('،', '!'), ('،', '؟'), ('،', '؛'), (':', '؟'), (':', '!'), (':', '؛')]
        EN_WHOLE_SENTENCE_MARK = [('.', ','), ('.', ';'), ('.', ':'), ('.', '?'), (
            '.', '!'), (',', '!'), (',', '?'), (',', ';'), (':', '?'), (':', '!'), (':', ';')]
        CN_WHOLE_SENTENCE_MARK = [('。', '，'), ('。', '；'), ('。', '：'), ('。', '？'), (
            '。', '！'), ('，', '！'), ('，', '？'), ('，', '；'), ('：', '？'), ('：', '！'), ('：', '；')]
        self.ALL_LANGUAGE_END_PUNCTION = list(
            set(AR_END_PUNCTION + EN_END_PUNCTION + CN_END_PUNCTION))
        NON_TERMINAL_ENDS = ['"', '”', ')']
        self.ALL_LANGUAGE_END_CLIP_REGEX = re.compile(".*" + "[" + re.escape("".join(
            self.ALL_LANGUAGE_END_PUNCTION)) + "] ?[" + re.escape("".join(NON_TERMINAL_ENDS)) + "]?")
        self.ALL_LANGUAGE_WHOLE_SENTENCE_MARK = list(
            set(AR_WHOLE_SENTENCE_MARK + EN_WHOLE_SENTENCE_MARK + CN_WHOLE_SENTENCE_MARK))


@dataclass
class PageTitleRuleSet:

    TITLE_TAG_XPATH = '//title/text()'

    META_TITLE_XPATHS = [
        '//meta[@property="og:title"]/@content',
        '//meta[@name="twitter:title"]/@content',
        '//meta[@property="twitter:title"]/@content'
    ]

    SUPPLEMENT_TITLE_XPATHS = [
        '//h1//text()',
        '//h2//text()',
        '//*[contains(@id, "title")]/text()',
        '//*[contains(@id, "Title")]/text()',
        '//*[contains(@id, "TITLE")]/text()',
        '//*[contains(@class, "title")]/text()',
        '//*[contains(@class, "Title")]/text()',
        '//*[contains(@class, "TITLE")]/text()'
    ]


@dataclass
class PublishTimeRuleSet:

    META_TIME_XPATHS = [
        '//meta[contains(@name, "og:time")]/@content',
        '//meta[contains(@name, "PubDate")]/@content',
        '//meta[contains(@name, "pubtime")]/@content',
        '//meta[contains(@name, "_pubtime")]/@content',
        '//meta[contains(@name, "apub:time")]/@content',
        '//meta[contains(@pubdate, "pubdate")]/@content',
        '//meta[contains(@name, "publishdate")]/@content',
        '//meta[contains(@name, "PublishDate")]/@content',
        '//meta[contains(@name, "sailthru.date")]/@content',
        '//meta[contains(@itemprop, "dateUpdate")]/@content',
        '//meta[contains(@name, "publication_date")]/@content',
        '//meta[contains(@itemprop, "datePublished")]/@content',
        '//meta[contains(@property, "og:release_date")]/@content',
        '//meta[contains(@name, "article_date_original")]/@content',
        '//meta[contains(@property, "og:published_time")]/@content',
        '//meta[contains(@property, "rnews:datePublished")]/@content',
        '//meta[contains(@name, "OriginalPublicationDate")]/@content',
        '//meta[contains(@name, "weibo: article:create_at")]/@content',
        '//meta[@name="Keywords" and contains(@content, ":")]/@content',
        '//meta[contains(@property, "article:published_time")]/@content'
    ]

    SUPPLEMENT_TIME_XPATHS = [
        '//div[@class="time fix"]//text()',
        '//span[@id="pubtime_baidu"]/text()',
        '//i[contains(@class, "time")]/text()',
        '//span[contains(text(), "时间")]/text()',
        '//div[contains(@class, "time")]//text()',
        '//span[contains(@class, "date")]/text()',
        '//div[contains(@class, "info")]//text()',
        '//span[contains(@class, "time")]/text()',
        '//div[contains(@class, "_time")]/text()',
        '//span[contains(@id, "paperdate")]/text()',
        '//em[contains(@id, "publish_time")]/text()',
        '//time[@data-testid="timestamp"]/@dateTime',
        '//span[contains(@id, "articleTime")]/text()',
        '//span[contains(@class, "pub_time")]/text()',
        '//span[contains(@class, "item-time")]/text()',
        '//span[contains(@class, "publishtime")]/text()',
        '//div[contains(@class, "news_time_source")]/text()'
    ]

    REGEX_TIME = [
        "(\d{1,2}月\d{1,2}日)",
        "(\d{2}年\d{1,2}月\d{1,2}日)",
        "(\d{4}年\d{1,2}月\d{1,2}日)",
        "(\d{2}[-|/|.]\d{1,2}[-|/|.]\d{1,2})",
        "(\d{4}[-|/|.]\d{1,2}[-|/|.]\d{1,2})",
        "(\d{1,2}月\d{1,2}日\s*?[2][0-3]:[0-5]?[0-9])",
        "(\d{1,2}月\d{1,2}日\s*?[0-1]?[0-9]:[0-5]?[0-9])",
        "(\d{2}年\d{1,2}月\d{1,2}日\s*?[2][0-3]:[0-5]?[0-9])",
        "(\d{4}年\d{1,2}月\d{1,2}日\s*?[2][0-3]:[0-5]?[0-9])",
        "(\d{2}年\d{1,2}月\d{1,2}日\s*?[0-1]?[0-9]:[0-5]?[0-9])",
        "(\d{4}年\d{1,2}月\d{1,2}日\s*?[0-1]?[0-9]:[0-5]?[0-9])",
        "(\d{1,2}月\d{1,2}日\s*?[1-24]\d时[0-60]\d分)([1-24]\d时)",
        "(\d{1,2}月\d{1,2}日\s*?[2][0-3]:[0-5]?[0-9]:[0-5]?[0-9])",
        "(\d{4}[-|/|.]\d{1,2}[-|/|.]\d{1,2}\s*?[2][0-3]:[0-5]?[0-9])",
        "(\d{1,2}月\d{1,2}日\s*?[0-1]?[0-9]:[0-5]?[0-9]:[0-5]?[0-9])",
        "(\d{2}[-|/|.]\d{1,2}[-|/|.]\d{1,2}\s*?[2][0-3]:[0-5]?[0-9])",
        "(\d{4}[-|/|.]\d{1,2}[-|/|.]\d{1,2}\s*?[0-1]?[0-9]:[0-5]?[0-9])",
        "(\d{2}[-|/|.]\d{1,2}[-|/|.]\d{1,2}\s*?[0-1]?[0-9]:[0-5]?[0-9])",
        "(\d{2}年\d{1,2}月\d{1,2}日\s*?[1-24]\d时[0-60]\d分)([1-24]\d时)",
        "(\d{4}年\d{1,2}月\d{1,2}日\s*?[1-24]\d时[0-60]\d分)([1-24]\d时)",
        "(\d{2}年\d{1,2}月\d{1,2}日\s*?[2][0-3]:[0-5]?[0-9]:[0-5]?[0-9])",
        "(\d{4}年\d{1,2}月\d{1,2}日\s*?[2][0-3]:[0-5]?[0-9]:[0-5]?[0-9])",
        "(\d{2}年\d{1,2}月\d{1,2}日\s*?[0-1]?[0-9]:[0-5]?[0-9]:[0-5]?[0-9])",
        "(\d{4}年\d{1,2}月\d{1,2}日\s*?[0-1]?[0-9]:[0-5]?[0-9]:[0-5]?[0-9])",
        "(\d{4}[-|/|.]\d{1,2}[-|/|.]\d{1,2}\s*?[1-24]\d时[0-60]\d分)([1-24]\d时)",
        "(\d{2}[-|/|.]\d{1,2}[-|/|.]\d{1,2}\s*?[1-24]\d时[0-60]\d分)([1-24]\d时)",
        "(\d{2}[-|/|.]\d{1,2}[-|/|.]\d{1,2}\s*?[2][0-3]:[0-5]?[0-9]:[0-5]?[0-9])",
        "(\d{4}[-|/|.]\d{1,2}[-|/|.]\d{1,2}\s*?[2][0-3]:[0-5]?[0-9]:[0-5]?[0-9])",
        "(\d{4}[-|/|.]\d{1,2}[-|/|.]\d{1,2}\s*?[0-1]?[0-9]:[0-5]?[0-9]:[0-5]?[0-9])",
        "(\d{2}[-|/|.]\d{1,2}[-|/|.]\d{1,2}\s*?[0-1]?[0-9]:[0-5]?[0-9]:[0-5]?[0-9])"
    ]


@dataclass
class PageContentRuleSet:
    BASE_PUNCTUATION = {'；', "'", ')', '“', '"', '《', '.', '”', ':', '?', '）',
                        ';', '？', '》', '(', '：', '、', '。', '（', '！', '，', '’', '‘', '%', '!', ','}
    USELESS_BLOCK_XPATH = ['//*[@class="UIlTO"]', '//*[@class="video-title"]',
                           '//*[@class="video-from"]', '//*[@class="wap_special"]']
    USELESS_TAG = ['img', 'style', 'script', 'link', 'video',
                   'iframe', 'source', 'picture', 'header', 'blockquote', 'footer']
    TAGS_CAN_BE_REMOVE_IF_EMPTY = [
        'source', 'pre', 'section', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span']
    USELESS_ATTR_KEYWORD = {'video-title', 'register', 'fenxiang', 'header', 'comment', 'foot', 'share', 'top', 'fixedNav', 'login', 'fixed-bar', 'logo',
                            'share', 'contribution', 'copyright', 'copy-right', 'disclaimer', 'recommend', 'related', 'footer', 'comment', 'social', 'submeta', 'report-infor'}
    CONTENT_ARRT_KEYWORD = ['Section0', 'describe_text', 'articleText', 'newscontents',
                            'detail', 'content', 'article', 'news_txt', 'post_text', 'markdown']
    DELETE_SENTENCE = {'用微信扫码二维码', '分享至好友和朋友圈', '分享至', '0'}
