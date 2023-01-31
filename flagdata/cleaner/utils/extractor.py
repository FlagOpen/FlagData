# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from .ruleset import PageTitleRuleSet, PublishTimeRuleSet, PageContentRuleSet
from .time_formatter import return_format_datetime
from lxml.html import etree, HtmlElement, fromstring
import re
from html import unescape
import math
import copy
import unicodedata
from collections import OrderedDict


class BasicExtractor:
    def __init__(self):
        self.save_key = str()
        self.need_clean = False

    @classmethod
    def get_subclasses(self):
        class_dict = OrderedDict()
        classes = self.__subclasses__()
        for subclass in classes:
            class_dict[subclass.__name__] = subclass
        return class_dict

    def extract(self, content, html_tree):
        raise NotImplementedError


class TitleExtractor(BasicExtractor):
    def __init__(self, save_key="pageTitle"):
        super().__init__()
        self.save_key = save_key
        self.rules = PageTitleRuleSet()

    def extract_title_from_meta(self, html_tree):
        for each_meta_xpath in self.rules.META_TITLE_XPATHS:
            title_temp = html_tree.xpath(each_meta_xpath)
            if title_temp:
                return ''.join(title_temp[0]).strip()
        return ''

    def extract_title_from_title_tag(self, html_tree):
        titles = html_tree.xpath(self.rules.TITLE_TAG_XPATH)
        if titles:
            handled_titles = sum(
                [[each_block for each_block in re.split(r'[-_|]', each_temp) if len(each_block) >= 3]
                    for each_temp in titles], []
            )
            try:
                return sorted(handled_titles, key=len)[-1].strip()
            except:
                return titles[0].strip()
        return ''

    def extract_title_from_other_tag(self, html_tree):
        title_list = []
        for each_xpath in self.rules.SUPPLEMENT_TITLE_XPATHS:
            title_temp = html_tree.xpath(each_xpath)
            if title_temp:
                title_list += [x for x in title_temp if len(x.strip()) > 0]
        if len(title_list) > 0:
            return [z for z in sorted(title_list, key=len) if '{{' not in z][-1].strip()
        return ''

    def extract(self, content, html_tree):
        page_title = ""
        meta_title = self.extract_title_from_meta(html_tree)
        if meta_title:
            page_title = unescape(meta_title)
        else:
            tag_title = self.extract_title_from_title_tag(html_tree)
            if tag_title:
                page_title = unescape(tag_title)
            else:
                other_xpath_title = self.extract_title_from_other_tag(
                    html_tree)
                if other_xpath_title:
                    page_title = unescape(other_xpath_title)
        extracted_data = {self.save_key: page_title}
        return extracted_data


class TimeExtractor(BasicExtractor):
    def __init__(self, save_key="pagePublishTime"):
        super().__init__()
        self.save_key = save_key
        self.rules = PublishTimeRuleSet()

    @staticmethod
    def get_valid_length_time(publish_time_list):
        if len(publish_time_list) > 0:
            length_valid_publish_time = [
                y for y in publish_time_list if len(y) >= 9]
            publish_time = length_valid_publish_time[0] \
                if length_valid_publish_time else publish_time_list[0]
        else:
            publish_time = ""
        return publish_time

    def extract_time_from_meta(self, html_tree):
        for each_meta_xpath in self.rules.META_TIME_XPATHS:
            publish_time_temp = html_tree.xpath(each_meta_xpath)
            if publish_time_temp:
                return ''.join(publish_time_temp).strip()
        return ''

    def extract_time_from_other_tag(self, html_tree):
        publish_time_list = []
        for each_xpath in self.rules.SUPPLEMENT_TIME_XPATHS:
            publish_time_temp = ''.join(html_tree.xpath(each_xpath)).strip()
            if publish_time_temp:
                publish_time_list += [re.findall(x, publish_time_temp)[0] for x in self.rules.REGEX_TIME
                                      if re.findall(x, publish_time_temp)]
        return self.get_valid_length_time(publish_time_list)

    def extract_time_from_html(self, content):
        publish_time_list = []
        for each_time_regex in self.rules.REGEX_TIME:
            publish_time_temp = re.findall(each_time_regex, content)
            if publish_time_temp:
                publish_time_list += publish_time_temp
        return self.get_valid_length_time(publish_time_list)

    def extract(self, content, html_tree):
        publish_time = ""
        meta_date = self.extract_time_from_meta(html_tree)
        if meta_date:
            publish_time = return_format_datetime(meta_date)
        else:
            tag_date = self.extract_time_from_other_tag(html_tree)
            if tag_date:
                publish_time = return_format_datetime(tag_date)
            else:
                html_date = self.extract_time_from_html(content)
                if html_date:
                    publish_time = return_format_datetime(html_date)
        extracted_data = {self.save_key: publish_time}
        return extracted_data


class ContentExtractor(BasicExtractor):
    def __init__(self, save_key="pageContent"):
        super().__init__()
        self.save_key = save_key
        self.need_clean = True  # to indicate page content for cleaning
        self.rules = PageContentRuleSet()

    @staticmethod
    def remove_space(article):
        return re.sub(' {3,}', '', article).strip()

    @staticmethod
    def remove_line_feed(article):
        return re.sub(r'(\n\s*)+\n+', '\n', article).strip()

    @staticmethod
    def remove_html_tag(article):
        regex_html = re.sub(r'\{\{.*?\}\}', '', re.sub(
            r'<(\S*?)[^>]*>.*?|<.*?/>', '', article, 2000, flags=re.I).strip()).strip()
        return regex_html

    def tag_node_iter(self, element):
        yield element
        for sub_element in element:
            if isinstance(sub_element, HtmlElement):
                yield from self.tag_node_iter(sub_element)

    @staticmethod
    def judge_tag_empty(tag):
        return not tag.getchildren() and not tag.text

    @staticmethod
    def delete_tag(tag):
        parent = tag.getparent()
        if parent is not None:
            parent.remove(tag)

    @staticmethod
    def drop_tag(tag):
        parent = tag.getparent()
        if parent is not None:
            tag.drop_tag()

    def tag_normalize(self, element):
        etree.strip_elements(element, self.rules.USELESS_TAG)
        for node in self.tag_node_iter(element):
            if node.tag.lower() in self.rules.TAGS_CAN_BE_REMOVE_IF_EMPTY and self.judge_tag_empty(node):
                self.delete_tag(node)
            if node.tag.lower() == 'p':
                etree.strip_tags(node, 'span')
                etree.strip_tags(node, 'strong')
            if node.tag.lower() == 'div' and not node.getchildren():
                node.tag = 'p'
            if node.tag.lower() == 'span' and not node.getchildren():
                node.tag = 'p'
            if node.tag.lower() == 'p' and not node.xpath('.//img'):
                if not (node.text and node.text.strip()):
                    self.drop_tag(node)
            class_name = node.get('class')
            if class_name:
                judge_list = [
                    each_attr for each_attr in self.rules.USELESS_ATTR_KEYWORD if each_attr in class_name]
                if len(judge_list) > 0:
                    self.delete_tag(node)
                    break
        return element

    def delete_useless_node_block(self, element):
        for noise_xpath in self.rules.USELESS_BLOCK_XPATH:
            nodes = element.xpath(noise_xpath)
            for node in nodes:
                self.delete_tag(node)
        return element

    def increase_tag_weight(self, ti, element):
        tag_class = element.get('class', '')
        regex_high_weight_keyword = re.compile(
            '|'.join(self.rules.CONTENT_ARRT_KEYWORD), flags=re.I)
        if regex_high_weight_keyword.search(tag_class):
            return 2 * ti
        return ti

    def get_all_text_of_element(self, element_list, element_text_cache_dict):
        if not isinstance(element_list, list):
            element_list = [element_list]
        text_list = []
        for element in element_list:
            element_flag = element.getroottree().getpath(element)
            if element_flag in element_text_cache_dict:
                text_list = element_text_cache_dict[element_flag]
            else:
                element_text_list = []
                for text in element.xpath('.//text()'):
                    text = text.strip()
                    if not text:
                        continue
                    clear_text = re.sub(' +', ' ', text, flags=re.S)
                    if text not in self.rules.DELETE_SENTENCE:
                        if '{{' not in text:
                            element_text_list.append(
                                clear_text.replace('\n', ''))
                element_text_cache_dict[element_flag] = element_text_list
                text_list.extend(element_text_list)
        return text_list

    def calc_text_density(self, element, element_text_cache_dict):
        ti_text = '\n'.join(self.get_all_text_of_element(
            element, element_text_cache_dict))
        ti = len(ti_text)
        ti = self.increase_tag_weight(ti, element)
        lti = len(''.join(self.get_all_text_of_element(
            element.xpath('.//a'), element_text_cache_dict)))
        tgi = len(element.xpath('.//*'))
        ltgi = len(element.xpath('.//a'))
        if (tgi - ltgi) == 0:
            return {'density': 0, 'ti_text': ti_text, 'ti': ti, 'lti': lti, 'tgi': tgi, 'ltgi': ltgi}
        density = (ti - lti) / (tgi - ltgi)
        return {'density': density, 'ti_text': ti_text, 'ti': ti, 'lti': lti, 'tgi': tgi, 'ltgi': ltgi}

    def calc_sbdi(self, text, ti, lti):
        sbi = 0
        for char in text:
            if char in self.rules.BASE_PUNCTUATION:
                sbi += 1
        sbdi = (ti - lti) / (sbi + 1)
        return sbdi or 1

    def original_parse(self, content, html_tree):
        all_p_content = html_tree.xpath(
            '//*[contains(@id, "_content") or contains(@class, "content") or contains(@class, "Content")]/p//text()')
        if all_p_content:
            eatract_result = self.remove_line_feed(''.join(all_p_content))
        else:
            block_content = html_tree.xpath(
                '//*[contains(@id, "_content") or contains(@class, "content") or contains(@class, "Content")]//text()')
            if block_content:
                eatract_result = self.remove_line_feed(''.join(block_content))
            else:
                eatract_result = self.remove_space(
                    self.remove_line_feed(self.remove_html_tag(content)))
        return eatract_result

    def extract(self, content, html_tree):
        page_content = ""
        node_info_dict = {}
        element_text_cache_dict = {}
        if 'isBaiJiaHao' in content:
            page_content = unescape(''.join(html_tree.xpath(
                'string(//*[@id="ssr-content"])'))).replace('百度首页登录', '')
            page_content = re.sub(
                '举报/反馈设为首页© Baidu 使用百度前必读 意见反馈 京ICP证\d+号 京公网安备\d+号', '', page_content).strip()
        else:
            try:
                # use deepcopy to keep integrity of original tree
                tmp_html_tree = copy.deepcopy(html_tree)
                pre_element_nodes = self.tag_normalize(tmp_html_tree)
                delete_noise_node = self.delete_useless_node_block(
                    pre_element_nodes)
                page_body = delete_noise_node.xpath('//body')[0]
                for node in self.tag_node_iter(page_body):
                    node_hash = hash(node)
                    density_info = self.calc_text_density(
                        node, element_text_cache_dict)
                    text_density = density_info['density']
                    ti_text = density_info['ti_text']
                    text_tag_count = len(node.xpath('.//p'))
                    sbdi = self.calc_sbdi(
                        ti_text, density_info['ti'], density_info['lti'])
                    node_info = {'ti': density_info['ti'], 'lti': density_info['lti'], 'tgi': density_info['tgi'], 'ltgi': density_info['ltgi'],
                                 'node': node, 'density': text_density, 'text': ti_text, 'text_tag_count': text_tag_count, 'sbdi': sbdi}
                    node_info_dict[node_hash] = node_info
                for node_hash, node_info in node_info_dict.items():
                    score = node_info['density'] * math.log10(
                        node_info['text_tag_count'] + 2) * math.log(node_info['sbdi'])
                    node_info_dict[node_hash]['score'] = score
                page_content = unescape(sorted(node_info_dict.items(
                ), key=lambda x: x[1]['score'], reverse=True)[0][1]['text'].strip())
            except:
                page_content = self.original_parse(content, html_tree)
        if len(page_content) < 1:
            page_content = self.original_parse(content, html_tree)
        extracted_data = {self.save_key: page_content}
        return extracted_data


class ExtractorPipeline:
    def __init__(self, *args):
        self.extractors = args

    def extract(self, content):
        # do basic normalization
        content = unicodedata.normalize('NFC', content)
        content = re.sub('</?br.*?>', '', content)
        html_tree = fromstring(content)
        gathered_extracted = dict()
        extracted_content = ""
        for extractor in self.extractors:
            extracted = extractor.extract(content, html_tree)
            if extractor.need_clean is True:
                extracted_content = extracted.get(extractor.save_key)
            gathered_extracted.update(extracted)
        return gathered_extracted, extracted_content
