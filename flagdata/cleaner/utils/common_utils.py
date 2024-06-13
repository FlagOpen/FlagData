import re
import unicodedata


def remove_control_chars(text):
    return ''.join(ch for ch in text if (unicodedata.category(ch) != 'Cc' or ch == '\n'))  #


def remove_extraspace(text):
    normalized_string = re.sub(r'[\xa0\u3000]+', ' ', text)
    normalized_string = re.sub(r'[\x00-\x09\x0B-\x0C\x0E-\x1F\x7F-\x9F]+', '', normalized_string)
    return normalized_string


def end_clip(text):
    chinese_punctuation = "。！？"
    english_punctuation = ".?!"
    all_punctuation = chinese_punctuation + english_punctuation
    if text[-1] in all_punctuation:
        return text
    else:
        match = re.search(r'[{}]'.format(re.escape(all_punctuation)), text[::-1])
        if match:
            return text[:len(text) - match.start()]
        else:
            return ""

    # 删除含有特定内容的句子


def remove_specific_patterns(text):
    for pattern in ["@article"]:
        text = re.sub(r'^.*' + re.escape(pattern) + r'.*$\n?', '', text, flags=re.MULTILINE)
    return text.strip()

    # 删除多行空格


def remove_unwanted_lines(text):
    return re.sub(r'\n\n+', '\n', text)

    # 删除数字比率大于0.5的文档


def drop_doc_below_ratio(text, ratio=0.5):
    def count_characters(doc):
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', doc)
        chinese_count = sum(len(char) for char in chinese_chars)
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        word_count = len(words)
        return chinese_count + word_count

    numbers = re.findall(r'\d+\.\d+|\d+', text)

    if count_characters(text) != 0:
        score = len(numbers) / count_characters(text)
        if score > ratio:
            return True
        else:
            return False
    else:
        return False


def drop_docs_exceeding_newline_proportion(text, max_newline_proportion=0.25):
    def count_newlines(doc):
        return doc.count('\n')

    def count_characters(doc):
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', doc)
        chinese_count = sum(len(char) for char in chinese_chars)
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        word_count = len(words)
        return chinese_count + word_count

    newline_count = count_newlines(text)
    char_count = count_characters(text)

    if char_count != 0:
        newline_proportion = newline_count / char_count
        if newline_proportion <= max_newline_proportion:
            return False
        else:
            return True
    else:
        return False
